/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
*/

/*! \file
 * Implementation of the GPUSPH core
 */

#include <cfloat> // FLT_EPSILON

#include <unistd.h> // getpid()
#include <sys/mman.h> // shm_open()/shm_unlink()
#include <fcntl.h> // O_* macros when opening files

#define GPUSPH_MAIN
#include "particledefine.h"
#undef GPUSPH_MAIN

// NOTE: including GPUSPH.h before particledefine.h does not compile.
// This inclusion problem should be solved
#include "GPUSPH.h"

// HostBuffer
#include "hostbuffer.h"

// GPUWorker
#include "GPUWorker.h"

// div_up
#include "utils.h"

// HotFile
#include "HotFile.h"

using namespace std;

// an empty set of PostProcessEngines, to be used when we want to save
// the particle system without running post-processing filters
// (e.g. when inspecting the particle system before each forces computation)
static const PostProcessEngineSet noPostProcess{};

GPUSPH* GPUSPH::getInstance() {
	// guaranteed to be destroyed; instantiated on first use
	static GPUSPH instance;
	// return a reference, not just a pointer
	return &instance;
}

GPUSPH::GPUSPH() :
	clOptions(NULL),
	gdata(NULL),
	problem(NULL),

	m_totalPerformanceCounter(NULL),
	m_intervalPerformanceCounter(NULL),
	m_multiNodePerformanceCounter(NULL),

	m_info_stream_name(),
	m_info_stream(NULL),

	m_rcBitmap(NULL),
	m_rcNotified(NULL),
	m_rcAddrs(NULL),

	m_peakParticleSpeed(0.0),
	m_peakParticleSpeedTime(0.0),

	initialized(false),
	repacked(false)
{
	openInfoStream();
	resetCommandTimes();
}

GPUSPH::~GPUSPH() {
	if (gdata->debug.benchmark_command_runtimes)
		showCommandTimes();

	closeInfoStream();
	// it would be useful to have a "fallback" deallocation but we have to check
	// that main did not do that already
	if (initialized) finalize();
}

void GPUSPH::resetCommandTimes()
{
	for (CommandName cmd = IDLE; cmd < NUM_COMMANDS; cmd = CommandName(cmd+1))
	{
		max_cmd_time[cmd] = cmd_time_duration(0);
		tot_cmd_time[cmd] = cmd_time_duration(0);
		cmd_calls[cmd] = 0;
	}
}

void GPUSPH::showCommandTimes()
{
	cout << "CMDTIMES:COMMAND\tCMD_NUM\tCALLS\tMAX(ms)\tTOT(ms)\n";
	for (CommandName cmd = IDLE; cmd < NUM_COMMANDS; cmd = CommandName(cmd+1))
	{
		if (!cmd_calls[cmd])
			continue;
		std::chrono::duration<double, std::milli> max_ms = max_cmd_time[cmd];
		std::chrono::duration<double, std::milli> tot_ms = tot_cmd_time[cmd];
		cout << "CMDTIMES:" << command_name[cmd] << "\t" << cmd << "\t"
			<< cmd_calls[cmd] << "\t"
			<< max_ms.count() << "\t" << tot_ms.count() << "\n";
	}
}

void GPUSPH::openInfoStream() {
	stringstream ss;
	ss << "GPUSPH-" << getpid();
	m_info_stream_name = ss.str();
	m_info_stream = NULL;
	int ret = shm_open(m_info_stream_name.c_str(), O_RDWR | O_CREAT, S_IRWXU | S_IRGRP | S_IROTH);
	if (ret < 0) {
		cerr << "WARNING: unable to open info stream " << m_info_stream_name << endl;
		return;
	}
	m_info_stream = fdopen(ret, "w");
	if (!m_info_stream) {
		cerr << "WARNING: unable to fdopen info stream " << m_info_stream_name << endl;
		close(ret);
		shm_unlink(m_info_stream_name.c_str());
		return;
	}
	cout << "Info stream: " << m_info_stream_name << endl;
	fputs("Initializing ...\n", m_info_stream);
	fflush(m_info_stream);
	fseek(m_info_stream, 0, SEEK_SET);
}

void GPUSPH::closeInfoStream() {
	if (m_info_stream) {
		shm_unlink(m_info_stream_name.c_str());
		fclose(m_info_stream);
		m_info_stream = NULL;
	}
}

bool GPUSPH::initialize(GlobalData *_gdata) {

	printf("Initializing...\n");

	gdata = _gdata;
	clOptions = gdata->clOptions;
	problem = gdata->problem;

	// For the new problem interface (compute worldorigin, init ODE, etc.)
	// In all cases, also runs the checks for dt, neib list size, etc
	// and creates the problem dir
	if (!problem->initialize()) {
		printf("Problem initialization failed. Aborting...\n");
		return false;
	}

	// sets the correct viscosity coefficient according to the one set in SimParams
	setViscosityCoefficient();

	m_totalPerformanceCounter = new IPPSCounter();
	m_intervalPerformanceCounter = new IPPSCounter();
	// only init if MULTI_NODE
	if (MULTI_NODE)
		m_multiNodePerformanceCounter = new IPPSCounter();

	// utility pointer
	SimParams *_sp = gdata->problem->simparams();

	// copy the options passed by command line to GlobalData
	if (isfinite(clOptions->tend))
		_sp->tend = clOptions->tend;

	if (clOptions->repack_maxiter)
		_sp->repack_maxiter = clOptions->repack_maxiter;

	// Update GlobalData's maxiter based on the run mode
	if (gdata->run_mode == REPACK)
		gdata->maxiter = _sp->repack_maxiter;
	else if (clOptions->maxiter > 0)
		gdata->maxiter = clOptions->maxiter;

	// update the GlobalData copies of the sizes of the domain
	gdata->worldOrigin = make_float3(problem->get_worldorigin());
	gdata->worldSize = make_float3(problem->get_worldsize());

	// get the grid size
	gdata->gridSize = problem->get_gridsize();

	// compute the number of cells, in ulong first (an overflow would make the comparison with MAX_CELLS pointless)
	ulong longNGridCells = (ulong) gdata->gridSize.x * gdata->gridSize.y * gdata->gridSize.z;
	if (longNGridCells > MAX_CELLS) {
		printf("FATAL: cannot handle %lu > %u cells\n", longNGridCells, MAX_CELLS);
		return false;
	}
	gdata->nGridCells = (uint)longNGridCells;

	// get the cell size
	gdata->cellSize = make_float3(problem->get_cellsize());

	printf(" - World origin: %g , %g , %g\n", gdata->worldOrigin.x, gdata->worldOrigin.y, gdata->worldOrigin.z);
	printf(" - World size:   %g x %g x %g\n", gdata->worldSize.x, gdata->worldSize.y, gdata->worldSize.z);
	printf(" - Cell size:    %g x %g x %g\n", gdata->cellSize.x, gdata->cellSize.y, gdata->cellSize.z);
	printf(" - Grid size:    %u x %u x %u (%s cells)\n", gdata->gridSize.x, gdata->gridSize.y, gdata->gridSize.z, gdata->addSeparators(gdata->nGridCells).c_str());
	printf(" - Cell linearization: %s,%s,%s\n", STR(COORD1), STR(COORD2), STR(COORD3));
	printf(" - Dp:   %g\n", gdata->problem->m_deltap);
	printf(" - R0:   %g\n", gdata->problem->physparams()->r0);


	// initial dt (or, just dt in case adaptive is disabled)
	gdata->dt = _sp->dt;
	gdata->dtadapt = _sp->simflags & ENABLE_DTADAPT;
	if (isfinite(gdata->clOptions->dt)) {
		float new_dt = gdata->clOptions->dt;
		printf("Time step %g specified on the command-line overrides %g\n", new_dt, gdata->dt);
		if (gdata->dtadapt)
			printf("\tfixed time-stepping will be used even though adapting time-stepping is enabled\n");
		gdata->dt = new_dt;
		gdata->dtadapt = false;
	}

	printf("Generating problem particles...\n");

	ifstream *hot_in = NULL;
	HotFile **hf = NULL;
	uint hot_nrank = 1;

	if (clOptions->resume_fname.empty()) {
		// get number of particles from problem file
		gdata->totParticles = problem->fill_parts();
	} else {
		gdata->totParticles = problem->fill_parts(false);
		// get number of particles from hot file
		struct stat statbuf;
		ostringstream err_msg;
		// check if the hotfile is part of a multi-node simulation
		size_t found = clOptions->resume_fname.find_last_of("/");
		if (found == string::npos)
			found = 0;
		else
			found++;
		string resume_file = clOptions->resume_fname.substr(found);
		string pre_fname, post_fname;
		// this is the case if the filename is of the form "hot_nX.Y_Z.bin" where X,Y,Z are integers
		if(resume_file.compare(0,5,"hot_n") == 0) {
			// get number of ranks from previous simulation
			pre_fname = clOptions->resume_fname.substr(0, found+5);
			found = resume_file.find_first_of(".")+1;
			size_t found2 = resume_file.find_first_of("_", 5);
			if (found == string::npos || found2 == string::npos || found > found2) {
				err_msg << "Malformed Hot start filename: " << resume_file << "\nNeeds to be of the form \"hot_nX.Y_ZZZZZ.bin\"";
				throw runtime_error(err_msg.str());
			}
			istringstream (resume_file.substr(found,found2-found)) >> hot_nrank;
			post_fname = resume_file.substr(found-1);
			cout << "Hot start has been written from a multi-node simulation with " << hot_nrank << " processes" << endl;
		}
		if(resume_file.compare(0,8,"repack_n") == 0) {
			// get number of ranks from previous simulation
			pre_fname = clOptions->resume_fname.substr(0, found+8);
			found = resume_file.find_first_of(".")+1;
			size_t found2 = resume_file.find_first_of("_", 8);
			if (found == string::npos || found2 == string::npos || found > found2) {
				err_msg << "Malformed Repack filename: " << resume_file << "\nNeeds to be of the form \"repack_nX.Y_ZZZZZ.bin\"";
				throw runtime_error(err_msg.str());
			}
			istringstream (resume_file.substr(found,found2-found)) >> hot_nrank;
			post_fname = resume_file.substr(found-1);
			cout << "Hot start has been written from a multi-node simulation with " << hot_nrank << " processes" << endl;
		}
		// allocate hot file arrays and file pointers
		hot_in = new ifstream[hot_nrank];
		hf = new HotFile*[hot_nrank];
		gdata->totParticles = 0;
		for (uint i = 0; i < hot_nrank; i++) {
			ostringstream fname;
			if (hot_nrank == 1)
				fname << clOptions->resume_fname;
			else
				fname << pre_fname << i << post_fname;
			cout << "Hot starting from " << fname.str() << "..." << endl;
			if (stat(fname.str().c_str(), &statbuf)) {
				// stat failed
				err_msg << "Hot start file " << fname.str() << " not found";
				throw runtime_error(err_msg.str());
			}
			/* enable automatic exception handling on failure */
			hot_in[i].exceptions(ifstream::failbit | ifstream::badbit);
			hot_in[i].open(fname.str().c_str());
			hf[i] = new HotFile(hot_in[i], gdata);
			hf[i]->readHeader(gdata->totParticles, gdata->problem->simparams()->numOpenBoundaries);
		}
	}

	// allocate the particles of the *whole* simulation

	// Allocate internal storage for moving bodies
	problem->allocate_bodies_storage();

	// the number of allocated particles will be bigger, to be sure it can contain particles being created
	// WARNING: particle creation in inlets also relies on this, do not disable if using inlets
	// round up to multiple of 4 to improve reductions' performances
	gdata->allocatedParticles = round_up(problem->max_parts(gdata->totParticles), 4U);

	// generate planes
	problem->copy_planes(gdata->s_hPlanes);

	{
		size_t numPlanes = gdata->s_hPlanes.size();
		if (numPlanes > 0) {
			if (!(problem->simparams()->simflags & ENABLE_PLANES))
				throw invalid_argument("planes present but ENABLE_PLANES not specified in framework flags");
			if (numPlanes > MAX_PLANES) {
				stringstream err; err << "FATAL: too many planes (" <<
					numPlanes << " > " << MAX_PLANES;
				throw runtime_error(err.str().c_str());
			}
		}
		if ((problem->simparams()->simflags & ENABLE_DEM) &&
			(problem->get_dem() == NULL))
		{
			throw invalid_argument("DEM support enabled, but no DEM defined!");
		}
	}

	// Create the Writers according to the WriterType
	// Should be done after the last fill operation
	createWriter();

	// allocate aux arrays for rollCallParticles()
	m_rcBitmap = (bool*) calloc( sizeof(bool) , gdata->allocatedParticles );
	m_rcNotified = (bool*) calloc( sizeof(bool) , gdata->allocatedParticles );
	m_rcAddrs = (uint*) calloc( sizeof(uint) , gdata->allocatedParticles );

	if (!m_rcBitmap) {
		fprintf(stderr,"FATAL: failed to allocate roll call bitmap\n");
		exit(1);
	}

	if (!m_rcNotified) {
		fprintf(stderr,"FATAL: failed to allocate roll call notified map\n");
		exit(1);
	}

	if (!m_rcAddrs) {
		fprintf(stderr,"FATAL: failed to allocate roll call particle address space\n");
		exit(1);
	}


	printf("Allocating shared host buffers...\n");
	// allocate cpu buffers, 1 per process
	size_t totCPUbytes = allocateGlobalHostBuffers();

	// pretty print
	printf("  allocated %s on host for %s particles (%s active)\n",
		gdata->memString(totCPUbytes).c_str(),
		gdata->addSeparators(gdata->allocatedParticles).c_str(),
		gdata->addSeparators(gdata->totParticles).c_str() );

	/* Now we either copy particle data from the Problem to the GPUSPH buffers,
	 * or, if it was requested, we load buffers from a HotStart file
	 */
	/* TODO FIXME copying data from the Problem doubles the host memory
	 * requirements, find some smart way to have the host fill the shared
	 * buffer directly.
	 */
	bool resumed = false;

	// initialize the buffers
	if (clOptions->resume_fname.empty()) {
		gdata->s_hBuffers.set_state_on_write("problem init");
		printf("Copying the particles to shared arrays...\n");
		printf("---\n");
		problem->copy_to_array(gdata->s_hBuffers);
		if (gdata->run_mode != REPACK) {
			if (problem->simparams()->turbmodel == KEPSILON)
				problem->init_keps(gdata->s_hBuffers, gdata->totParticles);
			problem->initializeParticles(gdata->s_hBuffers, gdata->totParticles);
		}
		printf("---\n");
	} else {
		gdata->t = hf[0]->get_t();
		if (gdata->t > 0) {
			gdata->s_hBuffers.set_state_on_write("resumed");
			gdata->iterations = hf[0]->get_iterations();
			gdata->dt = hf[0]->get_dt();
			for (uint i = 0; i < hot_nrank; i++) {
				hf[i]->load();
#if 0
				// for debugging, enable this and inspect contents
				const float4 *pos = gdata->s_hBuffers.getConstData<BUFFER_POS>();
				const particleinfo *info = gdata->s_hBuffers.getConstData<BUFFER_INFO>();
#endif
				hot_in[i].close();
				cerr << "Successfully restored hot start file " << i+1 << " / " << hot_nrank << endl;
				cerr << *hf[i];
			}

			cerr << "Restarting from t=" << gdata->t
				<< ", iteration=" << gdata->iterations
				<< ", dt=" << gdata->dt << endl;
			// warn about possible discrepancies in case of ODE objects
			if (problem->simparams()->numbodies) {
				cerr << "WARNING: simulation has rigid bodies and/or moving boundaries, resume will not give identical results" << endl;
			}
		} else {
			gdata->s_hBuffers.set_state_on_write("resumed from repack");
			gdata->iterations = 0;
			gdata->t = 0;
			for (uint i = 0; i < hot_nrank; i++) {
				hf[i]->load();
#if 0
				// for debugging, enable this and inspect contents
				const float4 *pos = gdata->s_hBuffers.getConstData<BUFFER_POS>();
				const particleinfo *info = gdata->s_hBuffers.getConstData<BUFFER_INFO>();
#endif
				hot_in[i].close();
				cerr << "Successfully restored repack file " << i+1 << " / " << hot_nrank << endl;
				cerr << *hf[i];
			}

			// Reset the arrays
			problem->resetBuffers(gdata->s_hBuffers, gdata->totParticles);

			cerr << "Restarting from a repack file"
				<< ", dt=" << gdata->dt << endl;
		}
		delete[] hf;
		delete[] hot_in;
		resumed = true;
	}
	gdata->s_hBuffers.clear_pending_state();

	cout << "RB First/Last Index:\n";
	for (uint i = 0 ; i < problem->simparams()->numforcesbodies; ++i) {
			cout << "\t" << gdata->s_hRbFirstIndex[i] << "\t" << gdata->s_hRbLastIndex[i] << endl;
	}

	// Initialize potential joints if there are floating bodies
	if (problem->simparams()->numbodies)
		problem->initializeObjectJoints();

	// Perform all those operations that require accessing the particles (e.g. find least obj id,
	// count fluid parts per cell, etc.)
	prepareProblem();

	// let the Problem partition the domain (with global device ids)
	// NOTE: this could be done before fill_parts(), as long as it does not need knowledge about the fluid, but
	// not before allocating the host buffers
	if (MULTI_DEVICE) {
		printf("Splitting the domain in %u partitions...\n", gdata->totDevices);
		// fill the device map with numbers from 0 to totDevices
		gdata->problem->fillDeviceMap();
		// here it is possible to save the device map before the conversion
		// gdata->saveDeviceMapToFile("linearIdx");
		if (MULTI_NODE) {
			// make the numbers globalDeviceIndices, with the least 3 bits reserved for the device number
			gdata->convertDeviceMap();
			// here it is possible to save the converted device map
			// gdata->saveDeviceMapToFile("");
		}
		printf("Striping is:  %s\n", (gdata->clOptions->striping ? "enabled" : "disabled") );
		printf("GPUDirect is: %s\n", (gdata->clOptions->gpudirect ? "enabled" : "disabled") );
		printf("MPI transfers are: %s\n", (gdata->clOptions->asyncNetworkTransfers ? "ASYNCHRONOUS" : "BLOCKING") );
	}

	// initialize CGs (or, the problem could directly write on gdata)
	if (gdata->problem->simparams()->numbodies > 0) {
		gdata->problem->get_bodies_cg();
	}

	if (!resumed && _sp->sph_formulation == SPH_GRENIER)
		problem->init_volume(gdata->s_hBuffers, gdata->totParticles);

	if (!resumed && (_sp->simflags & ENABLE_INTERNAL_ENERGY))
		problem->init_internal_energy(gdata->s_hBuffers, gdata->totParticles);

	if (!resumed && _sp->turbmodel > ARTIFICIAL)
		problem->init_turbvisc(gdata->s_hBuffers, gdata->totParticles);

	if (!resumed && _sp->rheologytype == GRANULAR)
		problem->init_effpres(gdata->s_hBuffers, gdata->totParticles);

	/* When starting a simulation with open boundaries, we need to
	 * initialize the array of the next ID for generated particles,
	 * and count the total number of open boundary vertices.
	 */
	if (_sp->simflags & ENABLE_INLET_OUTLET)
		gdata->numOpenVertices = initializeNextIDs(resumed);

	if (MULTI_DEVICE) {
		printf("Sorting the particles per device...\n");
		sortParticlesByHash();
	} else {
		// if there is something more to do, encapsulate in a dedicated method please
		gdata->s_hStartPerDevice[0] = 0;
		gdata->s_hPartsPerDevice[0] = gdata->processParticles[0] =
				gdata->totParticles;
	}

	for (uint d=0; d < gdata->devices; d++)
		printf(" - device at index %u has %s particles assigned and offset %s\n",
			d, gdata->addSeparators(gdata->s_hPartsPerDevice[d]).c_str(), gdata->addSeparators(gdata->s_hStartPerDevice[d]).c_str());

	// TODO this is where we would instance the Integrator class
	// for the time being, we will instead call a function that initializes
	// our own CommandSequences
	// Note that if gdata->run_mode == REPACK, the REPACKING_INTEGRATOR will be instantiated instead
	integrator = Integrator::instance(PREDITOR_CORRECTOR, gdata);

	// new Synchronizer; it will be waiting on #devices+1 threads (GPUWorkers + main)
	gdata->threadSynchronizer = new Synchronizer(gdata->devices + 1);

	printf("Starting workers...\n");

	// allocate workers
	gdata->GPUWORKERS.reserve(gdata->devices);
	for (uint d=0; d < gdata->devices; d++)
		gdata->GPUWORKERS.push_back( make_shared<GPUWorker>(gdata, d) );

	// actually start the threads
	for (uint d = 0; d < gdata->devices; d++)
		gdata->GPUWORKERS[d]->run_worker(); // begin of INITIALIZATION ***

	// The following barrier waits for GPUworkers to complete CUDA init, GPU allocation, subdomain and devmap upload

	gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***

	if (!gdata->keep_going)
		return false;

	// peer accessibility is checked and set in the initialization phase
	if (MULTI_GPU)
		printDeviceAccessibilityTable();

	return (initialized = true);
}

bool GPUSPH::finalize() {
	// TODO here, when there will be the Integrator
	// delete Integrator

	printf("Deallocating...\n");

	// stuff for rollCallParticles()
	free(m_rcBitmap);
	free(m_rcNotified);
	free(m_rcAddrs);

	// workers
	gdata->GPUWORKERS.clear();

	// Synchronizer
	delete gdata->threadSynchronizer;

	// host buffers
	deallocateGlobalHostBuffers();

	Writer::Destroy();

	// ...anything else?

	delete m_totalPerformanceCounter;
	delete m_intervalPerformanceCounter;
	if (m_multiNodePerformanceCounter)
		delete m_multiNodePerformanceCounter;

	initialized = false;

	return true;
}

template<>
void GPUSPH::runCommand<END_OF_INIT>(CommandStruct const& cmd)
{
	printf("Entering the main %s cycle\n", gdata->run_mode_desc());

	//  IPPS counter does not take the initial uploads into consideration
	m_totalPerformanceCounter->start();
	m_intervalPerformanceCounter->start();
	if (MULTI_NODE)
		m_multiNodePerformanceCounter->start();

	// write some info. This could replace "Entering the main simulation cycle"
	printStatus();
}

template<>
void GPUSPH::runCommand<END_OF_REPACKING>(CommandStruct const& cmd)
{
	gdata->t = -1;
	gdata->iterations = 0;
	repacked = true;
	check_write(true);
	gdata->keep_going = false;
	printf("Repacking algorithm is finished\n");
}

template<>
void GPUSPH::runCommand<TIME_STEP_PRELUDE>(CommandStruct const& cmd)
{
	printStatus(m_info_stream);
}

template<>
void GPUSPH::runCommand<HANDLE_HOTWRITE>(CommandStruct const& cmd)
{
	if (Writer::HotWriterPending())
		saveParticles(noPostProcess, "step n", HotWriteFlags);
}

template<>
void GPUSPH::runCommand<TIME_STEP_EPILOGUE>(CommandStruct const& cmd)
{
	// increase counters
	gdata->iterations++;
	m_totalPerformanceCounter->incItersTimesParts( gdata->processParticles[ gdata->mpi_rank ] );
	m_intervalPerformanceCounter->incItersTimesParts( gdata->processParticles[ gdata->mpi_rank ] );
	if (MULTI_NODE)
		m_multiNodePerformanceCounter->incItersTimesParts( gdata->totParticles );
	// to check, later, that the simulation is actually progressing
	double previous_t = gdata->t;
	gdata->t += gdata->dt;
	// buildneibs_freq?

	// choose minimum dt among the devices
	if (gdata->dtadapt) {
		gdata->dt = gdata->dts[0];
		for (uint d = 1; d < gdata->devices; d++)
			gdata->dt = min(gdata->dt, gdata->dts[d]);
		// if runnin multinode, should also find the network minimum
		if (MULTI_NODE)
			gdata->networkManager->networkFloatReduction(&(gdata->dt), 1, MIN_REDUCTION);
	}

	// check that dt is not too small (absolute)
	if (!gdata->dt) {
		throw DtZeroException(gdata->t, gdata->dt);
	} else if (gdata->dt < FLT_EPSILON) {
		fprintf(stderr, "FATAL: timestep %g under machine epsilon at iteration %lu - requesting quit...\n", gdata->dt, gdata->iterations);
		gdata->quit_request = true;
	}

	// check that dt is not too small (relative to t)
	if (gdata->t == previous_t) {
		fprintf(stderr, "FATAL: timestep %g too small at iteration %lu, time is still - requesting quit...\n", gdata->dt, gdata->iterations);
		gdata->quit_request = true;
	}

	//printf("Finished iteration %lu, time %g, dt %g\n", gdata->iterations, gdata->t, gdata->dt);

	// are we done?
	const bool we_are_done =
		// during simulation, ask the problem if we're done
		(gdata->run_mode == SIMULATE && gdata->problem->finished(gdata->t)) ||
		// if not, check if we've completed the number of iterations prescribed
		// (for repacking, or from the command line during simulation)
		gdata->iterations >= gdata->maxiter ||
		// and of course we're finished if a quit was requested
		gdata->quit_request;

	if (gdata->run_mode == REPACK && we_are_done) {
		// signal to the integrator that we are done with the repacking,
		// so we can proceed with the final
		integrator->we_are_done();
		// but don't keep going if a quit was requested
		if (gdata->quit_request)
			gdata->keep_going = false;
	} else {
		check_write(we_are_done);

		if (we_are_done)
			// NO dispatchCommand() after keep_going has been unset!
			gdata->keep_going = false;
	}
}

void GPUSPH::unknownCommand(CommandName cmd)
{
	string err = "FATAL: command " + to_string(cmd)
		+ " (" + getCommandName(cmd) + ") issued on host "
		"is not implemented";
	throw std::runtime_error(err);
}

template<>
void GPUSPH::runCommand<NUM_COMMANDS>(CommandStruct const& cmd)
{
	unknownCommand(cmd.command);
}

void
GPUSPH::dispatchCommand(CommandStruct cmd, flag_t flags)
{
	dispatchCommand(cmd.set_flags(flags));
}

bool GPUSPH::runSimulation() {
	bool all_ok = false;

	if (!initialized) return all_ok;

	all_ok = true;

	// doing first write
	printf("Performing first write...\n");
	doWrite(StepInfo(0));

	printf("Letting threads upload the subdomains...\n");
	gdata->threadSynchronizer->barrier(); // begins UPLOAD ***

	// here the Workers are uploading their subdomains

	// After next barrier, the workers will enter their simulation cycle, so it is recommended to set
	// nextCommand properly before the barrier (although should be already initialized to IDLE).
	// dispatchCommand(IDLE) would be equivalent, but this is more clear
	gdata->nextCommand = IDLE;
	gdata->threadSynchronizer->barrier(); // end of UPLOAD, begins SIMULATION ***
	gdata->threadSynchronizer->barrier(); // unlock CYCLE BARRIER 1

	integrator->start();

	const CommandStruct* cmd = nullptr;
	while (gdata->keep_going && (cmd = integrator->next_command()) ) try {
		dispatchCommand(*cmd);
	} catch (exception const& e) {
		cerr << e.what() << endl;
		all_ok = false;
		gdata->keep_going = false;
		if (MULTI_NODE)
			gdata->networkManager->sendKillRequest();
		// the loop is being ended by some exception, so we cannot guarantee that
		// all threads are alive. Force unlocks on all subsequent barriers to exit
		// as cleanly as possible without stalling
		gdata->threadSynchronizer->forceUnlock();
	}

	const char* run_desc = gdata->run_mode_desc();
	const char* run_desc_title = gdata->run_mode_Desc();

	// elapsed time, excluding the initialization
	printf("Elapsed time of %s cycle: %.2gs\n", run_desc,
		m_totalPerformanceCounter->getElapsedSeconds());

	// In multinode simulations we also print the global performance. To make only rank 0 print it, add
	// the condition (gdata->mpi_rank == 0)
	if (MULTI_NODE)
		printf("Global performance of the multinode %s: %.2g MIPPS\n", run_desc,
			m_multiNodePerformanceCounter->getMIPPS());

	// suggest max speed for next runs
	printf("Peak particle speed was ~%g m/s at %g s -> can set maximum vel %.2g for this problem\n",
		m_peakParticleSpeed, m_peakParticleSpeedTime, (m_peakParticleSpeed*1.1));

	// NO dispatchCommand() nor other barriers than the standard ones after the

	printf("%s end, cleaning up...\n", run_desc_title);

	// dispatchCommand(QUIT) would be equivalent, but this is more clear
	gdata->nextCommand = QUIT;
	gdata->threadSynchronizer->barrier(); // unlock CYCLE BARRIER 2
	gdata->threadSynchronizer->barrier(); // end of SIMULATION, begins FINALIZATION ***

	// just wait or...?

	gdata->threadSynchronizer->barrier(); // end of FINALIZATION ***

	// after the last barrier has been reached by all threads (or after the Synchronizer has been forcedly unlocked),
	// we wait for the threads to actually exit
	for (uint d = 0; d < gdata->devices; d++)
		gdata->GPUWORKERS[d]->join_worker();

	return all_ok;
}

// Finalize the computation of the total force and torque acting on moving bodies
// Add up the partial reductions obtained for each device
template<>
void GPUSPH::runCommand<REDUCE_BODIES_FORCES_HOST>(CommandStruct const& cmd)
{
	const size_t numforcesbodies = problem->simparams()->numforcesbodies;

	// Now sum up the partial forces and momentums computed in each gpu
	if (MULTI_GPU) {
		for (uint ob = 0; ob < numforcesbodies; ob ++) {
			gdata->s_hRbTotalForce[ob] = make_float3( 0.0f );
			gdata->s_hRbTotalTorque[ob] = make_float3( 0.0f );

			for (uint d = 0; d < gdata->devices; d++) {
				gdata->s_hRbTotalForce[ob] += gdata->s_hRbDeviceTotalForce[d*numforcesbodies + ob];
				gdata->s_hRbTotalTorque[ob] += gdata->s_hRbDeviceTotalTorque[d*numforcesbodies + ob];
			} // Iterate on devices
		} // Iterate on objects on which we compute forces
	}

	// if running multinode, also reduce across nodes
	if (MULTI_NODE) {
		// to minimize the overhead, we reduce the whole arrays of forces and torques in one command
		gdata->networkManager->networkFloatReduction((float*)gdata->s_hRbTotalForce, 3 * numforcesbodies, SUM_REDUCTION);
		gdata->networkManager->networkFloatReduction((float*)gdata->s_hRbTotalTorque, 3 * numforcesbodies, SUM_REDUCTION);
	}

}

/* Make a copy of the total forces, and let the problem override the applied forces, if necessary */
template<>
void GPUSPH::runCommand<BODY_FORCES_CALLBACK>(CommandStruct const& cmd)
{
	const int step = cmd.step.number;
	const float dt = cmd.dt(gdata);

	const size_t numforcesbodies = problem->simparams()->numforcesbodies;

	memcpy(gdata->s_hRbAppliedForce, gdata->s_hRbTotalForce, numforcesbodies*sizeof(float3));
	memcpy(gdata->s_hRbAppliedTorque, gdata->s_hRbTotalTorque, numforcesbodies*sizeof(float3));

	double t0 = gdata->t;
	double t1 = t0 + dt;;
	problem->bodies_forces_callback(t0, t1, step, gdata->s_hRbAppliedForce, gdata->s_hRbAppliedTorque);
}

template<>
void GPUSPH::runCommand<MOVE_BODIES>(CommandStruct const& cmd)
// GPUSPH::move_bodies(flag_t integrator_step)
{
	// TODO this function should also be ported to the CommandSequence architecture
	const int step = cmd.step.number;

	const float dt = cmd.dt(gdata);

	// Let the problem compute the new moving bodies data
	// TODO we pass step and gdata->dt, should be pass step and dt (which is already halved if necessary)?
	// instead?
	problem->bodies_timestep(gdata->s_hRbAppliedForce, gdata->s_hRbAppliedTorque, step, gdata->dt, gdata->t,
		gdata->s_hRbCgGridPos, gdata->s_hRbCgPos,
		gdata->s_hRbTranslations, gdata->s_hRbRotationMatrices, gdata->s_hRbLinearVelocities, gdata->s_hRbAngularVelocities);

	if (cmd.step.last)
		problem->post_timestep_callback(gdata->t);
}

// Allocate the shared buffers, i.e. those accessed by all workers
// Returns the number of allocated bytes.
// This does *not* include what was previously allocated (e.g. particles in problem->fillparts())
size_t GPUSPH::allocateGlobalHostBuffers()
{
	// define host buffers
	gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_POS_GLOBAL>();
	gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_POS>();
	gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_HASH>();
	gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_VEL>();
	gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_INFO>();

	if (gdata->debug.neibs)
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_NEIBSLIST>();

	if (gdata->debug.forces)
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_FORCES>();

	if (gdata->simframework->hasPostProcessOption(SURFACE_DETECTION, BUFFER_NORMALS))
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_NORMALS>();
	if (gdata->simframework->hasPostProcessOption(INTERFACE_DETECTION, BUFFER_NORMALS))
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_NORMALS>();

	if (gdata->simframework->hasPostProcessEngine(VORTICITY))
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_VORTICITY>();

	if (problem->simparams()->boundarytype == SA_BOUNDARY) {
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_BOUNDELEMENTS>();
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_VERTICES>();
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_GRADGAMMA>();
	}

	if (problem->simparams()->turbmodel == KEPSILON) {
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_TKE>();
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_EPSILON>();
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_TURBVISC>();
	}


	if (problem->simparams()->boundarytype == SA_BOUNDARY &&
		(problem->simparams()->simflags & ENABLE_INLET_OUTLET ||
		problem->simparams()->turbmodel == KEPSILON))
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_EULERVEL>();

	if (problem->simparams()->simflags & ENABLE_INLET_OUTLET)
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_NEXTID>();

	if (problem->simparams()->turbmodel == SPS)
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_SPS_TURBVISC>();

	if (NEEDS_EFFECTIVE_VISC(problem->simparams()->rheologytype))
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_EFFVISC>();

	if (problem->simparams()->rheologytype == GRANULAR) {
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_EFFPRES>();
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_JACOBI>();
	}

	if (problem->simparams()->sph_formulation == SPH_GRENIER) {
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_VOLUME>();
		// Only for debugging:
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_SIGMA>();
	}

	if (gdata->simframework->hasPostProcessEngine(CALC_PRIVATE)) {
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_PRIVATE>();
		if (gdata->simframework->hasPostProcessOption(CALC_PRIVATE, BUFFER_PRIVATE2))
			gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_PRIVATE2>();
		if (gdata->simframework->hasPostProcessOption(CALC_PRIVATE, BUFFER_PRIVATE4))
			gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_PRIVATE4>();
	}

	if (problem->simparams()->simflags & ENABLE_INTERNAL_ENERGY) {
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_INTERNAL_ENERGY>();
	}

	// number of elements to allocate
	const size_t numparts = gdata->allocatedParticles;

	const uint numcells = gdata->nGridCells;
	const size_t devcountCellSize = sizeof(devcount_t) * numcells;
	const size_t uintCellSize = sizeof(uint) * numcells;

	size_t totCPUbytes = 0;

	BufferList::iterator iter = gdata->s_hBuffers.begin();
	while (iter != gdata->s_hBuffers.end()) {
		if (iter->first == BUFFER_NEIBSLIST)
			totCPUbytes += iter->second->alloc(numparts*gdata->problem->simparams()->neiblistsize);
		else
			totCPUbytes += iter->second->alloc(numparts);
		++iter;
	}

	const size_t numbodies = gdata->problem->simparams()->numbodies;
	cout << "Numbodies : " << numbodies << "\n";
	if (numbodies > 0) {
		gdata->s_hRbCgGridPos = new int3 [numbodies];
		fill_n(gdata->s_hRbCgGridPos, numbodies, make_int3(0));
		gdata->s_hRbCgPos = new float3 [numbodies];
		fill_n(gdata->s_hRbCgPos, numbodies, make_float3(0.0f));
		gdata->s_hRbTranslations = new float3 [numbodies];
		fill_n(gdata->s_hRbTranslations, numbodies, make_float3(0.0f));
		gdata->s_hRbLinearVelocities = new float3 [numbodies];
		fill_n(gdata->s_hRbLinearVelocities, numbodies, make_float3(0.0f));
		gdata->s_hRbAngularVelocities = new float3 [numbodies];
		fill_n(gdata->s_hRbAngularVelocities, numbodies, make_float3(0.0f));
		gdata->s_hRbRotationMatrices = new float [numbodies*9];
		fill_n(gdata->s_hRbRotationMatrices, 9*numbodies, 0.0f);
		totCPUbytes += numbodies*(sizeof(int3) + 4*sizeof(float3) + 9*sizeof(float));
	}
	const size_t numforcesbodies = gdata->problem->simparams()->numforcesbodies;
	cout << "Numforcesbodies : " << numforcesbodies << "\n";
	if (numforcesbodies > 0) {
		gdata->s_hRbFirstIndex = new int [numforcesbodies];
		fill_n(gdata->s_hRbFirstIndex, numforcesbodies, 0);
		gdata->s_hRbLastIndex = new uint [numforcesbodies];
		fill_n(gdata->s_hRbLastIndex, numforcesbodies, 0);
		totCPUbytes += numforcesbodies*sizeof(uint);
		gdata->s_hRbTotalForce = new float3 [numforcesbodies];
		fill_n(gdata->s_hRbTotalForce, numforcesbodies, make_float3(0.0f));
		gdata->s_hRbAppliedForce = new float3 [numforcesbodies];
		fill_n(gdata->s_hRbAppliedForce, numforcesbodies, make_float3(0.0f));
		gdata->s_hRbTotalTorque = new float3 [numforcesbodies];
		fill_n(gdata->s_hRbTotalTorque, numforcesbodies, make_float3(0.0f));
		gdata->s_hRbAppliedTorque = new float3 [numforcesbodies];
		fill_n(gdata->s_hRbAppliedTorque, numforcesbodies, make_float3(0.0f));
		totCPUbytes += numforcesbodies*4*sizeof(float3);
		if (MULTI_GPU) {
			gdata->s_hRbDeviceTotalForce = new float3 [numforcesbodies*MAX_DEVICES_PER_NODE];
			fill_n(gdata->s_hRbDeviceTotalForce, numforcesbodies*MAX_DEVICES_PER_NODE, make_float3(0.0f));
			gdata->s_hRbDeviceTotalTorque = new float3 [numforcesbodies*MAX_DEVICES_PER_NODE];
			fill_n(gdata->s_hRbDeviceTotalTorque, numforcesbodies*MAX_DEVICES_PER_NODE, make_float3(0.0f));
			totCPUbytes += numforcesbodies*MAX_DEVICES_PER_NODE*2*sizeof(float3);
		}
		// In order to avoid tests and special case for mono GPU in GPUWorker::reduceRbForces the the per device
		// total arrays are aliased to the global total ones.
		else {
			gdata->s_hRbDeviceTotalForce = gdata->s_hRbTotalForce;
			gdata->s_hRbDeviceTotalTorque = gdata->s_hRbTotalTorque;
		}
	}

	const size_t numOpenBoundaries = gdata->problem->simparams()->numOpenBoundaries;
	cout << "numOpenBoundaries : " << numOpenBoundaries << "\n";

	// water depth computation array
	if (problem->simparams()->simflags & ENABLE_WATER_DEPTH) {
		gdata->h_IOwaterdepth = new uint* [MULTI_GPU ? MAX_DEVICES_PER_NODE : 1];
		for (uint i=0; i<(MULTI_GPU ? MAX_DEVICES_PER_NODE : 1); i++)
			gdata->h_IOwaterdepth[i] = new uint [numOpenBoundaries];
		gdata->h_maxIOwaterdepth = new uint [numOpenBoundaries];
	}

	PostProcessEngineSet const& enabledPostProcess = gdata->simframework->getPostProcEngines();
	for (PostProcessEngineSet::const_iterator flt(enabledPostProcess.begin());
		flt != enabledPostProcess.end(); ++flt) {
		flt->second->hostAllocate(gdata);
	}

	if (MULTI_DEVICE) {
		// deviceMap
		gdata->s_hDeviceMap = new devcount_t[numcells];
		memset(gdata->s_hDeviceMap, 0, devcountCellSize);
		totCPUbytes += devcountCellSize;

		// counters to help splitting evenly
		gdata->s_hPartsPerSliceAlongX = new uint[ gdata->gridSize.x ];
		gdata->s_hPartsPerSliceAlongY = new uint[ gdata->gridSize.y ];
		gdata->s_hPartsPerSliceAlongZ = new uint[ gdata->gridSize.z ];
		// initialize
		for (uint c=0; c < gdata->gridSize.x; c++) gdata->s_hPartsPerSliceAlongX[c] = 0;
		for (uint c=0; c < gdata->gridSize.y; c++) gdata->s_hPartsPerSliceAlongY[c] = 0;
		for (uint c=0; c < gdata->gridSize.z; c++) gdata->s_hPartsPerSliceAlongZ[c] = 0;
		// record used memory
		totCPUbytes += sizeof(uint) * (gdata->gridSize.x + gdata->gridSize.y + gdata->gridSize.z);

		// cellStarts, cellEnds, segmentStarts of all devices. Array of device pointers stored on host
		// For cell starts and ends, the actual per-device components will be done by each GPUWorker,
		// using cudaHostAlloc to allocate pinned memory
		gdata->s_dCellStarts = (uint**)calloc(gdata->devices, sizeof(uint*));
		gdata->s_dCellEnds =  (uint**)calloc(gdata->devices, sizeof(uint*));
		gdata->s_dSegmentsStart = (uint**)calloc(gdata->devices, sizeof(uint*));
		for (uint d=0; d < gdata->devices; d++)
			gdata->s_dSegmentsStart[d] = (uint*)calloc(4, sizeof(uint));


		// few bytes... but still count them
		totCPUbytes += gdata->devices * sizeof(uint*) * 3;
		totCPUbytes += gdata->devices * sizeof(uint) * 4;
	}
	return totCPUbytes;
}

// Deallocate the shared buffers, i.e. those accessed by all workers
void GPUSPH::deallocateGlobalHostBuffers() {
	gdata->s_hBuffers.clear();

	// Deallocating waterdepth-related arrays
	if (problem->simparams()->simflags & ENABLE_WATER_DEPTH) {
		delete[] gdata->h_maxIOwaterdepth;
		delete[] gdata->h_IOwaterdepth;
	}

	// Deallocating rigid bodies related arrays
	if (gdata->problem->simparams()->numbodies > 0) {
		delete [] gdata->s_hRbCgGridPos;
		delete [] gdata->s_hRbCgPos;
		delete [] gdata->s_hRbTranslations;
		delete [] gdata->s_hRbLinearVelocities;
		delete [] gdata->s_hRbAngularVelocities;
		delete [] gdata->s_hRbRotationMatrices;
	}
	if (gdata->problem->simparams()->numforcesbodies > 0) {
		delete [] gdata->s_hRbFirstIndex;
		delete [] gdata->s_hRbLastIndex;
		delete [] gdata->s_hRbTotalForce;
		delete [] gdata->s_hRbAppliedForce;
		delete [] gdata->s_hRbTotalTorque;
		delete [] gdata->s_hRbAppliedTorque;
		if (MULTI_DEVICE) {
			delete [] gdata->s_hRbDeviceTotalForce;
			delete [] gdata->s_hRbDeviceTotalTorque;
		}
	}

	// planes
	gdata->s_hPlanes.clear();

	// multi-GPU specific arrays
	if (MULTI_DEVICE) {
		delete[] gdata->s_hDeviceMap;
		delete[] gdata->s_hPartsPerSliceAlongX;
		delete[] gdata->s_hPartsPerSliceAlongY;
		delete[] gdata->s_hPartsPerSliceAlongZ;
		free(gdata->s_dCellEnds);
		free(gdata->s_dCellStarts);
		free(gdata->s_dSegmentsStart);
	}

}

/// Check consistency of buffers across multiple GPUs
/*! This verifies that the particle buffers have the same content for areas
 * that are shared across devices (i.e. subdomain edges).
 * \note This requires the code to be compiled with INSPECT_DEVICE_MEMORY,
 * to ensure that device buffers are accessible on host.
 */
void GPUSPH::checkBufferConsistency()
{
#ifdef INSPECT_DEVICE_MEMORY
	const devcount_t numdevs = gdata->devices;

#define NUM_CHECK_LISTS 2
	const char* buflist_name[NUM_CHECK_LISTS] = { "READ", "WRITE" };
	std::vector<const BufferList *> buflist(numdevs*NUM_CHECK_LISTS);

	for (devcount_t d = 0; d < numdevs; ++d) {
		buflist[NUM_CHECK_LISTS*d + 0] = &gdata->GPUWORKERS[d]->getBufferList().getReadBufferList();
		buflist[NUM_CHECK_LISTS*d + 1] = &gdata->GPUWORKERS[d]->getBufferList().getWriteBufferList();
	}

	std::vector<uint> numParticles(numdevs);
	std::vector<uint> numInternalParticles(numdevs);

	for (devcount_t d = 0; d < numdevs; ++d) {
		numParticles[d] = gdata->GPUWORKERS[d]->getNumParticles();
		numInternalParticles[d] = gdata->GPUWORKERS[d]->getNumInternalParticles();
	}

	std::vector<particleinfo const*> infoArray(numdevs);
	std::vector<float4 const*> posArray(numdevs);
	std::vector<float4 const*> velArray(numdevs);

	// TODO FIXME this is where we assume two devices, for faster calculation about
	// which particles we have to compare against which

	if (numdevs != 2)
		throw runtime_error("sorry, buffer consistency check is only supported for two devices at the moment");

	uint externalParticles[2] = {
		numParticles[0] - numInternalParticles[0],
		numParticles[1] - numInternalParticles[1]
	};
	uint edgeParticlesStart[2] = {
		numInternalParticles[0] - externalParticles[1],
		numInternalParticles[1] - externalParticles[0]
	};

	for (uint l = 0; l < NUM_CHECK_LISTS; ++l) {

		for (devcount_t d = 0; d < numdevs; ++d) {
			infoArray[d] = buflist[NUM_CHECK_LISTS*d + l]->getData<BUFFER_INFO>();
			posArray[d] = buflist[NUM_CHECK_LISTS*d + l]->getData<BUFFER_POS>();
			velArray[d] = buflist[NUM_CHECK_LISTS*d + l]->getData<BUFFER_VEL>();
		}

		// Check the external particles of each device against their counterparts on the devices where these are edge internal
		for (uint d = 0; d < numdevs; ++d) {
			const uint neib_d = 1 - d; // TODO FIXME numdevs == 2
			for (uint p = 0; p < externalParticles[d]; ++p) {
				uint offset = numInternalParticles[d] + p;
				uint neib_offset = edgeParticlesStart[neib_d] + p;

				const particleinfo info = infoArray[d][offset];
				const particleinfo neib_info = infoArray[neib_d][neib_offset];
				const float4 pos = posArray[d][offset];
				const float4 neib_pos = posArray[neib_d][neib_offset];
				const float4 vel = velArray[d][offset];
				const float4 neib_vel = velArray[neib_d][neib_offset];

				if (memcmp(&info, &neib_info, sizeof(info))) {
					printf("%s INFO mismatch @ iteration %d, command %d | %d: device %d external particle %d (offset %d, neib %d)\n",
						buflist_name[l], gdata->iterations, gdata->nextCommand, gdata->commandFlags,
						d, p, offset, neib_offset);
					printf("(%d %d %d %d) vs (%d %d %d %d)\n",
						info.x, info.y, info.z, info.w,
						neib_info.x, neib_info.y, neib_info.z, neib_info.w);
					break;
				}

				if (memcmp(&pos, &neib_pos, sizeof(pos))) {
					printf("%s POS mismatch @ iteration %d, command %d | %d: device %d external particle %d (offset %d, neib %d)\n",
						buflist_name[l], gdata->iterations, gdata->nextCommand, gdata->commandFlags,
						d, p, offset, neib_offset);
					printf("(%g %g %g %g) vs (%g %g %g %g)\n",
						pos.x, pos.y, pos.z, pos.w,
						neib_pos.x, neib_pos.y, neib_pos.z, neib_pos.w);
					break;
				}

				if (memcmp(&vel, &neib_vel, sizeof(vel))) {
					printf("%s VEL mismatch @ iteration %d, command %d | %d: device %d external particle %d (offset %d, neib %d)\n",
						buflist_name[l], gdata->iterations, gdata->nextCommand, gdata->commandFlags,
						d, p, offset, neib_offset);
					printf("(%g %g %g %g) vs (%g %g %g %g)\n",
						vel.x, vel.y, vel.z, vel.w,
						neib_vel.x, neib_vel.y, neib_vel.z, neib_vel.w);
					break;
				}
			}
		}
	}

#else
	throw runtime_error("cannot check buffer consistency without INSPECT_DEVICE_MEMORY");
#endif
}

/// Initialize the array holding the IDs of the next generated particles
/*! Each open boundary vertex, when generating a new particle, will assign it
 * its nextID value, and update the nextID by adding the total count of open
 * boundary vertices present in the simulation.
 *
 * \return the number of open boundary vertices that have been found
 */
uint GPUSPH::initializeNextIDs(bool resumed)
{
	gdata->s_hBuffers.set_state_on_write("init next id");
	const particleinfo *particleInfo(gdata->s_hBuffers.getConstData<BUFFER_INFO>());
	uint *nextID(gdata->s_hBuffers.getData<BUFFER_NEXTID>());
	// TODO we should skip CORNER vertices, but we do CORNER vertex detection on device
	// later. Consider doing it during this pass
	uint numOpenVertices = 0;
	/* next nextID to be assigned —ASSUME the max id at the beginning is (at most) totParticles - 1,
	 * we'll check and throw if it's not */
	uint runningNextID = gdata->totParticles;
	for (uint p = 0; p < gdata->totParticles; ++p)
	{
		const particleinfo pinfo(particleInfo[p]);
		const uint this_id = id(pinfo);

		if (!resumed && this_id >= gdata->totParticles)
			throw runtime_error(
				"found id " + std::to_string(this_id) +
				" ≥ " + std::to_string(gdata->totParticles));

		// We only set nextID for open boundary vertices
		/* TODO this is currently designed for SA_BOUNDARY, for different boundary conditions
		 * it should be adapted to initialize the nextIDs of whichever particles generate fluid.
		 */
		const bool produces_particles = (VERTEX(pinfo) && IO_BOUNDARY(pinfo));
		if (!produces_particles)
			continue;

		++numOpenVertices;

		if (!resumed) {
			// OK, set the next ID
			nextID[p] = runningNextID;

#if 0 // DEBUG
			printf("\tassigned next ID %u to vertex %u (index %u)\n",
				nextID[p], this_id, p);
#endif

			++runningNextID;
		}

	}
	gdata->s_hBuffers.clear_pending_state();
	return numOpenVertices;
}

// Sort the particles in-place (pos, vel, info) according to the device number;
// update counters s_hPartsPerDevice and s_hStartPerDevice, which will be used to upload
// and download the buffers. Finally, initialize s_dSegmentsStart
// Assumptions: problem already filled, deviceMap filled, particles copied in shared arrays
void GPUSPH::sortParticlesByHash() {
	// DEBUG: print the list of particles before sorting
	// for (uint p=0; p < gdata->totParticles; p++)
	//	printf(" p %d has id %u, dev %d\n", p, id(gdata->s_hInfo[p]), gdata->calcDevice(gdata->s_hPos[p]) );

	// count parts for each device, even in other nodes (s_hPartsPerDevice only includes devices in self node on)
	uint particlesPerGlobalDevice[MAX_DEVICES_PER_CLUSTER];

	// reset counters. Not using memset since sizes are smaller than 1Kb
	for (uint d = 0; d < MAX_DEVICES_PER_NODE; d++)    gdata->s_hPartsPerDevice[d] = 0;
	for (uint n = 0; n < MAX_NODES_PER_CLUSTER; n++)   gdata->processParticles[n]  = 0;
	for (uint d = 0; d < MAX_DEVICES_PER_CLUSTER; d++) particlesPerGlobalDevice[d] = 0;

	// TODO: move this in allocateGlobalBuffers...() and rename it, or use only here as a temporary buffer? or: just use HASH, sorting also for cells, not only for device
	devcount_t* m_hParticleKeys = new devcount_t[gdata->totParticles];

	// fill array with particle hashes (aka global device numbers) and increase counters
	const hashKey *particleHash = gdata->s_hBuffers.getConstData<BUFFER_HASH>();
	for (uint p = 0; p < gdata->totParticles; p++) {

		// compute containing device according to the particle's hash
		uint cellHash = cellHashFromParticleHash( particleHash[p] );
		devcount_t whichGlobalDev = gdata->s_hDeviceMap[ cellHash ];

		// that's the key!
		m_hParticleKeys[p] = whichGlobalDev;

		// increase node and globalDev counter (only useful for multinode)
		gdata->processParticles[gdata->RANK(whichGlobalDev)]++;

		particlesPerGlobalDevice[gdata->GLOBAL_DEVICE_NUM(whichGlobalDev)]++;

		// if particle is in current node, increment the device counters
		if (gdata->RANK(whichGlobalDev) == gdata->mpi_rank)
			// increment per-device counter
			gdata->s_hPartsPerDevice[ gdata->DEVICE(whichGlobalDev) ]++;

		//if (whichGlobalDev != 0)
		//printf(" ö part %u has key %u (n%dd%u) global dev %u \n", p, whichGlobalDev, gdata->RANK(whichGlobalDev), gdata->DEVICE(whichGlobalDev), gdata->GLOBAL_DEVICE_NUM(whichGlobalDev) );

	}

	// printParticleDistribution();

	// update s_hStartPerDevice with incremental sum (should do in specific function?)
	gdata->s_hStartPerDevice[0] = 0;
	// zero is true for the first node. For the next ones, need to sum the number of particles of the previous nodes
	if (MULTI_NODE)
		for (int prev_nodes = 0; prev_nodes < gdata->mpi_rank; prev_nodes++)
			gdata->s_hStartPerDevice[0] += gdata->processParticles[prev_nodes];
	for (uint d = 1; d < gdata->devices; d++)
		gdata->s_hStartPerDevice[d] = gdata->s_hStartPerDevice[d-1] + gdata->s_hPartsPerDevice[d-1];

	// *** About the algorithm being used ***
	//
	// Since many particles share the same key, what we need is actually a compaction rather than a sort.
	// A cycle sort would be probably the best performing in terms of reducing the number of writes.
	// A selection sort would be the easiest to implement but would yield more swaps than needed.
	// The following variant, hybrid with a counting sort, is implemented.
	// We already counted how many particles are there for each device (m_partsPerDevice[]).
	// We keep two pointers, leftB and rightB (B stands for boundary). The idea is that leftB is the place
	// where we are going to put the next element and rightB is being moved to "scan" the rest of the array
	// and select next element. Unlike selection sort, rightB is initialized at the end of the array and
	// being decreased; this way, each element is expected to be moved no more than twice (estimation).
	// Moreover, a burst of particles which partially overlaps the correct bucket is not entirely moved:
	// since rightB goes from right to left, the rightmost particles are moved while the overlapping ones
	// are not. All particles before leftB have already been compacted; leftB is incremented as long as there
	// are particles already in correct positions. When there is a bucket change (we track it with nextBucketBeginsAt)
	// rightB is reset to the end of the array.
	// Possible optimization: decrease maxIdx to the last non-correct element of the array (if there is a burst
	// of correct particles in the end, this avoids scanning them everytime) and update this while running.
	// The following implementation iterates on buckets explicitly instead of working solely on leftB and rightB
	// and detect the bucket change. Same operations, cleaner code.

	// init
	const uint maxIdx = (gdata->totParticles - 1);
	uint leftB = 0;
	uint rightB;
	uint nextBucketBeginsAt = 0;

	// NOTE: in the for cycle we want to iterate on the global number of devices, not the local (process) one
	// NOTE(2): we don't need to iterate in the last bucket: it should be already correct after the others. That's why "devices-1".
	// We might want to iterate on last bucket only for correctness check.
	// For each bucket (device)...
	for (uint currentGlobalDevice = 0; currentGlobalDevice < (gdata->totDevices - 1U); currentGlobalDevice++) {
		// compute where current bucket ends
		nextBucketBeginsAt += particlesPerGlobalDevice[currentGlobalDevice];
		// reset rightB to the end
		rightB = maxIdx;
		// go on until we reach the end of the current bucket
		while (leftB < nextBucketBeginsAt) {

			// translate from globalDeviceIndex to an absolute device index in 0..totDevices (the opposite convertDeviceMap does)
			uint currPartGlobalDevice = gdata->GLOBAL_DEVICE_NUM( m_hParticleKeys[leftB] );

			// if in the current position there is a particle *not* belonging to the bucket...
			if (currPartGlobalDevice != currentGlobalDevice) {

				// ...let's find a correct one, scanning from right to left
				while ( gdata->GLOBAL_DEVICE_NUM( m_hParticleKeys[rightB] ) != currentGlobalDevice) rightB--;

				// here it should never happen that (rightB <= leftB). We should throw an error if it happens
				particleSwap(leftB, rightB);
				swap(m_hParticleKeys[leftB], m_hParticleKeys[rightB]);
			}

			// already correct or swapped, time to go on
			leftB++;
		}
	}
	// delete array of keys (might be recycled instead?)
	delete[] m_hParticleKeys;

	// initialize the outer cells values in s_dSegmentsStart. The inner_edge are still uninitialized
	for (uint currentDevice = 0; currentDevice < gdata->devices; currentDevice++) {
		uint assigned_parts = gdata->s_hPartsPerDevice[currentDevice];
		printf("    d%u  p %u\n", currentDevice, assigned_parts);
		// this should always hold according to the current CELL_TYPE values
		gdata->s_dSegmentsStart[currentDevice][CELLTYPE_INNER_CELL ] = 		EMPTY_SEGMENT;
		// this is usually not true, since a device usually has neighboring cells; will be updated at first reorder
		gdata->s_dSegmentsStart[currentDevice][CELLTYPE_INNER_EDGE_CELL ] =	EMPTY_SEGMENT;
		// this is true and will change at first APPEND
		gdata->s_dSegmentsStart[currentDevice][CELLTYPE_OUTER_EDGE_CELL ] =	EMPTY_SEGMENT;
		// this is true and might change between a reorder and the following crop
		gdata->s_dSegmentsStart[currentDevice][CELLTYPE_OUTER_CELL ] =		EMPTY_SEGMENT;
	}

	// DEBUG: check if the sort was correct
	bool monotonic = true;
	bool count_c = true;
	uint hcount[MAX_DEVICES_PER_NODE];
	for (uint d=0; d < MAX_DEVICES_PER_NODE; d++)
		hcount[d] = 0;
	for (uint p=0; p < gdata->totParticles && monotonic; p++) {
		devcount_t cdev = gdata->s_hDeviceMap[ cellHashFromParticleHash(particleHash[p]) ];
		devcount_t pdev;
		if (p > 0) pdev = gdata->s_hDeviceMap[ cellHashFromParticleHash(particleHash[p-1]) ];
		if (p > 0 && cdev < pdev ) {
			printf(" -- sorting error: array[%d] has device n%dd%u, array[%d] has device n%dd%u (skipping next errors)\n",
				p-1, gdata->RANK(pdev), gdata->	DEVICE(pdev), p, gdata->RANK(cdev), gdata->	DEVICE(cdev) );
			monotonic = false;
		}
		// count particles of the current process
		if (gdata->RANK(cdev) == gdata->mpi_rank)
			hcount[ gdata->DEVICE(cdev) ]++;
	}
	// WARNING: the following check is only for particles of the current rank (multigpu, not multinode).
	// Each process checks its own particles.
	for (uint d=0; d < gdata->devices; d++)
		if (hcount[d] != gdata->s_hPartsPerDevice[d]) {
			count_c = false;
			printf(" -- sorting error: counted %d particles for device %d, but should be %d\n",
				hcount[d], d, gdata->s_hPartsPerDevice[d]);
		}
	if (monotonic && count_c)
		printf(" --- array OK\n");
	else
		printf(" --- array ERROR\n");
	// finally, print the list again
	//for (uint p=1; p < gdata->totParticles && monotonic; p++)
		//printf(" p %d has id %u, dev %d\n", p, id(gdata->s_hInfo[p]), gdata->calcDevice(gdata->s_hPos[p]) ); // */
}

// Swap two particles in all host arrays; used in host sort
void GPUSPH::particleSwap(uint idx1, uint idx2)
{
	BufferList::iterator iter = gdata->s_hBuffers.begin();
	while (iter != gdata->s_hBuffers.end()) {
			iter->second->swap_elements(idx1, idx2);
		++iter;
	}
}

void GPUSPH::setViscosityCoefficient()
{
	PhysParams *pp = gdata->problem->physparams();
	const SimParams *sp = gdata->problem->simparams();

	// Set visccoeff based on the viscosity model used
	switch (sp->rheologytype) {
	case INVISCID:
		// ensure that the viscous coefficients are NaN: they should never be used,
		// and if they are it's an error
		for (uint f = 0; f < pp->numFluids(); ++f)
			pp->visccoeff[f] = NAN;
		break;
	case NEWTONIAN:
		if (sp->compvisc == KINEMATIC) {
			for (uint f = 0; f < pp->numFluids(); ++f)
				pp->visccoeff[f] = pp->kinematicvisc[f];
		} else if (sp->compvisc == DYNAMIC) {
			for (uint f = 0; f < pp->numFluids(); ++f)
				pp->visccoeff[f] = pp->visc_consistency[f];
		}
		break;
	default:
		/* Generalized Newtonian: the stored coefficients are always the dynamic value
		 */
		for (uint f = 0; f < pp->numFluids(); ++f)
			pp->visccoeff[f] = pp->visc_consistency[f];
	}

	// Check and/or set bulk viscosity when using Español & Revenga
	if (sp->viscmodel == ESPANOL_REVENGA) {
		float shear, bulk;
		for (uint f = 0; f < pp->numFluids(); ++f) {
			shear = pp->visc_consistency[f];
			bulk = pp->bulkvisc[f];
			if (isnan(bulk)) pp->bulkvisc[f] = bulk = 0;
			if (bulk*3 > shear*5)
				throw runtime_error("fluid " + to_string(f) +
					" with bulk viscosity " + to_string(bulk) +
					" and shear viscosity " + to_string(shear) +
					" cannot be modelled with Español & Revenga");
			pp->visc2coeff[f] = bulk;
			// For Español & Revenga, in the constant dynamic viscosity case,
			// we can reduce device-side computations by storing
			// in visccoeff the coefficient of the v part, and
			// in visc2coeff the coefficient of the r part
#if 0 // disabled until server side is implemented
			if (sp->is_const_visc) {
				pp->visccoeff[f] = 5*shear/3 - bulk;
				pp->visc2coeff[f] = 5*(bulk+shear/3);
				if (sp->compvisc == KINEMATIC) {
					pp->visccoeff[f] /= pp->rho0[f];
					pp->visc2coeff[f] /= pp->rho0[f];
				}
			}
#endif
		}
	}

	// Set SPS factors from the coefficients set by the problem
	if (sp->turbmodel == SPS) {
		const double spsCs = pp->smagorinsky_constant;
		const double spsCi = pp->isotropic_sps_constant;
		const double dp = gdata->problem->get_deltap();
		if (isnan(pp->smagfactor)) {
			pp->smagfactor = spsCs*dp;
			pp->smagfactor *= pp->smagfactor; // (Cs*∆p)^2
			printf("Autocomputed SPS Smagorinsky factor %g from C_s = %g, ∆p = %g\n",
				pp->smagfactor, spsCs, dp);
		}
		if (isnan(pp->kspsfactor)) {
			pp->kspsfactor = (2*spsCi/3)*dp*dp; // (2/3) Ci ∆p^2
			printf("Autocomputed SPS isotropic factor %g from C_i = %g, ∆p = %g\n",
				pp->kspsfactor, spsCi, dp);
		}
	}
}

// creates the Writer according to the requested WriterType
void GPUSPH::createWriter()
{
	Writer::Create(gdata);
}

double GPUSPH::Wendland2D(const double r, const double h) {
	const double q = r/h;
	double temp = 1 - q/2.;
	temp *= temp;
	temp *= temp;
	return 7/(4*M_PI*h*h)*temp*(2*q + 1);
}

void GPUSPH::doWrite(WriteFlags const& write_flags)
{
	// TODO FIXME skip unnecessary work based on write_flags
	// (e.g. do not run whatever isn't needed by the HotWriter during a hot write)
	uint node_offset = gdata->s_hStartPerDevice[0];

	// WaveGages work by looking at neighboring SURFACE particles and averaging their z coordinates
	// NOTE: it's a standard average, not an SPH smoothing, so the neighborhood is arbitrarily fixed
	// at gage (x,y) ± 2 smoothing lengths
	// TODO should it be an SPH smoothing instead?

	GageList &gages = problem->simparams()->gage;
	double slength = problem->simparams()->slength;

	size_t numgages = gages.size();
	vector<double> gages_W(numgages, 0.);
	for (uint g = 0; g < numgages; ++g) {
		if (gages[g].w == 0.)
			gages_W[g] = DBL_MAX;
		else
			gages_W[g] = 0.;
		gages[g].z = 0.;
	}

	// energy in non-fluid particles + one for each fluid type
	// double4 with .x kinetic, .y potential, .z internal, .w currently ignored
	double4 energy[MAX_FLUID_TYPES+1] = {0.0f};

	// TODO: parallelize? (e.g. each thread tranlsates its own particles)
	double3 const& wo = problem->get_worldorigin();
	const float4 *lpos = gdata->s_hBuffers.getConstData<BUFFER_POS>();
	const hashKey* hash = gdata->s_hBuffers.getConstData<BUFFER_HASH>();
	const particleinfo *info = gdata->s_hBuffers.getConstData<BUFFER_INFO>();
	double4 *gpos = gdata->s_hBuffers.getData<BUFFER_POS_GLOBAL>();

	const float *intEnergy = gdata->s_hBuffers.getConstData<BUFFER_INTERNAL_ENERGY>();
	/* vel is only used to compute kinetic energy */
	const float4 *vel = gdata->s_hBuffers.getConstData<BUFFER_VEL>();
	const double3 gravity = make_double3(gdata->problem->physparams()->gravity);

	bool warned_nan_pos = false;

	// max particle speed only for this node only at time t
	float local_max_part_speed = 0;

	for (uint i = node_offset; i < node_offset + gdata->processParticles[gdata->mpi_rank]; i++) {
		const float4 pos = lpos[i];
		uint3 gridPos = gdata->calcGridPosFromCellHash( cellHashFromParticleHash(hash[i]) );
		// double-precision absolute position, without using world offset (useful for computing the potential energy)
		double4 dpos = make_double4(
			gdata->calcGlobalPosOffset(gridPos, as_float3(pos)) + wo,
			pos.w);
		const particleinfo pinfo = info[i];

		if (!warned_nan_pos && !(isfinite(dpos.x) && isfinite(dpos.y) && isfinite(dpos.z))) {
			fprintf(stderr, "WARNING: particle %u (id %u, type %u) has NAN position! (%g, %g, %g) @ (%u, %u, %u) = (%g, %g, %g) at iteration %lu, time %g\n",
				i, id(pinfo), PART_TYPE(pinfo),
				pos.x, pos.y, pos.z,
				gridPos.x, gridPos.y, gridPos.z,
				dpos.x, dpos.y, dpos.z,
				gdata->iterations, gdata->t);
			warned_nan_pos = true;
		}

		// if we're tracking internal energy, we're interested in all the energy
		// in the system, including kinetic and potential: keep track of that too
		if (intEnergy) {
			const double4 energies = dpos.w*make_double4(
				/* kinetic */ sqlength3(vel[i])/2,
				/* potential */ -dot3(dpos, gravity),
				/* internal */ intEnergy[i],
				/* TODO */ 0);
			int idx = FLUID(info[i]) ? fluid_num(info[i]) : MAX_FLUID_TYPES;
			energy[idx] += energies;
		}

		// for surface particles add the z coordinate to the appropriate wavegages
		if (numgages && SURFACE(info[i])) {
			for (uint g = 0; g < numgages; ++g) {
				const double gslength  = gages[g].w;
				const double r = sqrt((dpos.x - gages[g].x)*(dpos.x - gages[g].x) + (dpos.y - gages[g].y)*(dpos.y - gages[g].y));
				if (gslength > 0) {
					if (r < 2*gslength) {
						const double W = Wendland2D(r, gslength);
						gages_W[g] += W;
						gages[g].z += dpos.z*W;
					}
				}
				else {
					if (r < gages_W[g]) {
						gages_W[g] = r;
						gages[g].z = dpos.z;
					}
				}
			}
		}

		gpos[i] = dpos;

		// track peak speed
		local_max_part_speed = fmax(local_max_part_speed, length( as_float3(vel[i]) ));
	}

	// max speed: read simulation global for multi-node
	if (MULTI_NODE)
		// after this, local_max_part_speed actually becomes global_max_part_speed for time t only
		gdata->networkManager->networkFloatReduction(&(local_max_part_speed), 1, MAX_REDUCTION);
	// update peak
	if (local_max_part_speed > m_peakParticleSpeed) {
		m_peakParticleSpeed = local_max_part_speed;
		m_peakParticleSpeedTime = gdata->t;
	}

	WriterMap writers = Writer::StartWriting(gdata->t, write_flags);

	if (numgages) {
		for (uint g = 0 ; g < numgages; ++g) {
			/*cout << "Ng : " << g << " gage: " << gages[g].x << "," << gages[g].y << " r : " << gages[g].w << " z: " << gages[g].z
					<< " gparts :" << gage_parts[g] << endl;*/
			if (gages[g].w)
				gages[g].z /= gages_W[g];
		}
		//Write WaveGage information on one text file
		Writer::WriteWaveGage(writers, gdata->t, gages);
	}

	if (gdata->problem->simparams()->numforcesbodies > 0) {
		Writer::WriteObjectForces(writers, gdata->t, problem->simparams()->numforcesbodies,
			gdata->s_hRbTotalForce, gdata->s_hRbTotalTorque,
			gdata->s_hRbAppliedForce, gdata->s_hRbAppliedTorque);
	}

	if (gdata->problem->simparams()->numbodies > 0) {
		Writer::WriteObjects(writers, gdata->t);
	}

	PostProcessEngineSet const& enabledPostProcess = gdata->simframework->getPostProcEngines();
	for (PostProcessEngineSet::const_iterator flt(enabledPostProcess.begin());
		flt != enabledPostProcess.end(); ++flt) {
		flt->second->write(writers, gdata->t);
	}

	Writer::WriteEnergy(writers, gdata->t, energy);

	Writer::Write(writers,
		gdata->processParticles[gdata->mpi_rank],
		gdata->s_hBuffers,
		node_offset,
		gdata->t, gdata->simframework->hasPostProcessEngine(TESTPOINTS));

	Writer::MarkWritten(writers, gdata->t);
}

/*! Save the particle system to disk.
 *
 * This method downloads all necessary buffers from devices to host,
 * after running the defined post-process functions, and invokes the write-out
 * routine.
 */
void GPUSPH::saveParticles(
	PostProcessEngineSet const& enabledPostProcess,
	string const& state,
	WriteFlags const& write_flags)
{
	const SimParams * const simparams = problem->simparams();

	// set the buffers to be dumped
	flag_t which_buffers = BUFFER_POS | BUFFER_VEL | BUFFER_INFO | BUFFER_HASH;

	if (gdata->debug.neibs)
		which_buffers |= BUFFER_NEIBSLIST;
	if (gdata->debug.forces)
		which_buffers |= BUFFER_FORCES;

	if (simparams->simflags & ENABLE_INTERNAL_ENERGY)
		which_buffers |= BUFFER_INTERNAL_ENERGY;

	// get GradGamma
	if (simparams->boundarytype == SA_BOUNDARY)
		which_buffers |= BUFFER_GRADGAMMA | BUFFER_VERTICES | BUFFER_BOUNDELEMENTS;

	if (simparams->sph_formulation == SPH_GRENIER)
		which_buffers |= BUFFER_VOLUME | BUFFER_SIGMA;

	// get k and epsilon
	if (simparams->turbmodel == KEPSILON)
		which_buffers |= BUFFER_TKE | BUFFER_EPSILON | BUFFER_TURBVISC;

	// Get SPS turbulent viscocity
	if (simparams->turbmodel == SPS)
		which_buffers |= BUFFER_SPS_TURBVISC;

	// Get effective viscocity
	if (NEEDS_EFFECTIVE_VISC(simparams->rheologytype))
		which_buffers |= BUFFER_EFFVISC;

	// get granular effective pressure
	if (simparams->rheologytype == GRANULAR)
		which_buffers |= BUFFER_EFFPRES;

	// get Eulerian velocity
	if (simparams->simflags & ENABLE_INLET_OUTLET ||
		simparams->turbmodel == KEPSILON)
		which_buffers |= BUFFER_EULERVEL;

	// get nextIDs
	// this must always be done, not just for debugging, because
	// it is needed for propert restart. If the data is deemed
	// unnecessary under normal condition and needs to be fenced
	// behind a debug.inspect_nextid, the check for it must be
	// done in the writers themselves, not here
	if (simparams->simflags & ENABLE_INLET_OUTLET)
		which_buffers |= BUFFER_NEXTID;

	// run post-process filters and dump their arrays
	// TODO migrate post-processing commands to command structure
	for (auto const& flt : enabledPostProcess) {
		PostProcessType filter = flt.first;
		AbstractPostProcessEngine *engine = flt.second;

		dispatchCommand(POSTPROCESS, filter);

		engine->hostProcess(gdata);

		/* list of buffers that were updated in-place */
		const flag_t updated_buffers = engine->get_updated_buffers();
		/* list of buffers that were written from scratch */
		const flag_t written_buffers = engine->get_written_buffers();

		const flag_t buffers_to_sync = updated_buffers | written_buffers;

		// If the post-process engine wrote something, we need to do
		// a multi-device update as well as a buffer swap
		const bool need_update = (buffers_to_sync != NO_FLAGS);

		/* TODO FIXME ideally we would have a way to specify when,
		 * after a post-processing, buffers need to be uploaded to other
		 * devices as well.
		 * This IS needed e.g. after the INFO update from SURFACE_DETECTION,
		 * even during pre-write post-processing, since otherwise buffers
		 * get out of sync across devices. Doing this update here is
		 * suboptimal, but otherwise we'd need to pass the information
		 * about the need to update external particles, and which buffers,
		 * to the main loop, which would be a bit of a mess in itself;
		 * so let's do it here for the time being.
		 */
#if 1
		if (MULTI_DEVICE && need_update) {
			CommandStruct update(UPDATE_EXTERNAL);
			update.updating("step n", buffers_to_sync);
			dispatchCommand(update);
		}
#endif

		which_buffers |= buffers_to_sync;
	}

	// TODO: the performanceCounter could be "paused" here

	// we do not care about ephemeral buffers during a hotwrite —
	// they will not be saved anyway, so exclude them
	// TODO a better solution would be to ask each involved writer
	// during this saveParticles() call which buffers do they
	// actually care about
	if (write_flags.hot_write)
		which_buffers &= ~EPHEMERAL_BUFFERS;

	// dump what we want to save
	CommandStruct dump(DUMP);
	dump.reading(state, which_buffers);
	dispatchCommand(dump);

	// triggers Writer->write()
	doWrite(write_flags);
}

// scan and check the peak number of neighbors and the estimated number of interactions
template<>
void GPUSPH::runCommand<CHECK_NEIBSNUM>(CommandStruct const& cmd)
// GPUSPH::checkNeibsNum()
{
	static const uint maxPossibleFluidBoundaryNeibs = problem->simparams()->neibboundpos;
	static const uint maxPossibleVertexNeibs = problem->simparams()->boundarytype == SA_BOUNDARY ? problem->simparams()->neiblistsize - problem->simparams()->neibboundpos - 2 : 0;
	for (uint d = 0; d < gdata->devices; d++) {
		const uint currDevMaxFluidBoundaryNeibs = gdata->timingInfo[d].maxFluidBoundaryNeibs;
		const uint currDevMaxVertexNeibs = gdata->timingInfo[d].maxVertexNeibs;

		if (currDevMaxFluidBoundaryNeibs > maxPossibleFluidBoundaryNeibs ||
			currDevMaxVertexNeibs > maxPossibleVertexNeibs) {
			printf("WARNING: current max. neighbors numbers (%u | %u) greater than max possible neibs (%u | %u) at iteration %lu\n",
				currDevMaxFluidBoundaryNeibs, currDevMaxVertexNeibs, maxPossibleFluidBoundaryNeibs, maxPossibleVertexNeibs, gdata->iterations);
			printf("\tpossible culprit: %d (neibs: %d + %d | %d)\n", gdata->timingInfo[d].hasTooManyNeibs,
				gdata->timingInfo[d].hasMaxNeibs[PT_FLUID],
				gdata->timingInfo[d].hasMaxNeibs[PT_BOUNDARY],
				gdata->timingInfo[d].hasMaxNeibs[PT_VERTEX]);
		}

		if (currDevMaxFluidBoundaryNeibs > gdata->lastGlobalPeakFluidBoundaryNeibsNum)
			gdata->lastGlobalPeakFluidBoundaryNeibsNum = currDevMaxFluidBoundaryNeibs;
		if (currDevMaxVertexNeibs > gdata->lastGlobalPeakVertexNeibsNum)
			gdata->lastGlobalPeakVertexNeibsNum = currDevMaxVertexNeibs;

		gdata->lastGlobalNumInteractions += gdata->timingInfo[d].numInteractions;
	}

	gdata->last_buildneibs_iteration = gdata->iterations;
}

// find if new particles were created on any device
template<>
void GPUSPH::runCommand<CHECK_NEWNUMPARTS>(CommandStruct const& cmd)
// GPUSPH::checkNewNumParts()
{
	gdata->particlesCreated = gdata->particlesCreatedOnNode[0];
	for (uint d = 1; d < gdata->devices; d++)
		gdata->particlesCreated |= gdata->particlesCreatedOnNode[d];
	// if runnign multinode, should also find the network minimum
	if (MULTI_NODE)
		gdata->networkManager->networkBoolReduction(
			&(gdata->particlesCreated), 1);

	// update the it counter if new particles are created
	if (gdata->particlesCreated) {
		gdata->createdParticlesIterations++;

		/*** IMPORTANT: updateArrayIndices() is only useful to be able to dump
		 * the newly generated particles on the upcoming (if any) save. HOWEVER,
		 * it introduces significant issued when used in multi-GPU, due
		 * to the fact that generated particles are appended after the externals.
		 * A method to handle this better needs to be devised (at worst enabling
		 * this only as a debug feature in single-GPU mode). For the time being
		 * the code section is disabled.
		 */
#if 0
		// we also update the array indices, so that e.g. when saving
		// the newly created particles are visible
		// TODO this doesn't seem to impact performance noticeably
		// in single-GPU. If it is found to be too expensive on
		// multi-GPU (or especially multi-node) it might be necessary
		// to only do it when saving. It does not affect the simulation
		// anyway, since it will be done during the next buildNeibList()
		// call
		updateArrayIndices();
#endif
	}
}

//! Invoke system callbacks
/*! Currently this only calls the variable-gravity callback.
 * Since this is invoked in-between steps, the simulated time t
 * for which the problem should be set up should be the one
 * for the next timestep
 */
template<>
void GPUSPH::runCommand<RUN_CALLBACKS>(CommandStruct const& cmd)
// GPUSPH::doCallBacks(flag_t current_integrator_step)
{
	ProblemCore *pb = gdata->problem;

	double t_callback = gdata->t + cmd.dt(gdata);

	if (pb->simparams()->gcallback)
		gdata->s_varGravity = pb->g_callback(t_callback);
}

void GPUSPH::printStatus(FILE *out)
{
//#define ti timingInfo
	fprintf(out, "%s time t=%es, iteration=%s, dt=%es, %s parts (%.2g, cum. %.2g MIPPS), maxneibs %u+%u\n",
		gdata->run_mode_Desc(),
			//"mean %e neibs. in %es, %e neibs/s, max %u neibs\n"
			//"mean neib list in %es\n"
			//"mean integration in %es\n",
			gdata->t, gdata->addSeparators(gdata->iterations).c_str(), gdata->dt,
			gdata->addSeparators(gdata->totParticles).c_str(), m_intervalPerformanceCounter->getMIPPS(),
			m_totalPerformanceCounter->getMIPPS(),
			gdata->lastGlobalPeakFluidBoundaryNeibsNum,
			gdata->lastGlobalPeakVertexNeibsNum
			//ti.t, ti.iterations, ti.dt, ti.numParticles, (double) ti.meanNumInteractions,
			//ti.meanTimeInteract, ((double)ti.meanNumInteractions)/ti.meanTimeInteract, ti.maxNeibs,
			//ti.meanTimeNeibsList,
			//ti.meanTimeEuler
			);
	fflush(out);
	// output to the info stream is always overwritten
	if (out == m_info_stream)
		fseek(out, 0, SEEK_SET);
//#undef ti
}

void GPUSPH::printParticleDistribution()
{
	printf("Particle distribution for process %u at iteration %lu:\n", gdata->mpi_rank, gdata->iterations);
	for (uint d = 0; d < gdata->devices; d++) {
		printf(" - Device %u: %u internal particles, %u total\n", d, gdata->s_hPartsPerDevice[d], gdata->GPUWORKERS[d]->getNumParticles());
		// Uncomment the following to detail the segments of each device
		/*
		if (MULTI_DEVICE) {
			printf("   Internal particles start at:      %u\n", gdata->s_dSegmentsStart[d][0]);
			printf("   Internal edge particles start at: %u\n", gdata->s_dSegmentsStart[d][1]);
			printf("   External edge particles start at: %u\n", gdata->s_dSegmentsStart[d][2]);
			printf("   External particles start at:      %u\n", gdata->s_dSegmentsStart[d][3]);
		}
		*/
	}
	printf("   TOT:   %u particles\n", gdata->processParticles[ gdata->mpi_rank ]);
}

// print peer accessibility for all devices
void GPUSPH::printDeviceAccessibilityTable()
{
	printf("Peer accessibility table:\n");
	// init line
	printf("-");
	for (uint d = 0; d <= gdata->devices; d++) printf("--------");
	printf("\n");

	// header
	printf("| READ >|");
	for (uint d = 0; d < gdata->devices; d++)
		printf(" %u (%u) |", d, gdata->device[d]);
	printf("\n");

	// header line
	printf("-");
	for (uint d = 0; d <= gdata->devices; d++) printf("--------");
	printf("\n");

	// rows
	for (uint d = 0; d < gdata->devices; d++) {
		printf("|");
		printf(" %u (%u) |", d, gdata->device[d]);
		for (uint p = 0; p < gdata->devices; p++) {
			if (p == d)
				printf("   -   |");
			else
			if (gdata->s_hDeviceCanAccessPeer[d][p])
				printf("   Y   |");
			else
				printf("   n   |");
		}
		printf("\n");
	}

	// closing line
	printf("-");
	for (uint d = 0; d <= gdata->devices; d++) printf("--------");
	printf("\n");
}


// Do a roll call of particle IDs; useful after dumps if the filling was uniform.
// Notifies anomalies only once in the simulation for each particle ID
// NOTE: only meaningful in single-node (otherwise, there is no correspondence between indices and ids),
// with compact particle filling (i.e. no holes in the ID space) and in simulations without open boundaries
void GPUSPH::rollCallParticles()
{
	// everything's ok till now?
	bool all_normal = true;
	// warn the user about the first anomaly only
	bool first_double_warned = false;
	bool first_missing_warned = false;
	// set this to true if we want to warn for every anomaly (for deep debugging)
	const bool WARN_EVERY_TIME = false;

	// reset bitmap and addrs
	for (uint part_id = 0; part_id < gdata->processParticles[gdata->mpi_rank]; part_id++) {
		m_rcBitmap[part_id] = false;
		m_rcAddrs[part_id] = UINT_MAX;
	}

	// fill out the bitmap and check for duplicates
	const particleinfo *particleInfo = gdata->s_hBuffers.getConstData<BUFFER_INFO>();
	for (uint part_index = 0; part_index < gdata->processParticles[gdata->mpi_rank]; part_index++) {
		uint part_id = id(particleInfo[part_index]);
		if (m_rcBitmap[part_id] && !m_rcNotified[part_id]) {
			if (WARN_EVERY_TIME || !first_double_warned) {
				printf("WARNING: at iteration %lu, time %g particle ID %u is at indices %u and %u!\n",
					gdata->iterations, gdata->t, part_id, m_rcAddrs[part_id], part_index);
				first_double_warned = true;
			}
			// getchar(); // useful for debugging
			// printf("Press ENTER to continue...\n");
			all_normal = false;
			m_rcNotified[part_id] = true;
		}
		m_rcBitmap[part_id] = true;
		m_rcAddrs[part_id] = part_index;
	}
	// now check if someone is missing
	for (uint part_id = 0; part_id < gdata->processParticles[gdata->mpi_rank]; part_id++)
		if (!m_rcBitmap[part_id] && !m_rcNotified[part_id]) {
			if (WARN_EVERY_TIME || !first_missing_warned) {
				printf("WARNING: at iteration %lu, time %g particle ID %u was not found!\n",
					gdata->iterations, gdata->t, part_id);
				first_missing_warned = true;
			}
			// printf("Press ENTER to continue...\n");
			// getchar(); // useful for debugging
			m_rcNotified[part_id] = true;
			all_normal = false;
		}
	// if there was any warning...
	if (!all_normal) {
		printf("Recap of devices after roll call:\n");
		for (uint d = 0; d < gdata->devices; d++) {
			printf(" - device at index %u has %s particles assigned and offset %s\n", d,
					gdata->addSeparators(gdata->s_hPartsPerDevice[d]).c_str(),
					gdata->addSeparators(gdata->s_hStartPerDevice[d]).c_str() );
			// extra stuff for deeper debugging
			// uint last_idx = gdata->s_hStartPerDevice[d] + gdata->s_hPartsPerDevice[d] - 1;
			// uint first_idx = gdata->s_hStartPerDevice[d];
			// printf("   first part has idx %u, last part has idx %u\n", id(gdata->s_hInfo[first_idx]), id(gdata->s_hInfo[last_idx])); */
		}
	}
}

// update s_hStartPerDevice, s_hPartsPerDevice and totParticles
// Could go in GlobalData but would need another forward-declaration
template<>
void GPUSPH::runCommand<UPDATE_ARRAY_INDICES>(CommandStruct const& cmd)
// GPUPSH::updateArrayIndices()
{
	static const auto debug_dump = CommandStruct(DUMP).reading("sorted", BUFFER_INFO);

	uint processCount = 0;

	// just store an incremental counter
	for (uint d = 0; d < gdata->devices; d++) {
		gdata->s_hPartsPerDevice[d] = gdata->GPUWORKERS[d]->getNumInternalParticles();
		processCount += gdata->s_hPartsPerDevice[d];
	}

	// update che number of particles of the current process. Do we need to store the previous value?
	// uint previous_process_parts = gdata->processParticles[ gdata->mpi_rank ];
	gdata->processParticles[gdata->mpi_rank] = processCount;

	// allgather values, aka: receive values of other processes
	if (MULTI_NODE)
		gdata->networkManager->allGatherUints(&processCount, gdata->processParticles);

	// now update the offsets for each device:
	gdata->s_hStartPerDevice[0] = 0;
	for (int n = 0; n < gdata->mpi_rank; n++) // first shift s_hStartPerDevice[0] by means of the previous nodes...
		gdata->s_hStartPerDevice[0] += gdata->processParticles[n];
	for (uint d = 1; d < gdata->devices; d++) // ...then shift the other devices by means of the previous devices
		gdata->s_hStartPerDevice[d] = gdata->s_hStartPerDevice[d-1] + gdata->s_hPartsPerDevice[d-1];

	/* Checking the total number of particles can be done by rank 0 process only if there are no inlets/outlets,
	 * since its aim is just error checking. However, in presence of inlets every process should have the
	 * updated number of active particles, at least for coherent status printing; thus, every process counts
	 * the particles and only rank 0 checks for correctness. */
	// WARNING: in case #parts changes with no open boundaries, devices with MPI rank different than 0 will keep a
	// wrong newSimulationTotal. Is this wanted? Harmful?
	if (gdata->mpi_rank == 0 || gdata->problem->simparams()->simflags & ENABLE_INLET_OUTLET) {
		uint newSimulationTotal = 0;
		for (uint n = 0; n < gdata->mpi_nodes; n++)
			newSimulationTotal += gdata->processParticles[n];

		// number of particle may increase or decrease if there are respectively inlets or outlets
		// TODO this should be simplified, but it would be better to check separately
		// for < and >, based on the number of inlets and outlets, so we leave
		// it this way as a reminder
		if ( (newSimulationTotal < gdata->totParticles && gdata->problem->simparams()->simflags & ENABLE_INLET_OUTLET) ||
			 (newSimulationTotal > gdata->totParticles && gdata->problem->simparams()->simflags & ENABLE_INLET_OUTLET) ) {
			// printf("Number of total particles at iteration %u passed from %u to %u\n", gdata->iterations, gdata->totParticles, newSimulationTotal);
			gdata->totParticles = newSimulationTotal;
		} else if (newSimulationTotal != gdata->totParticles && gdata->mpi_rank == 0) {

			// Ideally, only warn and make a roll call if
			// - total number of particles increased without inlets, or
			// - total number of particles decreased without outlets and no-leak-warning option was not passed
			// However, we use joint flag and counter for open boundaries (either in or out), so the actual logic
			// is a little different: we warn and roll call if
			// - total number of particles increased without inlets nor outlets, or
			// - total number of particles decreased without inlets nor outlets and no-leak-warning option was not passed
			if (newSimulationTotal > gdata->totParticles || !clOptions->no_leak_warning) {

				printf("WARNING: at iteration %lu the number of particles changed from %u to %u for no known reason!\n",
					gdata->iterations, gdata->totParticles, newSimulationTotal);

				// who is missing? if single-node, do a roll call
				if (SINGLE_NODE) {
					dispatchCommand(debug_dump);
					rollCallParticles();
				}
			}

			// update totParticles to avoid dumping an outdated particle (and repeating the warning).
			// Note: updading *after* the roll call likely shows the missing particle(s) and the duplicate(s). Doing before it only shows the missing one(s)
			gdata->totParticles = newSimulationTotal;
		}
	}

	// in case estimateMaxInletsIncome() was slightly in defect (unlikely)
	// FIXME: like in other methods, we should avoid quitting only one process
	if (processCount > gdata->allocatedParticles) {
		printf( "FATAL: Number of total particles at iteration %lu (%u) exceeding allocated buffers (%u). Requesting immediate quit\n",
				gdata->iterations, processCount, gdata->allocatedParticles);
		gdata->quit_request = true;
	}
}

// perform post-filling operations
void GPUSPH::prepareProblem()
{
	const particleinfo *infos = gdata->s_hBuffers.getConstData<BUFFER_INFO>();
	const hashKey *hashes = gdata->s_hBuffers.getConstData<BUFFER_HASH>();

	//nGridCells

	// should write something more meaningful here
	printf("Preparing the problem...\n");

	// at the time being, we only need preparation for multi-device simulations
	if (!MULTI_DEVICE) return;

	for (uint p = 0; p < gdata->totParticles; p++) {
		// For DYN bounds, take into account also boundary parts; for other boundary types,
		// only cound fluid parts
		if ( problem->simparams()->boundarytype != LJ_BOUNDARY || FLUID(infos[p]) ) {
			const uint cellHash = cellHashFromParticleHash( hashes[p] );
			const uint3 cellCoords = gdata->calcGridPosFromCellHash( cellHash );
			// NOTE: s_hPartsPerSliceAlong* are only allocated if MULTI_DEVICE holds.
			// Change the loop accordingly if other operations are performed!
			gdata->s_hPartsPerSliceAlongX[ cellCoords.x ]++;
			gdata->s_hPartsPerSliceAlongY[ cellCoords.y ]++;
			gdata->s_hPartsPerSliceAlongZ[ cellCoords.z ]++;
		}
	}
}

template<>
void GPUSPH::runCommand<FIND_MAX_IOWATERDEPTH>(CommandStruct const& cmd)
// GPUSPH::findMaxWaterDepth()
{
	static const uint numOpenBoundaries = problem->simparams()->numOpenBoundaries;
	static uint *max_waterdepth = gdata->h_maxIOwaterdepth;

	// max over all devices per node
	for (uint ob = 0; ob < numOpenBoundaries; ob ++) {
		max_waterdepth[ob] = 0;
		for (uint d = 0; d < gdata->devices; d++)
			max_waterdepth[ob] = max(max_waterdepth[ob], gdata->h_IOwaterdepth[d][ob]);
	}
	// if we are in multi-node mode we need to run an mpi reduction over all nodes
	if (MULTI_NODE) {
		gdata->networkManager->networkIntReduction(
			(int*)max_waterdepth, numOpenBoundaries, MAX_REDUCTION);
	}
	// copy global value back to one array so that we can upload it again
	for (uint ob = 0; ob < numOpenBoundaries; ob ++)
		gdata->h_IOwaterdepth[0][ob] = max_waterdepth[ob];
}

template<>
void GPUSPH::runCommand<DEBUG_DUMP>(CommandStruct const& cmd)
{
	saveParticles(noPostProcess, cmd.src, cmd.step);
}

void GPUSPH::check_write(bool we_are_done)
{
	static PostProcessEngineSet const& enabledPostProcess = gdata->simframework->getPostProcEngines();
	// list of writers that need to write at this timestep
	ConstWriterMap writers = Writer::NeedWrite(gdata->t);

	// we need to write if any writer is configured to write at this timestep
	// i.e. if the writers list is not empty
	const bool need_write = !writers.empty();

	// do we want to write even if no writer is asking to?
	const bool force_write =
		// ask the problem if we want to write anyway
		gdata->problem->need_write(gdata->t) ||
		// always write if we're done with the simulation
		we_are_done ||
		// write if it was requested
		gdata->save_request;

	// reset save_request, we're going to satisfy it anyway
	if (force_write)
		gdata->save_request = false;

	if (need_write || force_write) {
		if (gdata->clOptions->nosave && !force_write) {
			// we want to avoid writers insisting we need to save,
			// so pretend we actually saved
			Writer::FakeMarkWritten(writers, gdata->t);
		} else {
			saveParticles(enabledPostProcess, "step n", force_write);

			// we generally want to print the current status and reset the
			// interval performance counter when writing. However, when writing
			// at every timestep, this can be very bothersome (lots and lots of
			// output) so we do not print the status if the only writer(s) that
			// have been writing have a frequency of 0 (write every timestep)
			// TODO the logic here could be improved; for example, we are not
			// considering the case of a single writer that writes at every timestep:
			// when do we print the status then?
			// TODO other enhancements would be to print who is writing (what)
			// during the print status
			double maxfreq = 0;
			ConstWriterMap::iterator it(writers.begin());
			ConstWriterMap::iterator end(writers.end());
			while (it != end) {
				double freq = it->second->get_write_freq();
				if (freq > maxfreq)
					maxfreq = freq;
				++it;
			}
			if (force_write || maxfreq > 0) {
				printStatus();
				m_intervalPerformanceCounter->restart();
			}
		}
	}
}

template<>
void GPUSPH::runCommand<JACOBI_RESET_STOP_CRITERION>(CommandStruct const& cmd)
{
	printf("JACOBI PREP\n");
	gdata->h_jacobiStop = false;
	gdata->h_jacobiCounter = 0;
}


template<>
void GPUSPH::runCommand<JACOBI_STOP_CRITERION>(CommandStruct const& cmd)
{
	// if we are in multi-node mode we need to run an mpi reduction over all nodes
	if (MULTI_NODE) {
		gdata->networkManager->networkFloatReduction(&(gdata->h_jacobiResidual), 1, MAX_REDUCTION);
		gdata->networkManager->networkFloatReduction(&(gdata->h_jacobiBackwardError), 1, MAX_REDUCTION);
	}

	if ((gdata->h_jacobiBackwardError < 1e-4 && gdata->h_jacobiResidual < 1e-4) || gdata->h_jacobiCounter > 100) {
		gdata->h_jacobiStop = true;
		printf("TRUE - counter = %d\terror[0] = %f\tresidual[0] = %f\n", gdata->h_jacobiCounter, gdata->h_jacobiBackwardError, gdata->h_jacobiResidual);
	} else {
		gdata->h_jacobiCounter++;
		printf("FALSE - counter = %d\terror[0] = %f\tresidual[0] = %f\n", gdata->h_jacobiCounter, gdata->h_jacobiBackwardError, gdata->h_jacobiResidual);
	}
}

//! Auxiliary class to time a command execution
//! The TimerObject itself checks the time from its own creation to its own destruction,
//! and updates two associated durations (max and total).
//! dispatchCommand() leverages this by creating a TimerObject before issuing the command,
//! so that on return from dispatchCommand() the TimerObject destructor updates the effective
//! runtime of the corresponding object.
//! \note Nested commands contribute to the calling command runtimes.
struct TimerObject
{
	using clock = GPUSPH::cmd_time_clock;
	using duration = GPUSPH::cmd_time_duration;

	duration &max_ref;
	duration &tot_ref;
	std::chrono::time_point<clock> start;

	TimerObject(duration &max_ref_, duration &tot_ref_) :
		max_ref(max_ref_),
		tot_ref(tot_ref_),
		start(clock::now())
	{ }

	~TimerObject()
	{
		auto stop = clock::now();
		auto duration = stop - start;
		tot_ref += duration;
		if (duration > max_ref)
			max_ref = duration;
	}
};

// set nextCommand, unlock the threads and wait for them to complete
void GPUSPH::dispatchCommand(CommandStruct const& cmd)
{
	shared_ptr<TimerObject> timer;
	if (gdata->debug.benchmark_command_runtimes) {
		++cmd_calls[cmd.command];
		timer = make_shared<TimerObject>(
			max_cmd_time[cmd.command],
			tot_cmd_time[cmd.command]);
	}

	// resetting the host buffers is useful to check if the arrays are completely filled
	/*/ if (cmd==DUMP) {
	 const uint float4Size = sizeof(float4) * gdata->totParticles;
	 const uint infoSize = sizeof(particleinfo) * gdata->totParticles;
	 memset(gdata->s_hPos, 0, float4Size);
	 memset(gdata->s_hVel, 0, float4Size);
	 memset(gdata->s_hInfo, 0, infoSize);
	 } */

	if (MULTI_NODE && gdata->networkManager->checkKillRequest())
		throw runtime_error("GPUSPH killed by MPI kill request");

	if (cmd.command > NUM_WORKER_COMMANDS) {
		switch (cmd.command) {
#define DEFINE_COMMAND(code, ...) \
			case code: \
				/* TODO if (dbg_step_printf) describeCommand<code>(cmd); */ \
				runCommand<code>(cmd); \
				break;
#include "define_host_commands.h"
#undef DEFINE_COMMAND
			default:
				unknownCommand(cmd.command);
		}
		return;
	}

	gdata->nextCommand = cmd;
	gdata->threadSynchronizer->barrier(); // unlock CYCLE BARRIER 2
	gdata->threadSynchronizer->barrier(); // wait for completion of last command and unlock CYCLE BARRIER 1

	if (!gdata->keep_going)
		throw runtime_error("GPUSPH aborted by worker thread");

	// Check buffer consistency after every call.
	// Don't even bother with the conditional if it's not enabled though.
	// TODO as things are now, all the calls from the first APPEND_EXTERNAL
	// to the first EULER will complain about inconsistency in the WRITE buffers.
	// With knowledge about which buffers are read/written to by each command, we
	// could restrict ourselves to check those buffers; with the upcoming new
	// Integrator and ParticleSystem classes, this will be easier, so let's not
	// waste too much time on it at the moment.
#ifdef INSPECT_DEVICE_MEMORY
	if (MULTI_DEVICE && gdata->debug.check_buffer_consistency)
		checkBufferConsistency();
#endif
}
