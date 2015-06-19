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

#include <float.h> // FLT_EPSILON

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

/* Include only the problem selected at compile time */
#include "problem_select.opt"

// HotFile
#include "HotFile.h"

/* Include all other opt file for show_version */
#include "gpusph_version.opt"
#include "fastmath_select.opt"
#include "compute_select.opt"

using namespace std;

GPUSPH* GPUSPH::getInstance() {
	// guaranteed to be destroyed; instantiated on first use
	static GPUSPH instance;
	// return a reference, not just a pointer
	return &instance;
}

GPUSPH::GPUSPH() {
	clOptions = NULL;
	gdata = NULL;
	problem = NULL;

	initialized = false;
	m_peakParticleSpeed = 0.0;
	m_peakParticleSpeedTime = 0.0;

	openInfoStream();
}

GPUSPH::~GPUSPH() {
	closeInfoStream();
	// it would be useful to have a "fallback" deallocation but we have to check
	// that main did not do that already
	if (initialized) finalize();
}

void GPUSPH::openInfoStream() {
	stringstream ss;
	ss << "GPUSPH-" << getpid();
	m_info_stream_name = ss.str();
	m_info_stream = NULL;
	int ret = shm_open(m_info_stream_name.c_str(), O_RDWR | O_CREAT, S_IRWXU);
	if (ret < 0) {
		cerr << "WARNING: unable to open info stream " << m_info_stream_name << endl;
		return;
	}
	m_info_stream = fdopen(ret, "w");
	if (!m_info_stream) {
		cerr << "WARNING: unable to fdopen info stream " << m_info_stream_name << endl;
		close(ret);
		shm_unlink(m_info_stream_name.c_str());
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

	// needed for the new problem interface (compute worldorigin, init ODE, etc.)
	if (!problem->initialize()) {
		printf("Problem initialization failed. Aborting...\n");
		return false;
	}

	// run post-construction functions
	problem->check_dt();
	problem->check_maxneibsnum();
	problem->create_problem_dir();
	problem->calculateFerrariCoefficient();

	printf("Problem calling set grid params\n");
	problem->set_grid_params();

	// sets the correct viscosity coefficient according to the one set in SimParams
	setViscosityCoefficient();

	m_totalPerformanceCounter = new IPPSCounter();
	m_intervalPerformanceCounter = new IPPSCounter();
	// only init if MULTI_NODE
	m_multiNodePerformanceCounter = NULL;
	if (MULTI_NODE)
		m_multiNodePerformanceCounter = new IPPSCounter();

	// utility pointer
	SimParams *_sp = gdata->problem->get_simparams();

	// copy the options passed by command line to GlobalData
	if (isfinite(clOptions->tend))
		_sp->tend = clOptions->tend;

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
#define STR(macro) #macro
#define COORD_NAME(coord) STR(coord)
	printf(" - Cell linearizazion: %s,%s,%s\n", COORD_NAME(COORD1), COORD_NAME(COORD2),
		COORD_NAME(COORD3));
#undef COORD_NAME
#undef STR
	printf(" - Dp:   %g\n", gdata->problem->m_deltap);
	printf(" - R0:   %g\n", gdata->problem->get_physparams()->r0);


	// initial dt (or, just dt in case adaptive is disabled)
	gdata->dt = _sp->dt;

	printf("Generating problem particles...\n");
	// allocate the particles of the *whole* simulation
	gdata->totParticles = problem->fill_parts();

	// Determine the highest ID per device for unambiguous particle creation
	for (uint d=0; d < gdata->devices; d++)
		gdata->highestDevId[d] = gdata->totParticles + GlobalData::GLOBAL_DEVICE_ID(gdata->mpi_rank, d);

	// Allocate internal storage for moving bodies
	problem->allocate_bodies_storage();

	// the number of allocated particles will be bigger, to be sure it can contain particles being created
	// WARNING: particle creation in inlets also relies on this, do not disable if using inlets
	gdata->allocatedParticles = problem->max_parts(gdata->totParticles);

	// generate planes, will be allocated in allocateGlobalHostBuffers()
	gdata->numPlanes = problem->fill_planes();

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

	// copy planes from the problem to the shared array, if there are any,
	// and convert them into the form requird for homogeneous accuracy
	if (gdata->numPlanes) {
		problem->copy_planes(gdata->s_hPlanes);

		/* domain centerpoint, used to find the plane point below */
		const double3 midPoint = problem->get_worldorigin() + problem->get_worldsize()/2;
		for (uint i = 0; i < gdata->numPlanes; ++i) {
			const double4 &p = gdata->s_hPlanes[i];
			double norm = length3(p);
			const double3 normal = as_double3(p)/norm;
			gdata->s_hPlaneNormal[i] = make_float3(normal);

			/* For the plane point, we pick the one closest to the center of the domain
			 * TODO find a better logic ? */

			/* note that to compute the distance we dot with the normal, and then
			 * add p.w/norm, since p.w is still not normalized */
			const double midDist = dot(midPoint, normal) + p.w/norm;
			double3 planePoint = midPoint - midDist*normal;

			problem->calc_grid_and_local_pos(planePoint,
				gdata->s_hPlanePointGridPos + i,
				gdata->s_hPlanePointLocalPos + i);
		}
	}

	/* Now we either copy particle data from the Problem to the GPUSPH buffers,
	 * or, if it was requested, we load buffers from a HotStart file
	 */
	/* TODO FIXME copying data from the Problem doubles the host memory
	 * requirements, find some smart way to have the host fill the shared
	 * buffer directly.
	 */
	bool resumed = false;

	if (clOptions->resume_fname.empty()) {
		printf("Copying the particles to shared arrays...\n");
		printf("---\n");
		problem->copy_to_array(gdata->s_hBuffers);
		printf("---\n");
	} else {
		struct stat statbuf;
		ifstream hot_in;
		ostringstream err_msg;
		HotFile *hf = NULL;
		const char *fname = clOptions->resume_fname.c_str();
		cout << "Hot starting from " << fname << "..." << endl;
		if (stat(fname, &statbuf)) {
			// stat failed
			err_msg << "Hot start file " << fname << " not found";
			throw runtime_error(err_msg.str());
		}
		/* enable automatic exception handling on failure */
		hot_in.exceptions(ifstream::failbit | ifstream::badbit);
		hot_in.open(fname);
		hf = new HotFile(hot_in, gdata);
		hf->load();
		cerr << "Successfully restored hot start file" << endl;
		cerr << *hf << endl;
		cerr << "Restarting from t=" << hf->get_t()
			<< ", iteration=" << hf->get_iterations()
			<< ", dt=" << hf->get_dt() << endl;
		// warn about possible discrepancies in case of ODE objects
		if (problem->get_simparams()->numbodies) {
			cerr << "WARNING: simulation has rigid bodies and/or moving boundaries, resume will not give identical results" << endl;
		}
		gdata->iterations = hf->get_iterations();
		gdata->dt = hf->get_dt();
		gdata->t = hf->get_t();
		hot_in.close();
		resumed = true;
	}

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
	if (gdata->problem->get_simparams()->numbodies > 0) {
		gdata->problem->get_bodies_cg();
	}

	// initialize values of k and e for k-e model
	if (!resumed && _sp->visctype == KEPSVISC)
		problem->init_keps(
			gdata->s_hBuffers.getData<BUFFER_TKE>(),
			gdata->s_hBuffers.getData<BUFFER_EPSILON>(),
			gdata->totParticles,
			gdata->s_hBuffers.getData<BUFFER_INFO>(),
			gdata->s_hBuffers.getData<BUFFER_POS>(),
			gdata->s_hBuffers.getData<BUFFER_HASH>());

	if (!resumed && _sp->sph_formulation == SPH_GRENIER)
		problem->init_volume(gdata->s_hBuffers, gdata->totParticles);

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

	// TODO
	//		// > new Integrator

	// new Synchronizer; it will be waiting on #devices+1 threads (GPUWorkers + main)
	gdata->threadSynchronizer = new Synchronizer(gdata->devices + 1);

	printf("Starting workers...\n");

	// allocate workers
	gdata->GPUWORKERS = (GPUWorker**)calloc(gdata->devices, sizeof(GPUWorker*));
	for (uint d=0; d < gdata->devices; d++)
		gdata->GPUWORKERS[d] = new GPUWorker(gdata, d);

	gdata->keep_going = true;

	// actually start the threads
	for (uint d = 0; d < gdata->devices; d++)
		gdata->GPUWORKERS[d]->run_worker(); // begin of INITIALIZATION ***

	// The following barrier waits for GPUworkers to complete CUDA init, GPU allocation, subdomain and devmap upload

	gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***

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
	for (uint d = 0; d < gdata->devices; d++)
		delete gdata->GPUWORKERS[d];

	free(gdata->GPUWORKERS);

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

bool GPUSPH::runSimulation() {
	if (!initialized) return false;

	// doing first write
	printf("Performing first write...\n");
	doWrite(true);

	printf("Letting threads upload the subdomains...\n");
	gdata->threadSynchronizer->barrier(); // begins UPLOAD ***

	// here the Workers are uploading their subdomains

	// After next barrier, the workers will enter their simulation cycle, so it is recommended to set
	// nextCommand properly before the barrier (although should be already initialized to IDLE).
	// doCommand(IDLE) would be equivalent, but this is more clear
	gdata->nextCommand = IDLE;
	gdata->threadSynchronizer->barrier(); // end of UPLOAD, begins SIMULATION ***
	gdata->threadSynchronizer->barrier(); // unlock CYCLE BARRIER 1

	// this is where we invoke initialization routines that have to be
	// run by the GPUWokers

	if (problem->get_simparams()->boundarytype == SA_BOUNDARY) {

		// compute neighbour list for the first time
		buildNeibList();

		// set density and other values for segments and vertices
		saBoundaryConditions(INITIALIZATION_STEP);

	}

	printf("Entering the main simulation cycle\n");

	//  IPPS counter does not take the initial uploads into consideration
	m_totalPerformanceCounter->start();
	m_intervalPerformanceCounter->start();
	if (MULTI_NODE)
		m_multiNodePerformanceCounter->start();

	// write some info. This could replace "Entering the main simulation cycle"
	printStatus();

	buildNeibList();

	FilterFreqList const& enabledFilters = gdata->simframework->getFilterFreqList();
	PostProcessEngineSet const& enabledPostProcess = gdata->simframework->getPostProcEngines();

	while (gdata->keep_going) {
		printStatus(m_info_stream);
		// when there will be an Integrator class, here (or after bneibs?) we will
		// call Integrator -> setNextStep

		// run enabled filters
		if (gdata->iterations > 0) {
			FilterFreqList::const_iterator flt(enabledFilters.begin());
			FilterFreqList::const_iterator flt_end(enabledFilters.end());
			while (flt != flt_end) {
				FilterType filter = flt->first;
				uint freq = flt->second; // known to be > 0
				if (gdata->iterations % freq == 0) {
					gdata->only_internal = true;
					doCommand(FILTER, NO_FLAGS, float(filter));
					// update before swapping, since UPDATE_EXTERNAL works on write buffers
					if (MULTI_DEVICE)
						doCommand(UPDATE_EXTERNAL, BUFFER_VEL | DBLBUFFER_WRITE);
					doCommand(SWAP_BUFFERS, BUFFER_VEL);
				}
				++flt;
			}
		}

		//			//(init bodies)


		// variable gravity
		if (problem->get_simparams()->gcallback) {
			// ask the Problem to update gravity, one per process
			doCallBacks();
			// upload on the GPU, one per device
			doCommand(UPLOAD_GRAVITY);
		}

		// for Grenier formulation, compute sigma and smoothed density
		if (problem->get_simparams()->sph_formulation == SPH_GRENIER) {
			// put READ vel in WRITE buffer
			doCommand(SWAP_BUFFERS, BUFFER_VEL);
			gdata->only_internal = true;

			// compute density and sigma, updating WRITE vel in-place
			doCommand(COMPUTE_DENSITY, INTEGRATOR_STEP_1);
			if (MULTI_DEVICE)
				doCommand(UPDATE_EXTERNAL, BUFFER_SIGMA | BUFFER_VEL | DBLBUFFER_WRITE);
			// restore vel buffer into READ position
			doCommand(SWAP_BUFFERS, BUFFER_VEL);
		}

		// for SPS viscosity, compute first array of tau and exchange with neighbors
		if (problem->get_simparams()->visctype == SPSVISC) {
			gdata->only_internal = true;
			doCommand(SPS, INTEGRATOR_STEP_1);
			if (MULTI_DEVICE)
				doCommand(UPDATE_EXTERNAL, BUFFER_TAU);
		}

		// compute forces only on internal particles
		gdata->only_internal = true;
		if (gdata->clOptions->striping && MULTI_DEVICE)
			doCommand(FORCES_ENQUEUE, INTEGRATOR_STEP_1);
		else
			doCommand(FORCES_SYNC, INTEGRATOR_STEP_1);

		// update forces of external particles
		if (MULTI_DEVICE)
			doCommand(UPDATE_EXTERNAL, POST_FORCES_UPDATE_BUFFERS | DBLBUFFER_WRITE);

		// if striping was active, now we want the kernels to complete
		if (gdata->clOptions->striping && MULTI_DEVICE)
			doCommand(FORCES_COMPLETE, INTEGRATOR_STEP_1);

		//MM		fetch/update forces on neighbors in other GPUs/nodes
		//				initially done trivial and slow: stop and read
		//			//reduce bodies

		// boundelements is swapped because the normals are updated in the moving objects case
		doCommand(SWAP_BUFFERS, BUFFER_BOUNDELEMENTS);

		// Take care of moving bodies
		// TODO: use INTEGRATOR_STEP
		move_bodies(1);

		// in the case of the summation density there is a neighbour loop in euler and so we can run on internal only
		if (!(problem->get_simparams()->simflags & ENABLE_DENSITY_SUM))
			// integrate also the externals
			gdata->only_internal = false;

		doCommand(EULER, INTEGRATOR_STEP_1);

		// summation density requires an update from the other GPUs.
		if (problem->get_simparams()->simflags & ENABLE_DENSITY_SUM) {
			if (MULTI_DEVICE) {
				doCommand(UPDATE_EXTERNAL, BUFFER_POS | BUFFER_VEL | BUFFER_EULERVEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_BOUNDELEMENTS | BUFFER_GRADGAMMA | DBLBUFFER_WRITE);
				// the following only need update after the first step, vel due to rhie and chow and gradgamma to save gam^n
				doCommand(UPDATE_EXTERNAL, BUFFER_VEL | BUFFER_GRADGAMMA | DBLBUFFER_READ);
			}
		}

		doCommand(SWAP_BUFFERS, BUFFER_BOUNDELEMENTS);

		//			//reduce bodies
		//MM		fetch/update forces on neighbors in other GPUs/nodes
		//				initially done trivial and slow: stop and read

		// variable gravity
		if (problem->get_simparams()->gcallback) {
			// ask the Problem to update gravity, one per process
			doCallBacks();
			// upload on the GPU, one per device
			doCommand(UPLOAD_GRAVITY);
		}

		// semi-analytical boundary conditions
		if (problem->get_simparams()->boundarytype == SA_BOUNDARY)
			saBoundaryConditions(INTEGRATOR_STEP_1);

		doCommand(SWAP_BUFFERS, POST_COMPUTE_SWAP_BUFFERS);

		// Here the first part of our time integration scheme is complete. All updated values
		// are now in the read buffers again.

		// for Grenier formulation, compute sigma and smoothed density
		if (problem->get_simparams()->sph_formulation == SPH_GRENIER) {
			// put READ vel in WRITE buffer
			doCommand(SWAP_BUFFERS, BUFFER_VEL);
			gdata->only_internal = true;

			// compute density and sigma, updating WRITE vel in-place
			doCommand(COMPUTE_DENSITY, INTEGRATOR_STEP_2);
			if (MULTI_DEVICE)
				doCommand(UPDATE_EXTERNAL, BUFFER_SIGMA | BUFFER_VEL | DBLBUFFER_WRITE);
			// restore vel buffer into READ position
			doCommand(SWAP_BUFFERS, BUFFER_VEL);
		}

		// for SPS viscosity, compute first array of tau and exchange with neighbors
		if (problem->get_simparams()->visctype == SPSVISC) {
			gdata->only_internal = true;
			doCommand(SPS, INTEGRATOR_STEP_2);
			if (MULTI_DEVICE)
				doCommand(UPDATE_EXTERNAL, BUFFER_TAU);
		}

		gdata->only_internal = true;
		if (gdata->clOptions->striping && MULTI_DEVICE)
			doCommand(FORCES_ENQUEUE, INTEGRATOR_STEP_2);
		else
			doCommand(FORCES_SYNC, INTEGRATOR_STEP_2);

		// update forces of external particles
		if (MULTI_DEVICE)
			doCommand(UPDATE_EXTERNAL, POST_FORCES_UPDATE_BUFFERS | DBLBUFFER_WRITE);

		// if striping was active, now we want the kernels to complete
		if (gdata->clOptions->striping && MULTI_DEVICE)
			doCommand(FORCES_COMPLETE, INTEGRATOR_STEP_2);

		// swap read and writes again because the write contains the variables at time n
		// boundelements is swapped because the normals are updated in the moving objects case
		doCommand(SWAP_BUFFERS, BUFFER_POS | BUFFER_VEL | BUFFER_VOLUME | BUFFER_TKE | BUFFER_EPSILON | BUFFER_BOUNDELEMENTS);

		// Take care of moving bodies
		// TODO: use INTEGRATOR_STEP
		move_bodies(2);

		// in the case of the summation density there is a neighbour loop in euler and so we can run on internal only
		if (!(problem->get_simparams()->simflags & ENABLE_DENSITY_SUM))
			// integrate also the externals
			gdata->only_internal = false;

		doCommand(EULER, INTEGRATOR_STEP_2);

		// summation density requires an update from the other GPUs.
		if (problem->get_simparams()->simflags & ENABLE_DENSITY_SUM) {
			if (MULTI_DEVICE) {
				doCommand(UPDATE_EXTERNAL, BUFFER_POS | BUFFER_VEL | BUFFER_EULERVEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_BOUNDELEMENTS | BUFFER_GRADGAMMA | DBLBUFFER_WRITE);
			}
		}

		// Euler needs always cg(n)
		if (problem->get_simparams()->numbodies > 0)
			doCommand(EULER_UPLOAD_OBJECTS_CG);

		doCommand(SWAP_BUFFERS, BUFFER_BOUNDELEMENTS);

		// semi-analytical boundary conditions
		if (problem->get_simparams()->boundarytype == SA_BOUNDARY)
			saBoundaryConditions(INTEGRATOR_STEP_2);

		// update inlet/outlet changes only after step 2
		// and check if a forced buildneibs is required (i.e. if particles were created)
		if (problem->get_simparams()->simflags & ENABLE_INLET_OUTLET){
			doCommand(DOWNLOAD_NEWNUMPARTS);

			gdata->particlesCreated = gdata->particlesCreatedOnNode[0];
			for (uint d = 1; d < gdata->devices; d++)
				gdata->particlesCreated |= gdata->particlesCreatedOnNode[d];
			// if runnign multinode, should also find the network minimum
			if (MULTI_NODE)
				gdata->networkManager->networkBoolReduction(&(gdata->particlesCreated), 1);

			// update the it counter if new particles are created
			if (gdata->particlesCreated)
				gdata->createdParticlesIterations++;
		}

		doCommand(SWAP_BUFFERS, POST_COMPUTE_SWAP_BUFFERS);

		// Here the second part of our time integration scheme is complete, i.e. the time-step is
		// fully computed. All updated values are now in the read buffers again.

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
		if (gdata->problem->get_simparams()->simflags & ENABLE_DTADAPT) {
			gdata->dt = gdata->dts[0];
			for (uint d = 1; d < gdata->devices; d++)
				gdata->dt = min(gdata->dt, gdata->dts[d]);
			// if runnin multinode, should also find the network minimum
			if (MULTI_NODE)
				gdata->networkManager->networkFloatReduction(&(gdata->dt), 1, MIN_REDUCTION);
		}

		// check that dt is not too small (absolute)
		if (!gdata->t) {
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
		bool finished = gdata->problem->finished(gdata->t);
		// list of writers that need to write
		ConstWriterMap writers = Writer::NeedWrite(gdata->t);
		// do we need to write?
		bool need_write = !writers.empty();
		// do we want to write anyway? (the problem want us to write, or we are done,
		// or we are quitting)
		bool force_write = gdata->problem->need_write(gdata->t) || finished || gdata->quit_request;
		if (gdata->save_request) {
			force_write = true;
			gdata->save_request = false;
		}

		// If we are about to quit, we want to save regardless --nosave option
		if (finished || gdata->quit_request)
			force_write = true;

		// Launch specific post processing kernels (vorticity, free surface detection , ...)
		// before writing to disk
		if (need_write || force_write) {

			//if (final_save)
			//	printf("Issuing final save...\n");

			// set the buffers to be dumped
			flag_t which_buffers = BUFFER_POS | BUFFER_VEL | BUFFER_INFO | BUFFER_HASH;

			// choose the read buffer for the double buffered arrays
			which_buffers |= DBLBUFFER_READ;

			// get GradGamma
			if (gdata->problem->get_simparams()->boundarytype == SA_BOUNDARY)
				which_buffers |= BUFFER_GRADGAMMA | BUFFER_VERTICES;

			if (gdata->problem->get_simparams()->sph_formulation == SPH_GRENIER)
				which_buffers |= BUFFER_VOLUME;

			// get k and epsilon
			if (gdata->problem->get_simparams()->visctype == KEPSVISC)
				which_buffers |= BUFFER_TKE | BUFFER_EPSILON | BUFFER_TURBVISC;

			// Get SPS turbulent viscocity
			if (gdata->problem->get_simparams()->visctype == SPSVISC)
				which_buffers |= BUFFER_SPS_TURBVISC;

			// get Eulerian velocity
			if (gdata->problem->get_simparams()->simflags & ENABLE_INLET_OUTLET ||
				gdata->problem->get_simparams()->visctype == KEPSVISC)
				which_buffers |= BUFFER_EULERVEL;

			// run post-process filters and dump their arrays
			for (PostProcessEngineSet::const_iterator flt(enabledPostProcess.begin());
				flt != enabledPostProcess.end(); ++flt) {
				PostProcessType filter = flt->first;
				gdata->only_internal = true;
				doCommand(POSTPROCESS, NO_FLAGS, float(filter));

				/* list of buffers that were updated in-place */
				const flag_t updated_buffers = flt->second->get_updated_buffers();
				/* list of buffers that were written in BUFFER_WRITE */
				const flag_t written_buffers = flt->second->get_written_buffers();
				/* TODO FIXME ideally we would have a way to specify when,
				 * after a post-processing, buffers need to be uploaded to other
				 * devices as well.
				 * This might be needed e.g. after the INFO update from SURFACE_DETECTION,
				 * although maybe not during pre-write post-processing
				 */
#if 0
				if (MULTI_DEVICE)
					doCommand(UPDATE_EXTERNAL, written_buffers | DBLBUFFER_WRITE);
#endif
				/* Swap the written buffers, so we can access the new data from
				 * DBLBUFFER_READ
				 */
				doCommand(SWAP_BUFFERS, written_buffers);
				which_buffers |= updated_buffers | written_buffers;
			}

			if ( force_write || !gdata->nosave) {
				// TODO: the performanceCounter could be "paused" here
				// dump what we want to save
				doCommand(DUMP, which_buffers);
				// triggers Writer->write()
				doWrite(force_write);
			} else
				// --nosave enabled, not final: just pretend we actually saved
				Writer::MarkWritten(gdata->t, true);

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

		// build neighbors list
		if (gdata->iterations % problem->get_simparams()->buildneibsfreq == 0 ||
			gdata->particlesCreated) {
			buildNeibList();
		}

		if (finished || gdata->quit_request)
			// NO doCommand() after keep_going has been unset!
			gdata->keep_going = false;
	}

	// elapsed time, excluding the initialization
	printf("Elapsed time of simulation cycle: %.2gs\n", m_totalPerformanceCounter->getElapsedSeconds());

	// In multinode simulations we also print the global performance. To make only rank 0 print it, add
	// the condition (gdata->mpi_rank == 0)
	if (MULTI_NODE)
		printf("Global performance of the multinode simulation: %.2g MIPPS\n", m_multiNodePerformanceCounter->getMIPPS());

	// suggest max speed for next runs
	printf("Peak particle speed was ~%g m/s at %g s -> can set maximum vel %.2g for this problem\n",
		m_peakParticleSpeed, m_peakParticleSpeedTime, (m_peakParticleSpeed*1.1));

	// NO doCommand() nor other barriers than the standard ones after the

	printf("Simulation end, cleaning up...\n");

	// doCommand(QUIT) would be equivalent, but this is more clear
	gdata->nextCommand = QUIT;
	gdata->threadSynchronizer->barrier(); // unlock CYCLE BARRIER 2
	gdata->threadSynchronizer->barrier(); // end of SIMULATION, begins FINALIZATION ***

	// just wait or...?

	gdata->threadSynchronizer->barrier(); // end of FINALIZATION ***

	// after the last barrier has been reached by all threads (or after the Synchronizer has been forcedly unlocked),
	// we wait for the threads to actually exit
	for (uint d = 0; d < gdata->devices; d++)
		gdata->GPUWORKERS[d]->join_worker();

	return true;
}


void GPUSPH::move_bodies(const uint step)
{
	// Get moving bodies data (position, linear and angular velocity ...)
	if (problem->get_simparams()->numbodies > 0) {
		// We have to reduce forces and torques only on bodies which requires it
		const size_t numforcesbodies = problem->get_simparams()->numforcesbodies;
		if (numforcesbodies > 0) {
			doCommand(REDUCE_BODIES_FORCES);

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

			/* Make a copy of the total forces, and let the problem override the applied forces, if necessary */
			memcpy(gdata->s_hRbAppliedForce, gdata->s_hRbTotalForce, numforcesbodies*sizeof(float3));
			memcpy(gdata->s_hRbAppliedTorque, gdata->s_hRbTotalTorque, numforcesbodies*sizeof(float3));

			double t0 = gdata->t;
			double t1 = t0;
			if (step == 1)
				t1 += gdata->dt/2.0;
			else
				t1 += gdata->dt;
			problem->bodies_forces_callback(t0, t1, step, gdata->s_hRbAppliedForce, gdata->s_hRbAppliedTorque);
		}

		// Let the problem compute the new moving bodies data
		problem->bodies_timestep(gdata->s_hRbAppliedForce, gdata->s_hRbAppliedTorque, step, gdata->dt, gdata->t,
			gdata->s_hRbCgGridPos, gdata->s_hRbCgPos,
			gdata->s_hRbTranslations, gdata->s_hRbRotationMatrices, gdata->s_hRbLinearVelocities, gdata->s_hRbAngularVelocities);

		if (step == 2)
			problem->post_timestep_callback(gdata->t);

		// Upload translation vectors and rotation matrices; will upload CGs after euler
		doCommand(UPLOAD_OBJECTS_MATRICES);
		// Upload objects linear and angular velocities
		doCommand(UPLOAD_OBJECTS_VELOCITIES);
		// Upload objects CG in forces only
		if (numforcesbodies)
			doCommand(FORCES_UPLOAD_OBJECTS_CG);
	} // if there are objects
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

#if _DEBUG_
	gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_FORCES>();
#endif

	if (gdata->simframework->hasPostProcessOption(SURFACE_DETECTION, BUFFER_NORMALS))
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_NORMALS>();
	if (gdata->simframework->hasPostProcessEngine(VORTICITY))
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_VORTICITY>();

	if (problem->m_simparams->boundarytype == SA_BOUNDARY) {
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_BOUNDELEMENTS>();
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_VERTICES>();
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_GRADGAMMA>();
	}

	if (problem->m_simparams->visctype == KEPSVISC) {
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_TKE>();
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_EPSILON>();
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_TURBVISC>();
	}

	if (problem->m_simparams->simflags & ENABLE_INLET_OUTLET ||
		problem->m_simparams->visctype == KEPSVISC)
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_EULERVEL>();

	if (problem->m_simparams->visctype == SPSVISC)
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_SPS_TURBVISC>();

	if (problem->m_simparams->sph_formulation == SPH_GRENIER) {
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_VOLUME>();
		// Only for debugging:
		//gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_SIGMA>();
	}

	if (gdata->simframework->hasPostProcessEngine(CALC_PRIVATE))
		gdata->s_hBuffers.addBuffer<HostBuffer, BUFFER_PRIVATE>();

	// number of elements to allocate
	const size_t numparts = gdata->allocatedParticles;

	const uint numcells = gdata->nGridCells;
	const size_t devcountCellSize = sizeof(devcount_t) * numcells;
	const size_t uintCellSize = sizeof(uint) * numcells;

	size_t totCPUbytes = 0;

	BufferList::iterator iter = gdata->s_hBuffers.begin();
	while (iter != gdata->s_hBuffers.end()) {
		totCPUbytes += iter->second->alloc(numparts);
		++iter;
	}


	if (gdata->numPlanes > 0) {
		if (gdata->numPlanes > MAX_PLANES) {
			printf("FATAL: unsupported number of planes (%u > %u)\n", gdata->numPlanes, MAX_PLANES);
			exit(1);
		}
		const size_t planeSize4 = sizeof(double4) * gdata->numPlanes;
		const size_t planeSize3 = sizeof(float3) * gdata->numPlanes; // same as for int3

		gdata->s_hPlanes = new double4[gdata->numPlanes];
		memset(gdata->s_hPlanes, 0, planeSize4);
		totCPUbytes += planeSize4;

		gdata->s_hPlaneNormal = new float3[gdata->numPlanes];
		memset(gdata->s_hPlaneNormal, 0, planeSize3);

		gdata->s_hPlanePointGridPos = new int3[gdata->numPlanes];
		memset(gdata->s_hPlanePointGridPos, 0, planeSize3);

		gdata->s_hPlanePointLocalPos = new float3[gdata->numPlanes];
		memset(gdata->s_hPlanePointLocalPos, 0, planeSize3);

		totCPUbytes += planeSize4 + 3*planeSize3;
	}

	const size_t numbodies = gdata->problem->get_simparams()->numbodies;
	std::cout << "Numbodies : " << numbodies << "\n";
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
	const size_t numforcesbodies = gdata->problem->get_simparams()->numforcesbodies;
	std::cout << "Numforcesbodies : " << numforcesbodies << "\n";
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
		// TODO: alloc pinned memory instead, with per-worker methods. See GPUWorker::asyncCellIndicesUpload()
		gdata->s_dCellStarts = (uint**)calloc(gdata->devices, sizeof(uint*));
		gdata->s_dCellEnds =  (uint**)calloc(gdata->devices, sizeof(uint*));
		gdata->s_dSegmentsStart = (uint**)calloc(gdata->devices, sizeof(uint*));

		// few bytes... but still count them
		totCPUbytes += gdata->devices * sizeof(uint*) * 3;

		for (uint d=0; d < gdata->devices; d++) {
			// NOTE: not checking errors, similarly to the other allocations.
			// Also cudaHostAllocWriteCombined flag was tested but led to ~0.1MIPPS performance loss, with and without streams.
			cudaHostAlloc( &(gdata->s_dCellStarts[d]), numcells * sizeof(uint), cudaHostAllocPortable );
			cudaHostAlloc( &(gdata->s_dCellEnds[d]), numcells * sizeof(uint), cudaHostAllocPortable );
			// same on non-pinned memory
			//gdata->s_dCellStarts[d] = (uint*)calloc(numcells, sizeof(uint));
			//gdata->s_dCellEnds[d] =   (uint*)calloc(numcells, sizeof(uint));
			totCPUbytes += uintCellSize * 2;

			gdata->s_dSegmentsStart[d] = (uint*)calloc(4, sizeof(uint));
			totCPUbytes += sizeof(uint) * 4;
		}
	}
	return totCPUbytes;
}

// Deallocate the shared buffers, i.e. those accessed by all workers
void GPUSPH::deallocateGlobalHostBuffers() {
	gdata->s_hBuffers.clear();

	// Deallocating rigid bodies related arrays
	if (gdata->problem->get_simparams()->numbodies > 0) {
		delete [] gdata->s_hRbCgGridPos;
		delete [] gdata->s_hRbCgPos;
		delete [] gdata->s_hRbTranslations;
		delete [] gdata->s_hRbLinearVelocities;
		delete [] gdata->s_hRbAngularVelocities;
		delete [] gdata->s_hRbRotationMatrices;
	}
	if (gdata->problem->get_simparams()->numforcesbodies > 0) {
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
	if (gdata->numPlanes > 0) {
		delete[] gdata->s_hPlanes;
		delete[] gdata->s_hPlaneNormal;
		delete[] gdata->s_hPlanePointGridPos;
		delete[] gdata->s_hPlanePointLocalPos;
	}

	// multi-GPU specific arrays
	if (MULTI_DEVICE) {
		delete[] gdata->s_hDeviceMap;
		delete[] gdata->s_hPartsPerSliceAlongX;
		delete[] gdata->s_hPartsPerSliceAlongY;
		delete[] gdata->s_hPartsPerSliceAlongZ;
		// cells
		for (uint d = 0; d < gdata->devices; d++) {
			cudaFreeHost(gdata->s_dCellStarts[d]);
			cudaFreeHost(gdata->s_dCellEnds[d]);
			//delete[] gdata->s_dCellStarts[d];
			//delete[] gdata->s_dCellEnds[d];
			// same on non-pinned memory
			delete[] gdata->s_dSegmentsStart[d];
		}
		delete[] gdata->s_dCellEnds;
		delete[] gdata->s_dCellStarts;
		delete[] gdata->s_dSegmentsStart;
	}
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
	for (uint p = 0; p < gdata->totParticles; p++) {

		// compute containing device according to the particle's hash
		uint cellHash = cellHashFromParticleHash( gdata->s_hBuffers.getData<BUFFER_HASH>()[p] );
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
	for (uint currentGlobalDevice = 0; currentGlobalDevice < (gdata->totDevices - 1); currentGlobalDevice++) {
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
				std::swap(m_hParticleKeys[leftB], m_hParticleKeys[rightB]);
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
		devcount_t cdev = gdata->s_hDeviceMap[ cellHashFromParticleHash(gdata->s_hBuffers.getData<BUFFER_HASH>()[p]) ];
		devcount_t pdev;
		if (p > 0) pdev = gdata->s_hDeviceMap[ cellHashFromParticleHash(gdata->s_hBuffers.getData<BUFFER_HASH>()[p-1]) ];
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

// set nextCommand, unlock the threads and wait for them to complete
void GPUSPH::doCommand(CommandType cmd, flag_t flags, float arg)
{
	// resetting the host buffers is useful to check if the arrays are completely filled
	/*/ if (cmd==DUMP) {
	 const uint float4Size = sizeof(float4) * gdata->totParticles;
	 const uint infoSize = sizeof(particleinfo) * gdata->totParticles;
	 memset(gdata->s_hPos, 0, float4Size);
	 memset(gdata->s_hVel, 0, float4Size);
	 memset(gdata->s_hInfo, 0, infoSize);
	 } */

	gdata->nextCommand = cmd;
	gdata->commandFlags = flags;
	gdata->extraCommandArg = arg;
	gdata->threadSynchronizer->barrier(); // unlock CYCLE BARRIER 2
	gdata->threadSynchronizer->barrier(); // wait for completion of last command and unlock CYCLE BARRIER 1
}

void GPUSPH::setViscosityCoefficient()
{
	PhysParams *pp = gdata->problem->get_physparams();
	ViscosityType vt = gdata->problem->get_simparams()->visctype;

	// Set visccoeff based on the viscosity model used
	switch (vt) {
		case ARTVISC:
			for (uint f = 0; f < pp->numFluids(); ++f)
				pp->visccoeff[f] = pp->artvisccoeff;
			break;

		case KINEMATICVISC:
		case SPSVISC:
			for (uint f = 0; f < pp->numFluids(); ++f)
				pp->visccoeff[f] = 4*pp->kinematicvisc[f];
			break;

		case KEPSVISC:
		case DYNAMICVISC:
			for (uint f = 0; f < pp->numFluids(); ++f)
				pp->visccoeff[f] = pp->kinematicvisc[f];
			break;

		default:
			throw runtime_error(string("Don't know how to set viscosity coefficient for chosen viscosity type!"));
			break;
	}

	// Set SPS factors from coefficients, if they were not set
	// by the problem
	if (vt == SPSVISC) {
		// TODO physparams should have configurable Cs, Ci
		// rather than configurable smagfactor, kspsfactor, probably
		const double spsCs = 0.12;
		const double spsCi = 0.0066;
		const double dp = gdata->problem->get_deltap();
		if (isnan(pp->smagfactor)) {
			pp->smagfactor = spsCs*dp;
			pp->smagfactor *= pp->smagfactor; // (Cs*∆p)^2
		}
		if (isnan(pp->kspsfactor))
			pp->kspsfactor = (2*spsCi/3)*dp*dp; // (2/3) Ci ∆p^2
	}
}

// creates the Writer according to the requested WriterType
void GPUSPH::createWriter()
{
	Writer::Create(gdata);
}

void GPUSPH::doWrite(bool force)
{
	uint node_offset = gdata->s_hStartPerDevice[0];

	// WaveGages work by looking at neighboring SURFACE particles and averaging their z coordinates
	// NOTE: it's a standard average, not an SPH smoothing, so the neighborhood is arbitrarily fixed
	// at gage (x,y) ± 2 smoothing lengths
	// TODO should it be an SPH smoothing instead?

	GageList &gages = problem->get_simparams()->gage;
	double slength = problem->get_simparams()->slength;

	size_t numgages = gages.size();

	std::vector<double2> gage_llimit, gage_ulimit; // neighborhood limits
	std::vector<uint> gage_parts;
	GageList::iterator gage = gages.begin();
	GageList::iterator gage_end = gages.end();
	while (gage != gage_end) {
		gage_llimit.push_back(make_double2(*gage) - 2*slength);
		gage_ulimit.push_back(make_double2(*gage) + 2*slength);
		gage_parts.push_back(0);
		gage->z = 0;
		++gage;
	}

	// TODO: parallelize? (e.g. each thread tranlsates its own particles)
	double3 const& wo = problem->get_worldorigin();
	const float4 *lpos = gdata->s_hBuffers.getData<BUFFER_POS>();
	const particleinfo *info = gdata->s_hBuffers.getData<BUFFER_INFO>();
	double4 *gpos = gdata->s_hBuffers.getData<BUFFER_POS_GLOBAL>();

	bool warned_nan_pos = false;

	// max particle speed only for this node only at time t
	float local_max_part_speed = 0;

	for (uint i = node_offset; i < node_offset + gdata->processParticles[gdata->mpi_rank]; i++) {
		const float4 pos = lpos[i];
		double4 dpos;
		uint3 gridPos = gdata->calcGridPosFromCellHash( cellHashFromParticleHash(gdata->s_hBuffers.getData<BUFFER_HASH>()[i]) );
		dpos.x = ((double) gdata->cellSize.x)*(gridPos.x + 0.5) + (double) pos.x + wo.x;
		dpos.y = ((double) gdata->cellSize.y)*(gridPos.y + 0.5) + (double) pos.y + wo.y;
		dpos.z = ((double) gdata->cellSize.z)*(gridPos.z + 0.5) + (double) pos.z + wo.z;
		dpos.w = pos.w;

		if (!warned_nan_pos && !(isfinite(dpos.x) && isfinite(dpos.y) && isfinite(dpos.z))) {
			fprintf(stderr, "WARNING: particle %u (id %u) has NAN position! (%g, %g, %g) @ (%u, %u, %u) = (%g, %g, %g)\n",
				i, id(info[i]),
				pos.x, pos.y, pos.z,
				gridPos.x, gridPos.y, gridPos.z,
				dpos.x, dpos.y, dpos.z);
			warned_nan_pos = true;
		}

		// for surface particles add the z coordinate to the appropriate wavegages
		if (numgages && SURFACE(info[i])) {
			for (uint g = 0; g < numgages; ++g) {
				if ((dpos.x > gage_llimit[g].x) && (dpos.x < gage_ulimit[g].x) &&
					(dpos.y > gage_llimit[g].y) && (dpos.y < gage_ulimit[g].y)) {
						gage_parts[g]++;
						gages[g].z += dpos.z;
				}
			}

		}

		gpos[i] = dpos;

		// track peak speed
		local_max_part_speed = fmax(local_max_part_speed, length( as_float3(gdata->s_hBuffers.getData<BUFFER_VEL>()[i]) ));
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

	Writer::SetForced(force);

	if (numgages) {
		for (uint g = 0 ; g < numgages; ++g) {
			gages[g].z /= gage_parts[g];
		}
		//Write WaveGage information on one text file
		Writer::WriteWaveGage(gdata->t, gages);
	}

	if (gdata->problem->get_simparams()->numforcesbodies > 0) {
		Writer::WriteObjectForces(gdata->t, problem->get_simparams()->numforcesbodies,
			gdata->s_hRbTotalForce, gdata->s_hRbTotalTorque,
			gdata->s_hRbAppliedForce, gdata->s_hRbAppliedTorque);
	}

	if (gdata->problem->get_simparams()->numbodies > 0) {
		Writer::WriteObjects(gdata->t);
	}

	Writer::Write(
		gdata->processParticles[gdata->mpi_rank],
		gdata->s_hBuffers,
		node_offset,
		gdata->t, gdata->simframework->hasPostProcessEngine(TESTPOINTS));
	Writer::MarkWritten(gdata->t);

	// TODO: enable energy computation and dump
	/*calc_energy(m_hEnergy,
		m_dPos[m_currentPosRead],
		m_dVel[m_currentVelRead],
		m_dInfo[m_currentInfoRead],
		m_numParticles,
		m_physparams->numFluids());
	m_writer->write_energy(m_simTime, m_hEnergy);*/

	// always reset force-saving
	if (force)
		Writer::SetForced(false);
}

void GPUSPH::buildNeibList()
{
	// run most of the following commands on all particles
	gdata->only_internal = false;

	doCommand(CALCHASH);
	doCommand(SORT);
	doCommand(REORDER);

	// get the new number of particles: with inlet/outlets, they
	// may have changed because of incoming/outgoing particle, otherwise
	// some particles might have been disabled (and discarded) for flying
	// out of the domain
	doCommand(DOWNLOAD_NEWNUMPARTS);

	// swap all double buffers
	doCommand(SWAP_BUFFERS, gdata->simframework->getAllocPolicy()->get_multi_buffered());

	// if running on multiple GPUs, update the external cells
	if (MULTI_DEVICE) {
		// copy cellStarts, cellEnds and segments on host
		doCommand(DUMP_CELLS);
		doCommand(UPDATE_SEGMENTS);

		// here or later, before update indices: MPI_Allgather (&sendbuf,sendcount,sendtype,&recvbuf, recvcount,recvtype,comm)
		// maybe overlapping with dumping cells (run async before dumping the cells)

		// update particle offsets
		updateArrayIndices();
		// crop external cells
		doCommand(CROP);
		// append fresh copies of the externals
		// NOTE: this imports also particle hashes without resetting the high bits, which are wrong
		// until next calchash; however, they are filtered out when using the particle hashes.
		doCommand(APPEND_EXTERNAL, IMPORT_BUFFERS);
		// update the newNumParticles device counter
		if (problem->get_simparams()->simflags & ENABLE_INLET_OUTLET)
			doCommand(UPLOAD_NEWNUMPARTS);
	} else
		updateArrayIndices();

	// update vertID->particleIndex lookup table for vertex connectivity, on *all* particles
	if (problem->get_simparams()->boundarytype == SA_BOUNDARY) {
		gdata->only_internal = false;
		doCommand(SA_UPDATE_VERTIDINDEX);
	}

	// build neib lists only for internal particles
	gdata->only_internal = true;
	doCommand(BUILDNEIBS);

	if (MULTI_DEVICE && problem->get_simparams()->boundarytype == SA_BOUNDARY)
		doCommand(UPDATE_EXTERNAL, BUFFER_VERTPOS);

	// scan and check the peak number of neighbors and the estimated number of interactions
	const uint maxPossibleNeibs = gdata->problem->get_simparams()->maxneibsnum;
	gdata->lastGlobalPeakNeibsNum = 0;
	for (uint d = 0; d < gdata->devices; d++) {
		const uint currDevMaxNeibs = gdata->timingInfo[d].maxNeibs;

		if (currDevMaxNeibs > maxPossibleNeibs)
			printf("WARNING: current max. neighbors numbers %u greather than MAXNEIBSNUM (%u) at iteration %lu\n",
				currDevMaxNeibs, maxPossibleNeibs, gdata->iterations);

		if (currDevMaxNeibs > gdata->lastGlobalPeakNeibsNum)
			gdata->lastGlobalPeakNeibsNum = currDevMaxNeibs;

		gdata->lastGlobalNumInteractions += gdata->timingInfo[d].numInteractions;
	}
}

void GPUSPH::doCallBacks()
{
	Problem *pb = gdata->problem;

	if (pb->m_simparams->gcallback)
		gdata->s_varGravity = pb->g_callback(gdata->t);
}

void GPUSPH::printStatus(FILE *out)
{
//#define ti timingInfo
	fprintf(out, "Simulation time t=%es, iteration=%s, dt=%es, %s parts (%.2g, cum. %.2g MIPPS), maxneibs %u\n",
			//"mean %e neibs. in %es, %e neibs/s, max %u neibs\n"
			//"mean neib list in %es\n"
			//"mean integration in %es\n",
			gdata->t, gdata->addSeparators(gdata->iterations).c_str(), gdata->dt,
			gdata->addSeparators(gdata->totParticles).c_str(), m_intervalPerformanceCounter->getMIPPS(),
			m_totalPerformanceCounter->getMIPPS(),
			gdata->lastGlobalPeakNeibsNum
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
// NOTE: only meaningful in singlenode (otherwise, there is no correspondence between indices and ids)
void GPUSPH::rollCallParticles()
{
	bool all_normal = true;

	// reset bitmap and addrs
	for (uint idx = 0; idx < gdata->processParticles[gdata->mpi_rank]; idx++) {
		m_rcBitmap[idx] = false;
		m_rcAddrs[idx] = UINT_MAX;
	}

	// fill out the bitmap and check for duplicates
	for (uint pos = 0; pos < gdata->processParticles[gdata->mpi_rank]; pos++) {
		uint idx = id(gdata->s_hBuffers.getData<BUFFER_INFO>()[pos]);
		if (m_rcBitmap[idx] && !m_rcNotified[idx]) {
			printf("WARNING: at iteration %lu, time %g particle idx %u is in pos %u and %u!\n",
					gdata->iterations, gdata->t, idx, m_rcAddrs[idx], pos);
			// getchar(); // useful for debugging
			// printf("Press ENTER to continue...\n");
			all_normal = false;
			m_rcNotified[idx] = true;
		}
		m_rcBitmap[idx] = true;
		m_rcAddrs[idx] = pos;
	}
	// now check if someone is missing
	for (uint idx = 0; idx < gdata->processParticles[gdata->mpi_rank]; idx++)
		if (!m_rcBitmap[idx] && !m_rcNotified[idx]) {
			printf("WARNING: at iteration %lu, time %g particle idx %u was not found!\n",
					gdata->iterations, gdata->t, idx);
			// printf("Press ENTER to continue...\n");
			// getchar(); // useful for debugging
			m_rcNotified[idx] = true;
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
void GPUSPH::updateArrayIndices() {
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
	if (gdata->mpi_rank == 0 || gdata->problem->get_simparams()->simflags & ENABLE_INLET_OUTLET) {
		uint newSimulationTotal = 0;
		for (uint n = 0; n < gdata->mpi_nodes; n++)
			newSimulationTotal += gdata->processParticles[n];

		// number of particle may increase or decrease if there are respectively inlets or outlets
		// TODO this should be simplified, but it would be better to check separately
		// for < and >, based on the number of inlets and outlets, so we leave
		// it this way as a reminder
		if ( (newSimulationTotal < gdata->totParticles && gdata->problem->get_simparams()->simflags & ENABLE_INLET_OUTLET) ||
			 (newSimulationTotal > gdata->totParticles && gdata->problem->get_simparams()->simflags & ENABLE_INLET_OUTLET) ) {
			// printf("Number of total particles at iteration %u passed from %u to %u\n", gdata->iterations, gdata->totParticles, newSimulationTotal);
			gdata->totParticles = newSimulationTotal;
		} else if (newSimulationTotal != gdata->totParticles && gdata->mpi_rank == 0) {
			printf("WARNING: at iteration %lu the number of particles changed from %u to %u for no known reason!\n",
				gdata->iterations, gdata->totParticles, newSimulationTotal);

			// who is missing? if single-node, do a roll call
			if (SINGLE_NODE) {
				doCommand(DUMP, BUFFER_INFO | DBLBUFFER_READ);
				rollCallParticles();
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
	const particleinfo *infos = gdata->s_hBuffers.getData<BUFFER_INFO>();
	const hashKey *hashes = gdata->s_hBuffers.getData<BUFFER_HASH>();

	//nGridCells

	// should write something more meaningful here
	printf("Preparing the problem...\n");

	// at the time being, we only need preparation for multi-device simulations
	if (!MULTI_DEVICE) return;

	for (uint p = 0; p < gdata->totParticles; p++) {
		// For DYN bounds, take into account also boundary parts; for other boundary types,
		// only cound fluid parts
		if ( problem->get_simparams()->boundarytype == DYN_BOUNDARY || FLUID(infos[p])) {
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

void GPUSPH::saBoundaryConditions(flag_t cFlag)
{
	if (gdata->simframework->getBCEngine() == NULL)
		throw runtime_error("no boundary conditions engine loaded");

	// initially data is in read so swap to write
	if (cFlag & INITIALIZATION_STEP) {
		doCommand(SWAP_BUFFERS, BUFFER_INFO);
		doCommand(IDENTIFY_CORNER_VERTICES);
		if (MULTI_DEVICE)
			doCommand(UPDATE_EXTERNAL, BUFFER_INFO | DBLBUFFER_WRITE);

		doCommand(SWAP_BUFFERS, BUFFER_VERTICES);
		doCommand(FIND_CLOSEST_VERTEX);
		if (MULTI_DEVICE)
			doCommand(UPDATE_EXTERNAL, BUFFER_VERTICES | DBLBUFFER_WRITE);

		doCommand(SWAP_BUFFERS, BUFFER_VEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_POS | BUFFER_EULERVEL | BUFFER_INFO);
	}

	// impose open boundary conditions
	if (problem->get_simparams()->simflags & ENABLE_INLET_OUTLET) {
		// reduce the water depth at pressure outlets if required
		// if we have multiple devices then we need to run a global max on the different gpus / nodes
		if (MULTI_DEVICE && problem->get_simparams()->simflags & ENABLE_WATER_DEPTH) {
			// each device gets his waterdepth array from the gpu
			doCommand(DOWNLOAD_IOWATERDEPTH);
			int* n_IOwaterdepth = new int[problem->get_simparams()->numOpenBoundaries];
			// max over all devices per node
			for (uint ob = 0; ob < problem->get_simparams()->numOpenBoundaries; ob ++) {
				n_IOwaterdepth[ob] = 0;
				for (uint d = 0; d < gdata->devices; d++)
					n_IOwaterdepth[ob] = max(n_IOwaterdepth[ob], gdata->h_IOwaterdepth[d][ob]);
			}
			// if we are in multi-node mode we need to run an mpi reduction over all nodes
			if (MULTI_NODE) {
				gdata->networkManager->networkIntReduction((int*)n_IOwaterdepth, problem->get_simparams()->numOpenBoundaries, MAX_REDUCTION);
			}
			// copy global value back to one array so that we can upload it again
			for (uint ob = 0; ob < problem->get_simparams()->numOpenBoundaries; ob ++)
				gdata->h_IOwaterdepth[0][ob] = n_IOwaterdepth[ob];
			// upload the global max value to the devices
			doCommand(UPLOAD_IOWATERDEPTH);
		}
		gdata->only_internal = false;
		doCommand(SWAP_BUFFERS, BUFFER_POS);
		doCommand(IMPOSE_OPEN_BOUNDARY_CONDITION);
		doCommand(SWAP_BUFFERS, BUFFER_POS);
	}

	gdata->only_internal = true;

	if (!(cFlag & INITIALIZATION_STEP))
		doCommand(SWAP_BUFFERS, BUFFER_VERTICES);

	// compute boundary conditions on segments and detect outgoing particles at open boundaries
	doCommand(SA_CALC_SEGMENT_BOUNDARY_CONDITIONS, cFlag);
	if (MULTI_DEVICE)
		doCommand(UPDATE_EXTERNAL, POST_SA_SEGMENT_UPDATE_BUFFERS | DBLBUFFER_WRITE);

	// check if new particles were created and check for ID overflow
	if (cFlag & INTEGRATOR_STEP_2) {
		// compute the offset for id cloning and check for possible overflows
		// NOTE: the formula is
		//    new_id = vertex_id + initial_parts - initial_fluid_parts +
		//             (num_iterations_with_part_creations * numVertexParticles)
		// but we replace (initial_parts - initial_fluid_parts) with numInitialNonFluidParticles.
		if (problem->get_simparams()->simflags & ENABLE_INLET_OUTLET) {

			//gdata->newIDsOffset = gdata->numInitialNonFluidParticles +
			//	(gdata->numVertices * gdata->createdParticlesIterations);

			for (uint d = 0; d < gdata->devices; d++) {
				if (UINT_MAX - gdata->numVertices < gdata->highestDevId[d]) {
					fprintf(stderr, " FATAL: possible ID overflow in particle creation after iteration %lu on device %d - requesting quit...\n", gdata->iterations, d);
					gdata->quit_request = true;
				}
			}
		}
	}

	// compute boundary conditions on vertices including mass variation and create new particles at open boundaries
	doCommand(SA_CALC_VERTEX_BOUNDARY_CONDITIONS, cFlag);
	if (MULTI_DEVICE)
		doCommand(UPDATE_EXTERNAL, POST_SA_VERTEX_UPDATE_BUFFERS | DBLBUFFER_WRITE);

	// check if we need to delete some particles which passed through open boundaries
	if (problem->get_simparams()->simflags & ENABLE_INLET_OUTLET && (cFlag & INTEGRATOR_STEP_2)) {
		doCommand(DISABLE_OUTGOING_PARTS);
		if (MULTI_DEVICE)
			doCommand(UPDATE_EXTERNAL, BUFFER_POS | BUFFER_VERTICES | DBLBUFFER_WRITE);
	}

	// swap changed buffers back so that read contains the new data
	if (cFlag & INITIALIZATION_STEP)
		doCommand(SWAP_BUFFERS, BUFFER_VEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_POS | BUFFER_EULERVEL | BUFFER_GRADGAMMA | BUFFER_VERTICES);
}
