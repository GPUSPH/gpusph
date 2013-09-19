/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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


/*! \mainpage GPUSPH Developer's documentation
 *
 *
 * \section intro_sec  Introduction
 *
 * GPUSPH is a CUDA-based 3D SPH simulator (STUB).
 *
 * \section compile_sec Compiling and installing
 *
 * See "make help" (STUB).
 *
 * \section quick_links Internal links
 * - \ref main \n
 * - ParticleSystem
 *
 * \section links Links
 * - <a href="http://www.stack.nl/~dimitri/doxygen/manual.html">Complete Doxygen manual</a>
 * - <a href="http://www.nvidia.com/object/cuda_gpus.html">GPUs and compute capabilites</a>
 *
 *
 * GPUSPH is a CUDA-based 3D SPH simulator (FIX).
 *
 * This document was generated with Doxygen.\n
 *
 */
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <time.h>
#include <string.h>
#include <signal.h>

#define GPUSPH_MAIN
#include "particledefine.h"
#undef GPUSPH_MAIN

// NOTE: including GPUSPH.h before particledefine.h does not compile.
// This inclusion problem should be solved
#include "GPUSPH.h"

#include "ParticleSystem.h"
#include "Problem.h"

// Writer types
#include "TextWriter.h"
#include "VTKWriter.h"
#include "VTKLegacyWriter.h"
#include "CustomTextWriter.h"
#include "UDPWriter.h"

/* Include only the problem selected at compile time */
#include "problem_select.opt"

using namespace std;

FILE *timing_log = NULL;

ParticleSystem *psystem = 0;

// timing
TimingInfo  const* timingInfo = NULL;
char title[256];

// viewing parameters
float3 worldOrigin;
float3 worldSize;
float3 camera_pos;
float3 target_pos;
float3 camera_up;
enum rotation_mode { ROT_NONE, ROT_ORT, ROT_VEC };
rotation_mode rotating = ROT_NONE;

#define view_angle 60.0
/* cotg(view_angle/2) */
#define view_trig (1.0/tan(M_PI*view_angle/360))

const float3 x_axis(make_float3(1, 0, 0));
const float3 y_axis(make_float3(0, 1, 0));
const float3 z_axis(make_float3(0, 0, 1));

float near_plane = 0.1;
float far_plane = 100;

float3 box_corner[8];

Problem *problem;

void cleanup(void)
{
	if (psystem)
		delete psystem;
	if (timing_log != NULL)
		fclose(timing_log);
}

void quit(int ret)
{
	double elapsed_sec = (clock() - timingInfo->startTime)/CLOCKS_PER_SEC;
	printf("\nTotal time %es, throughput %.4g MIPPS\n", elapsed_sec, timingInfo->getMIPPS());
	printf("Quitting\n");
	cleanup();
	exit(ret);
}

void show_timing(int ret)
{
#define ti timingInfo
	printf(
		"\nt=%es dt=%es %u parts.\n"
		"%e neibs. in %es, mean %e neibs/s, max %u neibs\n"
		"%e ints., %e ints/s, mean %e ints/s)\n"
		"integration in %es (mean %es)\n"
		"throughput %.4g MIPPS\n",
		ti->t, ti->dt, ti->numParticles,
		(double)ti->numInteractions, ti->timeNeibsList, ti->meanTimeNeibsList, ti->maxNeibs,
		(double)ti->meanNumInteractions, ti->numInteractions/ti->timeInteract, ti->meanNumInteractions/ti->meanTimeInteract,
		ti->timeEuler, ti->meanTimeEuler,
		ti->getMIPPS());
	fflush(stdout);
#undef ti
}

/* Command line options */
Options clOptions;

// TODO: delete this, use "make list-problems" to list problems instead
void problem_list(void) {
	cout << "GPUSph problems:\n";
	cout << "\tDamBreak3D\n";
	cout << "\tOpenCoast\n";
	cout << "\tTestTopo\n";
	cout << "\tWaveTank\n";
	cout << "FIXME: this list is static, use \"make problem-list\" for an updated one\n";
	cout << endl;
}


void parse_options(int argc, char **argv)
{
	const char *arg(NULL);

	/* skip arg 0 (program name) */
	argv++; argc--;

	while (argc > 0) {
		arg = *argv;
		argv++;
		argc--;
		if (!strcmp(arg, "--device")) {
			/* read the next arg as an integer */
			sscanf(*argv, "%d", &(clOptions.device));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--deltap")) {
			/* read the next arg as a float */
			sscanf(*argv, "%f", &(clOptions.deltap));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--tend")) {
			/* read the next arg as a float */
			sscanf(*argv, "%f", &(clOptions.tend));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--dem")) {
			clOptions.dem = std::string(*argv);
			argv++;
			argc--;
		} else if (!strcmp(arg, "--")) {
			cout << "Skipping unsupported option " << arg << endl;
		} else {
			cout << "Fatal: Unknown option: " << arg << endl;
			exit(0);

			// Left for future dynamic loading:
			/*if (clOptions.problem.empty()) {
				clOptions.problem = std::string(arg);
			} else {
				cout << "Problem " << arg << " selected after problem " << clOptions.problem << endl;
			}*/
		}
	}

	clOptions.problem = std::string( QUOTED_PROBLEM );
	cout << "Compiled for problem \"" << QUOTED_PROBLEM << "\"" << endl;

	// Left for future dynamic loading:
	/*if (clOptions.problem.empty()) {
		problem_list();
		exit(0);
	}*/
}


void init(const char *arg)
{
	problem = new PROBLEM(clOptions);

	/* TODO do it this way for all options? */
	if (isfinite(clOptions.tend))
		problem->get_simparams()->tend = clOptions.tend;

	// set the influence radii - should be encapsulated somewhere as this is physics logic
	SimParams *sp = problem->get_simparams();
	sp->influenceRadius = sp->kernelradius * sp->slength;
	sp->nlInfluenceRadius = sp->influenceRadius * sp->nlexpansionfactor;
	sp->nlSqInfluenceRadius = sp->nlInfluenceRadius * sp->nlInfluenceRadius;

	psystem = new ParticleSystem(problem);

	psystem->printPhysParams();
	psystem->printSimParams();

	// filling simulation domain with particles
	uint numParticles = problem->fill_parts();
	uint maxParticles = problem->max_parts(numParticles);
	psystem->allocate(numParticles, maxParticles);
	problem->copy_to_array(psystem->m_hPos, psystem->m_hVel, psystem->m_hInfo);
	psystem->setArray(ParticleSystem::POSITION);
	psystem->setArray(ParticleSystem::VELOCITY);
	psystem->setArray(ParticleSystem::INFO);

	uint numPlanes = problem->fill_planes();
	if (numPlanes > 0) {
		if (numPlanes > MAXPLANES) {
			fprintf(stderr, "Number of planes too high: %u > %u\n", numPlanes, MAXPLANES);
			exit(1);
		}
		psystem->allocate_planes(numPlanes);
		problem->copy_planes(psystem->m_hPlanes, psystem->m_hPlanesDiv);
		psystem->setPlanes();
	}
	psystem->setOutlets();
	psystem->setInlets();

	timingInfo = psystem->markStart();
}

void get_arrays(bool need_write)
{
	psystem->getArray(ParticleSystem::POSITION, need_write);
	psystem->getArray(ParticleSystem::VELOCITY, need_write);
	psystem->getArray(ParticleSystem::INFO, need_write);
	if (need_write) {
		if (problem->m_simparams.vorticity)
			psystem->getArray(ParticleSystem::VORTICITY, need_write);

		if (problem->m_simparams.savenormals)
			psystem->getArray(ParticleSystem::NORMALS, need_write);
	}
}

void do_write()
{
	#define ti timingInfo
	printf(	"\nSaving file at t=%es iterations=%ld dt=%es %u parts.\n"
			"mean %e neibs. in %es, %e neibs/s, max %u neibs\n"
			"mean neib list in %es\n"
			"mean integration in %es\n"
			"throughput %.4g MIPPS\n",
			ti->t, ti->iterations, ti->dt, ti->numParticles, (double) ti->meanNumInteractions,
			ti->meanTimeInteract, ((double)ti->meanNumInteractions)/ti->meanTimeInteract, ti->maxNeibs,
			ti->meanTimeNeibsList,
			ti->meanTimeEuler,
			ti->getMIPPS());
	fflush(stdout);
	#undef ti
	if (problem->m_simparams.gage.size() > 0) {
		psystem->writeWaveGage();
	}
	psystem->writeToFile();
}

// commented out for possible future use
/*void display()
{
	if (!bPause)
	{
		try {
			timingInfo = psystem->PredcorrTimeStep(true);
		} catch (TimingException &e) {
			fprintf(stderr, "[%g]: %s (dt = %g)\n", e.simTime, e.what(), e.dt);
			quit(1);
		}
#ifdef TIMING_LOG
		fprintf(timing_log,"%9.4e\t%9.4e\t%9.4e\t%9.4e\t%9.4e\n", timingInfo.t, timingInfo.dt,
				timingInfo.timeInteract, timingInfo.timeEuler, timingInfo.timeNeibsList);
		fflush(timing_log);
#endif
	}

	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	bool finished = problem->finished(timingInfo->t);

	bool need_display = displayEnabled && problem->need_display(timingInfo->t);
	bool need_write = problem->need_write(timingInfo->t) || finished;
	problem->write_rbdata(timingInfo->t);

	if (stepping_mode) {
		need_display = true;
		bPause = true;
	}

	if (need_display || need_write)
	{
		get_arrays(need_write);
		if (need_write)
			do_write();
	}

	if (displayEnabled)
	{
		psystem->drawParts(show_boundary, show_floating, view_field);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		problem->draw_boundary(psystem->getTime());
		problem->draw_inlets();
		problem->draw_outlets();
		problem->draw_axis();

		char s[1024];
		size_t len = sprintf(s, "t=%7.4es", timingInfo->t, timingInfo->dt);
		if (stepping_mode)
			len += sprintf(s + len, "    (stepping mode)");
		else if (bPause)
			len += sprintf(s + len, "    (paused)");
		displayStatus(s);

		glutSwapBuffers();
	}

	// view transform
	// look();

	glutSwapBuffers();

#define ti timingInfo
	switch (timing) {
		case M_INTERACTION:
			sprintf(title, "t=%7.2es dt=%7.2es %d parts. %7.2eint. : %7.2eint./s (mean %7.2eint./s) (maxneibs %d)",
				ti->t, ti->dt, ti->numParticles, (double) ti->meanNumInteractions,
				((double) ti->numInteractions)/ti->timeInteract,
				((double) ti->meanNumInteractions)/ti->meanTimeInteract, ti->maxNeibs);
			break;

		case M_NEIBSLIST:
			sprintf(title, "t=%7.2es dt=%7.2es %d parts. %7.2e neibs in %7.2es (mean %7.2es) (maxneibs %d)",
				ti->t, ti->dt, ti->numParticles, (double) ti->numInteractions,
				ti->timeNeibsList, ti->meanTimeNeibsList, ti->maxNeibs);
			break;

		case M_EULER:
			sprintf(title, "t=%7.2es dt=%7.2es %d parts. integration in %7.2es (mean %7.2es)",
				ti->t, ti->dt, ti->numParticles, ti->timeEuler, ti->meanTimeEuler);
			break;

		case M_MEAN:
			sprintf(title, "%7.2e interactions (%7.2eint./s) - Neibs list %7.2es - Euler %7.2es",
				(double) ti->meanNumInteractions, (double) ti->meanNumInteractions/ti->meanTimeInteract,
				ti->meanTimeNeibsList, ti->meanTimeEuler);
			break;

		case M_IPPS:
			sprintf(title, "t=%7.2es dt=%7.2es %10u parts. %10lu iters. %7.2g MIPPS\n",
				ti->t, ti->dt, ti->numParticles, ti->iterations,
				ti->getMIPPS());
			break;

		case M_NOTIMING:
			title[0] = '\0';
			break;
	}
#undef ti

	// leave the "Hit space to start" message until unpaused
	if (!bPause)
		glutSetWindowTitle(title);

	glutSwapBuffers();

	glutReportErrors();

	// Taking a screenshot
	if (displayEnabled && (problem->need_screenshot(timingInfo->t) || screenshotNow))
	{
		glscreenshot->TakeScreenshot(timingInfo->t);
		if (screenshotNow) {
			cout << "Screenshot @ " << timingInfo->t << endl;
			screenshotNow = false;
		}
	}

	if (finished)
		quit(0);
} */

void console_loop(void)
{
	int error = 0;
	bool finished = false;
	while (!finished) {
		try {
			timingInfo = psystem->PredcorrTimeStep(true);
		} catch (TimingException &e) {
			fprintf(stderr, "[%g] :::ERROR::: %s (dt = %g)\n", e.simTime, e.what(), e.dt);
			finished = true;
			error = 1;
		}

		finished |= problem->finished(timingInfo->t);

		bool need_write = problem->need_write(timingInfo->t) || finished;

		if (need_write)
		{
			get_arrays(need_write);
			do_write();
		}
	}

	if (finished)
		quit(error);
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
oldMain( int argc, char** argv)
{
	if (sizeof(uint) != 2*sizeof(short)) {
		fprintf(stderr, "Fatal: this architecture does not have uint = 2 short\n");
		exit(1);
	}
	signal(SIGINT, quit);
	signal(SIGUSR1, show_timing);

	parse_options(argc, argv);

	init(clOptions.problem.c_str());

	// do an initial write
	get_arrays(true);
	do_write();

	console_loop();


	quit(0);

	return 0;
}

/***  here follow the new methods - all that's above should be checked and moved or deleted ***/

GPUSPH& GPUSPH::getInstance() {
	// guaranteed to be destroyed; instantiated on first use
	static GPUSPH instance;
	// return a reference, not just a pointer
	return instance;
}

GPUSPH::GPUSPH() {
	clOptions = NULL;
	gdata = NULL;
	problem = NULL;
	psystem = NULL;
	initialized = false;
}

GPUSPH::~GPUSPH() {
	// it would be useful to have a "fallback" deallocation but we have to check
	// that main did not do that already
	if (initialized) finalize();
}

bool GPUSPH::initialize(GlobalData *_gdata) {

	printf("Initializing...\n");

	gdata = _gdata;
	clOptions = gdata->clOptions;
	problem = gdata->problem;

	m_performanceCounter = new IPPSCounter();

	// utility pointer
	SimParams *_sp = gdata->problem->get_simparams();

	// copy the options passed by command line to GlobalData
	gdata->nosave = clOptions->nosave;
	if (isfinite(clOptions->tend))
		_sp-> tend = clOptions->tend;

	// update the GlobalData copies of the sizes of the domain
	gdata->worldOrigin = problem->get_worldorigin();
	gdata->worldSize = problem->get_worldsize();
	// TODO: re-enable the followin after the WriterType rampage is over
	// gdata->writerType = problem->get_writertype();

	// initialize the influence radius and its derived parameters
	_sp->influenceRadius = _sp->kernelradius * _sp->slength;
	_sp->nlInfluenceRadius = _sp->influenceRadius * _sp->nlexpansionfactor;
	_sp->nlSqInfluenceRadius = _sp->nlInfluenceRadius * _sp->nlInfluenceRadius;

	// compute the size of grid
	gdata->gridSize.x = (uint) (gdata->worldSize.x / _sp->influenceRadius);
	gdata->gridSize.y = (uint) (gdata->worldSize.y / _sp->influenceRadius);
	gdata->gridSize.z = (uint) (gdata->worldSize.z / _sp->influenceRadius);

	// compute the number of cells
	gdata->nGridCells = gdata->gridSize.x * gdata->gridSize.y * gdata->gridSize.z;

	// since the gridsize was obtained by truncation, make the cellSize and exact divisor
	gdata->cellSize.x = gdata->worldSize.x / gdata->gridSize.x;
	gdata->cellSize.y = gdata->worldSize.y / gdata->gridSize.y;
	gdata->cellSize.z = gdata->worldSize.z / gdata->gridSize.z;

	printf(" - World origin: %g , %g , %g\n", gdata->worldOrigin.x, gdata->worldOrigin.y, gdata->worldOrigin.z);
	printf(" - World size:   %g x %g x %g\n", gdata->worldSize.x, gdata->worldSize.y, gdata->worldSize.z);
	printf(" - Cell size:    %g x %g x %g\n", gdata->cellSize.x, gdata->cellSize.y, gdata->cellSize.z);
	printf(" - Grid size:    %u x %u x %u (%s cells)\n", gdata->gridSize.x, gdata->gridSize.y, gdata->gridSize.z, gdata->addSeparators(gdata->nGridCells).c_str());


	// initial dt (or, just dt in case adaptive is enabled)
	gdata->dt = _sp->dt;

	// the initial assignment is arbitrary, just need to be complementary
	// (caveat: as long as these pointers, and not just 0 and 1 values, are always used)
	gdata->currentPosRead = gdata->currentVelRead = gdata->currentInfoRead = 0;
	gdata->currentPosWrite = gdata->currentVelWrite = gdata->currentInfoWrite = 1;

	// check the number of moving boundaries
	if (problem->m_mbnumber > MAXMOVINGBOUND) {
		printf("FATAL: unsupported number of moving boundaries (%u > %u)\n", problem->m_mbnumber, MAXMOVINGBOUND);
		return false;
	}

	// compute mbdata size
	gdata->s_mbDataSize = problem->m_mbnumber * sizeof(float4);

	// sets the correct viscosity coefficient according to the one set in SimParams
	setViscosityCoefficient();

	// create the Writer according to the WriterType
	createWriter();

	// TODO: writeSummary

	// we should need no PS anymore
	//		> new PS
	// psystem = new ParticleSystem(gdata);
	//			PS constructor updates cellSize, worldSize, nCells, etc. in gdata
	//			PS creates new writer. Do it outside?

	// no: done
	//		> new Problem

	printf("Generating problem particles...\n");
	// allocate the particles of the *whole* simulation
	gdata->totParticles = problem->fill_parts();

	// the number of allocated particles will be bigger, to be sure it can contain particles being created
	gdata->allocatedParticles = problem->max_parts(gdata->totParticles);

	// TODO:  before allocating everything else we should check if the number of particles is too high.
	// Not only the mere number of particles should be compared to a hardcoded system limit, but a good
	// estimation of the required memory should be computed
	if (gdata->totParticles >= MAXPARTICLES) {
		fprintf(stderr, "Cannot handle %u > %u particles, sorry\n", gdata->totParticles, MAXPARTICLES);
		// exit(1);
		return false;
	}

	// allocate aux arrays for rollCallParticles()
	m_rcBitmap = (bool*) calloc( sizeof(bool) , gdata->totParticles );
	m_rcNotified = (bool*) calloc( sizeof(bool) , gdata->totParticles );
	m_rcAddrs = (uint*) calloc( sizeof(uint) , gdata->totParticles );

	// Self-explicative check - but could be moved elsewhere
	uint _maxneibsnum = gdata->problem->get_simparams()->maxneibsnum;
	if (_maxneibsnum % NEIBINDEX_INTERLEAVE != 0) {
		fprintf(stderr, "The maximum number of neibs per particle (%u) should be a multiple of NEIBINDEX_INTERLEAVE (%u)\n",
				_maxneibsnum, NEIBINDEX_INTERLEAVE);
		// exit(1);
		return false;
	}

	printf("Allocating shared host buffers...\n");
	// allocate cpu buffers, 1 per process
	size_t totCPUbytes = allocateGlobalHostBuffers(); // TODO was partially implemented
	uint extra = gdata->allocatedParticles - gdata->totParticles;
	if (extra == 0)
		printf("  allocated %.2g Gb on host for %s particles\n", (ulong)totCPUbytes/1000000000.0F, gdata->addSeparators(gdata->allocatedParticles).c_str());
	else
		printf("  allocated %.2g Gb on host for %s particles (%s for initial filling + %s for particle creation)\n", (ulong)totCPUbytes/1000000000.0F,
			gdata->addSeparators(gdata->allocatedParticles).c_str(), gdata->addSeparators(gdata->totParticles).c_str(), gdata->addSeparators(extra).c_str());

	// let the Problem partition the domain (with global device ids)
	// NOTE: this could be done before fill_parts(), as long as it does not need knowledge about the fluid, but
	// not before allocating the host buffers
	if (gdata->devices > 1) {
		printf("Splitting the domain in %u partitions...\n", gdata->devices);
		gdata->problem->fillDeviceMap(gdata);
	}

	printf("Copying the particles to shared arrays...\n");
	printf("---\n");
	// Check: is this mandatory? this requires double the memory!
	// copy particles from problem to GPUSPH buffers
	problem->copy_to_array(gdata->s_hPos, gdata->s_hVel, gdata->s_hInfo);
	printf("---\n");

	if (gdata->devices > 1) {
		printf("Sorting the particles per device...\n");
		sortParticlesByHash();
	} else {
		// if there is something more to do, encapsulate in a dedicated method please
		gdata->s_hStartPerDevice[0] = 0;
		gdata->s_hPartsPerDevice[0] = gdata->totParticles;
	}

	for (uint d=0; d < gdata->devices; d++)
		printf(" - device at index %u has %s particles assigned and offset %s\n",
			d, gdata->addSeparators(gdata->s_hPartsPerDevice[d]).c_str(), gdata->addSeparators(gdata->s_hStartPerDevice[d]).c_str());

	// TODO
	//		// > new Integrator

	// TODO: read DEM file. setDemTexture() will be called from the GPUWokers instead

	// new Synchronizer; it will be waiting on #devices+1 threads (GPUWorkers + main)
	gdata->threadSynchronizer = new Synchronizer(gdata->devices + 1);

	printf("Starting workers...\n");

	// allocate workers
	gdata->GPUWORKERS = (GPUWorker**)calloc(gdata->devices, sizeof(GPUWorker*));
	for (int d=0; d < gdata->devices; d++)
		gdata->GPUWORKERS[d] = new GPUWorker(gdata, d);

	gdata->keep_going = true;

	// actually start the threads
	for (int d=0; d < gdata->devices; d++)
		gdata->GPUWORKERS[d]->run_worker();

	// The following barrier waits for GPUworkers to complete CUDA init, GPU allocation, subdomain and devmap upload

	gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***

	return (initialized = true);
}

bool GPUSPH::finalize() {
	// TODO here, when there will be the Integrator
	// delete Integrator

	printf("Deallocating...\n");

	// suff for rollCallParticles()
	free(m_rcBitmap);
	free(m_rcNotified);
	free(m_rcAddrs);

	// workers
	for (int d=0; d < gdata->devices; d++)
			delete gdata->GPUWORKERS[d];

	delete gdata->GPUWORKERS;

	// Synchronizer
	delete gdata->threadSynchronizer;

	// host buffers
	deallocateGlobalHostBuffers();

	// ParticleSystem which, in turn, deletes the Writer
	//delete psystem;

	delete gdata->writer;

	// ...anything else?

	delete m_performanceCounter;

	initialized = false;

	return true;
}

bool GPUSPH::runSimulation() {
	if (!initialized) return false;

	// short form
	const bool inoutlets = (gdata->problem->get_physparams()->inlets > 0 || gdata->problem->get_physparams()->outlets > 0);

	// doing first write
	printf("Performing first write...\n");
	doWrite();

	printf("Letting threads upload the subdomains...\n");
	gdata->threadSynchronizer->barrier(); // begins UPLOAD ***

	// here the Workers are uploading their subdomains

	// After next barrier, the workers will enter their simulation cycle, so it is recommended to set
	// nextCommand properly before the barrier (although should be already initialized to IDLE).
	// doCommand(IDLE) would be equivalent, but this is more clear
	gdata->nextCommand = IDLE;
	gdata->threadSynchronizer->barrier();  // end of UPLOAD, begins SIMULATION ***
	gdata->threadSynchronizer->barrier();  // unlock CYCLE BARRIER 1

	printf("Entering the main simulation cycle\n");

	//  IPPS counter does not take the initial uploads into consideration
	m_performanceCounter->start();

	// write some info. This could replace "Entering the main simulation cycle"
	printStatus();

	while (gdata->keep_going) {
		// when there will be an Integrator class, here (or after bneibs?) we will
		// call Integrator -> setNextStep

		// build neighbors list
		if (gdata->iterations % problem->get_simparams()->buildneibsfreq == 0) {
			// run most of the following commands on all particles
			gdata->only_internal = false;

			doCommand(CALCHASH);
			doCommand(SORT);
			doCommand(REORDER);
			if (inoutlets && gdata->iterations > 0)
				doCommand(DOWNLOAD_NEWNUMPARTS);
			// swap pos, vel and info double buffers
			gdata->swapDeviceBuffers(BUFFER_POS | BUFFER_VEL | BUFFER_INFO);

			// if running on multiple GPUs, update the external cells
			if (gdata->devices > 1) {
				// copy cellStarts, cellEnds and segments on host
				doCommand(DUMP_CELLS);
				doCommand(UPDATE_SEGMENTS);
				// update particle offsets
				updateArrayIndices();
				// crop external cells
				doCommand(CROP);
				// append fresh copies of the externals
				doCommand(APPEND_EXTERNAL);
				// update the newNumParticles device counter
				if (inoutlets)
					doCommand(UPLOAD_NEWNUMPARTS);
			}

			// build neib lists only for internal particles
			gdata->only_internal = true;
			doCommand(BUILDNEIBS);
		}

		uint mlsfreq = problem->get_simparams()->mlsfreq;
		if (mlsfreq > 0 && gdata->iterations > 0 && (gdata->iterations % mlsfreq == 0)) {
			gdata->only_internal = true;
			doCommand(MLS);
			// update before swapping, since UPDATE_EXTERNAL works on write buffers
			if (gdata->devices > 1)
				doCommand(UPDATE_EXTERNAL, BUFFER_VEL);
			gdata->swapDeviceBuffers(BUFFER_VEL);
		}

		uint shepardfreq = problem->get_simparams()->shepardfreq;
		if (shepardfreq > 0 && gdata->iterations > 0 && (gdata->iterations % shepardfreq == 0)) {
			gdata->only_internal = true;
			doCommand(SHEPARD);
			// update before swapping, since UPDATE_EXTERNAL works on write buffers
			if (gdata->devices > 1)
				doCommand(UPDATE_EXTERNAL, BUFFER_VEL);
			gdata->swapDeviceBuffers(BUFFER_VEL);
		}

	//			//(init bodies)

		// compute forces only on internal particles
		gdata->only_internal = true;
		doCommand(FORCES, INTEGRATOR_STEP_1);
		// update forces of external particles
		if (gdata->devices > 1)
			doCommand(UPDATE_EXTERNAL, BUFFER_FORCES);

	//MM		fetch/update forces on neighbors in other GPUs/nodes
	//				initially done trivial and slow: stop and read
	//			//reduce bodies

		// moving boundaries
		if (problem->get_simparams()->mbcallback) {
			// ask the Problem to update mbData, one per process
			doCallBacks();
			// upload on the GPU, one per device
			doCommand(UPLOAD_MBDATA, INTEGRATOR_STEP_1);
		}

		// variable gravity
		if (problem->get_simparams()->gcallback) {
			// ask the Problem to update gravity, one per process
			doCallBacks();
			// upload on the GPU, one per device
			doCommand(UPLOAD_GRAVITY, INTEGRATOR_STEP_1);
		}

		// integrate also the externals
		gdata->only_internal = false;
		doCommand(EULER, INTEGRATOR_STEP_1);

		// this made sense for testing and running EULER on internals only
		//if (gdata->devices > 1)
			//doCommand(UPDATE_EXTERNAL, BUFFER_POS | BUFFER_VEL);

	//			//reduce bodies
	//			//callbacks (bounds, gravity)
	//MM		fetch/update forces on neighbors in other GPUs/nodes
	//				initially done trivial and slow: stop and read

		gdata->only_internal = true;
		doCommand(FORCES, INTEGRATOR_STEP_2);
		// update forces of external particles
		if (gdata->devices > 1)
			doCommand(UPDATE_EXTERNAL, BUFFER_FORCES);

	//			//reduce bodies

		// moving boundaries
		if (problem->get_simparams()->mbcallback) {
			// ask the Problem to update mbData, one per process
			doCallBacks();
			// upload on the GPU, one per device
			doCommand(UPLOAD_MBDATA, INTEGRATOR_STEP_2);
		}

		// variable gravity
		if (problem->get_simparams()->gcallback) {
			// ask the Problem to update gravity, one per process
			doCallBacks();
			// upload on the GPU, one per device
			doCommand(UPLOAD_GRAVITY, INTEGRATOR_STEP_2);
		}

		// integrate also the externals
		gdata->only_internal = false;
		doCommand(EULER, INTEGRATOR_STEP_2);

		// update inlet/outlet changes only after step 2
		if (inoutlets)
			doCommand(DOWNLOAD_NEWNUMPARTS);

		// this made sense for testing and running EULER on internals only
		//if (gdata->devices > 1)
			//doCommand(UPDATE_EXTERNAL, BUFFER_POS | BUFFER_VEL);

	//			//reduce bodies

		gdata->swapDeviceBuffers(BUFFER_POS | BUFFER_VEL);

		// choose minimum dt among the devices
		if (gdata->problem->get_simparams()->dtadapt) {
			gdata->dt = gdata->dts[0];
			for (int d=1; d < gdata->devices; d++)
				gdata->dt = min(gdata->dt, gdata->dts[d]);
			// TODO for multinode
			//MM			send dt
			//MM			gather dts, chose min, broadcast (if rank 0)
			//MM			receive global dt
		}

		// TODO				check/throw dtZero exception

		// increase counters
		gdata->iterations++;
		gdata->t += gdata->dt;
		// buildneibs_freq?

		//printf("Finished iteration %lu, time %g, dt %g\n", gdata->iterations, gdata->t, gdata->dt);

		bool finished = gdata->problem->finished(gdata->t);
		bool need_write = (!gdata->nosave) && (gdata->problem->need_write(gdata->t) || finished);
		//bool need_write = (!nosave) && gdata->iterations % 10 == 0;
		need_write &= (!gdata->nosave);


		if (need_write) {
			// TODO: the performanceCounter could be "paused" here
			// ask workers to dump their subdomains and wait for it to complete
			doCommand(DUMP, BUFFER_POS | BUFFER_VEL | BUFFER_INFO | DBLBUFFER_READ );
			// triggers Writer->write()
			doWrite();
			// usual status
			printStatus();
		}

		if (finished || gdata->quit_request) {
			printStatus();
			// regardless --nosave is enabled
			printf("Issuing final save...\n");
			// set the buffers to be dumped
			uint which_buffers = BUFFER_POS | BUFFER_VEL | BUFFER_INFO;
			// choose the read buffer for the double buffered arrays
			which_buffers |= DBLBUFFER_READ;
			// compute and dump voriticity if set
			if (gdata->problem->get_simparams()->vorticity) {
				doCommand(VORTICITY);
				which_buffers |= BUFFER_VORTICITY;
			}
			// compute and dump normals if set
			// Warning: in the original code, buildneibs is called before surfaceParticle(). However, here should be safe
			// not to call, since it has been called at least once for sure
			if (gdata->problem->get_simparams()->surfaceparticle) {
				doCommand(SURFACE_PARTICLES);
				gdata->swapDeviceBuffers(BUFFER_INFO);
				which_buffers |= BUFFER_NORMALS;
			}
			// dumping AFTER normals, since it also affects the particleInfo
			doCommand(DUMP, which_buffers );
			doWrite();
			// NO doCommand() after keep_going has been unset!
			gdata->keep_going = false;
		}
	}

	// NO doCommand() nor other barriers than the standard ones after the

	printf("Simulation end, cleaning up...\n");

	// doCommand(QUIT) would be equivalent, but this is more clear
	gdata->nextCommand = QUIT;
	gdata->threadSynchronizer->barrier();  // unlock CYCLE BARRIER 2
	gdata->threadSynchronizer->barrier();  // end of SIMULATION, begins FINALIZATION ***

	// just wait or...?

	gdata->threadSynchronizer->barrier();  // end of FINALIZATION ***

	// after the last barrier has been reached by all threads (or after the Synchronizer has been forcedly unlocked),
	// we wait for the threads to actually exit
	for (int d=0; d < gdata->devices; d++)
		gdata->GPUWORKERS[d]->join_worker();

	return true;
}

// Allocate the shared buffers, i.e. those accessed by all workers
// Returns the number of allocated bytes.
// This does *not* include what was previously allocated (e.g. particles in problem->fillparts())
long unsigned int GPUSPH::allocateGlobalHostBuffers()
{

	long unsigned int numparts = gdata->allocatedParticles;
	unsigned int numcells = gdata->nGridCells;
	const uint float3Size = sizeof(float3) * numparts;
	const uint float4Size = sizeof(float4) * numparts;
	const uint infoSize = sizeof(particleinfo) * numparts;
	const uint ucharCellSize = sizeof(uchar) * numcells;
	const uint uintCellSize = sizeof(uint) * numcells;

	long unsigned int totCPUbytes = 0;

	// allocate pinned memory
	//s_hPos = (float*)
	// test also cudaHostAllocWriteCombined
	//cudaHostAlloc(&s_hPos, memSize4, cudaHostAllocPortable);
	//cudaMallocHost(&s_hPos, memSize4, cudaHostAllocPortable);
	gdata->s_hPos = new float4[numparts];
	memset(gdata->s_hPos, 0, float4Size);
	totCPUbytes += float4Size;

	gdata->s_hVel = new float4[numparts];
	memset(gdata->s_hVel, 0, float4Size);
	totCPUbytes += float4Size;

	gdata->s_hInfo = new particleinfo[numparts];
	memset(gdata->s_hInfo, 0, infoSize);
	totCPUbytes += infoSize;

	if (problem->m_simparams.vorticity) {
		gdata->s_hVorticity = new float3[numparts];
		memset(gdata->s_hVorticity, 0, float3Size);
		totCPUbytes += float3Size;
	}

	if (problem->m_simparams.surfaceparticle) {
		gdata->s_hNormals = new float4[numparts];
		memset(gdata->s_hNormals, 0, float4Size);
		totCPUbytes += float4Size;
	}

	/*dump_hPos = new float4[numparts];
	memset(dump_hPos, 0, float4Size);
	totCPUbytes += float4Size;

	dump_hVel = new float4[numparts];
	memset(dump_hVel, 0, float4Size);
	totCPUbytes += float4Size;

	dump_hInfo = new particleinfo[numparts];
	memset(dump_hInfo, 0, infoSize);
	totCPUbytes += infoSize;*/

	if (gdata->devices > 1) {
		// deviceMap
		gdata->s_hDeviceMap = new uchar[numcells];
		memset(gdata->s_hDeviceMap, 0, ucharCellSize);
		totCPUbytes += ucharCellSize;

		// cellStarts, cellEnds, segmentStarts of all devices. Array of device pointers stored on host
		gdata->s_dCellStarts = (uint**)calloc(gdata->devices, sizeof(uint*));
		gdata->s_dCellEnds =  (uint**)calloc(gdata->devices, sizeof(uint*));
		gdata->s_dSegmentsStart = (uint**)calloc(gdata->devices, sizeof(uint*));

		// few bytes... but still count them
		totCPUbytes += gdata->devices * sizeof(uint*) * 3;

		for (uint d=0; d < gdata->devices; d++) {
			gdata->s_dCellStarts[d] = (uint*)calloc(numcells, sizeof(uint));
			gdata->s_dCellEnds[d] =   (uint*)calloc(numcells, sizeof(uint));
			totCPUbytes += uintCellSize * 2;

			gdata->s_dSegmentsStart[d] = (uint*)calloc(4, sizeof(uint));
			totCPUbytes += sizeof(uint) * 4;
		}
	}
	return totCPUbytes;
}

// Deallocate the shared buffers, i.e. those accessed by all workers
void GPUSPH::deallocateGlobalHostBuffers()
{
	//cudaFreeHost(s_hPos); // pinned memory
	delete [] gdata->s_hPos;
	delete [] gdata->s_hVel;
	delete [] gdata->s_hInfo;
	if (problem->m_simparams.vorticity)
		delete [] gdata->s_hVorticity;
	if (problem->m_simparams.surfaceparticle)
		delete [] gdata->s_hNormals;
	// multi-GPU specific arrays
	if (gdata->devices>1) {
		delete [] gdata->s_hDeviceMap;
		// cells
		for (int d=0; d < gdata->devices; d++) {
			delete [] gdata->s_dCellStarts[d];
			delete [] gdata->s_dCellEnds[d];
			delete [] gdata->s_dSegmentsStart[d];
		}
		delete [] gdata->s_dCellEnds;
		delete [] gdata->s_dCellStarts;
		delete [] gdata->s_dSegmentsStart;
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

	// reset counters. Not using memset since its size is typically lower than 1Kb
	for (uint d=0; d < MAX_DEVICES_PER_CLUSTER; d++)
		gdata->s_hPartsPerDevice[d] = 0;

	// TODO: move this in allocateGlobalBuffers...() and rename it, or use only here as a temporary buffer?
	uchar* m_hParticleHashes = new uchar[gdata->totParticles];

	// fill array with particle hashes (aka global device numbers)
	for (uint p=0; p < gdata->totParticles; p++) {
		// compute cell according to the particle's position and to the deviceMap
		uchar whichDev = gdata->calcDevice(gdata->s_hPos[p]);
		// that's the key!
		m_hParticleHashes[p] = whichDev;
		// increment per-device counter
		gdata->s_hPartsPerDevice[whichDev]++;
	}

	// update s_hStartPerDevice with incremental sum (should do in specific function?)
	gdata->s_hStartPerDevice[0] = 0;
	for (uint d=1; d < gdata->devices; d++)
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
	// are particles already in correct positions. When there is a bucket change (we know because of )
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
	for (uint currentDevice=0; currentDevice < (gdata->devices-1); currentDevice++) {
		// compute where current bucket ends
		  //nextBucketBeginsAt += gdata->s_hPartsPerDevice[currentDevice];
		nextBucketBeginsAt = gdata->s_hStartPerDevice[currentDevice+1];
		// reset rightB to the end
		rightB = maxIdx;
		// go on until we reach the end of the current bucket
		while (leftB < nextBucketBeginsAt) {
			// if in the current position there is a particle *not* belonging to the bucket...
			if (m_hParticleHashes[leftB] != currentDevice) {
				// ...let's find a correct one, scanning from right to left
				while (m_hParticleHashes[rightB] != currentDevice) rightB--;
				// here it should never happen that (rightB <= leftB). We should throw an error if it happens
				particleSwap(leftB, rightB);
				std::swap(m_hParticleHashes[leftB], m_hParticleHashes[rightB]);
			}
			// already correct or swapped, time to go on
			leftB++;
		}
	}
	// delete array of keys (might be recycled instead?)
	delete [] m_hParticleHashes;

	// initialize the outer cells values in s_dSegmentsStart. The inner_edge are still uninitialized
	for (uint currentDevice=0; currentDevice < gdata->devices; currentDevice++) {
		uint assigned_parts = gdata->s_hPartsPerDevice[currentDevice];
		printf("    d%u  p %u\n", currentDevice, assigned_parts);
		// this should always hold according to the current CELL_TYPE values
		gdata->s_dSegmentsStart[currentDevice][CELLTYPE_INNER_CELL]			= EMPTY_SEGMENT;
		// this is usually not true, since a device usually has neighboring cells; will be updated at first reorder
		gdata->s_dSegmentsStart[currentDevice][CELLTYPE_INNER_EDGE_CELL]	= EMPTY_SEGMENT;
		// this is true and will change at first APPEND
		gdata->s_dSegmentsStart[currentDevice][CELLTYPE_OUTER_EDGE_CELL]	= EMPTY_SEGMENT;
		// this is true and might change between a reorder and the following crop
		gdata->s_dSegmentsStart[currentDevice][CELLTYPE_OUTER_CELL]			= EMPTY_SEGMENT;
	}

	// DEBUG: check if the sort was correct
	/*bool monotonic = true;
	bool count_c = true;
	uint hcount[MAX_DEVICES_PER_CLUSTER];
	for (uint d=0; d<gdata->devices; d++)
		hcount[d] = 0;
	for (uint p=0; p < gdata->totParticles && monotonic; p++) {
		uint cdev = gdata->calcDevice(gdata->s_hPos[p]);
		if (p > 0 && cdev < gdata->calcDevice(gdata->s_hPos[p-1])) {
			printf(" -- sorting error: array[%d] has device %d, array[%d] has device %d\n",
				p-1, gdata->calcDevice(gdata->s_hPos[p-1]), p, cdev );
			monotonic = false;
		}
		hcount[cdev]++;
	}
	for (uint d=0; d<gdata->devices; d++)
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

// Swap two particles in shared arrays (pos, vel, pInfo); used in host sort
void GPUSPH::particleSwap(uint idx1, uint idx2)
{
	// could keep a counter
	swap( gdata->s_hPos[idx1],  gdata->s_hPos[idx2] );
	swap( gdata->s_hVel[idx1],  gdata->s_hVel[idx2] );
	swap( gdata->s_hInfo[idx1], gdata->s_hInfo[idx2] );
}

// set nextCommand, unlock the threads and wait for them to complete
void GPUSPH::doCommand(CommandType cmd, uint flags)
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
	gdata->threadSynchronizer->barrier(); // unlock CYCLE BARRIER 2
	gdata->threadSynchronizer->barrier(); // wait for completion of last command and unlock CYCLE BARRIER 1
}

void GPUSPH::setViscosityCoefficient()
{
	PhysParams *pp = gdata->problem->get_physparams();
	// Setting visccoeff
	switch (gdata->problem->get_simparams()->visctype) {
		case ARTVISC:
			pp->visccoeff = pp->artvisccoeff;
			break;

		case KINEMATICVISC:
		case SPSVISC:
			pp->visccoeff = 4.0*pp->kinematicvisc;
			break;

		case DYNAMICVISC:
			pp->visccoeff = pp->kinematicvisc;
			break;
	}
}

// creates the Writer according to the requested WriterType
void GPUSPH::createWriter()
{
	//gdata->writerType = gdata->problem->get_writertype();
	//switch(gdata->writerType) {
	switch (gdata->problem->get_writertype()) {
		case Problem::TEXTWRITER:
			gdata->writerType = TEXTWRITER;
			gdata->writer = new TextWriter(gdata->problem);
			break;

		case Problem::VTKWRITER:
			gdata->writerType = VTKWRITER;
			gdata->writer = new VTKWriter(gdata->problem);
			gdata->writer->setGlobalData(gdata); // VTK also supports writing the device index
			break;

		case Problem::VTKLEGACYWRITER:
			gdata->writerType = VTKLEGACYWRITER;
			gdata->writer = new VTKLegacyWriter(gdata->problem);
			break;

		case Problem::CUSTOMTEXTWRITER:
			gdata->writerType = CUSTOMTEXTWRITER;
			gdata->writer = new CustomTextWriter(gdata->problem);
			break;

		case Problem::UDPWRITER:
			gdata->writerType = UDPWRITER;
			gdata->writer = new UDPWriter(gdata->problem);
			break;

		default:
			//stringstream ss;
			//ss << "Writer not supported";
			//throw runtime_error(ss.str());
			printf("Writer not supported\n");
			break;
	}
}

void GPUSPH::doWrite()
{
	gdata->writer->write(gdata->totParticles, gdata->s_hPos, gdata->s_hVel, gdata->s_hInfo,
		gdata->s_hVorticity, gdata->t, gdata->problem->get_simparams()->testpoints,
		gdata->s_hNormals);
	gdata->problem->mark_written(gdata->t);
	// TODO: enable energy computation and dump
	/*calc_energy(m_hEnergy,
		m_dPos[m_currentPosRead],
		m_dVel[m_currentVelRead],
		m_dInfo[m_currentInfoRead],
		m_numParticles,
		m_physparams->numFluids);
	m_writer->write_energy(m_simTime, m_hEnergy);*/
}

void GPUSPH::doCallBacks()
{
	Problem *pb = gdata->problem;

	float addendum = gdata->dt;
	if (gdata->commandFlags == INTEGRATOR_STEP_1)
		addendum /= 2.0f;

	if (pb->m_simparams.mbcallback)
		gdata->s_mbData = pb->get_mbdata(
			gdata->t + addendum,
			gdata->dt/2.0f,
			gdata->iterations == 0);

	if (pb->m_simparams.gcallback)
		gdata->s_varGravity = pb->g_callback(gdata->t);
}

void GPUSPH::printStatus()
{
//#define ti timingInfo
	printf(	"Simulation time t=%es, iteration=%s, dt=%es, %s parts (%.2g MIPPS)\n",
			//"mean %e neibs. in %es, %e neibs/s, max %u neibs\n"
			//"mean neib list in %es\n"
			//"mean integration in %es\n",
			gdata->t, gdata->addSeparators(gdata->iterations).c_str(), gdata->dt,
			gdata->addSeparators(gdata->totParticles).c_str(), m_performanceCounter->getMIPPS(gdata->iterations * gdata->totParticles)
			//ti.t, ti.iterations, ti.dt, ti.numParticles, (double) ti.meanNumInteractions,
			//ti.meanTimeInteract, ((double)ti.meanNumInteractions)/ti.meanTimeInteract, ti.maxNeibs,
			//ti.meanTimeNeibsList,
			//ti.meanTimeEuler
			);
	fflush(stdout);
//#undef ti
}

// Do a roll call of particle IDs; useful after dumps if no in/outlets are there and if the filling was uniform.
// Notifies anomalies only once in the simulation for each particle ID
void GPUSPH::rollCallParticles()
{
	bool all_normal = true;

	// reset bitmap and addrs
	for (uint idx = 0; idx < gdata->totParticles; idx++) {
		m_rcBitmap[idx] = false;
		m_rcAddrs[idx] = 0xFFFFFFFF;
	}

	// fill out the bitmap and check for duplicates
	for (uint pos = 0; pos < gdata->totParticles; pos++) {
		uint idx = id(gdata->s_hInfo[pos]);
		if (m_rcBitmap[idx] && !m_rcNotified[idx]) {
			printf("WARNING: at iteration %d, time %g particle idx %u is in pos %u and %u! Wait for ENTER...\n",
				gdata->iterations, gdata->t, idx, m_rcAddrs[idx], pos );
			// getchar(); // useful for debugging
			all_normal = false;
			m_rcNotified[idx] = true;
		}
		m_rcBitmap[idx] = true;
		m_rcAddrs[idx] = pos;
	}
	// now check if someone is missing
	for (uint idx = 0; idx < gdata->totParticles; idx++)
		if (!m_rcBitmap[idx] && !m_rcNotified[idx]) {
			printf("WARNING: at iteration %d, time %g particle idx %u was not found! Wait for ENTER...\n",
				gdata->iterations, gdata->t, idx);
			// getchar(); // useful for debugging
			m_rcNotified[idx] = true;
			all_normal = false;
		}
	// if there was any warning...
	if (!all_normal) {
		printf("Recap of devices after roll call:\n");
		for (uint d=0; d < gdata->devices; d++) {
				printf(" - device at index %u has %s particles assigned and offset %s\n",
					d, gdata->addSeparators(gdata->s_hPartsPerDevice[d]).c_str(), gdata->addSeparators(gdata->s_hStartPerDevice[d]).c_str());
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
	uint count = 0;
	// this should always hold
	gdata->s_hStartPerDevice[0] = 0;
	// just store an incremental counter
	for (int d=0; d < gdata->devices; d++) {
		count += gdata->s_hPartsPerDevice[d] = gdata->GPUWORKERS[d]->getNumInternalParticles();
		if (d > 0)
			gdata->s_hStartPerDevice[d] = gdata->s_hStartPerDevice[d-1] + gdata->s_hPartsPerDevice[d-1];
	}
	// number of particle may increase or decrease if there are respectively inlets or outlets
	if ( (count < gdata->totParticles && gdata->problem->get_physparams()->outlets > 0) ||
		 (count > gdata->totParticles && gdata->problem->get_physparams()->inlets > 0) ) {
		printf("Number of total particles at iteration %u passed from %u to %u\n", gdata->iterations, gdata->totParticles, count);
		gdata->totParticles = count;
	} else
	if (count != gdata->totParticles) {
		printf("WARNING: at iteration %u the number of particles changed from %u to %u for no known reason!\n", gdata->iterations, gdata->totParticles, count);

		// who is missing?
		doCommand(DUMP, BUFFER_INFO | DBLBUFFER_READ );
		rollCallParticles();

		// update totParticles to avoid dumping an outdated particle (and repeating the warning).
		// Note: updading *after* the roll call likely shows the missing particle(s) and the duplicate(s). Doing before it only shows the missing one(s)
		gdata->totParticles = count;
	}
	// in case estimateMaxInletsIncome() was slightly in defect (unlikely)
	if (count > gdata->allocatedParticles) {
		printf("FATAL: Number of total particles at iteration %u (%u) exceeding allocated buffers (%u). Requesting immediate quit\n",
			gdata->iterations, count, gdata->allocatedParticles);
		gdata->quit_request = true;
	}
}
