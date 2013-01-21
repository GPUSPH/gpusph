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
#include <time.h>

#define GPUSPH_MAIN
#include "particledefine.h"
#undef GPUSPH_MAIN

// NOTE: including GPUSPH.h before particledefine.h does not compile.
// This inclusion problem should be solved
#include "GPUSPH.h"

#include "ParticleSystem.h"
#include "Problem.h"

/* Include only the problem selected at compile time */
#include "problem_select.opt"

using namespace std;

FILE *timing_log = NULL;

ParticleSystem *psystem = 0;

// timing
TimingInfo  timingInfo;
clock_t		start_time;

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
	double elapsed_sec = (clock() - start_time)/CLOCKS_PER_SEC;
	printf("\nTotal time %es\n", elapsed_sec);
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
		"integration in %es (mean %es)\n",
		ti.t, ti.dt, ti.numParticles,
		(double)ti.numInteractions, ti.timeNeibsList, ti.meanTimeNeibsList, ti.maxNeibs,
		(double)ti.meanNumInteractions, ti.numInteractions/ti.timeInteract, ti.meanNumInteractions/timingInfo.meanTimeInteract,
		ti.timeEuler, ti.meanTimeEuler);
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

	psystem = new ParticleSystem(problem);

	psystem->printPhysParams();
	psystem->printSimParams();

	// filling simulation domain with particles
	uint numParticles = problem->fill_parts();
	psystem->allocate(numParticles);
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

	start_time = clock();
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
			"mean integration in %es\n",
			ti.t, ti.iterations, ti.dt, ti.numParticles, (double) ti.meanNumInteractions,
			ti.meanTimeInteract, ((double)ti.meanNumInteractions)/ti.meanTimeInteract, ti.maxNeibs,
			ti.meanTimeNeibsList,
			ti.meanTimeEuler);
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

	bool finished = problem->finished(timingInfo.t);

	bool need_display = displayEnabled && problem->need_display(timingInfo.t);
	bool need_write = problem->need_write(timingInfo.t) || finished;
	problem->write_rbdata(timingInfo.t);

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
		problem->draw_axis();

		char s[1024];
		size_t len = sprintf(s, "t=%7.4es", timingInfo.t, timingInfo.dt);
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

	switch (timing) {
		case M_INTERACTION:
			sprintf(title, "t=%7.2es dt=%7.2es %d parts. %7.2eint. : %7.2eint./s (mean %7.2eint./s) (maxneibs %d)", timingInfo.t, timingInfo.dt,
			timingInfo.numParticles, (double) timingInfo.meanNumInteractions,
			((double) timingInfo.numInteractions)/timingInfo.timeInteract,
			((double) timingInfo.meanNumInteractions)/timingInfo.meanTimeInteract, timingInfo.maxNeibs);
			break;

		case M_NEIBSLIST:
			sprintf(title, "t=%7.2es dt=%7.2es %d parts. %7.2e neibs in %7.2es (mean %7.2es) (maxneibs %d)",timingInfo.t, timingInfo.dt,
			timingInfo.numParticles, (double) timingInfo.numInteractions,
			timingInfo.timeNeibsList,
			timingInfo.meanTimeNeibsList, timingInfo.maxNeibs);
			break;

		case M_EULER:
			sprintf(title, "t=%7.2es dt=%7.2es %d parts. integration in %7.2es (mean %7.2es)", timingInfo.t, timingInfo.dt,
			timingInfo.numParticles, timingInfo.timeEuler, timingInfo.meanTimeEuler);
			break;

		case M_MEAN:
			sprintf(title, "%7.2e interactions (%7.2eint./s) - Neibs list %7.2es - Euler %7.2es", (double) timingInfo.meanNumInteractions,
				(double) timingInfo.meanNumInteractions/timingInfo.meanTimeInteract, timingInfo.meanTimeNeibsList, timingInfo.meanTimeEuler);
			break;

		case M_NOTIMING:
			title[0] = '\0';
			break;
	}

	// leave the "Hit space to start" message until unpaused
	if (!bPause)
		glutSetWindowTitle(title);

	glutSwapBuffers();

	glutReportErrors();

	// Taking a screenshot
	if (displayEnabled && (problem->need_screenshot(timingInfo.t) || screenshotNow))
	{
		glscreenshot->TakeScreenshot(timingInfo.t);
		if (screenshotNow) {
			cout << "Screenshot @ " << timingInfo.t << endl;
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

		finished |= problem->finished(timingInfo.t);

		bool need_write = problem->need_write(timingInfo.t) || finished;

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
main( int argc, char** argv)
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
	if (!initialized)
		finalize();
}

bool GPUSPH::initialize(GlobalData *_gdata) {
	gdata = _gdata;
	clOptions = gdata->clOptions;
	problem = gdata->problem;

	//		> new PS
	psystem = new ParticleSystem(gdata);
	//			PS constructor updates cellSize, worldSize, nCells, etc. in gdata
	//			PS creates new writer. Do it outside?

	// no: done
	//		> new Problem

	// TODO: allocate and fill the device map. Only ints?
	//			problem > createDeviceMap
	//			global dev id, bit edging
	//		//GPUSPH > createUploadMask (64 bit per cell hash, 1 bit per device)

	// allocate the particles of the *whole* simulation
	gdata->totParticles = problem->fill_parts();
	// allocate cpu buffers, 1 per process
	allocateCPUBuffers(); // TODO
	// To check: is this mandatory? this requires double memory!
	//	copy particles from problem to GPUSPH buffers
	problem->copy_to_array(gdata->s_hPos, gdata->s_hVel, gdata->s_hInfo);

	// TODO
	//		GPUSPH > calcHashHost (partid, cell id, cell device)
	//		GPUSPH > hostSort

	// TODO
	//		// > new Integrator

	// TO check: this was done in main. Move in GPUSPH, or even in GlobalData
	///		> new Synchronizer

	gdata->GPUWORKERS = (GPUWorker**)calloc(gdata->devices, sizeof(GPUWorker*));
	for (int d=0; d < gdata->devices; d++)
		gdata->GPUWORKERS[d] = new GPUWorker(gdata, d);

	// TODO
	//		+ start workers

	// DO in GPUworkers
	//		GPUWorkers > checkCUDA (before allocating everything; put CC in common structure)
	//		GPUWorkers > allocateGPU
	//		GPUWorkers > uploadSubdomains (cell by cell, light optimizations)
	//			incl. edging!
	//		GPUWorkers > createCompactDevMap (from global devmap to 2bits/dev)
	//		GPUWorkers > uploadCompactDevMap (2 bits per cell, to be elaborated on this)

	// TODO barrier to wait for their init or other mechanism

	initialized = true;
}

bool GPUSPH::finalize() {
	//GPUWorkers > deallocateGPU
	//		// delete Integrator
	//		delete Synchronizer
	//		delete Workers
	deallocateCPUBuffers();
	//		problem > deallocate (every process allocates everything)
	//		delete Problem
	//		delete PS
	//			delete Writer
	// ...
	initialized = false;
}

bool GPUSPH::runSimulation() {
	//while (keep_going)
	//			// Integrator > setNextStep
	//			// run next SimulationStep (workers do it, w barrier)
	//			// or -----
	//			> buildNeibslist
	//				k>  calcHash
	//					2 bits from compactDevMap + usual
	//				k>  sort_w_ids
	//				k>  reorderDataAndFindCellStart
	//				swap3
	//				k>  buildNeibslist
	//			k>  shepard && swap1
	//			k>  mls && swap
	//			//set mvboundaries and gravity
	//			//(init bodies)
	//			k>  forces (in dt1; write on internals)
	//MM		fetch/update forces on neighbors in other GPUs/nodes
	//				initially done trivial and slow: stop and read
	//			//reduce bodies
	//			k>  euler (also on externals)
	//			//reduce bodies
	//			//callbacks (bounds, gravity)
	//MM		fetch/update forces on neighbors in other GPUs/nodes
	//				initially done trivial and slow: stop and read
	//			k>  forces (in dt2; write on internals)
	//			//reduce bodies
	//			k>  euler (also on externals)
	//			//reduce bodies
	//			swap2
	//			increase t bz dt and iter (in every process
	//			if dtadapt
	//				dt = min (dt1, dt2)
	//MM			send dt
	//MM			gather dts, chose min, broadcast (if rank 0)
	//MM			receive global dt
	//				check/throw dtZero exception
	//		problem > finished(t);
	//		problem > need_write(t)
	//		if (need_write)
	//			> get_arrays
	//				ask workers to dump arrays
	//			> do_write
	//				printf info
	//				ps > writeToFile
}

// Returns the number of allocated bytes.
// This does *not* include what was previously allocated (e.g. particles in problem->fillparts())
long unsigned int GPUSPH::allocateCPUBuffers() {

	long unsigned int numparts = gdata->totParticles;
	const uint float4Size = sizeof(float4) * numparts;
	const uint infoSize = sizeof(particleinfo) * numparts;

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

	/*dump_hPos = new float4[numparts];
	memset(dump_hPos, 0, float4Size);
	totCPUbytes += float4Size;

	dump_hVel = new float4[numparts];
	memset(dump_hVel, 0, float4Size);
	totCPUbytes += float4Size;

	dump_hInfo = new particleinfo[numparts];
	memset(dump_hInfo, 0, infoSize);
	totCPUbytes += infoSize;*/

	return totCPUbytes;
}

void GPUSPH::deallocateCPUBuffers() {
	//cudaFreeHost(s_hPos); // pinned memory
	delete [] gdata->s_hPos;
	delete [] gdata->s_hVel;
	delete [] gdata->s_hInfo;
}
