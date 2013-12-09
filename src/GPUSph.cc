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

#include <GL/glew.h>
#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include "gl_screenshot.h"

#define GPUSPH_MAIN
#include "particledefine.h"
#undef GPUSPH_MAIN

#include "ParticleSystem.h"
#include "Problem.h"

/* Include only the problem selected at compile time */
#include "problem_select.opt"

/* Include all other opt file for show_version */
#include "gpusph_version.opt"
#include "fastmath_select.opt"
#include "dbg_select.opt"
#include "compute_select.opt"

using namespace std;

FILE *timing_log = NULL;

int ox, oy; float oz = 0.9;

GLdouble oldx, oldy, oldz;
GLdouble newx, newy, newz;
GLdouble modelview[16];
GLdouble projection[16];
GLint viewport[4];

int buttonState = 0;
const float inertia = 0.5;

int mode = 0;
bool displayEnabled = true;
bool bPause = true;
bool stepping_mode = false;
bool show_boundary = false;
bool show_floating = false;

enum { M_VIEW = 0, M_MOVE};
int view_field = ParticleSystem::VM_NORMAL;
enum { M_INTERACTION = 0, M_NEIBSLIST, M_EULER, M_MEAN, M_IPPS, M_NOTIMING};
int timing = M_IPPS;

ParticleSystem *psystem = 0;

CScreenshot *glscreenshot = 0;

float modelView[16];

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

bool screenshotNow = false;

void show_version()
{
	static const char dbg_or_rel[] =
#if defined(_DEBUG_)
		"Debug";
#else
		"Release";
#endif

	printf("GPUSPH version %s\n", GPUSPH_VERSION);
	printf("%s version %s fastmath for compute capability %u.%u\n",
		dbg_or_rel,
		FASTMATH ? "with" : "without",
		COMPUTE/10, COMPUTE%10);
	printf("Compiled for problem \"%s\"\n", QUOTED_PROBLEM);
}

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

void reset_display(void)
{
	camera_pos = target_pos = worldOrigin + worldSize/2;
	camera_pos.y += worldSize.x*view_trig/2;
	camera_pos.z += worldSize.z*view_trig/2;

	float3 camvec = camera_pos - target_pos;

	camera_up = rotate(camvec, cross(camvec, z_axis), M_PI/2);
}

void reset_target(void)
{
	if (rotating != ROT_NONE)
		return;

	float z;
	double tx, ty, tz;
	glReadPixels(viewport[2]/2, viewport[3]/2, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z);
	if (z == 1)
		return;

	gluUnProject(viewport[2]/2, viewport[3]/2, z, modelview,
			projection, viewport, &tx, &ty, &tz);

	target_pos.x = tx;
	target_pos.y = ty;
	target_pos.z = tz;
}

/* Command line options */
Options clOptions;

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
			sscanf(*argv, "%lf", &(clOptions.deltap));
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
		} else if (!strcmp(arg, "--dir")) {
			clOptions.dir = std::string(*argv);
			argv++;
			argc--;
		} else if (!strcmp(arg, "--console")) {
			clOptions.console = true;
		} else if (!strcmp(arg, "--version")) {
			show_version();
			exit(0);
		} else if (!strncmp(arg, "--", 2)) {
			cerr << "Skipping unsupported option " << arg << endl;
		} else {
			cerr << "Fatal: Unknown option: " << arg << endl;
			exit(1);

			// Left for future dynamic loading:
			/*if (clOptions.problem.empty()) {
				clOptions.problem = std::string(arg);
			} else {
				cout << "Problem " << arg << " selected after problem " << clOptions.problem << endl;
			}*/
		}
	}

	clOptions.problem = std::string( QUOTED_PROBLEM );

	// Left for future dynamic loading:
	/*if (clOptions.problem.empty()) {
		problem_list();
		exit(0);
	}*/
}


void init(const char *arg)
{
	problem = new PROBLEM(clOptions);
	problem->check_dt();

	printf("Problem calling set grid params\n");
	problem->set_grid_params();

	/* TODO do it this way for all options? */
	if (isfinite(clOptions.tend))
		problem->get_simparams()->tend = clOptions.tend;

	worldOrigin = make_float3(problem->get_worldorigin());
	worldSize = make_float3(problem->get_worldsize());

	box_corner[0] = worldOrigin;
	box_corner[1] = box_corner[0];
	box_corner[1].x += worldSize.x;
	box_corner[2] = box_corner[0];
	box_corner[2].y += worldSize.y;
	box_corner[3] = box_corner[0];
	box_corner[3].z += worldSize.z;
	box_corner[4] = box_corner[1];
	box_corner[4].z += worldSize.z;
	box_corner[5] = box_corner[2];
	box_corner[5].z += worldSize.z;
	box_corner[6] = box_corner[1];
	box_corner[6].y += worldSize.y;
	box_corner[7] = box_corner[6];
	box_corner[7].z += worldSize.z;


	reset_display();

	psystem = new ParticleSystem(problem);

	// filling simulation domain with particles
	uint numParticles = problem->fill_parts();
	psystem->allocate(numParticles);

	psystem->printPhysParams();
	psystem->printSimParams();

	problem->copy_to_array(psystem->m_hPos, psystem->m_hVel, psystem->m_hInfo, psystem->m_hParticleHash);
	psystem->setArray(ParticleSystem::POSITION);
	psystem->setArray(ParticleSystem::VELOCITY);
	psystem->setArray(ParticleSystem::INFO);
	psystem->setArray(ParticleSystem::HASH);

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

	glscreenshot = new CScreenshot(problem->get_dirname());

	timingInfo = psystem->markStart();
}


void initGL()
{
	glewInit();
	if (!glewIsSupported("GL_VERSION_1_4")) {
		fprintf(stderr, "Required OpenGL extensions missing.");
		exit(-1);
	}

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	glClearColor(1, 1, 1, 0);

	glutReportErrors();
}

void set_old(int x, int y)
{
	ox = x; oy = y;

	float z;
	glReadPixels(x, viewport[3] - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z);
	if (z == 1) {
		glReadPixels(viewport[2]/2, viewport[3]/2, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &z);
		if (z == 1)
			z = oz;
	}
	oz = z;
	gluUnProject(x, viewport[3] - y, oz, modelview,
			projection, viewport, &oldx, &oldy, &oldz );
}

void set_new(int x, int y)
{
	gluUnProject(x, viewport[3] - y, oz, modelview,
			projection, viewport, &newx, &newy, &newz );
}

void update_projection(void)
{
	// get the projection matrix
	glGetDoublev( GL_PROJECTION_MATRIX, projection );
	// get the modelview matrix
	glGetDoublev( GL_MODELVIEW_MATRIX, modelview );
}

void look(bool update=true)
{
	near_plane = HUGE_VAL;
	far_plane = 0;
	float d = HUGE_VAL-HUGE_VAL;

#define DIST_CHECK(expr) do { \
	d = expr; \
	if (d < near_plane) near_plane = d; \
	if (d > far_plane) far_plane = d; \
} while (0)

	DIST_CHECK(fabs(camera_pos.x - worldSize.x));
	DIST_CHECK(fabs(camera_pos.x - worldSize.x - worldOrigin.x));
	DIST_CHECK(fabs(camera_pos.y - worldSize.y));
	DIST_CHECK(fabs(camera_pos.y - worldSize.y - worldOrigin.y));
	DIST_CHECK(fabs(camera_pos.z - worldSize.z));
	DIST_CHECK(fabs(camera_pos.z - worldSize.z - worldOrigin.z));

	for (uint i = 0; i < 8; ++i)
		DIST_CHECK(length(camera_pos - box_corner[i]));

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(view_angle, GLdouble(viewport[2])/viewport[3], near_plane, far_plane);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(camera_pos.x, camera_pos.y, camera_pos.z,
			target_pos.x, target_pos.y, target_pos.z,
			camera_up.x, camera_up.y, camera_up.z);

	if (update)
		update_projection();

	reset_target();
}

/* heads up display code */
void viewOrtho(int x, int y){ // Set Up An Ortho View
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glOrtho(0, x , 0, y , -1, 1);
	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
}

void viewPerspective() // Set Up A Perspective View
{
	glMatrixMode(GL_PROJECTION); // Select Projection
	glPopMatrix(); // Pop The Matrix
	glMatrixMode(GL_MODELVIEW); // Select Modelview
	glPopMatrix(); // Pop The Matrix
}

void displayStatus(char *s) {
	int len, i;
	viewOrtho(viewport[2], viewport[3]); //Starting to draw the HUD
	glRasterPos2i(10, 10);
	/*
	float3 tank_size = make_float3(problem->m_size);
	glRasterPos3f(tank_size.x, tank_size.y, 0.0);
	*/
	len = (int) strlen(s);
	for (i = 0; i < len; i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, s[i]);
	}
	viewPerspective(); //switch back to 3D drawing
}

void get_arrays(bool need_write)
{
	psystem->getArray(ParticleSystem::POSITION, need_write);
	psystem->getArray(ParticleSystem::HASH, need_write);
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

void display()
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
		problem->draw_axis();

		char s[1024];
		size_t len = sprintf(s, "t=%7.4es", timingInfo->t); // , timingInfo->dt);
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
}

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


void reshape(int w, int h)
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	glViewport(0, 0, w, h);
	viewport[2] = w;
	viewport[3] = h;

	look();
}



void zoom(float factor)
{
	float3 delta = factor*(camera_pos - target_pos);
	camera_pos += delta;
	target_pos += delta;

	look();
}

void pan(/* float dx, float dy */)
{
	float3 delta = make_float3(newx-oldx, newy-oldy, newz-oldz);

	camera_pos -= delta;
	target_pos -= delta;

	look();
}

void rotate(float x, float y, float dx, float dy)
{
	float3 camvec = camera_pos - target_pos;

	/* vector from old to mid */
	float dox = ox - viewport[2]/2;
	float doy = oy - viewport[3]/2;

	float dol2 = dox*dox+doy*doy;

	/* square length of vector from old to new */
	float dl2 = dx*dx+dy*dy;

	if (dol2*dl2 == 0)
		return;

	/* angle between old-to-mid and new-to-old */
	float dot = (dox*dx + doy*dy)/sqrt(dol2*dl2);
	float angle = acos(fabs(dot));

#define axis_tol 32
	/* Otherwise, if the rotating axis hasn't been chosen yet and we're
	   moving approximately in the line passing through the midpoint, rotate
	   around the camera_up or the ortogonal vector.
	 */
	if (rotating == ROT_NONE) {
		if (angle < M_PI_4 ||
				(fabs(dox) < axis_tol && fabs(doy) < axis_tol))
			rotating = ROT_ORT;
		else
			rotating = ROT_VEC;
	}

	// FIXME can rotating be still ROT_NONE here?

	if (rotating == ROT_ORT) {
		float tx = -dx*M_PI;
		float ty = dy*M_PI;

		/* Force rotation to one direction if there's a strong preference */
#define rot_factor 1.44
		if (fabs(tx) > rot_factor*fabs(ty))
			ty = 0;
		if (fabs(ty) > rot_factor*fabs(tx))
			tx = 0;
#undef rot_factor

		float3 ort = cross(camvec, camera_up);
		camvec = rotate(camvec, camera_up, tx);
		camvec = rotate(camvec, ort, ty);
		camera_up = rotate(camera_up, ort, ty);

		camera_pos = target_pos + camvec;
	} else {
		/* find the angle old-to-mid v new-to-mid */
		dx = x - viewport[2]/2;
		dy = y - viewport[3]/2;
		dl2 = dx*dx+dy*dy;
		dot = (dox*dy - doy*dx)/sqrt(dol2*dl2);
		angle = asin(dot);
		camera_up = rotate(camera_up, camvec, angle);
	}

	look();
}


void motion(int x, int y)
{
	float dx, dy;

	dx = (float)(x - ox)/viewport[2];
	dy = (float)(y - oy)/viewport[3];

	set_new(x, y);

	if (buttonState == 3) {
		zoom(dy);
	} else if (buttonState & 2) {
		pan(/* dx, dy */);
	} else if (buttonState & 1) { // left button
		rotate(x, y, dx, dy);
	}

	set_old(x, y);

	glutPostRedisplay();
}


void mouse(int button, int state, int x, int y)
{
	int mods;

	set_old(x, y);

	if (state == GLUT_UP) {
		buttonState = 0;
		reset_target();
		glutPostRedisplay();
		return;
	}

	/* buttons 3 and 4 correspond to
	   zoom in/zoom out (scrollwheel)
	   */
	if (button == 3) {
		zoom(-1.0/16);
		return;
	}
	if (button == 4) {
		zoom(1.0/16);
		return;
	}


	// state == GLUTDOWN
	buttonState |= 1<<button;

	mods = glutGetModifiers();
	if (mods & GLUT_ACTIVE_SHIFT) {
		buttonState = 2;
	} else if (mods & GLUT_ACTIVE_CTRL) {
		buttonState = 3;
	}

	rotating = ROT_NONE;
}


// commented out to remove unused parameter warnings in Linux
void key(unsigned char key, int /*x*/, int /*y*/)
{
	float3 camvec(camera_pos - target_pos);

	switch (key)
	{
	case ' ':
		bPause = !bPause;
		printf("%saused\n", bPause ? "P" : "Unp");
		break;

	case 13:
		stepping_mode = !stepping_mode;
		if (stepping_mode) {
			printf("Stepping\n");
		} else {
			printf("Running\n");
			bPause = false;
		}
		break;

	case '\033':
	case 'q':
		quit(0);
		break;

	case 'b':
		show_boundary = !show_boundary;
		printf("%showing boundaries\n",
				show_boundary ? "S" : "Not s");
		break;

	case 'f':
		show_floating = !show_floating;
		printf("%showing boundaries\n",
				show_boundary ? "S" : "Not s");
		break;

	case 'v':
		view_field = ParticleSystem::VM_VELOCITY;
		printf("Showing velocity\n");
		break;

	case 'p':
		view_field = ParticleSystem::VM_PRESSURE;
		printf("Showing pressure\n");
		break;

	case 'd':
		view_field = ParticleSystem::VM_DENSITY;
		printf("Showing density\n");
		break;

	case 'n':
		view_field = ParticleSystem::VM_NORMAL;
		printf("Showing normal\n");
		break;

	case 'o':
		view_field = ParticleSystem::VM_VORTICITY;
		printf("Showing vorticity magnitude\n");
		break;

	case 'X':
		if (camvec.y == 0 && camvec.z == 0)
			camera_pos.x = target_pos.x - camvec.x;
		else {
			camera_pos = target_pos - length(camvec)*x_axis;
		}
		camera_up = z_axis;
		printf("Looking from X\n");
		look();
		break;

	case 'Y':
		if (camvec.x == 0 && camvec.z == 0)
			camera_pos.y = target_pos.y - camvec.y;
		else {
			camera_pos = target_pos + length(camvec)*y_axis;
		}
		camera_up = z_axis;
		printf("Looking from Y\n");
		look();
		break;

	case 'Z':
		if (camvec.x == 0 && camvec.y == 0)
			camera_pos.z = target_pos.z - camvec.z;
		else {
			camera_pos = target_pos + length(camvec)*z_axis;
		}
		camera_up = y_axis;
		printf("Looking from Z\n");
		look();
		break;


	case 'x':
		camera_up = rotate(camvec, cross(camvec, x_axis), M_PI/2);
		printf("x axis up\n");
		look();
		break;

	case 'y':
		camera_up = rotate(camvec, cross(camvec, y_axis), M_PI/2);
		printf("y axis up\n");
		look();
		break;

	case 'z':
		camera_up = rotate(camvec, cross(camvec, z_axis), M_PI/2);
		printf("z axis up\n");
		look();
		break;

	case '+':
		printf("zooming in\n");
		zoom(-0.25);
		break;

	case '-':
		printf("zooming out\n");
		zoom(0.25);
		break;

	case '0':
		printf("resetting display out\n");
		reset_display();
		look();
		break;

	case 'r':
		displayEnabled = !displayEnabled;
		printf("Display %sabled\n", displayEnabled ? "en" : "dis");
		break;

	case 'i':
		timing = M_INTERACTION;
		printf("Title: interaction\n");
		break;

	case 'l':
		timing = M_NEIBSLIST;
		printf("Title: neighbours\n");
		break;

	case 'e':
		timing = M_EULER;
		printf("Title: Euler\n");
		break;

	case 'm':
		timing = M_MEAN;
		printf("Title: mean\n");
		break;

	case 's':
		screenshotNow = true;
		printf("Screenshotting\n");
		break;

	case 't':
		timing = M_NOTIMING;
		printf("Title: none\n");
		break;
	}

	fflush(stdout);
	glutPostRedisplay();
}


void idle(void)
{
	reset_target();
	glutPostRedisplay();
}


void mainMenu(int i)
{
	key((unsigned char) i, 0, 0);
}


void initMenus()
{
	glutCreateMenu(mainMenu);
	glutAddMenuEntry("Toggle view boundary [b]", 'b');
	glutAddMenuEntry("Toggle view pressure [p]", 'p');
	glutAddMenuEntry("Toggle view velocity [v]", 'v');
	glutAddMenuEntry("Toggle view density [d]", 'd');
	glutAddMenuEntry("Toggle animation [ ]", ' ');
	glutAddMenuEntry("Quit (esc)", '\033');
	glutAttachMenu(GLUT_RIGHT_BUTTON);
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
	show_version();

	init(clOptions.problem.c_str());

	// do an initial write
	get_arrays(true);
	do_write();

	if (clOptions.console) {
		console_loop();
	} else {
		glutInit(&argc, argv);
		glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
		glutInitWindowSize(800, 600);
		viewport[0] = 0;
		viewport[1] = 0;
		viewport[2] = 800;
		viewport[3] = 600;
		glutCreateWindow("GPUSPH:  Hit Space Bar to start!");

#ifdef TIMING_LOG
		timing_log = fopen("timing.txt","w");
#endif

		initGL();

		initMenus();

		glutDisplayFunc(display);
		glutReshapeFunc(reshape);
		glutMouseFunc(mouse);
		glutMotionFunc(motion);
		glutKeyboardFunc(key);
		glutIdleFunc(idle);

 		glutMainLoop();
	}

	quit(0);

	return 0;
}


