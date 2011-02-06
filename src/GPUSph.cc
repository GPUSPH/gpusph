//
// File:   particles.cc
// Author: alexis
//
// Created on 28 avril 2008, 02:59
//

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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
bool show_boundary = false;

enum { M_VIEW = 0, M_MOVE};
int view_field = ParticleSystem::VM_NORMAL;
enum { M_INTERACTION = 0, M_NEIBSLIST, M_EULER, M_MEAN, M_NOTIMING};
int timing = M_NEIBSLIST;

ParticleSystem *psystem = 0;

CScreenshot *glscreenshot = 0;

float modelView[16];

// timing
TimingInfo  timingInfo;
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

Problem *problem;

bool screenshotNow = false;

void cleanup(void)
{
	if (psystem)
		delete psystem;
	if (timing_log != NULL)
		fclose(timing_log);
}

void quit(int ret)
{
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
		} else if (!strcmp(arg, "--console")) {
			clOptions.console = true;
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
/*#define CHECK(string) \
	if (!strcasecmp(arg, #string)) \
	problem = new string(clOptions)
#define CHECK3D(string) \
	if (!strcasecmp(arg, #string)) \
		problem = new string##3D(clOptions); \
	else CHECK(string##3D)
#define CHECKTEST(string) \
	if (!strcasecmp(arg, #string)) \
		problem = new string##Test(clOptions); \
	else CHECK(string##Test)
#define CHECKTEST3D(string) \
	if (!strcasecmp(arg, #string)) \
		problem = new string##Test3D(clOptions); \
	else CHECK3D(string##Test)

	CHECK3D(DamBreak);
	else CHECK(OpenChannel);
	else CHECK(TestTopo);
	else CHECK(WaveTank);
	else {
		cerr << "unknown problem " << arg << endl;
		problem_list();
		exit(1);
	}
#undef CHECK
#undef CHECK3D
#undef CHECKTEST
#undef CHECKTEST3D*/

	problem = new PROBLEM(clOptions);

	/* TODO do it this way for all options? */
	if (isfinite(clOptions.tend))
		problem->get_simparams().tend = clOptions.tend;

	worldOrigin = problem->get_worldorigin();
	worldSize = problem->get_worldsize();

	reset_display();

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


	glscreenshot = new CScreenshot(problem->get_dirname());

}


void initGL()
{
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 GL_VERSION_1_5 GL_ARB_multitexture GL_ARB_vertex_buffer_object")) {
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
	float d1 = length(camera_pos - worldOrigin);
	float d2 = length(camera_pos - (worldOrigin+worldSize));
	float d3 = length(camera_pos - worldOrigin+make_float3(0, 0, worldSize.z));
	float d4 = length(camera_pos - worldOrigin+make_float3(0, worldSize.y, 0));
	float d5 = length(camera_pos - worldOrigin+make_float3(worldSize.x, 0, 0));
	float d6 = length(camera_pos - worldOrigin+make_float3(worldSize.x, 0, worldSize.z));
	float d7 = length(camera_pos - worldOrigin+make_float3(0, worldSize.y, worldSize.z));
	float d8 = length(camera_pos - worldOrigin+make_float3(worldSize.x, worldSize.y, 0));

	far_plane = 1.1*max(
			max(max(d1, d2), max(d3, d4)),
			max(max(d5, d6), max(d7, d8)));
	gluPerspective(view_angle, viewport[2] / viewport[3], near_plane, far_plane);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	gluLookAt(camera_pos.x, camera_pos.y, camera_pos.z,
			target_pos.x, target_pos.y, target_pos.z,
			camera_up.x, camera_up.y, camera_up.z);

	if (update)
		update_projection();

	reset_target();
}

void display()
{
	if (!bPause)
	{
		timingInfo = psystem->PredcorrTimeStep(true);
#ifdef TIMING_LOG
		fprintf(timing_log,"%9.4e\t%9.4e\t%9.4e\t%9.4e\t%9.4e\n", timingInfo.t, timingInfo.dt,
				timingInfo.timeInteract, timingInfo.timeEuler, timingInfo.timeNeibsList);
#endif
	}

	// render
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	bool finished = problem->finished(timingInfo.t);
	bool need_display = displayEnabled && problem->need_display(timingInfo.t);
	bool need_write = problem->need_write(timingInfo.t) || finished;
	if (need_display || need_write)
	{
		psystem->getArray(ParticleSystem::POSITION);
		psystem->getArray(ParticleSystem::VELOCITY);
	    psystem->getArray(ParticleSystem::INFO);
		if (need_write) {
			if (problem->m_simparams.vorticity)
				psystem->getArray(ParticleSystem::VORTICITY);
			psystem->writeToFile();
		}

	}

	if (displayEnabled)
	{
		psystem->drawParts(show_boundary, view_field);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		problem->draw_boundary(psystem->getTime());
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

	// Writing to file
	if (finished)
		quit(0);
}

void console_loop(void)
{
	while (true) {
		timingInfo = psystem->PredcorrTimeStep(true);

		bool finished = problem->finished(timingInfo.t);
		bool need_write = problem->need_write(timingInfo.t) || finished;

		if (need_write)
		{
			psystem->getArray(ParticleSystem::POSITION);
			psystem->getArray(ParticleSystem::VELOCITY);
			psystem->getArray(ParticleSystem::INFO);

			if (problem->m_simparams.vorticity)
				psystem->getArray(ParticleSystem::VORTICITY);

			psystem->writeToFile();
			#define ti timingInfo
			printf(	"\nSaving file at nt=%es iterations=%ld dt=%es %u parts.\n"
					"mean %e neibs. in %es, %e neibs/s, max %u neibs\n"
					"mean integration in %es \n",
					ti.t, ti.iterations, ti.dt, ti.numParticles, (double) ti.meanNumInteractions,
					ti.meanTimeInteract, ((double)ti.meanNumInteractions)/ti.meanTimeInteract, ti.maxNeibs,
					ti.meanTimeEuler);
			#undef ti
		}

		if (problem->finished(timingInfo.t))
			break;
	}

	if (problem->finished(timingInfo.t))
		exit(0);
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
		printf("Stepping\n");
		psystem->PredcorrTimeStep(true);
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
	signal(SIGINT, quit);
	signal(SIGUSR1, show_timing);

	parse_options(argc, argv);

	if (clOptions.console) {
		init(clOptions.problem.c_str());
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
		init(clOptions.problem.c_str());

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


