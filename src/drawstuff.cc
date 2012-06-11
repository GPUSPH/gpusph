/*************************************************************************
 *                                                                       *
 * Open Dynamics Engine, Copyright (C) 2001-2003 Russell L. Smith.       *
 * All rights reserved.  Email: russ@q12.org   Web: www.q12.org          *
 *                                                                       *
 * This library is free software; you can redistribute it and/or         *
 * modify it under the terms of EITHER:                                  *
 *   (1) The GNU Lesser General Public License as published by the Free  *
 *       Software Foundation; either version 2.1 of the License, or (at  *
 *       your option) any later version. The text of the GNU Lesser      *
 *       General Public License is included with this library in the     *
 *       file LICENSE.TXT.                                               *
 *   (2) The BSD-style license that is included with this library in     *
 *       the file LICENSE-BSD.TXT.                                       *
 *                                                                       *
 * This library is distributed in the hope that it will be useful,       *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the files    *
 * LICENSE.TXT and LICENSE-BSD.TXT for more details.                     *
 *                                                                       *
 *************************************************************************/

/*

 simple graphics.

 the following command line flags can be used (typically under unix)
 -notex              Do not use any textures
 -noshadow[s]        Do not draw any shadows
 -pause              Start the simulation paused
 -texturepath <path> Inform an alternative textures path

 TODO
 ----

 manage openGL state changes better

 */

#include <math.h>

#ifdef __APPLE__
#include <OpenGl/gl.h>
#include <OpenGl/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#include "drawstuff.h"

//***************************************************************************
// misc

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

// constants to convert degrees to radians and the reverse
#define RAD_TO_DEG (180.0/M_PI)
#define DEG_TO_RAD (M_PI/180.0)

//***************************************************************************
// misc mathematics stuff

static void normalizeVector3(float v[3]) {
	float len = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	if (len <= 0.0f) {
		v[0] = 1;
		v[1] = 0;
		v[2] = 0;
	} else {
		len = 1.0f / (float) sqrt(len);
		v[0] *= len;
		v[1] *= len;
		v[2] *= len;
	}
}

static void crossProduct3(float res[3], const float a[3], const float b[3]) {
	float res_0 = a[1] * b[2] - a[2] * b[1];
	float res_1 = a[2] * b[0] - a[0] * b[2];
	float res_2 = a[0] * b[1] - a[1] * b[0];
	// Only assign after all the calculations are over to avoid incurring memory aliasing
	res[0] = res_0;
	res[1] = res_1;
	res[2] = res_2;
}

//***************************************************************************
// the current drawing state (for when the user's step function is drawing)

static float color[4] = { 0, 0, 0, 0 }; // current r,g,b,alpha color

//***************************************************************************
// OpenGL utility stuff

static void setCamera(float x, float y, float z, float h, float p, float r) {
	glMatrixMode (GL_MODELVIEW);
	glLoadIdentity();
	glRotatef(90, 0, 0, 1);
	glRotatef(90, 0, 1, 0);
	glRotatef(r, 1, 0, 0);
	glRotatef(p, 0, 1, 0);
	glRotatef(-h, 0, 0, 1);
	glTranslatef(-x, -y, -z);
}

// sets the material color, not the light color

static void setColor(float r, float g, float b, float alpha) {
	GLfloat light_ambient[4], light_diffuse[4], light_specular[4];
	light_ambient[0] = r * 0.3f;
	light_ambient[1] = g * 0.3f;
	light_ambient[2] = b * 0.3f;
	light_ambient[3] = alpha;
	light_diffuse[0] = r * 0.7f;
	light_diffuse[1] = g * 0.7f;
	light_diffuse[2] = b * 0.7f;
	light_diffuse[3] = alpha;
	light_specular[0] = r * 0.2f;
	light_specular[1] = g * 0.2f;
	light_specular[2] = b * 0.2f;
	light_specular[3] = alpha;
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, light_ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, light_diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, light_specular);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 5.0f);
}

static void setTransform(const float pos[3], const float R[12]) {
	GLfloat matrix[16];
	matrix[0] = R[0];
	matrix[1] = R[4];
	matrix[2] = R[8];
	matrix[3] = 0;
	matrix[4] = R[1];
	matrix[5] = R[5];
	matrix[6] = R[9];
	matrix[7] = 0;
	matrix[8] = R[2];
	matrix[9] = R[6];
	matrix[10] = R[10];
	matrix[11] = 0;
	matrix[12] = pos[0];
	matrix[13] = pos[1];
	matrix[14] = pos[2];
	matrix[15] = 1;
	glPushMatrix();
	glMultMatrixf(matrix);
}

static void setTransform(const double pos[3], const double R[12]) {
	GLdouble matrix[16];
	matrix[0] = R[0];
	matrix[1] = R[4];
	matrix[2] = R[8];
	matrix[3] = 0;
	matrix[4] = R[1];
	matrix[5] = R[5];
	matrix[6] = R[9];
	matrix[7] = 0;
	matrix[8] = R[2];
	matrix[9] = R[6];
	matrix[10] = R[10];
	matrix[11] = 0;
	matrix[12] = pos[0];
	matrix[13] = pos[1];
	matrix[14] = pos[2];
	matrix[15] = 1;
	glPushMatrix();
	glMultMatrixd(matrix);
}

static void drawConvex(float *_planes, unsigned int _planecount, float *_points,
		unsigned int _pointcount, unsigned int *_polygons) {
	unsigned int polyindex = 0;
	for (unsigned int i = 0; i < _planecount; ++i) {
		unsigned int pointcount = _polygons[polyindex];
		polyindex++;
		glBegin (GL_POLYGON);
		glNormal3f(_planes[(i * 4) + 0], _planes[(i * 4) + 1],
				_planes[(i * 4) + 2]);
		for (unsigned int j = 0; j < pointcount; ++j) {
			glVertex3f(_points[_polygons[polyindex] * 3],
					_points[(_polygons[polyindex] * 3) + 1],
					_points[(_polygons[polyindex] * 3) + 2]);
			polyindex++;
		}
		glEnd();
	}
}

static void drawConvex(double *_planes, unsigned int _planecount,
		double *_points, unsigned int _pointcount, unsigned int *_polygons) {
	unsigned int polyindex = 0;
	for (unsigned int i = 0; i < _planecount; ++i) {
		unsigned int pointcount = _polygons[polyindex];
		polyindex++;
		glBegin (GL_POLYGON);
		glNormal3d(_planes[(i * 4) + 0], _planes[(i * 4) + 1],
				_planes[(i * 4) + 2]);
		for (unsigned int j = 0; j < pointcount; ++j) {
			glVertex3d(_points[_polygons[polyindex] * 3],
					_points[(_polygons[polyindex] * 3) + 1],
					_points[(_polygons[polyindex] * 3) + 2]);
			polyindex++;
		}
		glEnd();
	}
}

static void drawBox(const float sides[3]) {
	float lx = sides[0] * 0.5f;
	float ly = sides[1] * 0.5f;
	float lz = sides[2] * 0.5f;

	// sides
	glBegin (GL_TRIANGLE_STRIP);
	glNormal3f(-1, 0, 0);
	glVertex3f(-lx, -ly, -lz);
	glVertex3f(-lx, -ly, lz);
	glVertex3f(-lx, ly, -lz);
	glVertex3f(-lx, ly, lz);
	glNormal3f(0, 1, 0);
	glVertex3f(lx, ly, -lz);
	glVertex3f(lx, ly, lz);
	glNormal3f(1, 0, 0);
	glVertex3f(lx, -ly, -lz);
	glVertex3f(lx, -ly, lz);
	glNormal3f(0, -1, 0);
	glVertex3f(-lx, -ly, -lz);
	glVertex3f(-lx, -ly, lz);
	glEnd();

	// top face
	glBegin (GL_TRIANGLE_FAN);
	glNormal3f(0, 0, 1);
	glVertex3f(-lx, -ly, lz);
	glVertex3f(lx, -ly, lz);
	glVertex3f(lx, ly, lz);
	glVertex3f(-lx, ly, lz);
	glEnd();

	// bottom face
	glBegin(GL_TRIANGLE_FAN);
	glNormal3f(0, 0, -1);
	glVertex3f(-lx, -ly, -lz);
	glVertex3f(-lx, ly, -lz);
	glVertex3f(lx, ly, -lz);
	glVertex3f(lx, -ly, -lz);
	glEnd();
}

// This is recursively subdivides a triangular area (vertices p1,p2,p3) into
// smaller triangles, and then draws the triangles. All triangle vertices are
// normalized to a distance of 1.0 from the origin (p1,p2,p3 are assumed
// to be already normalized). Note this is not super-fast because it draws
// triangles rather than triangle strips.

static void drawPatch(float p1[3], float p2[3], float p3[3], int level) {
	int i;
	if (level > 0) {
		float q1[3], q2[3], q3[3]; // sub-vertices
		for (i = 0; i < 3; i++) {
			q1[i] = 0.5f * (p1[i] + p2[i]);
			q2[i] = 0.5f * (p2[i] + p3[i]);
			q3[i] = 0.5f * (p3[i] + p1[i]);
		}
		float length1 = (float) (1.0
				/ sqrt(q1[0] * q1[0] + q1[1] * q1[1] + q1[2] * q1[2]));
		float length2 = (float) (1.0
				/ sqrt(q2[0] * q2[0] + q2[1] * q2[1] + q2[2] * q2[2]));
		float length3 = (float) (1.0
				/ sqrt(q3[0] * q3[0] + q3[1] * q3[1] + q3[2] * q3[2]));
		for (i = 0; i < 3; i++) {
			q1[i] *= length1;
			q2[i] *= length2;
			q3[i] *= length3;
		}
		drawPatch(p1, q1, q3, level - 1);
		drawPatch(q1, p2, q2, level - 1);
		drawPatch(q1, q2, q3, level - 1);
		drawPatch(q3, q2, p3, level - 1);
	} else {
		glNormal3f(p1[0], p1[1], p1[2]);
		glVertex3f(p1[0], p1[1], p1[2]);
		glNormal3f(p2[0], p2[1], p2[2]);
		glVertex3f(p2[0], p2[1], p2[2]);
		glNormal3f(p3[0], p3[1], p3[2]);
		glVertex3f(p3[0], p3[1], p3[2]);
	}
}

// draw a sphere of radius 1

static int sphere_quality = 1;

static void drawSphere() {
	// icosahedron data for an icosahedron of radius 1.0
# define ICX 0.525731112119133606f
# define ICZ 0.850650808352039932f
	static GLfloat idata[12][3] = { { -ICX, 0, ICZ }, { ICX, 0, ICZ }, { -ICX,
			0, -ICZ }, { ICX, 0, -ICZ }, { 0, ICZ, ICX }, { 0, ICZ, -ICX }, { 0,
			-ICZ, ICX }, { 0, -ICZ, -ICX }, { ICZ, ICX, 0 }, { -ICZ, ICX, 0 }, {
			ICZ, -ICX, 0 }, { -ICZ, -ICX, 0 } };

	static int index[20][3] = { { 0, 4, 1 }, { 0, 9, 4 }, { 9, 5, 4 },
			{ 4, 5, 8 }, { 4, 8, 1 }, { 8, 10, 1 }, { 8, 3, 10 }, { 5, 3, 8 }, {
					5, 2, 3 }, { 2, 7, 3 }, { 7, 10, 3 }, { 7, 6, 10 }, { 7, 11,
					6 }, { 11, 0, 6 }, { 0, 1, 6 }, { 6, 1, 10 }, { 9, 0, 11 },
			{ 9, 11, 2 }, { 9, 2, 5 }, { 7, 2, 11 }, };

	static GLuint listnum = 0;
	if (listnum == 0) {
		listnum = glGenLists(1);
		glNewList(listnum, GL_COMPILE);
		glBegin (GL_TRIANGLES);
		for (int i = 0; i < 20; i++) {
			drawPatch(&idata[index[i][2]][0], &idata[index[i][1]][0],
					&idata[index[i][0]][0], sphere_quality);
		}
		glEnd();
		glEndList();
	}
	glCallList(listnum);
}

static void drawTriangle(const float *v0, const float *v1, const float *v2,
		int solid) {
	float u[3], v[3], normal[3];
	u[0] = v1[0] - v0[0];
	u[1] = v1[1] - v0[1];
	u[2] = v1[2] - v0[2];
	v[0] = v2[0] - v0[0];
	v[1] = v2[1] - v0[1];
	v[2] = v2[2] - v0[2];
	crossProduct3(normal, u, v);
	normalizeVector3(normal);

	glBegin(solid ? GL_TRIANGLES : GL_LINE_STRIP);
	glNormal3fv(normal);
	glVertex3fv(v0);
	glVertex3fv(v1);
	glVertex3fv(v2);
	glEnd();
}

static void drawTriangle(const double *v0, const double *v1, const double *v2,
		int solid) {
	float u[3], v[3], normal[3];
	u[0] = float(v1[0] - v0[0]);
	u[1] = float(v1[1] - v0[1]);
	u[2] = float(v1[2] - v0[2]);
	v[0] = float(v2[0] - v0[0]);
	v[1] = float(v2[1] - v0[1]);
	v[2] = float(v2[2] - v0[2]);
	crossProduct3(normal, u, v);
	normalizeVector3(normal);

	glBegin(solid ? GL_TRIANGLES : GL_LINE_STRIP);
	glNormal3fv(normal);
	glVertex3dv(v0);
	glVertex3dv(v1);
	glVertex3dv(v2);
	glEnd();
}

// draw a capped cylinder of length l and radius r, aligned along the x axis

static int capped_cylinder_quality = 3;

static void drawCapsule(float l, float r) {
	int i, j;
	float tmp, nx, ny, nz, start_nx, start_ny, a, ca, sa;
	// number of sides to the cylinder (divisible by 4):
	const int n = capped_cylinder_quality * 4;

	l *= 0.5;
	a = float(M_PI * 2.0) / float(n);
	sa = (float) sin(a);
	ca = (float) cos(a);

	// draw cylinder body
	ny = 1;
	nz = 0; // normal vector = (0,ny,nz)
	glBegin (GL_TRIANGLE_STRIP);
	for (i = 0; i <= n; i++) {
		glNormal3d(ny, nz, 0);
		glVertex3d(ny * r, nz * r, l);
		glNormal3d(ny, nz, 0);
		glVertex3d(ny * r, nz * r, -l);
		// rotate ny,nz
		tmp = ca * ny - sa * nz;
		nz = sa * ny + ca * nz;
		ny = tmp;
	}
	glEnd();

	// draw first cylinder cap
	start_nx = 0;
	start_ny = 1;
	for (j = 0; j < (n / 4); j++) {
		// get start_n2 = rotated start_n
		float start_nx2 = ca * start_nx + sa * start_ny;
		float start_ny2 = -sa * start_nx + ca * start_ny;
		// get n=start_n and n2=start_n2
		nx = start_nx;
		ny = start_ny;
		nz = 0;
		float nx2 = start_nx2, ny2 = start_ny2, nz2 = 0;
		glBegin(GL_TRIANGLE_STRIP);
		for (i = 0; i <= n; i++) {
			glNormal3d(ny2, nz2, nx2);
			glVertex3d(ny2 * r, nz2 * r, l + nx2 * r);
			glNormal3d(ny, nz, nx);
			glVertex3d(ny * r, nz * r, l + nx * r);
			// rotate n,n2
			tmp = ca * ny - sa * nz;
			nz = sa * ny + ca * nz;
			ny = tmp;
			tmp = ca * ny2 - sa * nz2;
			nz2 = sa * ny2 + ca * nz2;
			ny2 = tmp;
		}
		glEnd();
		start_nx = start_nx2;
		start_ny = start_ny2;
	}

	// draw second cylinder cap
	start_nx = 0;
	start_ny = 1;
	for (j = 0; j < (n / 4); j++) {
		// get start_n2 = rotated start_n
		float start_nx2 = ca * start_nx - sa * start_ny;
		float start_ny2 = sa * start_nx + ca * start_ny;
		// get n=start_n and n2=start_n2
		nx = start_nx;
		ny = start_ny;
		nz = 0;
		float nx2 = start_nx2, ny2 = start_ny2, nz2 = 0;
		glBegin(GL_TRIANGLE_STRIP);
		for (i = 0; i <= n; i++) {
			glNormal3d(ny, nz, nx);
			glVertex3d(ny * r, nz * r, -l + nx * r);
			glNormal3d(ny2, nz2, nx2);
			glVertex3d(ny2 * r, nz2 * r, -l + nx2 * r);
			// rotate n,n2
			tmp = ca * ny - sa * nz;
			nz = sa * ny + ca * nz;
			ny = tmp;
			tmp = ca * ny2 - sa * nz2;
			nz2 = sa * ny2 + ca * nz2;
			ny2 = tmp;
		}
		glEnd();
		start_nx = start_nx2;
		start_ny = start_ny2;
	}
}

// draw a cylinder of length l and radius r, aligned along the z axis

static void drawCylinder(float l, float r, float zoffset) {
	int i;
	float tmp, ny, nz, a, ca, sa;
	const int n = 24; // number of sides to the cylinder (divisible by 4)

	l *= 0.5;
	a = float(M_PI * 2.0) / float(n);
	sa = (float) sin(a);
	ca = (float) cos(a);

	// draw cylinder body
	ny = 1;
	nz = 0; // normal vector = (0,ny,nz)
	glBegin (GL_TRIANGLE_STRIP);
	for (i = 0; i <= n; i++) {
		glNormal3d(ny, nz, 0);
		glVertex3d(ny * r, nz * r, l + zoffset);
		glNormal3d(ny, nz, 0);
		glVertex3d(ny * r, nz * r, -l + zoffset);
		// rotate ny,nz
		tmp = ca * ny - sa * nz;
		nz = sa * ny + ca * nz;
		ny = tmp;
	}
	glEnd();

	// draw top cap
	glShadeModel (GL_FLAT);
	ny = 1;
	nz = 0; // normal vector = (0,ny,nz)
	glBegin (GL_TRIANGLE_FAN);
	glNormal3d(0, 0, 1);
	glVertex3d(0, 0, l + zoffset);
	for (i = 0; i <= n; i++) {
		if (i == 1 || i == n / 2 + 1)
			setColor(color[0] * 0.75f, color[1] * 0.75f, color[2] * 0.75f,
					color[3]);
		glNormal3d(0, 0, 1);
		glVertex3d(ny * r, nz * r, l + zoffset);
		if (i == 1 || i == n / 2 + 1)
			setColor(color[0], color[1], color[2], color[3]);

		// rotate ny,nz
		tmp = ca * ny - sa * nz;
		nz = sa * ny + ca * nz;
		ny = tmp;
	}
	glEnd();

	// draw bottom cap
	ny = 1;
	nz = 0; // normal vector = (0,ny,nz)
	glBegin(GL_TRIANGLE_FAN);
	glNormal3d(0, 0, -1);
	glVertex3d(0, 0, -l + zoffset);
	for (i = 0; i <= n; i++) {
		if (i == 1 || i == n / 2 + 1)
			setColor(color[0] * 0.75f, color[1] * 0.75f, color[2] * 0.75f,
					color[3]);
		glNormal3d(0, 0, -1);
		glVertex3d(ny * r, nz * r, -l + zoffset);
		if (i == 1 || i == n / 2 + 1)
			setColor(color[0], color[1], color[2], color[3]);

		// rotate ny,nz
		tmp = ca * ny + sa * nz;
		nz = -sa * ny + ca * nz;
		ny = tmp;
	}
	glEnd();
}

//***************************************************************************
// motion model

// current camera position and orientation
static float view_xyz[3]; // position x,y,z
static float view_hpr[3]; // heading, pitch, roll (degrees)

// initialize the above variables

static void initMotionModel() {
	view_xyz[0] = 2;
	view_xyz[1] = 0;
	view_xyz[2] = 1;
	view_hpr[0] = 180;
	view_hpr[1] = 0;
	view_hpr[2] = 0;
}

static void wrapCameraAngles() {
	for (int i = 0; i < 3; i++) {
		while (view_hpr[i] > 180)
			view_hpr[i] -= 360;
		while (view_hpr[i] < -180)
			view_hpr[i] += 360;
	}
}

// call this to update the current camera position. the bits in `mode' say
// if the left (1), middle (2) or right (4) mouse button is pressed, and
// (deltax,deltay) is the amount by which the mouse pointer has moved.

void dsMotion(int mode, int deltax, int deltay) {
	float side = 0.01f * float(deltax);
	float fwd = (mode == 4) ? (0.01f * float(deltay)) : 0.0f;
	float s = (float) sin(view_hpr[0] * DEG_TO_RAD);
	float c = (float) cos(view_hpr[0] * DEG_TO_RAD);

	if (mode == 1) {
		view_hpr[0] += float(deltax) * 0.5f;
		view_hpr[1] += float(deltay) * 0.5f;
	} else {
		view_xyz[0] += -s * side + c * fwd;
		view_xyz[1] += c * side + s * fwd;
		if (mode == 2 || mode == 5)
			view_xyz[2] += 0.01f * float(deltay);
	}
	wrapCameraAngles();
}

//***************************************************************************
// drawing loop stuff

static void drawPyramidGrid() {
	// setup stuff
	glEnable (GL_LIGHTING);
	glDisable (GL_TEXTURE_2D);
	glShadeModel (GL_FLAT);
	glEnable (GL_DEPTH_TEST);
	glDepthFunc (GL_LESS);

	// draw the pyramid grid
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			glPushMatrix();
			glTranslatef((float) i, (float) j, (float) 0);
			if (i == 1 && j == 0)
				setColor(1, 0, 0, 1);
			else if (i == 0 && j == 1)
				setColor(0, 0, 1, 1);
			else
				setColor(1, 1, 0, 1);
			const float k = 0.03f;
			glBegin (GL_TRIANGLE_FAN);
			glNormal3f(0, -1, 1);
			glVertex3f(0, 0, k);
			glVertex3f(-k, -k, 0);
			glVertex3f(k, -k, 0);
			glNormal3f(1, 0, 1);
			glVertex3f(k, k, 0);
			glNormal3f(0, 1, 1);
			glVertex3f(-k, k, 0);
			glNormal3f(-1, 0, 1);
			glVertex3f(-k, -k, 0);
			glEnd();
			glPopMatrix();
		}
	}
}

void dsSetViewpoint(float xyz[3], float hpr[3]) {
	if (xyz) {
		view_xyz[0] = xyz[0];
		view_xyz[1] = xyz[1];
		view_xyz[2] = xyz[2];
	}
	if (hpr) {
		view_hpr[0] = hpr[0];
		view_hpr[1] = hpr[1];
		view_hpr[2] = hpr[2];
		wrapCameraAngles();
	}
}

void dsGetViewpoint(float xyz[3], float hpr[3]) {
	if (xyz) {
		xyz[0] = view_xyz[0];
		xyz[1] = view_xyz[1];
		xyz[2] = view_xyz[2];
	}
	if (hpr) {
		hpr[0] = view_hpr[0];
		hpr[1] = view_hpr[1];
		hpr[2] = view_hpr[2];
	}
}

void dsSetColor(float red, float green, float blue) {
	color[0] = red;
	color[1] = green;
	color[2] = blue;
	color[3] = 1;
}

void dsSetColorAlpha(float red, float green, float blue, float alpha) {
	color[0] = red;
	color[1] = green;
	color[2] = blue;
	color[3] = alpha;
}

void dsDrawBox(const float pos[3], const float R[12], const float sides[3]) {
	glShadeModel (GL_FLAT);
	setTransform(pos, R);
	drawBox(sides);
	glPopMatrix();
}

void dsDrawConvex(const float pos[3], const float R[12], float *_planes,
		unsigned int _planecount, float *_points, unsigned int _pointcount,
		unsigned int *_polygons) {
	glShadeModel (GL_FLAT);
	setTransform(pos, R);
	drawConvex(_planes, _planecount, _points, _pointcount, _polygons);
	glPopMatrix();
}

void dsDrawSphere(const float pos[3], const float R[12], float radius) {
	glEnable (GL_NORMALIZE);
	glShadeModel (GL_SMOOTH);
	setTransform(pos, R);
	glScaled(radius, radius, radius);
	drawSphere();
	glPopMatrix();
	glDisable(GL_NORMALIZE);
}

void dsDrawTriangle(const float pos[3], const float R[12], const float *v0,
		const float *v1, const float *v2, int solid) {
	glShadeModel (GL_FLAT);
	setTransform(pos, R);
	drawTriangle(v0, v1, v2, solid);
	glPopMatrix();
}

void dsDrawCylinder(const float pos[3], const float R[12], float length,
		float radius) {
	glShadeModel (GL_SMOOTH);
	setTransform(pos, R);
	drawCylinder(length, radius, 0);
	glPopMatrix();
}

void dsDrawCapsule (const float pos[3], const float R[12],
		float length, float radius)
{
	glShadeModel (GL_SMOOTH);
	setTransform (pos,R);
	drawCapsule (length,radius);
	glPopMatrix();
}

void dsDrawLine(const float pos1[3], const float pos2[3]) {
	glColor3f(color[0], color[1], color[2]);
	glDisable (GL_LIGHTING);
	glLineWidth(2);
	glShadeModel (GL_FLAT);
	glBegin (GL_LINES);
	glVertex3f(pos1[0], pos1[1], pos1[2]);
	glVertex3f(pos2[0], pos2[1], pos2[2]);
	glEnd();
}

void dsDrawBox(const double pos[3], const double R[12], const double sides[3]) {
	int i;
	float pos2[3], R2[12], fsides[3];
	for (i = 0; i < 3; i++)
		pos2[i] = (float) pos[i];
	for (i = 0; i < 12; i++)
		R2[i] = (float) R[i];
	for (i = 0; i < 3; i++)
		fsides[i] = (float) sides[i];
	dsDrawBox(pos2, R2, fsides);
}

void dsDrawConvex (const double pos[3], const double R[12],
		double *_planes,unsigned int _planecount,
		double *_points,
		unsigned int _pointcount,
		unsigned int *_polygons)
{
	glShadeModel (GL_FLAT);
	setTransform (pos,R);
	drawConvex(_planes,_planecount,_points,_pointcount,_polygons);
	glPopMatrix();
}

void dsDrawSphere(const double pos[3], const double R[12], float radius) {
	int i;
	float pos2[3], R2[12];
	for (i = 0; i < 3; i++)
		pos2[i] = (float) pos[i];
	for (i = 0; i < 12; i++)
		R2[i] = (float) R[i];
	dsDrawSphere(pos2, R2, radius);
}

void dsDrawTriangle(const double pos[3], const double R[12], const double *v0,
		const double *v1, const double *v2, int solid) {
	int i;
	float pos2[3], R2[12];
	for (i = 0; i < 3; i++)
		pos2[i] = (float) pos[i];
	for (i = 0; i < 12; i++)
		R2[i] = (float) R[i];

	glShadeModel (GL_FLAT);
	setTransform(pos2, R2);
	drawTriangle(v0, v1, v2, solid);
	glPopMatrix();
}

void dsDrawCylinder(const double pos[3], const double R[12], float length,
		float radius) {
	int i;
	float pos2[3], R2[12];
	for (i = 0; i < 3; i++)
		pos2[i] = (float) pos[i];
	for (i = 0; i < 12; i++)
		R2[i] = (float) R[i];
	dsDrawCylinder(pos2, R2, length, radius);
}

void dsDrawCapsule(const double pos[3], const double R[12], float length,
		float radius) {
	int i;
	float pos2[3], R2[12];
	for (i = 0; i < 3; i++)
		pos2[i] = (float) pos[i];
	for (i = 0; i < 12; i++)
		R2[i] = (float) R[i];
	dsDrawCapsule(pos2, R2, length, radius);
}

void dsDrawLine(const double _pos1[3], const double _pos2[3]) {
	int i;
	float pos1[3], pos2[3];
	for (i = 0; i < 3; i++)
		pos1[i] = (float) _pos1[i];
	for (i = 0; i < 3; i++)
		pos2[i] = (float) _pos2[i];
	dsDrawLine(pos1, pos2);
}

void dsSetSphereQuality(int n) {
	sphere_quality = n;
}

void dsSetCapsuleQuality(int n) {
	capped_cylinder_quality = n;
}

void dsSetDrawMode(int mode) {
	switch (mode) {
	case DS_POLYFILL:
		glPolygonMode(GL_FRONT, GL_FILL);
		break;
	case DS_WIREFRAME:
		glPolygonMode(GL_FRONT, GL_LINE);
		break;
	}
}
