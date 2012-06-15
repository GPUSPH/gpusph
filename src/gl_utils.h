/*
 * gl_utils.h
 *
 *  Created on: 14 juin 2012
 *      Author: alexis
 */

#ifndef _GL_UTILS_H
#define _GL_UTILS_H

#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif
#include "Point.h"
#include "Vector.h"


static void
GLSetTransform(const float pos[3], const float R[12])
{
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


static void
GLSetTransform(const double pos[3], const double R[12])
{
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


static void
GLSetTransform(const Point &pt, const float R[12])
{
	float pos[3];
	pos[0] = pt(0);
	pos[1] = pt(1);
	pos[2] = pt(2);
	GLSetTransform(pos, R);
}


static void
GLetTransform(const Point &pt, const double R[12])
{
	double pos[3];
	pos[0] = pt(0);
	pos[1] = pt(1);
	pos[2] = pt(2);
	GLSetTransform(pos, R);
}


static void
GLDrawLine(const float p1[3], const float p2[3])
{
	glBegin(GL_LINES);
	glVertex3f(p1[0], p1[1], p1[2]);
	glVertex3f(p2[0], p2[1], p2[2]);
	glEnd();
}


static void
GLDrawQuad(const Point& p1, const Point& p2, Point& p3, const Point& p4)
{
	glBegin(GL_QUADS);
	glVertex3f((float) p1(0), (float) p1(1), (float) p1(2));
	glVertex3f((float) p2(0), (float) p2(1), (float) p2(2));
	glVertex3f((float) p3(0), (float) p3(1), (float) p3(2));
	glVertex3f((float) p4(0), (float) p4(1), (float) p4(2));
	glEnd();
}



static void
GLDrawCircle(const double r, const double z)
{
	#define CIRCLE_LINES 36
	const double angle = 2.0*M_PI/CIRCLE_LINES;
	glBegin(GL_POLYGON);
	for (int i = 0; i < CIRCLE_LINES; ++i) {
			double u = i*angle;
			glVertex3f(r*cos(u), r*sin(u), z);
		}
	glEnd();
	#undef CIRCLE_LINES
}


static void
GLDrawBox(float lx, float ly, float lz)
{
	lx *= 0.5f;
	ly *= 0.5f;
	lz *= 0.5f;

	// sides
	glBegin(GL_QUADS);
	glVertex3f (-lx,-ly,-lz);
	glVertex3f (-lx,-ly,lz);
	glVertex3f (-lx,ly,lz);
	glVertex3f (-lx,ly,-lz);
	glEnd();
	glBegin(GL_QUADS);
	glVertex3f (lx,-ly,-lz);
	glVertex3f (lx,-ly,lz);
	glVertex3f (lx,ly,lz);
	glVertex3f (lx,ly,-lz);
	glEnd();
	glBegin(GL_QUADS);
	glVertex3f (-lx,-ly,lz);
	glVertex3f (lx,-ly,lz);
	glVertex3f (lx,ly,lz);
	glVertex3f (-lx,ly,lz);
	glEnd();
	glBegin(GL_QUADS);
	glVertex3f (-lx,-ly,-lz);
	glVertex3f (-lx,ly,-lz);
	glVertex3f (lx,ly,-lz);
	glVertex3f (lx,-ly,-lz);
	glEnd();
}


static void
GLDrawUnitSphere(void)
{
	/* The parametric equation of the the sphere centered in (0, 0, 0),
	   of radius R is :
		x(u,v) = R cos(v) cos(u)
		y(u,v) = R cos(v) sin(u)
		z(u,v) = R sin(v)
	*/

	#define SPHERE_CIRCLES 9
	#define CIRCLE_LINES 36
	double angle1 = 2.0*M_PI/SPHERE_CIRCLES;
	double angle2 = 2.0*M_PI/CIRCLE_LINES;
	for (int i = 0; i <= SPHERE_CIRCLES; ++i) {
		const double v = i*angle1;
		const double z = sin(v);
		glBegin(GL_POLYGON);
		for (int j = 0; j < CIRCLE_LINES; ++j) {
			double u = j*angle2;
			glVertex3f(cos(v)*cos(u), cos(v)*sin(u),  z);
		}
		glEnd();
	}

	for (int i = 0; i < SPHERE_CIRCLES/2; i ++) {
		const double u = i*angle1;
		const double cosu = cos(u);
		const double sinu = sin(u);
		glBegin(GL_POLYGON);
		for (int j = 0; j < CIRCLE_LINES; ++j) {
			double v = j*angle2;
			glVertex3f(cos(v)*cos(u), cos(v)*sin(u), sin(v));
		}
		glEnd();
	}
	#undef SPHERE_CIRCLES
	#undef CIRCLE_LINES
}


static void
GLDrawCylinder(const float r, const float h)
{
	#define CIRCLES_NUM 4
	#define LINES_NUM	10
	const double dz = h/(2*CIRCLES_NUM);
	for (int i = -CIRCLES_NUM; i <= CIRCLES_NUM; ++i) {
		GLDrawCircle(r, i*dz);
	}

	const double angle2 = 2.0*M_PI/LINES_NUM;
	for (int i = 0; i < LINES_NUM; i++) {
		double u = i*angle2;
		const float p1[3] = {r*cos(u), r*sin(u), 0.0};
		const float p2[3] = {r*cos(u), r*sin(u), h};
		GLDrawLine(p1, p2);
	}

	#undef CIRCLES_NUM
	#undef LINES_NUM
}
#endif /* GL_UTILS_H_ */
