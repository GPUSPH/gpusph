/* 
 * File:   Torus.cc
 * Author: alexis
 * 
 * Created on 8 septembre 2011, 18:52
 */

#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif
#include <cmath>
#include <cstdlib>

#include "Torus.h"


Torus::Torus()
{
	m_R = 0.0;
	m_r = 0.0;
}


Torus::Torus(const Point& center, const Vector& axis, const double R, const double r)
{
	m_center = center;
	Vector axisdir = axis;
	axisdir.normalize();
	m_R = R;
	m_r = r;
	
	Vector v(0, 0, 1);
	const double angle = acos(axisdir*v);
	Vector rotdir = axisdir.cross(v);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);
	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();
}


Torus::Torus(const Point& center, const double R, const double r, const EulerParameters& ep)
{
	m_center = center;
	m_R = R;
	m_r = r;
	
	m_ep = ep;
	m_ep.ComputeRot();
}


Torus::~Torus() 
{
}


double
Torus::Volume(const double dx) const
{
	const double r = m_r + dx/2.0;
	const double volume = 2.0*M_PI*M_PI*r*r*m_R;
	return volume;
}


void
Torus::SetInertia(const double dx)
{
	const double r = m_r + dx/2.0;
	m_inertia[0] = m_mass*(5.0/8.0*m_R*m_R + 1.0/2.0*r*r);
	m_inertia[1] = m_inertia[0];
	m_inertia[0] = m_mass*(3.0/4.0*m_R*m_R + r*r);
}


void
Torus::FillBorder(PointVect& points, const double dx)
{
	const int ntheta = (int) ceil(M_PI*m_r/dx);
	const double dtheta = M_PI/ntheta;
	
	for (int i = 0; i <= ntheta; i++) {
		const double theta = i*dtheta;
		const double z = m_r*cos(theta);
		FillDiskBorder(points, m_ep, m_center, m_R + sqrt(m_r*m_r - z*z), z, dx, 2.0*M_PI*rand()/RAND_MAX);
  	 }
	for (int i = 1; i < ntheta; i++) {
		const double theta = i*dtheta;
		const double z = m_r*cos(theta);
		FillDiskBorder(points, m_ep, m_center, m_R - sqrt(m_r*m_r - z*z), z, dx, 2.0*M_PI*rand()/RAND_MAX);
  	 }
}


int
Torus::Fill(PointVect& points, const double dx, const bool fill)
{
	int nparts = 0;
	const int ntheta = (int) ceil(M_PI*m_r/dx);
	const double dtheta = M_PI/ntheta;
	
	for (int i = 0; i <= ntheta; i++) {
		const double theta = i*dtheta;
		const double z = m_r*cos(theta);
		nparts += FillDisk(points, m_ep, m_center, m_R - sqrt(m_r*m_r - z*z), 
					m_R + sqrt(m_r*m_r - z*z), z, dx, fill);
  	 }

	return nparts;
}

// TODO: FIXME
bool
Torus::IsInside(const Point& p, const double dx) const
{	
	Point lp = m_ep.TransposeRot(p - m_center);
	const double h = m_r + dx;
	bool isinside = false;
	const double z = lp(2);
	if (z > -h && z < h) {
		const double rmin2 = (m_R - sqrt(m_r*m_r - z*z) - dx)*(m_R - sqrt(m_r*m_r - z*z) - dx);
		const double rmax2 = (m_R + sqrt(m_r*m_r - z*z) + dx)*(m_R + sqrt(m_r*m_r - z*z) + dx);
		if (lp(0)*lp(0) + lp(1)*lp(1) < rmax2 && lp(0)*lp(0) + lp(1)*lp(1) > rmin2)
			isinside = true;
	}
	
	return isinside;
}


void
Torus::GLDraw(const EulerParameters& ep, const Point& cg) const
{
	/* The parametric equation of the torus of axis (0, 0, 1),
	   of radius R, and tube radius r is :
		x(u,v) = (R + r cos(v)) cos(u)
		y(u,v) = (R + r cos(v)) sin(u)
		z(u,v) = r sin(v)
	*/
	
	#define TORUS_CIRCLES 9
	#define CIRCLE_LINES 36
	double angle1 = 2.0*M_PI/TORUS_CIRCLES;
	double angle2 = 2.0*M_PI/CIRCLE_LINES;
	for (int i = 0; i <= TORUS_CIRCLES; ++i) {
		const double v = i*angle1;
		const double z = m_r*sin(v);
		glBegin(GL_POLYGON);
		for (int j=0; j < CIRCLE_LINES; ++j) {
			double u = j*angle2;
			Point p = ep.Rot(Point((m_R + m_r*cos(v))*cos(u), (m_R + m_r*cos(v))*sin(u), z));
			p += cg;
			glVertex3f(p(0), p(1), p(2));
		}
		glEnd();
	}
	
	for (int i = 0; i <= TORUS_CIRCLES; i ++) {
		const double u = i*angle1;
		const double cosu = cos(u);
		const double sinu = sin(u);
		glBegin(GL_POLYGON);
		for (int j=0; j < CIRCLE_LINES; ++j) {
			double v = j*angle2;
			Point p = m_ep.Rot(Point((m_R + m_r*cos(v))*cosu, (m_R + m_r*cos(v))*sinu, m_r*sin(v)));
			p += m_center;
			glVertex3f(p(0), p(1), p(2));
		}
		glEnd();
	}
	#undef TORUS_CIRCLES
	#undef CIRCLE_LINES
}


void
Torus::GLDraw(void) const
{
	GLDraw(m_ep, m_center);
}