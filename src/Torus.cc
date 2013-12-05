/*
 * File:   Torus.cc
 * Author: alexis
 *
 * Created on 8 septembre 2011, 18:52
 */

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
	Vector rotdir = -axisdir.cross(v);
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
	m_inertia[0] = m_mass*(5.0/8.0*r*r + 1.0/2.0*m_R*m_R);
	m_inertia[1] = m_inertia[0];
	m_inertia[2] = m_mass*(3.0/4.0*m_r*m_r + m_R*m_R);
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


bool
Torus::IsInside(const Point& p, const double dx) const
{
	Point lp = m_ep.TransposeRot(p - m_center);

	const double Rmin = m_R - m_r - dx;
	const double Rmax = m_R + m_r + dx;
	const double rxy = sqrt(lp(0)*lp(0) + lp(1)*lp(1));
	const double radsq = (rxy - m_R)*(rxy - m_R) + lp(2)*lp(2);
	if (radsq < Rmin*Rmin)
		return false;
	if (radsq > Rmax*Rmax)
		return false;

	return true;
}
