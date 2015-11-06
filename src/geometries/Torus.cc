/*
 * File:   Torus.cc
 * Author: alexis
 *
 * Created on 8 septembre 2011-2013, 18:52
 */

#include <cmath>
#include <cstdlib>

#include "Torus.h"


Torus::Torus()
{
	m_R = 0.0;
	m_r = 0.0;
	m_ep = EulerParameters();
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

	setEulerParameters(ep);
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

void Torus::setEulerParameters(const EulerParameters &ep)
{
	m_ep = ep;
	m_ep.ComputeRot();

	dQuaternion q;
	for (int i = 0; i < 4; i++)
		q[i] = m_ep(i);

	dQtoR(q, m_ODERot);
}

// TODO: now returning cubic container, should return minimum parallelepiped instead
// by taking into account the EulerParameters
void Torus::getBoundingBox(Point &output_min, Point &output_max)
{
	Point corner_origin = m_center + Point(-m_R, -m_R, -m_R);
	getBoundingBoxOfCube(output_min, output_max, corner_origin,
		Vector(m_R, 0, 0), Vector(0, m_R, 0), Vector(0, 0, m_R));
}

void Torus::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_center += poff;
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

void
Torus::FillIn(PointVect& points, const double dx, const int _layers)
{
	// NOTE - TODO
	// XProblem calls FillIn with negative number of layers to fill rects in the opposite
	// direction as the normal. Cubes and other primitives do not support it. This is a
	// temporary workaround until we decide a common policy for the filling of DYNAMIC
	// boundary layers consistent for any geometry.
	int layers = abs(_layers);

	Torus inner = Torus(m_center, m_R, m_r - ((double)layers + 0.5)*dx, m_ep);
	PointVect inpoints;

	Fill(inpoints, dx, true);
	inner.Unfill(inpoints, 0);
	points.insert(points.end(), inpoints.begin(), inpoints.end());
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

	const double r = m_r + dx;
	double temp = m_R - sqrt(lp(0)*lp(0) + lp(1)*lp(1));
	temp *= temp;
	temp += lp(2)*lp(2) - r*r;
	if (temp < 0)
		return true;

	return false;
}


void
Torus::ODEBodyCreate(dWorldID ODEWorld, const double dx, dSpaceID ODESpace)
{
	m_ODEBody = dBodyCreate(ODEWorld);
	dMassSetZero(&m_ODEMass);
	SetInertia(dx);
	dMassSetParameters (&m_ODEMass, m_mass, 0.0, 0.0, 0.0,
		m_inertia[0], m_inertia[1], m_inertia[2], 0.0, 0.0, 0.0);
	dBodySetMass(m_ODEBody, &m_ODEMass);
	dBodySetPosition(m_ODEBody, m_center(0), m_center(1), m_center(2));
	dQuaternion q;
	m_ep.ToODEQuaternion(q);
	dBodySetQuaternion(m_ODEBody, q);
}
