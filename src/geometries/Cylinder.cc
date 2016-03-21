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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "Cylinder.h"

Cylinder::Cylinder(void)
{
	m_center = Point(0,0,0);
	m_h = 1;
	m_r = 1;
	m_ep = EulerParameters();
}


Cylinder::Cylinder(const Point& origin, const double radius, const Vector& height)
{
	m_origin = origin;
	m_center = m_origin + 0.5*height;
	m_r = radius;
	m_h = height.norm();

	Vector v(0, 0, 1);
	const double angle = acos(height*v/m_h);
	Vector rotdir = -height.cross(v);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);

	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();
}


Cylinder::Cylinder(const Point& origin, const double radius, const double height, const EulerParameters& ep)
{
	m_origin = origin;
	m_h = height;
	m_r = radius;

	setEulerParameters(ep);
}


Cylinder::Cylinder(const Point& origin, const Vector& radius, const Vector& height)
{
	if (fabs(radius*height) > 1e-8*radius.norm()*height.norm()) {
		std::cout << "Trying to construct a cylinder with non perpendicular radius and axis\n";
		exit(1);
	}
	m_origin = origin;
	m_center = m_origin + 0.5*height;
	m_r = radius.norm();
	m_h = height.norm();

	Vector v(0, 0, 1);
	const double angle = acos(height*v/m_h);
	Vector rotdir = height.cross(v);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);

	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();
}


double
Cylinder::Volume(const double dx) const
{
	const double r = m_r + dx/2.0;
	const double h = m_h + dx;
	const double volume = M_PI*r*r*h;
	return volume;
}


void
Cylinder::SetInertia(const double dx)
{
	const double r = m_r + dx/2.0;
	const double h = m_h + dx;
	m_inertia[0] = m_mass/12.0*(3*r*r + h*h);
	m_inertia[1] = m_inertia[0];
	m_inertia[2] = m_mass/2.0*r*r;
}

void Cylinder::setEulerParameters(const EulerParameters &ep)
{
	m_ep = ep;
	m_ep.ComputeRot();

	m_center = m_origin + m_ep.Rot(0.5*m_h*Vector(0, 0, 1));

	dQuaternion q;
	for (int i = 0; i < 4; i++)
		q[i] = m_ep(i);
	dQtoR(q, m_ODERot);
}

void Cylinder::getBoundingBox(Point &output_min, Point &output_max)
{
	Point corner_origin = m_origin - Vector( -m_r, -m_r, 0.0 );
	getBoundingBoxOfCube(output_min, output_max, corner_origin,
		Vector(2*m_r, 0, 0), Vector(0, 2*m_r, 0), Vector(0, 0, m_h) );
}

void Cylinder::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_origin += poff;
	m_center += poff;
}

void
Cylinder::ODEBodyCreate(dWorldID ODEWorld, const double dx, dSpaceID ODESpace)
{
	m_ODEBody = dBodyCreate(ODEWorld);
	dMassSetZero(&m_ODEMass);
	dMassSetCylinderTotal(&m_ODEMass, m_mass, 3, m_r +dx/2.0, m_h + dx);
	dBodySetMass(m_ODEBody, &m_ODEMass);
	dBodySetPosition(m_ODEBody, m_center(0), m_center(1), m_center(2));
	dQuaternion q;
	m_ep.ToODEQuaternion(q);
	dBodySetQuaternion(m_ODEBody, q);
	if (ODESpace)
		ODEGeomCreate(ODESpace, dx);
}


void
Cylinder::ODEGeomCreate(dSpaceID ODESpace, const double dx) {
	m_ODEGeom = dCreateCylinder(ODESpace, m_r, m_h);
	if (m_ODEBody)
		dGeomSetBody(m_ODEGeom, m_ODEBody);
	else {
		dGeomSetPosition(m_ODEGeom, m_center(0), m_center(1), m_center(2));
		dQuaternion q;
		m_ep.ToODEQuaternion(q);
		dGeomSetQuaternion(m_ODEGeom, q);
	}
}


void
Cylinder::FillBorder(PointVect& points, const double dx, const bool bottom, const bool top)
{
	m_origin(3) = m_center(3);
	const int nz = (int) ceil(m_h/dx);
	const double dz = m_h/nz;
	for (int i = 0; i <= nz; i++)
		FillDiskBorder(points, m_ep, m_origin, m_r, i*dz, dx, 2.0*M_PI*rand()/RAND_MAX);
	if (bottom)
		FillDisk(points, m_ep, m_origin, m_r - dx, 0.0, dx, true);
	if (top)
		FillDisk(points, m_ep, m_origin, m_r - dx, nz*dz, dx, true);
}


int
Cylinder::Fill(PointVect& points, const double dx, const bool fill)
{
	m_origin(3) = m_center(3);
	int nparts = 0;
	const int nz = (int) ceil(m_h/dx);
	const double dz = m_h/nz;
	for (int i = 0; i <= nz; i++)
		nparts += FillDisk(points, m_ep, m_origin, m_r, i*dz, dx, fill);

	return nparts;
}

void
Cylinder::FillIn(PointVect& points, const double dx, const int layers)
{
	FillIn(points, dx, layers, true);
}


void
Cylinder::FillIn(PointVect& points, const double dx, const int _layers, const bool fill_tops)
{
	// NOTE - TODO
	// XProblem calls FillIn with negative number of layers to fill rects in the opposite
	// direction as the normal. Cubes and other primitives do not support it. This is a
	// temporary workaround until we decide a common policy for the filling of DYNAMIC
	// boundary layers consistent for any geometry.
	int layers = abs(_layers);

	m_origin(3) = m_center(3);

	for (uint l = 0; l < layers; l++) {

		const double smaller_r = m_r - l * dx;
		const double smaller_h = m_h - l * 2 * dx;

		const int nz = (int) ceil(smaller_h/dx);
		const double dz = smaller_h/nz;
		for (int i = 0; i <= nz; i++)
			FillDiskBorder(points, m_ep, m_origin, smaller_r, i*dz, dx, 2.0*M_PI*rand()/RAND_MAX);
		// fill "bottom"
		if (fill_tops)
			FillDisk(points, m_ep, m_origin, smaller_r - dx, l * dx, dx, true);
		// fill "top"
		if (fill_tops)
			FillDisk(points, m_ep, m_origin, smaller_r - dx, nz*dz + l * dx, dx, true);
	}
	return;
}

bool
Cylinder::IsInside(const Point& p, const double dx) const
{
	Point lp = m_ep.TransposeRot(p - m_origin);
	const double r = m_r + dx;
	const double h = m_h + dx;
	bool inside = false;
	if (lp(0)*lp(0) + lp(1)*lp(1) < r*r && lp(2) > - dx && lp(2) < h)
		inside = true;

	return inside;
}
