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
#include <cstdlib>

#include "Sphere.h"


Sphere::Sphere(void)
{
	m_center = Point(0,0,0);
	m_r = 1.0;
}


Sphere::Sphere(const Point& center, const double radius)
{
	m_center = center;
	m_r = radius;
	const double ep[4] = {1.0, 0.0, 0.0, 0.0};
	m_ep = EulerParameters(ep);
	m_ep.ComputeRot();
}


double
Sphere::Volume(const double dx) const
{
	const double r = m_r + dx/2.0;
	const double volume = 4.0/3.0*M_PI*r*r*r;
	return volume;
}


void
Sphere::SetInertia(const double dx)
{
	const double r = m_r + dx/2.0;
	m_inertia[0] = 2.0*m_mass/5.0*r*r;
	m_inertia[1] = m_inertia[0];
	m_inertia[2] = m_inertia[0];
}


void
Sphere::ODEBodyCreate(dWorldID ODEWorld, const double dx, dSpaceID ODESpace)
{
	m_ODEBody = dBodyCreate(ODEWorld);
	dMassSetZero(&m_ODEMass);
	dMassSetSphereTotal(&m_ODEMass, m_mass, m_r + dx/2.0);
	dBodySetMass(m_ODEBody, &m_ODEMass);
	dBodySetPosition(m_ODEBody, m_center(0), m_center(1), m_center(2));
	if (ODESpace)
		ODEGeomCreate(ODESpace, dx);
}


void
Sphere::ODEGeomCreate(dSpaceID ODESpace, const double dx) {
	m_ODEGeom = dCreateSphere(ODESpace, m_r + dx/2.0);
	if (m_ODEBody)
		dGeomSetBody(m_ODEGeom, m_ODEBody);
	else
		dGeomSetPosition(m_ODEGeom,  m_center(0), m_center(1), m_center(2));
}


void
Sphere::FillBorder(PointVect& points, const double dx)
{
	const double angle = dx/m_r;
	const int nc = (int) ceil(M_PI/angle); //number of layers
	const double dtheta = M_PI/nc;

	for (int i = - nc; i <= nc; ++i) {
		FillDiskBorder(points, m_ep, m_center, m_r*sin(i*dtheta), m_r*cos(i*dtheta), dx, 2.0*M_PI*rand()/RAND_MAX);
	}
}


void
Sphere::FillIn(PointVect& points, const double dx, const int layers)
{
	for (int l = 0; l < layers; l++) {
		const double r = m_r - l*dx;
		double angle = dx/r;
		const int nc = (int) ceil(M_PI/angle);
		const double dtheta = M_PI/nc;

		for (int i = - nc; i <= nc; ++i) {
			FillDiskBorder(points, m_ep, m_center, r*sin(i*dtheta), r*cos(i*dtheta), dx, 2.0*M_PI*rand()/RAND_MAX);
		}
	}
	return;
}


int
Sphere::Fill(PointVect& points, const double dx, const bool fill)
{
	int nparts = 0;
	const double angle = dx/m_r;
	const int nc = (int) ceil(M_PI/angle); //number of layers
	const double dtheta = M_PI/nc;

	for (int i = - nc; i <= nc; ++i) {
		nparts += FillDisk(points, m_ep, m_center, m_r*sin(i*dtheta), m_r*cos(i*dtheta), dx, fill);
	}

	return nparts;
}


bool
Sphere::IsInside(const Point& p, const double dx) const
{
	Point lp = p - m_center;
	const double r = m_r + dx;
	bool inside = false;
	if (lp(0)*lp(0) + lp(1)*lp(1) + lp(2)*lp(2) < r*r)
		inside = true;

	return inside;
}
