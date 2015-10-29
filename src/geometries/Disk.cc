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

#include "Disk.h"


Disk::Disk(void)
{
	m_center = Point();
	m_r = 0.0;
}


Disk::Disk(const Point& center, double radius, const Vector& normaldir)
{
	m_center = center;
	m_r = radius;
	Vector axisdir = normaldir;
	axisdir.normalize();

	Vector v(0, 0, 1);
	const double angle = acos(axisdir*v);
	Vector rotdir = axisdir.cross(v);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);
	// equivalent to setEulerParameters(EP(...)), but deprecated contructor anyway
	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();
}


Disk::Disk(const Point& center, double radius, const EulerParameters& ep)
{
	m_center = center;
	m_r = radius;

	setEulerParameters(ep);
}


Disk::Disk(const Point& center, const Vector& radius, const Vector& normaldir)
{
	if (radius*normaldir > 1.e-8*radius.norm()*normaldir.norm()) {
		std::cout << "Trying to construct a disk with non perpendicular radius and normal direction\n";
		exit(1);
	}

	m_center = center;
	m_r = radius.norm();

	Vector axisdir = normaldir;
	axisdir.normalize();

	Vector v(0, 0, 1);
	const double angle = acos(axisdir*v);
	Vector rotdir = axisdir.cross(v);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);
	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();
}


double
Disk::Volume(const double dx) const
{
	const double r = m_r + dx/2.0;
	const double volume = M_PI*r*r*dx;
	return volume;
}


void
Disk::SetInertia(const double dx)
{
	const double r = m_r + dx/2.0;
	const double h = dx;
	m_inertia[0] = m_mass/12.0*(3*r*r + h*h);
	m_inertia[1] = m_inertia[0];
	m_inertia[2] = m_mass/2.0*r*r;
}

void Disk::setEulerParameters(const EulerParameters &ep)
{
	m_ep = ep;
	m_ep.ComputeRot();
}

// TODO: now returning cubic container, should return minimum parallelepiped instead
// (of delta_p thickness, altough automatic world size will add) by using a vector radius
void Disk::getBoundingBox(Point &output_min, Point &output_max)
{
	Point corner_origin = m_center - Vector( -m_r, -m_r, -m_r );
	getBoundingBoxOfCube(output_min, output_max, corner_origin,
		Vector(2*m_r, 0, 0), Vector(0, 2*m_r, 0), Vector(0, 0, -2*m_r) );
}

void Disk::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_center += poff;
}

void
Disk::FillBorder(PointVect& points, const double dx)
{
	FillDiskBorder(points, m_ep, m_center, m_r, 0.0, dx, 0.0);
}


int
Disk::Fill(PointVect& points, const double dx, const bool fill)
{
	return FillDisk(points, m_ep, m_center, m_r, 0.0, dx, 0.0, fill);
}


bool
Disk::IsInside(const Point& p, const double dx) const
{
	Point lp = m_ep.TransposeRot(p - m_center);
	const double r = m_r + dx;
	bool inside = false;
	if (lp(0)*lp(0) + lp(1)*lp(1) < r*r && lp(2) > -dx && lp(2) < dx)
		inside = true;

	return inside;
}
