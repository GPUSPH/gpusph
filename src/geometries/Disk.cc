/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "Disk.h"


Disk::Disk(void)
{
	m_center = Point();
	m_r = 0.0;
}

//! Construct a disk of the given radius, normal to the z axis
Disk::Disk(const Point& center, double radius) : Disk(center, radius, Vector(0, 0, 1))
{ }

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
	const double dz = Object::world_dimensions == 3 ? dx : 1;
	const double volume = M_PI*r*r*dz;
	return volume;
}


void
Disk::SetInertia(const double dx)
{
	const double r = m_r + dx/2.0;
	const double h = dx;
	if (Object::world_dimensions == 3) {
		m_inertia[0] = m_mass/12.0*(3*r*r + h*h);
		m_inertia[1] = m_inertia[0];
		m_inertia[2] = m_mass/2.0*r*r;
	} else {
		throw std::runtime_error(std::string(__func__) + " not implemented in " + std::to_string(Object::world_dimensions) + " dimensions");
	}
}

void
Disk::FillIn(PointVect &points, const double dx, const int layers)
{
	switch (Object::world_dimensions) {
	case 2: FillIn2D(points, dx, layers); break;
	case 3: FillIn3D(points, dx, layers); break;
	default: throw std::runtime_error("can't FillIn a Disk in " + std::to_string(Object::world_dimensions) + " dimensions");
	}
}

/// Fill the disk boundary with layers of particles,
/// towards the inside (+) or outside (-) depending on the layers sign.
void
Disk::FillIn2D(PointVect &points, const double dx, const int layers)
{
	int _layers = abs(layers);

	// shift towards the inside
	const double signed_dx = (layers > 0 ? -dx : dx);

	// First layer: on the boundary
	FillBorder(points, dx);
	// NOTE: pre-decrementing causes (_layers-1) layers to be filled. This
	// is correct since the first layer was already filled
	while (--_layers > 0) {
		Disk layer(m_center, m_r + _layers*signed_dx);
		layer.SetPartMass(m_center(3));
		layer.FillBorder(points, dx);
	}
}


/// Fill a disk with layers of particles, from the surface
/// to the direction of the normal vector. Use a negative
/// value of layers to FillIn the opposite direction
void
Disk::FillIn3D(PointVect &points, const double dx, const int layers)
{
	throw std::runtime_error("Disk::FillIn not implemented in 3D");
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
	const double dz = Object::world_dimensions == 3 ? m_r : 0;
	Point corner_origin = m_center - Vector( m_r, m_r, dz );
	getBoundingBoxOfCube(output_min, output_max, corner_origin,
		Vector(2*m_r, 0, 0), Vector(0, 2*m_r, 0), Vector(0, 0, 2*dz) );
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
	return FillDisk(points, m_ep, m_center, 0.0, m_r, 0.0, dx, fill);
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
