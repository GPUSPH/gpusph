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

// for smart pointers
#include <memory>

#include "chrono_select.opt"
#if USE_CHRONO == 1
#include "chrono/physics/ChBodyEasy.h"
#endif

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
	if (fabs(radius*height) > 1e-8*radius.norm()*height.norm())
		throw std::invalid_argument("Trying to construct a cylinder with non perpendicular radius and axis");
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

	// Point mass is stored in the fourth component of m_center. Store and restore it after the rotation
	const double point_mass = m_center(3);
	m_center = m_origin + m_ep.Rot(0.5*m_h*Vector(0, 0, 1));
	m_center(3) = point_mass;
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
	uint layers = abs(_layers);

	m_origin(3) = m_center(3);

	if (layers*dx > m_r) {
		std::cerr << "WARNING: Cylinder FillIn with " << layers << " layers and " << dx << " stepping > radius " << m_r << " replaced by Fill" << std::endl;
		Fill(points, dx, true);
		return;
	}

	if (2*layers*dx > m_h) {
		std::cerr << "WARNING: Cylinder FillIn with " << layers << " layers and " << dx << " stepping > half-height " << (m_h/2) << " replaced by Fill" << std::endl;
		Fill(points, dx, true);
		return;
	}


	for (uint l = 0; l < layers; l++) {

		const double smaller_r = m_r - l * dx;
		const double smaller_h = m_h - l * 2 * dx;

		const int nz = (int) ceil(smaller_h/dx);
		const double dz = smaller_h/nz;
		for (int i = 0; i <= nz; i++)
			FillDiskBorder(points, m_ep, m_origin, smaller_r, (i + l)*dz, dx, 2.0*M_PI*rand()/RAND_MAX);
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

#if USE_CHRONO == 1
/* Create a cube Chrono body inside a specified Chrono physical system. If
 * collide is true this method also enables collision detection in Chrono.
 * Here we have to specialize this function for the Cylinder because the Chrono cylinder
 * is by default in the Y direction and ours in the Z direction.
 *	\param bodies_physical_system : Chrono physical system
 *	\param dx : particle spacing
 *	\param collide : add collision handling
 */
void
Cylinder::BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide,
	const ::chrono::ChQuaternion<> & orientation_diff)
{
	// Check if the physical system is valid
	if (!bodies_physical_system)
		throw std::runtime_error("Cube::BodyCreate Trying to create a body in an invalid physical system!\n");

	// Creating a new Chrono object
	m_body = std::make_shared< ::chrono::ChBodyEasyCylinder >( m_r + dx/2.0, m_h + dx, m_mass/Volume(dx), collide );
	m_body->SetPos(::chrono::ChVector<>(m_center(0), m_center(1), m_center(2)));
	m_body->SetRot(orientation_diff*m_ep.ToChQuaternion());

	m_body->SetCollide(collide);
	m_body->SetBodyFixed(m_isFixed);
	// mass is automatically set according to density

	// Add the body to the physical system
	bodies_physical_system->AddBody(m_body);
}
#endif
