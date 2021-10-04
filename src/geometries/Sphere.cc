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

#include <cstdlib>

// for smart pointers
#include <memory>

#include "Sphere.h"

#if USE_CHRONO == 1
#include "chrono/physics/ChBodyEasy.h"
#include "chrono/physics/ChSystem.h"
#endif
#include "EulerParametersQuaternion.h"


Sphere::Sphere(void)
{
	m_center = Point(0,0,0);
	m_r = 1.0;
	m_ep = EulerParameters();
}


Sphere::Sphere(const Point& center, const double radius)
{
	m_center = center;
	m_r = radius;
	m_ep = EulerParameters();
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

void Sphere::setEulerParameters(const EulerParameters &ep)
{
	m_ep = EulerParameters(ep);
	m_ep.ComputeRot();
}

void Sphere::getBoundingBox(Point &output_min, Point &output_max)
{
	Point corner_origin = m_center + Point(-m_r, -m_r, -m_r);
	getBoundingBoxOfCube(output_min, output_max, corner_origin,
		Vector(m_r, 0, 0), Vector(0, m_r, 0), Vector(0, 0, m_r));
}

void Sphere::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_center += poff;
}


void
Sphere::FillBorder(PointVect& points, const double dx)
{
	FillBorder(points, dx, m_r);
}

int
Sphere::FillBorder(PointVect& points, const double dx, const double r, const bool fill)
{
	int nparts = 0;
	const double angle = dx/r;
	const int nc = (int) ceil(M_PI/angle); //number of layers
	const double dtheta = M_PI/nc;

	for (int i = 0; i <= nc; ++i) {
		nparts += FillDiskBorder(points, m_ep, m_center, r*sin(i*dtheta), r*cos(i*dtheta), dx, 2.0*M_PI*rand() / RAND_MAX);
	}
	return nparts;
}


void
Sphere::FillIn(PointVect& points, const double dx, const int _layers)
{
	// NOTE - TODO
	// XProblem calls FillIn with negative number of layers to fill rects in the opposite
	// direction as the normal. Cubes and other primitives do not support it. This is a
	// temporary workaround until we decide a common policy for the filling of DYNAMIC
	// boundary layers consistent for any geometry.
	int layers = abs(_layers);

	for (int l = 0; l < layers; l++) {
		const double r = m_r - l*dx;
		double angle = dx/r;
		const int nc = (int) ceil(M_PI/angle);
		const double dtheta = M_PI/nc;

		for (int i = 0; i <= nc; ++i) {
			FillDiskBorder(points, m_ep, m_center, r*sin(i*dtheta), r*cos(i*dtheta), dx, 2.0*M_PI*rand()/RAND_MAX);
		}
	}
	return;
}


int
Sphere::Fill(PointVect& points, const double dx, const bool fill)
{
	int nparts = 0;
	int nc = round(m_r / dx);
	double distance = m_r / (nc);

	for (int i = 0; i <= nc; ++i) {
		nparts += FillBorder(points, dx, i*distance);
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

#if USE_CHRONO == 1
/* Create a Chrono box body.
 *	\param dx : particle spacing
 */
void
Sphere::BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide,
	const EulerParameters & orientation_diff)
{
	// Check if the physical system is valid
	if (!bodies_physical_system)
		throw std::runtime_error("Sphere::BodyCreate Trying to create a body in an invalid physical system!\n");

	// Creating a new Chrono object
	m_body = chrono_types::make_shared< ::chrono::ChBodyEasySphere > ( m_r + dx/2.0, m_mass/Volume(dx), collide );
	m_body->SetPos(::chrono::ChVector<>(m_center(0), m_center(1), m_center(2)));
	m_body->SetRot(EulerParametersQuaternion(orientation_diff*m_ep));

	m_body->SetCollide(collide);
	m_body->SetBodyFixed(m_isFixed);
	// mass is automatically set according to density

	// Add the body to the physical system
	bodies_physical_system->AddBody(m_body);
}
#endif
