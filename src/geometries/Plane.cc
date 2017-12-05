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

#include "Plane.h"

Plane::Plane(const double a, const double b, const double c, const double d)
{
	m_a = a;
	m_b = b;
	m_c = c;
	m_d = d;
	m_norm = sqrt(m_a * m_a + m_b * m_b + m_c * m_c);
}

void Plane::SetInertia(const double dx)
{
	throw std::runtime_error("Trying to set inertia on a plane!");
}

void Plane::FillBorder(PointVect& points, const double dx)
{
	printf("WARNING: FillBorder not implemented for planes!\n");
}

int Plane::Fill(PointVect& points, const double dx, const bool fill)
{
	printf("WARNING: Fill not implemented for planes!\n");
	return 0;
}

void Plane::FillIn(PointVect& points, const double dx, const int layers)
{
	printf("WARNING: FillIn not implemented for planes!\n");
}

bool Plane::IsInside(const Point& p, const double dx) const
{
	const double distance = (m_a * p(0) + m_b * p(1) + m_c * p(2) + m_d) / m_norm;
	return (distance > -dx);
}

void Plane::setEulerParameters(const EulerParameters &ep)
{
	throw std::runtime_error("Trying to set EulerParameters on a plane!");
}

// It is not really meaningful to GPUSPH to have a bounding box with infinities,
// but at least it is correct...
void Plane::getBoundingBox(Point &output_min, Point &output_max)
{
	if (m_a == 0 && m_b == 0) {
		output_min = Point(INFINITY, INFINITY, m_d/m_c);
		output_max = Point(INFINITY, INFINITY, m_d/m_c);
	} else
	if (m_a == 0 && m_c == 0) {
		output_min = Point(INFINITY, m_d/m_b, INFINITY);
		output_max = Point(INFINITY, m_d/m_b, INFINITY);
	} else
	if (m_b == 0 && m_c == 0) {
		output_min = Point(m_d/m_a, INFINITY, INFINITY);
		output_max = Point(m_d/m_a, INFINITY, INFINITY);
	} else
		output_min = Point(INFINITY, INFINITY, INFINITY);
		output_max = Point(INFINITY, INFINITY, INFINITY);
}

void Plane::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	// also update center although it has little meaning for a plane
	m_center += poff;
	printf("m_d was %g, off %g %g %g\n", m_d, offset.x, offset.y, offset.z);
	m_d += m_a * offset.x + m_b * offset.y + m_c * offset.z;
	printf("m_d now is %g\n", m_d);
}

void Plane::ODEBodyCreate(dWorldID ODEWorld, const double dx, dSpaceID ODESpace)
{
	throw std::runtime_error("Trying to instantiate an ODEBody for a plane!");
}

void Plane::ODEGeomCreate(dSpaceID ODESpace, const double dx)
{
	// NOTE: inverting m_d because ODE uses a+b+c=d
	m_ODEGeom = dCreatePlane(ODESpace, m_a, m_b, m_c, -m_d);
}
