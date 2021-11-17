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
#include <cfloat>
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
	throw std::runtime_error("FillBorder not implemented for planes!\n");
}

int Plane::Fill(PointVect& points, const double dx, const bool fill)
{
	throw std::runtime_error("Fill not implemented for planes!\n");
	return 0;
}

void Plane::FillIn(PointVect& points, const double dx, const int layers)
{
	throw std::runtime_error("FillIn not implemented for planes!\n");
}

bool Plane::IsInside(const Point& p, const double dx) const
{
	const double distance = (m_a * p(0) + m_b * p(1) + m_c * p(2) + m_d) / m_norm;
	// the particle is inside if the (signed) distance is larger than -dx,
	// i.e. distance + dx > 0
	// but we want to account for small variations, so we instead check against
	// -FLT_EPSILON*dx
#if 0
	bool inside = (distance > -dx);
#else
	bool inside = (distance + dx > FLT_EPSILON*dx);
#endif
	return inside;
}

void Plane::setEulerParameters(const EulerParameters &ep)
{
	throw std::runtime_error("Trying to set EulerParameters on a plane!");
}

// It is not really meaningful to GPUSPH to have a bounding box with infinities,
// but at least it is correct...
void Plane::getBoundingBox(Point &output_min, Point &output_max)
{
	// as a general rule, a plane spans the entire coordinate system
	output_min = Point(-INFINITY, -INFINITY, -INFINITY);
	output_max = Point(INFINITY, INFINITY, INFINITY);

	// there are exceptions for planes parallel to the coordinate system,
	// in which case we only take the half-plane
	// “above” (if the nonzero axis coefficient is positive)
	// or “below” (if the nonzero axis coefficient is negative)

	// the half plane in this case is give by {x,y,z} = -m_d/{m_a, m_b, m_c}
	// depending on which one is nonzero.
	// The trick we use in this case is that since only one of a, b, c is nonzero,
	// the sum of all three is equal to the nonzeroone:

	const double nonzero = m_a + m_b + m_c;
	const double coord = -m_d/nonzero;

	// we update output_min if the coefficient is positive, output_max if it's negative
	Point& changed = nonzero > 0 ? output_min : output_max;

	if (m_a == 0 && m_b == 0) {
		changed(2) = coord;
	} else if (m_a == 0 && m_c == 0) {
		changed(3) = coord;
	} else if (m_b == 0 && m_c == 0) {
		changed(0) = coord;
	}
	// else there is more than one nonzero coefficient, so there's nothing to do
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
