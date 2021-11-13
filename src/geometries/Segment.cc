/*  Copyright (c) 2021 INGV, EDF, UniCT, JHU

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

#include "Segment.h"


/// Default onstructor
Segment::Segment(void) :
	m_origin(0, 0, 0),
	m_lx(0),
	m_vx(0, 0, 0)
{}


/// Constructor
/*!	The segment is built from a starting point and vectors
	\param origin : starting point of the segment
	\param vx : vector
*/
Segment::Segment(const Point& origin, const Vector& vx) :
	m_origin(origin),
	m_lx(vx.norm()),
	m_vx(vx)
{
	m_center = m_origin + 0.5*m_vx;

	Vector z(0, 0, 1);
	const double angle = acos(m_vx*z/m_lx);
	Vector rotdir = -m_vx.cross(z);
	if (rotdir.norm() == 0)
		rotdir = Vector(0, 1, 0);

	m_ep = EulerParameters(rotdir, angle);
	m_ep.ComputeRot();
}


Segment::Segment(const Point &origin, const double lx, const EulerParameters &ep)
{
	m_origin = origin;

	m_lx = lx;

	m_vx = Vector(lx, 0, 0);
	setEulerParameters(ep);
}


double
Segment::Volume(const double dx) const
{
	const double lx = m_lx + dx;
	const double ly = Object::world_dimensions > 1 ? dx : 1;
	const double lz = Object::world_dimensions > 2 ? dx : 1;
	const double volume = lx*ly*lz;
	return volume;
}


void
Segment::SetInertia(const double dx)
{
	const double lx = m_lx + dx;
	const double ly = dx;
	const double lz = dx;
	switch (Object::world_dimensions) {
	case 1:
		// do we even care about moments of inertia in 1D,
		// given we won't have rotations?
		printf("TODO: %s verify 1D", __func__);
		m_inertia[0] = m_mass/12.0*lx*ly*ly;
		m_inertia[1] = 1.0; // should this be 0 or NAN?
		m_inertia[2] = 1.0; // should this be 0 or NAN?
		break;
	case 2:
		printf("TODO: %s verify 2D", __func__);
		m_inertia[0] = m_mass/12.0*lx*ly*ly;
		m_inertia[1] = m_mass/12.0*ly*lx*lx;
		m_inertia[2] = 1.0; // should this be 0 or NAN?
		break;
	case 3:
		m_inertia[0] = m_mass/12.0*(ly*ly + lz*lz);
		m_inertia[1] = m_mass/12.0*(lx*lx + lz*lz);
		m_inertia[2] = m_mass/12.0*(lx*lx + ly*ly);
		break;
	}
}


void
Segment::FillBorder(PointVect& points, const double dx)
{
	m_origin(3) = m_center(3);
	points.push_back(m_origin);
	points.push_back(m_origin + m_vx);
}


int
Segment::Fill(PointVect& points, const double dx, const bool fill)
{
	m_origin(3) = m_center(3);
	int nparts = 0;

	int nx = max((int) round(m_lx/dx), 1);
	int startx = 0;
	int endx = nx;

	Point origin = m_origin;

	if (default_filling_method == BORDER_TANGENT) {
		endx--;
		origin += 0.5*m_vx/nx;
	}

	for (int i = startx; i <= endx; i++) {
		Point p = origin + i*m_vx/nx;
		if (fill)
			points.push_back(p);
		nparts++;
	}

	return nparts;
}

void
Segment::FillIn(PointVect &points, const double dx, const int layers)
{
	switch (Object::world_dimensions) {
	case 1: FillIn1D(points, dx, layers); break;
	case 2: FillIn2D(points, dx, layers); break;
	// TODO decide what to do in 3D
	default: throw std::runtime_error("can't FillIn a Segment in " + std::to_string(Object::world_dimensions) + " dimensions");
	}
}

/// Place particles starting from the segment vertices
/// towards the inside (+) or outside (-) depending on the layers sign.
void
Segment::FillIn1D(PointVect &points, const double dx, const int layers)
{
	printf("TODO: %s verify 1D", __func__);

	// if the layers would fill the segment, just use a fill
	if (dx*layers >= m_lx) {
		printf("Segment FillIn converted to Fill\n");
		Fill(points, dx);
	}

	int _layers = abs(layers);

	Vector x_shift = m_vx;
	x_shift.normalize();

	// shift towards the inside
	const double signed_dx = (layers > 0 ? dx : -dx);

	m_origin(3) = m_center(3);
	Point o1 = m_origin;
	Point o2 = m_origin + m_vx;

	if (default_filling_method == BORDER_TANGENT) {
		// shift by half a dp
		o1 += x_shift*signed_dx/2;
		o2 -= x_shift*signed_dx/2;
	}

	while (_layers-- > 0) {
		points.push_back(o1 + _layers*signed_dx*x_shift);
		points.push_back(o2 - _layers*signed_dx*x_shift);
	}
}


/// Fill layers copies of a Segment in the direction of the normal vector
/// (conventionally, segments are oriented in the x direction,
///  so the (rotated) y vector is considered the normal)
void
Segment::FillIn2D(PointVect &points, const double dx, const int layers)
{
	int _layers = abs(layers);

	Vector normal = m_ep.Rot(Vector(0, 1, 0));

	Vector unitshift(layers > 0 ? normal : -normal);

	const double dy2 = layers < 0 ? -dx/2 : 0;
	const Point fill_origin =
		default_filling_method == BORDER_CENTERED ? m_origin
		: m_origin + m_ep.Rot( Vector(0, dy2, 0) );
	while (_layers-- > 0) {
		Segment layer(fill_origin + dx*_layers*unitshift, m_vx);
		layer.SetPartMass(m_center(3));
		layer.Fill(points, dx, true);
	}
}

bool
Segment::IsInside(const Point& p, const double dx) const
{
	Point lp = m_ep.TransposeRot(p - m_origin);
	const double lx = m_lx + dx;
	bool inside = false;
	if (lp(0) > -dx && lp(0) < lx && lp(1) > -dx && lp(1) < dx &&
		lp(2) > -dx && lp(2) < dx)
		inside = true;

	return inside;
}

void Segment::setEulerParameters(const EulerParameters &ep)
{
	m_ep = ep;
	m_ep.ComputeRot();

	m_vx = m_lx*m_ep.Rot(Vector(1, 0, 0));

	m_center = m_origin + .5*m_vx;
}

void Segment::getBoundingBox(Point &output_min, Point &output_max)
{
	getBoundingBoxOfCube(output_min, output_max, m_origin,
		m_vx, Vector(0, 0, 0), Vector(0, 0, 0) );
}

void Segment::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_center += poff;
	m_origin += poff;
}
