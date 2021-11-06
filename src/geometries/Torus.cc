/*  Copyright (c) 2011-2017 INGV, EDF, UniCT, JHU

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
/*
 * File:   Torus.cc
 * Author: alexis
 *
 * Created on 8 septembre 2011-2013, 18:52
 */

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

	// i = 0
	FillDiskBorder(points, m_ep, m_center, m_R + sqrt(m_r*m_r - m_r*m_r), m_r, dx, 2.0*M_PI*rand()/RAND_MAX);
	for (int i = 1; i < ntheta; i++) {
		const double theta = i*dtheta;
		const double z = m_r*cos(theta);
		FillDiskBorder(points, m_ep, m_center, m_R + sqrt(m_r*m_r - z*z), z, dx, 2.0*M_PI*rand()/RAND_MAX);
		FillDiskBorder(points, m_ep, m_center, m_R - sqrt(m_r*m_r - z*z), z, dx, 2.0*M_PI*rand()/RAND_MAX);
	}
	// i = ntheta
	FillDiskBorder(points, m_ep, m_center, m_R + sqrt(m_r*m_r - m_r*m_r), -m_r, dx, 2.0*M_PI*rand()/RAND_MAX);
}

void
Torus::FillIn(PointVect& points, const double dx, const int layers)
{
	const int lmin = layers < 0 ? layers + 1 : 0;
	const int lmax = layers < 0 ? 0 : layers - 1;

	// TODO FIXME check for extreme cases that result in filling or overlaps in the inner rings
	for (int l = lmin; l <= lmax; ++l) {
		Torus inner = Torus(m_center, m_R, m_r - l*dx, m_ep);
		inner.FillBorder(points, dx);
	}
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
