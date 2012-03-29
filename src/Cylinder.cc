/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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
	std::cout << " angle " << angle << "\n";
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
	
	m_ep = ep;
	m_ep.ComputeRot();
	
	m_center = m_origin + m_ep.Rot(0.5*m_h*Vector(0, 0, 1));
	m_origin.print();
	m_center.print();
}


Cylinder::Cylinder(const Point& origin, const Vector& radius, const Vector& height)
{	
	if (abs(radius*height) > 1e-8*radius.norm()*height.norm()) {
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


void
Cylinder::FillBorder(PointVect& points, const double dx, const bool bottom, const bool top)
{
	m_origin(3) = m_center(3);
	const int nz = (int) ceil(m_h/dx);
	const double dz = m_h/nz;
	for (int i = 0; i <= nz; i++)
		FillDiskBorder(points, m_ep, m_origin, m_r, i*dz, dx, 2.0*M_PI*rand()/RAND_MAX);
	if (bottom)
		FillDisk(points, m_ep, m_origin, m_r - dx, 0.0, dx, 0.0);
	if (top)
		FillDisk(points, m_ep, m_origin, m_r - dx, nz*dz, dx, 0.0);
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


void
Cylinder::GLDraw(const EulerParameters& ep, const Point& cg) const
{
	Point origin = cg - 0.5*m_h*ep.Rot(Vector(0, 0, 1));
	
	#define CIRCLES_NUM 6
	#define LINES_NUM	10
	const double dz = m_h/CIRCLES_NUM;
	for (int i = 0; i <= CIRCLES_NUM; ++i) {
		GLDrawCircle(ep, origin, m_r, i*dz);
	}
	
	const double angle2 = 2.0*M_PI/LINES_NUM;
	for (int i = 0; i < LINES_NUM; i++) {
		double u = i*angle2;
		Point p1 = ep.Rot(Point(m_r*cos(u), m_r*sin(u), 0.0));
		p1 += origin;
		Point p2 = ep.Rot(Point(m_r*cos(u), m_r*sin(u), m_h));
		p2 += origin;
		GLDrawLine(p1, p2);
	}
	
	#undef CIRCLES_NUM
	#undef LINES_NUM
}


void
Cylinder::GLDraw(void) const
{
	GLDraw(m_ep, m_center);
}