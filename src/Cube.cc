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

#include "Cube.h"
#include "Rect.h"


Cube::Cube(void)
{
	m_origin = Point(0, 0, 0);
	m_vx = Vector(0, 0, 0);
	m_vy = Vector(0, 0, 0);
	m_vz = Vector(0, 0, 0);
}


Cube::Cube(const Point &origin, const double lx, const double ly, const double lz, const EulerParameters &ep)
{
	m_origin = origin;
	
	m_ep = ep;
	m_ep.ComputeRot();
	m_lx = lx;
	m_ly = ly;
	m_lz = lz;
	
	m_vx = m_lx*m_ep.Rot(Vector(1, 0, 0));
	m_vy = m_ly*m_ep.Rot(Vector(0, 1, 0));
	m_vz = m_lz*m_ep.Rot(Vector(0, 0, 1));
	
	m_center = m_origin + 0.5*m_ep.Rot(Vector(m_lx, m_ly, m_lz));
	m_origin.print();
	m_center.print();
}


Cube::Cube(const Point& origin, const Vector& vx, const Vector& vy, const Vector& vz)
{
	if (abs(vx*vy) > 1e-8*vx.norm()*vy.norm() || abs(vx*vz) > 1e-8*vx.norm()*vz.norm() 
		|| abs(vy*vz) > 1e-8*vy.norm()*vz.norm()) {
		std::cout << "Trying to construct a cube with non perpendicular vectors\n";
		exit(1);
	}
	
	m_origin = origin;
	m_vx = vx;
	m_lx = m_vx.norm();
	m_vy = vy;
	m_ly = m_vy.norm();
	m_vz = vz;
	m_lz = m_vz.norm();
	m_center = m_origin + 0.5*(m_vx + m_vy + m_vz);
	
	Vector axis;
	double mat[9];
	mat[0] = m_vx(0)/m_lx;
	mat[3] = m_vx(1)/m_lx;
	mat[6] = m_vx(2)/m_lx;
	mat[1] = m_vy(0)/m_ly;
	mat[4] = m_vy(1)/m_ly;
	mat[7] = m_vy(2)/m_ly;
	mat[2] = m_vz(0)/m_lz;
	mat[5] = m_vz(1)/m_lz;
	mat[8] = m_vz(2)/m_lz;
	
	double trace = mat[0] + mat[4] + mat[8];
	double cs = 0.5*(trace - 1.0);
	double angle = acos(cs);  // in [0,PI]

	if (angle > 0.0)
	{
		if (angle < M_PI)
		{
			axis(0) = mat[7] - mat[5];
			axis(1) = mat[2] - mat[6];
			axis(2) = mat[3] - mat[1];
			axis /= axis.norm();
		}
		else
		{
			// angle is PI
			double halfInverse;
			if (mat[0] >= mat[4])
			{
				// r00 >= r11
				if (mat[0] >= mat[8])
				{
					// r00 is maximum diagonal term
					axis(0) = 0.5*sqrt(1.0 + mat[0] - mat[4] - mat[8]);
					halfInverse = 0.5/axis(0);
					axis(1) = halfInverse*mat[1];
					axis(2) = halfInverse*mat[2];
				}
				else
				{
					// r22 is maximum diagonal term
					axis(2) = 0.5*sqrt(1.0 + mat[8] - mat[0] - mat[4]);
					halfInverse = 0.5/axis(2);
					axis(0) = halfInverse*mat[2];
					axis(1) = halfInverse*mat[5];
				}
			}
			else
			{
				// r11 > r00
				if (mat[4] >= mat[8])
				{
					// r11 is maximum diagonal term
					axis(1) = 0.5*sqrt(1.0 + + mat[4] - mat[0] - mat[8]);
					halfInverse  = 0.5/axis(1);
					axis(0) = halfInverse*mat[1];
					axis(2) = halfInverse*mat[5];
				}
				else
				{
					// r22 is maximum diagonal term
					axis(2) = 0.5*sqrt(1.0 + mat[8] - mat[0] - mat[4]);
					halfInverse = 0.5/axis(2);
					axis(0) = halfInverse*mat[2];
					axis(1) = halfInverse*mat[5];
				}
			}
		}
	}
	else
	{
		// The angle is 0 and the matrix is the identity.  Any axis will
		// work, so just use the x-axis.
		axis(0) = 1.0;
		axis(1) = 0.0;
		axis(2) = 0.0;
	}
	
	m_ep = EulerParameters(axis, angle);
	m_ep.ComputeRot();
}


double
Cube::Volume(const double dx) const
{
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double lz = m_lz + dx;
	const double volume = lx*ly*lz;
	return volume;
}


void
Cube::SetInertia(const double dx)
{
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double lz = m_lz + dx;
	m_inertia[0] = m_mass/12.0*(ly*ly + lz*lz);
	m_inertia[1] = m_mass/12.0*(lx*lx + lz*lz);
	m_inertia[2] = m_mass/12.0*(lx*lx + ly*ly);
}


void
Cube::FillBorder(PointVect& points, const double dx, const int face_num, const bool *edges_to_fill)
{
	Point   rorigin;
	Vector  rvx, rvy;
	m_origin(3) = m_center(3);
	
	switch(face_num){
		case 0:
			rorigin = m_origin;
			rvx = m_vx;
			rvy = m_vz;
			break;
		case 1:
			rorigin = m_origin + m_vx;
			rvx = m_vy;
			rvy = m_vz;
			break;
		case 2:
			rorigin = m_origin + m_vx + m_vy;
			rvx = -m_vx;
			rvy = m_vz;
			break;
		case 3:
			rorigin = m_origin + m_vy;
			rvx = -m_vy;
			rvy = m_vz;
			break;
		case 4:
			rorigin = m_origin;
			rvx = m_vx;
			rvy = m_vy;
			break;
		case 5:
			rorigin = m_origin + m_vz;
			rvx = m_vx;
			rvy = m_vy;
			break;
	}

	Rect rect = Rect(rorigin, rvx, rvy);
	rect.Fill(points, dx, edges_to_fill);
}


void
Cube::FillBorder(PointVect& points, const double dx, const bool fill_top_face)
{
	m_origin(3) = m_center(3);
	
	bool edges_to_fill[6][4] =
		{   {true, true, true, true},
			{true, false, true, false},
			{true, true, true, true},
			{true, false, true, false},
			{false, false, false, false},
			{false, false, false, false} };

	int last_face = 6;
	if (!fill_top_face)
		last_face --;
	for (int face_num = 0; face_num < last_face; face_num++)
			FillBorder(points, dx, face_num, edges_to_fill[face_num]);
}


int
Cube::Fill(PointVect& points, const double dx, const bool fill_faces, const bool fill)
{
	m_origin(3) = m_center(3);
	int nparts = 0;
	
	const int nx = (int) (m_lx/dx);
	const int ny = (int) (m_ly/dx);
	const int nz = (int) (m_lz/dx);

	int startx = 0;
	int starty = 0;
	int startz = 0;
	int endx = nx;
	int endy = ny;
	int endz = nz;

	if (!fill_faces){
		startx++;
		starty++;
		startz++;
		endx --;
		endy --;
		endz --;
	}

	for (int i = startx; i <= endx; i++)
		for (int j = starty; j <= endy; j++)
			for (int k = startz; k <= endz; k++) {
				Point p = m_origin + i/((double) nx)*m_vx + j/((double) ny)*m_vy + k/((double) nz)*m_vz;
				if (fill)
					points.push_back(p);
				nparts ++;
			}
	return nparts;
}


void
Cube::InnerFill(PointVect& points, const double dx)
{
	m_origin(3) = m_center(3);
	const int nx = (int) (m_lx/dx);
	const int ny = (int) (m_ly/dx);
	const int nz = (int) (m_lz/dx);

	int startx = 0;
	int starty = 0;
	int startz = 0;
	int endx = nx;
	int endy = ny;
	int endz = nz;

	for (int i = startx; i < endx; i++)
		for (int j = starty; j < endy; j++)
			for (int k = startz; k < endz; k++) {
				Point p = m_origin + (i + 0.5)*m_vx/nx + (j + 0.5)*m_vy/ny + (k + 0.5)*m_vz/nz;
				points.push_back(p);
			}
	return;
}


bool
Cube::IsInside(const Point& p, const double dx) const
{
	Point lp = m_ep.TransposeRot(p - m_origin);
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double lz = m_lz + dx;
	bool inside = false;
	if (lp(0) > -dx && lp(0) < lx && lp(1) > -dx && lp(1) < ly &&
		lp(2) > -dx && lp(2) < lz )
		inside = true;
	
	return inside;
}


void
Cube::GLDraw(const EulerParameters& ep, const Point& cg) const
{
	Point origin = cg - 0.5*ep.Rot(Vector(m_lx, m_ly, m_lz));
	
	Point p1, p2, p3, p4;
	p1 = Point(0, 0, 0);
	p2 = Point(m_lx, 0, 0);
	p3 = Point(m_lx, 0, m_lz);
	p4 = Point(0, 0, m_lz);
	GLDrawQuad(ep, p1, p2, p3, p4, origin);
	
	p1 = Point(0, m_ly, 0);
	p2 = Point(m_lx, m_ly, 0);
	p3 = Point(m_lx, m_ly, m_lz);
	p4 = Point(0, m_ly, m_lz);
	GLDrawQuad(ep, p1, p2, p3, p4, origin);
	
	p1 = Point(0, 0, 0);
	p2 = Point(0, m_ly, 0);
	p3 = Point(0, m_ly, m_lz);
	p4 = Point(0, 0, m_lz);
	GLDrawQuad(ep, p1, p2, p3, p4, origin);
		
	p1 = Point(m_lx, 0, 0);
	p2 = Point(m_lx, m_ly, 0);
	p3 = Point(m_lx, m_ly, m_lz);
	p4 = Point(m_lx, 0, m_lz);
	GLDrawQuad(ep, p1, p2, p3, p4, origin);
}


void
Cube::GLDraw(void) const
{
	GLDraw(m_ep, m_center);
}
