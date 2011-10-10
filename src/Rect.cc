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

#include "Rect.h"


Rect::Rect(void)
{
	m_origin = Point(0, 0, 0);
	m_vx = Vector(0, 0, 0);
	m_vy = Vector(0, 0, 0);
}


Rect::Rect(const Point& origin, const Vector& vx, const Vector& vy)
{
	if (abs(vx*vy) > 1.e-8*vx.norm()*vy.norm()) {
		std::cout << "Trying to construct a rectangle with non perpendicular vectors\n";
		exit(1);
	}
	
	m_origin = origin;
	m_vx = vx;
	m_vy = vy;
	m_lx = vx.norm();
	m_ly = vy.norm();
	m_center = m_origin + 0.5*m_vx + 0.5*m_vy;
	
	Vector vz = m_vx.cross(m_vy);
	vz.normalize();
	
	Vector axis;
	double mat[8];
	mat[0] = m_vx(0)/m_lx;
	mat[3] = m_vx(1)/m_lx;
	mat[6] = m_vx(2)/m_lx;
	mat[1] = m_vy(0)/m_ly;
	mat[4] = m_vy(1)/m_ly;
	mat[7] = m_vy(2)/m_ly;
	mat[2] = vz(0);
	mat[5] = vz(1);
	mat[8] = vz(2);
	
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


Rect::Rect(const Point &origin, const double lx, const double ly, const EulerParameters &ep)
{
	m_origin = origin;
	
	m_ep = ep;
	m_ep.ComputeRot();
	m_lx = lx;
	m_ly = ly;
	
	m_vx = m_lx*m_ep.Rot(Vector(1, 0, 0));
	m_vy = m_ly*m_ep.Rot(Vector(0, 1, 0));
	
	m_center = m_origin + m_ep.Rot(Vector(0.5*m_lx, 0.5*m_ly, 0.0));
	m_origin.print();
	m_center.print();
}


double
Rect::Volume(const double dx) const
{
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double volume = lx*ly*dx;
	return volume;
}


void
Rect::SetInertia(const double dx)
{
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double lz = dx;
	m_inertia[0] = m_mass/12.0*(ly*ly + lz*lz);
	m_inertia[1] = m_mass/12.0*(lx*lx + lz*lz);
	m_inertia[2] = m_mass/12.0*(lx*lx + ly*ly);
}


void
Rect::FillBorder(PointVect& points, const double dx,
		const bool populate_first, const bool populate_last, const int edge_num)
{
	Point		origin;
	Vector		dir;

	m_origin(3) = m_center(3);
	switch(edge_num){
		case 0:
			origin = m_origin;
			dir = m_vx;
			break;
		case 1:
			origin = m_origin + m_vx;
			dir = m_vy;
			break;
		case 2:
			origin = m_origin + m_vx + m_vy;
			dir = - m_vx;
			break;
		case 3:
			origin = m_origin + m_vy;
			dir = - m_vy;
			break;
	}

	int nx = (int) (dir.norm()/dx);
	int startx = 0;
	int endx = nx;

	if (!populate_first){
		startx++;
	}
	
	if (!populate_last){
		endx--;
	}

	for (int i = startx; i <= endx; i++) {
		Point p = origin + i*dir/nx;
		points.push_back(p);
	}
}


void
Rect::FillBorder(PointVect& points, const double dx)
{
	m_origin(3) = m_center(3);
	FillBorder(points, dx, false, false, 0);
	FillBorder(points, dx, true, true, 1);
	FillBorder(points, dx, false, false, 2);
	FillBorder(points, dx, true, true, 3);
}


int
Rect::Fill(PointVect& points, const double dx, const bool fill_egdes, const bool fill)
{
	m_origin(3) = m_center(3);
	int nparts = 0;
	
	int nx = (int) (m_lx/dx);
	int ny = (int) (m_ly/dx);
	int startx = 0;
	int starty = 0;
	int endx = nx;
	int endy = ny;

	if (!fill_egdes){
		startx++;
		starty++;
		endx --;
		endy --;
	}

	for (int i = startx; i <= endx; i++)
		for (int j = starty; j <= endy; j++) {
			Point p = m_origin + i*m_vx/nx + j*m_vy/ny;
			if (fill)
				points.push_back(p);
			nparts++;
		}

	return nparts;
}


int
Rect::Fill(PointVect& points, const double dx, const bool fill)
{
	return Fill(points, dx, true, fill);
}


void
Rect::Fill(PointVect& points, const double dx, const bool* edges_to_fill)
{
	m_origin(3) = m_center(3);
	
	Fill(points, dx, false, true);

	for (int border_num = 0; border_num < 4; border_num++) {
		if (edges_to_fill[border_num])
			FillBorder(points, dx, true, false, border_num);
		}

	return;
}


bool
Rect::IsInside(const Point& p, const double dx) const
{	
	Point lp = m_ep.TransposeRot(p - m_origin);
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	bool inside = false;
	if (lp(0) > -dx && lp(0) < lx && lp(1) > -dx && lp(1) < ly &&
		lp(2) > -dx && lp(2) < dx)
		inside = true;
	
	return inside;
}


void
Rect::GLDraw(const EulerParameters& ep, const Point &cg) const
{
	Point origin = cg - ep.Rot(Vector(0.5*m_lx, 0.5*m_ly, 0.0));
	
	Point p1, p2, p3, p4;
	p1 = Point(0, 0, 0);
	p2 = Point(m_lx, 0, 0);
	p3 = Point(m_lx, m_ly, 0);
	p4 = Point(0, m_ly, 0);
	GLDrawQuad(ep, p1, p2, p3, p4, origin);
}


void
Rect::GLDraw(void) const
{
	GLDraw(m_ep, m_center);
}
