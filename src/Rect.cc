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

#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "Rect.h"
#include "Point.h"
#include "Segment.h"
#include "Vector.h"

Rect::Rect(void)
{
	origin = Point(0, 0, 0);
	vx = Vector(0, 0, 0);
	vy = Vector(0, 0, 0);
}

Rect::Rect(const Point &p, const Vector& v1, const Vector& v2)
{
	origin = p;
	vx = v1;
	vy = v2;
}


double
Rect::SetPartMass(double dx, double rho)
{
	int nx = (int) (vx.norm()/dx);
	double deltax = vx.norm()/((double) nx);
	int ny = (int) (vy.norm()/dx);
	double deltay = vy.norm()/((double) ny);
	double mass = dx*deltax*deltay*rho;

	origin(3) = mass;
	return mass;
}


void
Rect::SetPartMass(double mass)
{
	origin(3) = mass;
}


void
Rect::FillBorder(PointVect& points, double dx,
		bool populate_first, bool populate_last, int edge_num)
{
	Segment   seg;
	Point   start;
	Point   end;

	switch(edge_num){
		case 0:
			start = origin;
			end = start + vx;
			break;
		case 1:
			start = origin + vx;
			end = start + vy;
			break;
		case 2:
			start = origin + vx + vy;
			end = start - vx;
			break;
		case 3:
			start = origin + vy;
			end = start - vy;
			break;
	}

	seg = Segment(start, end);
	seg.FillBorder(points, dx, populate_first, populate_last);
}


void
Rect::FillBorder(PointVect& points, double dx, bool fill_top)
{
	FillBorder(points, dx, false, false, 0);
	FillBorder(points, dx, true, true, 1);
	if (fill_top)
		FillBorder(points, dx, false, false, 2);
	FillBorder(points, dx, true, true, 3);
}


void
Rect::Fill(PointVect& points, double dx, bool fill_egdes)
{
	int nx = (int) (vx.norm()/dx);
	int ny = (int) (vy.norm()/dx);
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
			Point p = origin + i*vx/nx + j*vy/ny;
			points.push_back(p);
		}

	return;
}


void
Rect::Fill(PointVect& points, double dx, bool *edges_to_fill)
{
	Fill(points, dx, false);

	for (int border_num = 0; border_num < 4; border_num++) {
		if (edges_to_fill[border_num])
			FillBorder(points, dx, true, false, border_num);
		}

	return;
}

void
Rect::GLDrawQuad(const Point& p1, const Point& p2,
		const Point& p3, const Point& p4)
{
	glVertex3f((float) p1(0), (float) p1(1), (float) p1(2));
	glVertex3f((float) p2(0), (float) p2(1), (float) p2(2));
	glVertex3f((float) p3(0), (float) p3(1), (float) p3(2));
	glVertex3f((float) p4(0), (float) p4(1), (float) p4(2));
}


void
Rect::GLDraw(void)
{
	Point p1, p2, p3, p4;
	glBegin(GL_QUADS);
	{
		p1 = origin;
		p2 = p1 + vx;
		p3 = p2 + vy;
		p4 = p3 - vx;
		GLDrawQuad(p1, p2, p3, p4);
	}
	glEnd();
}
