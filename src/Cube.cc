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

#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "Rect.h"

Cube::Cube(void)
{
	origin = Point(0, 0, 0);
	vx = Vector(0, 0, 0);
	vy = Vector(0, 0, 0);
	vz = Vector(0, 0, 0);
}


Cube::Cube(const Point &p, const Vector& v1, const Vector& v2, const Vector& v3)
{
	origin = p;
	vx = v1;
	vy = v2;
	vz = v3;
}


double
Cube::SetPartMass(double dx, double rho)
{
	int nx = (int) (vx.norm()/dx);
	double deltax = vx.norm()/((double) nx);
	int ny = (int) (vy.norm()/dx);
	double deltay = vy.norm()/((double) ny);
	int nz = (int) (vz.norm()/dx);
	double deltaz = vz.norm()/((double) nz);
	double mass = deltax*deltay*deltaz*rho;

	origin(3) = mass;
	return mass;
}


void
Cube::SetPartMass(double mass)
{
	origin(3) = mass;
}


void
Cube::FillBorder(PointVect& points, double dx, int face_num, bool *edges_to_fill)
{
	Point   rorigin;
	Vector  rvx, rvy;

	switch(face_num){
		case 0:
			rorigin = origin;
			rvx = vx;
			rvy = vz;
			break;
		case 1:
			rorigin = origin + vx;
			rvx = vy;
			rvy = vz;
			break;
		case 2:
			rorigin = origin + vx + vy;
			rvx = -vx;
			rvy = vz;
			break;
		case 3:
			rorigin = origin + vy;
			rvx = -vy;
			rvy = vz;
			break;
		case 4:
			rorigin = origin;
			rvx = vx;
			rvy = vy;
			break;
		case 5:
			rorigin = origin + vz;
			rvx = vx;
			rvy = vy;
			break;
	}

	Rect rect = Rect(rorigin, rvx, rvy);
	rect.Fill(points, dx, edges_to_fill);
}


void
Cube::FillBorder(PointVect& points, double dx, bool fill_top_face)
{
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
	//FillBorder(points, dx, 0, edges_to_fill[0]);
	//FillBorder(points, dx, 1, edges_to_fill[1]);
	//FillBorder(points, dx, 4, edges_to_fill[4]);
	//FillBorder(points, dx, 5, edges_to_fill[5]);
}


void
Cube::Fill(PointVect& points, double dx, bool fill_faces)
{
	int nx = (int) (vx.norm()/dx);
	int ny = (int) (vy.norm()/dx);
	int nz = (int) (vz.norm()/dx);

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

	int counter = 0;
	for (int i = startx; i <= endx; i++)
		for (int j = starty; j <= endy; j++)
			for (int k = startz; k <= endz; k++) {
				Point p = origin + i/((double) nx)*vx + j/((double) ny)*vy + k/((double) nz)*vz;
				points.push_back(p);
				counter++;
			}
	return;
}

void
Cube::InnerFill(PointVect& points, double dx)
{
	int nx = (int) (vx.norm()/dx);
	int ny = (int) (vy.norm()/dx);
	int nz = (int) (vz.norm()/dx);

	int startx = 0;
	int starty = 0;
	int startz = 0;
	int endx = nx;
	int endy = ny;
	int endz = nz;

	int counter = 0;
	for (int i = startx; i < endx; i++)
		for (int j = starty; j < endy; j++)
			for (int k = startz; k < endz; k++) {
				Point p = origin + (i+0.5)*vx/nx + (j+0.5)*vy/ny + (k+0.5)*vz/nz;
				points.push_back(p);
				counter++;
			}
	return;
}

void
Cube::GLDrawQuad(const Point& p1, const Point& p2,
		const Point& p3, const Point& p4)
{
	glVertex3f((float) p1(0), (float) p1(1), (float) p1(2));
	glVertex3f((float) p2(0), (float) p2(1), (float) p2(2));
	glVertex3f((float) p3(0), (float) p3(1), (float) p3(2));
	glVertex3f((float) p4(0), (float) p4(1), (float) p4(2));
}


void
Cube::GLDraw(void)
{
	Point p1, p2, p3, p4;
	glBegin(GL_QUADS);
	{
		p1 = origin;
		p2 = p1 + vx;
		p3 = p2 + vz;
		p4 = p3 - vx;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin + vy;
		p2 = p1 + vx;
		p3 = p2 + vz;
		p4 = p3 - vx;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin;
		p2 = p1 + vy;
		p3 = p2 + vz;
		p4 = p3 - vy;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin + vx;
		p2 = p1 + vy;
		p3 = p2 + vz;
		p4 = p3 - vy;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin;
		p2 = p1 + vx;
		p3 = p2 + vy;
		p4 = p3 - vx;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin + vz;
		p2 = p1 + vx;
		p3 = p2 + vy;
		p4 = p3 - vx;
		GLDrawQuad(p1, p2, p3, p4);
	}
	glEnd();
}
