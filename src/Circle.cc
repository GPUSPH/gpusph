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

#include <math.h>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "Circle.h"
#include "Rect.h"

Circle::Circle(void)
{
	center = Point();
	radius = Vector();
	normal = Vector(0,0,1);
}

Circle::Circle(const Point &p, const Vector &r, const Vector &u)
{
	center = p;
	radius = r;
	normal = u;
}

double
Circle::SetPartMass(double dx, double rho)
{
	// FIXME
	double mass = dx*dx*dx*rho;
	center(3) = mass;
	return mass;
}

void
Circle::SetPartMass(double mass)
{
	center(3) = mass;
}

void
Circle::FillBorder(PointVect& points, double dx)
{
	int np = round(2*PI*radius.norm()/dx);
	for (int i = 0; i < np; ++i) {
		Point pt = center + radius.rotated((double)(2*i*PI)/np, normal);
		points.push_back(pt);
	}
}

void
Circle::Fill(PointVect& points, double dx, bool fill_edge)
{
	PointVect rectpts;
	Vector rad1(radius);
	Vector rad2(radius.rotated(PI/2, normal));
    Rect rect(center - rad1 - rad2, 2*rad1, 2*rad2);
	rect.SetPartMass(center(3));
	rect.Fill(rectpts, dx, true);


	double r = radius.normSquared();
	for (unsigned int i=0; i < rectpts.size(); ++i) {
		Point pt(rectpts[i]);
		if (center.DistSquared(pt) <= 1.01*r) {
			points.push_back(pt);
		}
	}

}

void
Circle::GLDraw(void)
{
	glBegin(GL_LINES);
#define CIRCLE_LINES 360
	for (int i=0; i < CIRCLE_LINES; ++i) {
		Point pt = center + radius.rotated((double)(2*i*PI)/CIRCLE_LINES, normal);
		glVertex3f(pt(0), pt(1), pt(2));
	}
#undef CIRCLE_LINES
	glEnd();
}
