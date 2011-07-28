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

#include "Sphere.h"
#include "Circle.h"


Sphere::Sphere(void)
{
  	center = Point(0,0,0);
  	radius = Vector(1,0,0); // equatorial direction
  	height = Vector(0,0,1); //this is the polar direction
  	//  note: norm of radius and height must be equal for a sphere
}

Sphere::Sphere(const Point &c, const Vector &r, const Vector &h)
{
 	center = c;
 	radius = r;
 	height = h;
}

double
Sphere::SetPartMass(double dx, double rho)
{
  double mass = dx*dx*dx*rho;
  center(3) = mass;
  return mass;
}

void
Sphere::SetPartMass(double mass)
{
	center(3) = mass;
}

void
Sphere::FillBorder(PointVect& points, double dx)
{
  	double angle = dx/radius.norm();
    angle /= 2;
  	int nc = round(3.1415927/angle); //number of layers

  	for (int i = 1; i < (nc); ++i) {
  	 	Circle c(center -height*cos(i*angle),radius.rotated(i*angle,height)*sin(i*angle), height);
  	 	c.FillBorder(points, dx);
  	 }
}

void
Sphere::Fill(PointVect& points, double dx)
{
   	double angle = dx/radius.norm();

   	int nc = round(3.1415927/angle) ;//number of layers

   	for (int i=0; i < (nc); ++i) {
   	      Circle c(center-height*cos(i*angle), radius.rotated(i*angle,height)*sin(i*angle), height);
   	      c.Fill(points,dx,true);
   	}
}

void
Sphere::GLDraw(void)
{
	Circle(center, radius, height).GLDraw();
	Circle(center, height, radius).GLDraw();

}