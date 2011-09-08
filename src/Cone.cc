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

//Created by Andrew 12/2009
//To set to a cone,  let radiust be a really small number, but not zero; otherwise, use for a frustum

#include <math.h>
#include "Cone.h"
#include "Circle.h"


Cone::Cone(void)
{
	center = Point();
	radiust = Vector();
	radiusb = Vector();
	height = Vector(0,0,1);
}


Cone::Cone(const Point &c, const Vector &rb, const Vector &rt, const Vector &h)
{
	center = c;
	radiusb = rb;
	radiust = rt;
	height = h;
}


double
Cone::SetPartMass(double dx, double rho)
{
	int nc = round(height.norm()/dx);
	int nb = round(radiusb.norm()/dx);
	int nt = round(radiust.norm()/dx);
	double dh = height.norm()/nc;
	double rb = radiusb.norm()/nb;
	double rt = radiust.norm()/nt;
	double mass = dh*((rb*rb)+(rt*rt)+(rt*rb))*rho;
	center(3) = mass;
	return mass;
}


void
Cone::SetPartMass(double mass)
{
	center(3) = mass;
}

void

Cone::FillBorder(PointVect& points, double dx, bool bottom, bool top)
{


  int nc = round(height.norm()/dx)*10;  //number of circles
	if (bottom) {
		Circle c(center, radiusb, height);
		c.Fill(points, dx, true);
	}
	for (int i = 1; i < nc; ++i) {
	  double alpha = dx/(2*radiusb.norm());
	  double eta = dx/(2*radiust.norm());
	  Vector offset = i*height/nc;
	  Circle c(center + offset, radiusb.rotated(i*alpha, height) - (i*(radiusb.rotated(i*alpha, height)-radiust.rotated(i*eta, height)))/nc, height);
		c.FillBorder(points, dx);
	}
	if (top) {
	  Circle c(center + height, radiust, height);
	  c.Fill(points, dx, true);
	}
}


void
Cone::Fill(PointVect& points, double dx)
{

  int nc = round(height.norm()/dx)*10;
	for (int i = 0; i < nc; ++i) {
	    double alpha = dx/(2*(radiusb.norm()-(i*(radiusb.norm()-radiust.norm()))/nc));
	    double eta = dx/(2*radiust.norm());
	    Vector offset = i*height/nc;
	    Circle c(center + offset, radiusb.rotated(i*alpha, height) - (i*(radiusb.rotated(i*alpha, height)-radiust.rotated(i*eta, height)))/nc, height);
		c.Fill(points, dx, true);
	}
}


void
Cone::GLDraw(void)
{
	Circle(center + height, radiust, height).GLDraw();
	Circle(center, radiusb, height).GLDraw();
}


