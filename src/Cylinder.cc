#include <math.h>
#include <stdlib.h>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "Cylinder.h"
#include "Circle.h"

Cylinder::Cylinder(void)
{
	center = Point(0,0,0);
	radius = Vector(0,0,0);
	height = Vector(0,0,1);
}

Cylinder::Cylinder(const Point &c, const Vector &r, const Vector &h)
{
	center = c;
	radius = r;
	height = h;
}

double
Cylinder::SetPartMass(double dx, double rho)
{
	// FIXME
	int nc = round(height.norm()/dx);
	double dh = height.norm()/nc;
	double mass = dh*dx*dx*rho;
	center(3) = mass;
	return mass;
}

void
Cylinder::SetPartMass(double mass)
{
	center(3) = mass;
}

void
Cylinder::FillBorder(PointVect& points, double dx, bool bottom, bool top)
{
	/* stagger circles */
	double angle = dx/radius.norm();
	angle /= 2;

	int nc = round(height.norm()/dx);
	if (bottom) {
		Circle c(center, radius, height);
		c.Fill(points, dx, true);
	}
	for (int i = 1; i < nc; ++i) {
		Vector offset = i*height/nc;
		Circle c(center + offset, radius.rotated(i*angle, height), height);
		c.FillBorder(points, dx);
	}
	if (top) {
		Circle c(center + height, radius.rotated(nc*angle, height), height);
		c.Fill(points, dx, true);
	}
}

void
Cylinder::Fill(PointVect& points, double dx)
{
	int nparts = 0;
	int nr = (int) ceil(radius.norm()/dx);
	for (int i = 0; i <= nr; i++) {
		double r = i*dx;
		int nc = (int) (2.0*M_PI*r/dx);
		#define THETARAND 2.0*M_PI/RAND_MAX
		double theta0 = THETARAND*rand();
		#undef THETARAND
		for (int j = 0; j < nc; j++) {
			double theta = theta0 + 2.0*M_PI*j/nc;
			Point p = center;
			p(0) += r*cos(theta);
			p(1) += r*sin(theta);
			p(3) = center(3);

			float z = center(2);
			float heightn = height.norm();
			while (z <= heightn ) {
				p(2) = z;
				points.push_back(p);
				nparts++;
				z += dx;
				}
		}
	  }
}

void
Cylinder::GLDraw(void)
{
	Circle(center, radius, height).GLDraw();
	Circle(center+height, radius, height).GLDraw();

	// FIXME this is horrible
	for (int i = 0; i < 10; i++) {
		glBegin(GL_LINES);
		Point pt = center + radius.rotated(i*2*M_PI/10.0, height);
		glVertex3f(pt(0), pt(1), pt(2));
		pt += height;
		glVertex3f(pt(0), pt(1), pt(2));
		glEnd();
	}
}
