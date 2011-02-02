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
