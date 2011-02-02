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