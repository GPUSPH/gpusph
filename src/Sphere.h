#ifndef _SPHERE_H
#define	_SPHERE_H

#include "Point.h"
#include "Vector.h"

class Sphere {
	private:
		Point	center;
		Vector	radius, height;

	public:
		Sphere(void);
		Sphere(const Point &center, const Vector &radius, const Vector &height);
		~Sphere(void) {};

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);

		void FillBorder(PointVect& points, double dx);
		void Fill(PointVect& points, double dx);

		void GLDraw(void);
};

#endif	/* _SPHERE_H */

