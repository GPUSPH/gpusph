#ifndef _CYLINDER_H
#define	_CYLINDER_H

#include "Point.h"
#include "Vector.h"

class Cylinder {
	private:
		Point	center;
		Vector	radius, height;

	public:
		Cylinder(void);
		Cylinder(const Point &center, const Vector &radius, const Vector &height);
		~Cylinder(void) {};

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);

		void FillBorder(PointVect& points, double dx, bool bottom, bool top);
		void Fill(PointVect& points, double dx);

		void GLDraw(void);
};

#endif	/* _CYLINDER_H */

