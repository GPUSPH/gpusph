#ifndef _CIRCLE_H
#define	_CIRCLE_H

#include "Point.h"
#include "Vector.h"

#define PI 3.14159265358979323846

class Circle {
	private:
		Point	center;
		Vector	radius;
		Vector	normal;

	public:
		Circle(void);
		Circle(const Point &p, const Vector &r, const Vector &u);
		~Circle(void) {};

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);
		void FillBorder(PointVect& points, double dx);
		void Fill(PointVect& points, double dx, bool fill_edge=true);
		void GLDraw(void);

};
#endif	/* _CIRCLE_H */


