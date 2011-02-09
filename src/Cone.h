// Created by Andrew 12/2009
#ifndef _CONE_H
#define	_CONE_H

#include "Point.h"
#include "Vector.h"
#include "Circle.h"

#define PI 3.14159265358979323846

// TODO: the Cone object is broken, fix it

class Cone {
	private:
		Point	center;
		Vector	radiust, radiusb, height;

	public:
		Cone(void);
		Cone(const Point &center, const Vector &radiusb, const Vector &radiust, const Vector &height);
		~Cone(void) {};

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);

		void FillBorder(PointVect& points, double dx, bool bottom, bool top);
		void Fill(PointVect& points, double dx);

		void GLDraw(void);
};

#endif	/* _CONE_H */
