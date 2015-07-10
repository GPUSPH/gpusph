/*
 * File:   Torus.h
 * Author: alexis
 *
 * Created on 8 septembre 2011, 18:52
 */

#ifndef TORUS_H
#define	TORUS_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"


class Torus: public Object {
	private:
		double	m_R;
		double	m_r;

	public:
		Torus();
		Torus(const Point&, const Vector&, const double, const double);
		Torus(const Point&, const double, const double, const EulerParameters&);
		virtual ~Torus();

		double Volume(const double) const;
		void SetInertia(const double);

		void setEulerParameters(const EulerParameters &ep);
		void getBoundingBox(Point &output_min, Point &output_max);
		void shift(const double3 &offset);

		void FillBorder(PointVect&, const double);

		int Fill(PointVect&, const double, const bool fill = true);

		void FillIn(PointVect& points, const double dx, const int layers);

		bool IsInside(const Point &, const double) const;
};

#endif	/* TORUS_H */

