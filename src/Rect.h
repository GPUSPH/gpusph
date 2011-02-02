/*
 * File:   Rect.h
 * Author: alexis
 *
 * Created on 14 juin 2008, 17:04
 */

#ifndef _RECT_H
#define	_RECT_H

#include "Point.h"
#include "Vector.h"

class Rect {
	private:
		Point   origin;
		Vector  vx, vy;

	public:
		Rect(void);
		Rect(const Point& p, const Vector& v1, const Vector& v2);
		~Rect(void) {};

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);
		void FillBorder(PointVect& points, double dx, bool fill_top = true);
		void FillBorder(PointVect& points, double dx, bool populate_first,
				bool populate_last, int edge_num);
		void Fill(PointVect& points, double dx, bool fill_edges);
		void Fill(PointVect& points, double dx, bool* edges_to_fill);
		void GLDrawQuad(const Point& p1, const Point& p2, const Point& p3, const Point& p4);
		void GLDraw(void);
};
#endif	/* _RECT_H */
