/*
 * File:   Cube.h
 * Author: alexis
 *
 * Created on 14 juin 2008, 18:04
 */

#ifndef _CUBE_H
#define	_CUBE_H

#include "Point.h"
#include "Vector.h"

class Cube {
	private:
		Point	origin;
		Vector	vx, vy, vz;

	public:
		Cube(void);
		Cube(const Point& p, const Vector& v1, const Vector& v2, const Vector& v3);
		~Cube(void) {};

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);
		void FillBorder(PointVect& points, double dx, int face_num, bool* edges_to_fill);
		void FillBorder(PointVect& points, double dx, bool fill_top_face);
		void Fill(PointVect& points, double dx, bool fill_faces);
		void InnerFill(PointVect& points, double dx);
		void GLDrawQuad(const Point& p1, const Point& p2, const Point& p3, const Point& p4);
		void GLDraw(void);
};

#endif	/* _CUBE_H */

