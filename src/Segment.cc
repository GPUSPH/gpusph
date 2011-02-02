/*
 *  Line.cpp
 *  NNS
 *
 *  Created by Alexis Herault on 27/07/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */

#include <math.h>

#include "Segment.h"
#include "Point.h"
#include "Vector.h"

/*! Constructor from 2 points pnt1 and pnt2.
	The segement is oriented from pnt1 to pnt2.
	\param pnt1 : constant reference to pnt1
	\param pnt2 : constant reference to pnt2
*/
Segment::Segment(const Point &pnt1, const Point &pnt2)
{
	// Setting corner points
	A = pnt1;
	B = pnt2;

	// Setting barycentre
	G = (A + B)/2;
	u = Vector(G, B);
	length = u.norm();
	u = u/length;
	n = u.Normal();
	return;
}


/*! Distance of point point pnt from
	\param pnt : constant reference to point pnt
	\return distance from segment
*/
double Segment::Dist(const Point &pnt)
{
	double l = u*Vector(G, pnt)/length;
	double h = n*Vector(A, pnt);

	if (l > 1)
		return B.Dist(pnt);
	else if (l < -1)
		return A.Dist(pnt);
	else
		return fabs(h);
}


/*! Orthogonal projection of point pnt on segment
	\param pnt : constant reference to point pnt
	\return distance from segment barycenter and orthogonal projection of pnt
	on the segment
*/
double Segment::Proj(const Point &pnt)
{
	return u*Vector(G, pnt)/length;
}


/*! Orthogonal projection of point p on normal direction of segment
	\param pnt : constant reference to point pnt
	\return distance from segment and orthogonal projection of pnt
	on normal direction of the segment
*/
double Segment::NormalProj(const Point &pnt)
{
	return n*Vector(A, pnt);
}


/// Direction of segment
/*!	\return direction of segment
*/
Vector Segment::Dir()
{
	return u;
}


/// Normal direction of segment
/*!	\return normal direction of segment
*/
Vector Segment::Normal()
{
	return n;
}


bool Segment::Inside(const Point &pnt, double dx)
{
	return Dist(pnt) == 0;
}


/// Fill the segment with particles
/*!	Fill the segment with copy of particle every dx and add
	the new particles to simulation particles vector and
	eventually to boundary particle vector.
	\param part : particle to duplicate
	\param partvect : particles vector
	\param dx : distance between two particles on segment
	\param bound_partvect : boundary particle vector
	\param populate_first : if true, put a particle on first point of segment
	\param populate_last : if true, put a particle on last point of segment
	First and last point of segment are defined according to segment orientation.
*/
void Segment::FillBorder(PointVect& points, double dx,
						bool populate_first, bool populate_last)
{
	int nparts = (int) (2*length/dx);

	int idx1 = 1;
	int idx2 = nparts - 1;
	if (populate_last)
		idx2++;
	if (populate_first)
		idx1--;
	for (double i = idx1; i <= idx2; i++) {
		double l = 2*i*length/nparts;
		Point P = A + l*u;
		points.push_back(P);
		}
	return;
}




/// Return the bounding box of segment. A REFAIRE !!!
/*!	\param box : array for bounding box coordinates
	\param dx : grow the bounding box of dx in each direction
*/
void Segment::BoundingBox(double *box, double dx)
{
	box[0] = std::min(A(0), B(0)) - dx;
	box[1] = std::max(A(0), B(0)) + dx;
	box[2] = std::min(A(1), B(1)) - dx;
	box[3] = std::max(A(1), B(1)) + dx;
	box[4] = std::min(A(2), B(2)) - dx;
	box[5] = std::max(A(3), B(3)) + dx;
	return;
}
