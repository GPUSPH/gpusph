/*
 *  Line.h
 *  NNS
 *
 *  Created by Alexis Herault on 27/07/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */


#ifndef _SEGMENT_H_
#define _SEGMENT_H_

#include "Point.h"
#include "Vector.h"


/// 2D segment object.
/*!
2D Segment class provide :
	- segment creation
	- distance of a Point from segment
	- projection of a Point on the segment
	- projection of a Point on the normal direction of segment
	- direction of the segment
	- normal direction of the segment
	- filling the segment with particles
	- bounding box of the segment
*/
class Segment {
	private:
		double	length;		///< length
		Point	A, B;		///< starting and ending Point
		Point	G;			///< barycentre
		Vector	u, n;		///< unitary director and normal vectors

	public:
		Segment(void) {};
		Segment(const Point &, const Point &);
		~Segment(void) {};

		/*! \name
			Geometrical functions
		*/
		//\{
		double Dist(const Point &);
		double Proj(const Point &);
		double NormalProj(const Point &);
		bool Inside(const Point &, double);
		Vector Dir(void);
		Vector Normal(void);
		double Length(void) { return length;};

		//\}

		/*! \name
			Filling functions
		*/
		//\{
		void FillBorder(PointVect &, double);
		void FillBorder(PointVect &, double, bool, bool);
		void BoundingBox(double *, double);
		//\}
};
#endif
