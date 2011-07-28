/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
*/

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
