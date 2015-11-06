/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

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
 *  Point.h
 *
 *  Created by Alexis Herault on 27/07/06.
 *
 */

#ifndef _POINT_H
#define _POINT_H

#include "vector_math.h"

#include <vector>
#include "Vector.h"
#include "ode/ode.h"

class Vector;

/// 3D point classe.
/*!	This class provide :
		- standard operators for point coordinates
		- translation of vector V
		- distance calculation between points
		- access to coordinates values
*/
class Point {
	private:
		double	x[4];		///< coordinates of point and mass

	public:
		Point(double xx = 0, double yy = 0, double zz = 0, double m = 0);
		Point(const Point &);
		Point(const double3 &);
		Point(const double4 &);
		Point(const float3 &);
		Point(const float4 &);
		Point(const float *);
		Point(const double *);
		~Point(void) {};

		double4 toDouble4() const;

		void SetCoord(double *);
		void SetCoord(double, double, double);
		void SetMass(const double _newVal);

		Point Rot(const dMatrix3);
		Point TransposeRot(const dMatrix3);

		/*! \name
			Distance calculation
		*/
		//\{
		double Dist(void) const;
		double DistSquared(void) const;

		double Dist(const Point &) const;
		double DistSquared(const Point &) const;
		//\}

		/*! \name
			Overloaded operators
		*/
		//\{
		Point   &operator=(const Point &);
		Point   &operator=(double *);
		Point   &operator+=(const Point &);
		Point   &operator+=(const Vector &);
		Point   &operator+=(const double &);
		Point   &operator-=(const Point &);
		Point	&operator-=(const Vector &);
		Point   &operator-=(const double &);
		Point	&operator*=(double);
		Point	&operator/=(double);
		double  &operator()(int);
		double  operator()(int) const;
		//\}


		/*! \name
			Overloaded operators
		*/
		//\{
		friend Point operator+(const Point &, const Point &);
		friend Point operator+(const Point &, const Vector &);
		friend Point operator-(const Point &, const Point &);
		friend Point operator-(const Point &, const Vector &);
		friend Point operator*(double, const Point &);
		friend Point operator/(const Point &, double);
		friend Point operator-(const Point &);
		//\}

		/*! \name
			Distance calculation
		*/
		//\{
		friend double dist(const Point &, const Point &);
		friend double distsq(const Point &, const Point &);
		//\}

		// DEBUG
		void print(void);
};

float4 make_float4(const Point &);
double4 make_double4(const Point &);

float3 make_float3(const Point &);
double3 make_double3(const Point &);

void make_dvector3(const Point &, dVector3);

void make_dvector4(const Point &, dVector4);

typedef std::vector<Point> PointVect;

// Small utility functions, useful for bbox and world size computation
// Writes in pmin and pmax the per-element minima and maxima
inline void setMinMaxPerElement(Point &pmin, Point &pmax, Point a)
{
	for (uint elem = 0; elem < 3; elem++) {
		pmin(elem) = fmin( pmin(elem), a(elem));
		pmax(elem) = fmax( pmax(elem), a(elem));
	}
}
// Writes in pmin the per-element minima
inline void setMinPerElement(Point &pmin, Point a)
{
	for (uint elem = 0; elem < 3; elem++)
		pmin(elem) = fmin( pmin(elem), a(elem));
}
// Writes in pmax the per-element maxima
inline void setMaxPerElement(Point &pmax, Point a)
{
	for (uint elem = 0; elem < 3; elem++)
		pmax(elem) = fmax( pmax(elem), a(elem));
}

#endif
