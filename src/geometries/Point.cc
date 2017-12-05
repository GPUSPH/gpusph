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
 *  Point.cpp
 *  NNS
 *
 *  Created by Alexis Herault on 27/07/06.
 *
 */

#include <cmath>

#include "Point.h"
#include "Vector.h"

/// Constructor
/*!	\param xx : x coordinate of point
	\param yy : y coordinate of point
*/
Point::Point(double xx, double yy, double zz, double m)
{
	x[0] = xx;
	x[1] = yy;
	x[2] = zz;
	x[3] = m;
}


/// Copy constructor
/*!	\param pnt : source point
*/
Point::Point(const Point &pnt)
{
	x[0] = pnt.x[0];
	x[1] = pnt.x[1];
	x[2] = pnt.x[2];
	x[3] = pnt.x[3];
}

/// Constructor from double3
/*!	\param pt : double3
  the fourth component (mass) is initialized to 0
*/
Point::Point(const double3 &pt)
{
	x[0] = pt.x;
	x[1] = pt.y;
	x[2] = pt.z;
	x[3] = 0;
}

/// Constructor from double4
/*!	\param pt : double4
*/
Point::Point(const double4 &pt)
{
	x[0] = pt.x;
	x[1] = pt.y;
	x[2] = pt.z;
	x[3] = pt.w;
}

/// Constructor from float3
/*!	\param pt : float3
  the fourth component (mass) is initialized to 0
*/
Point::Point(const float3 &pt)
{
	x[0] = pt.x;
	x[1] = pt.y;
	x[2] = pt.z;
	x[3] = 0;
}

/// Constructor from float4
/*!	\param pt : float4
*/
Point::Point(const float4 &pt)
{
	x[0] = pt.x;
	x[1] = pt.y;
	x[2] = pt.z;
	x[3] = pt.w;
}


/// Constructor from double coordinates array
/*!	\param xx : coordinates array
*/
Point::Point(const double *xx)
{
	x[0] = xx[0];
	x[1] = xx[1];
	x[2] = xx[2];
	x[3] = xx[3];
}


/// Constructor from float coordinates array
/*!	\param xx : coordinates array
*/
Point::Point(const float *xx)
{
	x[0] = xx[0];
	x[1] = xx[1];
	x[2] = xx[2];
	x[3] = xx[3];
}

double4 Point::toDouble4() const
{
	return make_double4(x[0], x[1], x[2], x[3]);
}

Point
Point::Rot(const dMatrix3 rot)
{
	Point res;
	res(0) = rot[0]*x[0] + rot[1]*x[1] + rot[2]*x[2];
	res(1) = rot[4]*x[0] + rot[5]*x[1] + rot[6]*x[2];
	res(2) = rot[8]*x[0] + rot[9]*x[1] + rot[10]*x[2];

	return res;
}


Point
Point::TransposeRot(const dMatrix3 rot)
{
	Point res;
	res(0) = rot[0]*x[0] + rot[4]*x[1] + rot[8]*x[2];
	res(1) = rot[1]*x[0] + rot[5]*x[1] + rot[9]*x[2];
	res(2) = rot[2]*x[0] + rot[6]*x[1] + rot[10]*x[2];

	return res;
}


void
Point::SetCoord(double *data)
{
	x[0] = data[0];
	x[1] = data[1];
	x[2] = data[2];
}

void Point::SetCoord(double X, double Y, double Z)
{
	x[0] = X;
	x[1] = Y;
	x[2] = Z;
}

void Point::SetMass(const double _newVal)
{
	x[3] = _newVal;
}

/// Squared distance from origin
/*!	\return the squared distance from origin
*/
double Point::DistSquared(void) const
{
	return (x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}


/// Distance from origin
/*!	\return the distance from origin
*/
double Point::Dist(void) const
{
	return sqrt(DistSquared());
}


/// Squared distance from point
/*!	\param pnt : point
	\return the squared distance from point
*/
double Point::DistSquared(const Point &pnt) const
{
	return (pnt.x[0] - x[0])*(pnt.x[0] - x[0]) + (pnt.x[1] - x[1])*(pnt.x[1] - x[1])
				+ (pnt.x[2] - x[2])*(pnt.x[2] - x[2]);
}


/// Distance from point
/*!	\param pnt : point
	\return the distance from point
*/
double Point::Dist(const Point &pnt) const
{
	return sqrt(DistSquared(pnt));
}


/// Affectation operator
/*!	\param source : source point
	\return point = source
*/
Point &Point::operator=(const Point &source)
{
	x[0] = source.x[0];
	x[1] = source.x[1];
	x[2] = source.x[2];
	x[3] = source.x[3];
	return *this;
}


/// Affectation operator for coordinates array
/*!	\param source : source data
	\return point = source
*/
Point &Point::operator=(double *source)
{
	x[0] = source[0];
	x[1] = source[1];
	x[2] = source[2];
	x[3] = source[3];
	return *this;
}


/// Return a reference to value of coordinate i
/*!	\param i : coordinate number
	\return reference to coordinate
*/
double &Point::operator() (int i)
{
	return x[i];
}


/// Return the value of coordinate i
/*!	\param i : coordinate number
	\return coordinate value
*/
double Point::operator() (int i) const
{
	return x[i];
}


/// Define the Point+=Point operator
/*!	\param pnt : point
	\return this = this + point
*/
Point &Point::operator+=(const Point &pnt)
{
	x[0] += pnt.x[0];
	x[1] += pnt.x[1];
	x[2] += pnt.x[2];
	return *this;
}


/// Define the Point+=Vector2D operator
/*!	\param vect : vect
	\return this = this + vect
*/
Point &Point::operator+=(const Vector &vect)
{
	x[0] += vect(0);
	x[1] += vect(1);
	x[2] += vect(2);
	return *this;
}


/// Define the Point+=double operator
/*!	\param vect : dbl
	\return this = this + dbl
*/
Point &Point::operator+=(const double &dbl)
{
	x[0] += dbl;
	x[1] += dbl;
	x[2] += dbl;
	return *this;
}


/// Define the Point-=Point operator
/*!	\param pnt : point
	\return this = this - pnt
*/
Point &Point::operator-=(const Point &pnt)
{
	x[0] -= pnt.x[0];
	x[1] -= pnt.x[1];
	x[2] -= pnt.x[2];
	return *this;
}


/// Define the Point-=Vector2D operator
/*!	\param vect : vect
	\return this = this - vect
*/
Point &Point::operator-=(const Vector &vect)
{
	x[0] -= vect(0);
	x[1] -= vect(1);
	x[2] -= vect(2);
	return *this;
}


/// Define the Point-=double operator
/*!	\param vect : dbl
	\return this = this - dbl
*/
Point &Point::operator-=(const double &dbl)
{
	x[0] -= dbl;
	x[1] -= dbl;
	x[2] -= dbl;
	return *this;
}


/// Define the Point*=double operator
/*!	\param k : scalarr
	\return point = this*k;
*/
Point &Point::operator*=(double k)
{
	x[0] *= k;
	x[1] *= k;
	x[2] *= k;
	return *this;
}


/// Define the Point/=double operator
/*!	\param k : scalar
	\return point = this/k
*/
Point &Point::operator/=(double k)
{
	x[0] /= k;
	x[1] /= k;
	x[2] /= k;
	return *this;
}


/// Define the + operator for Point
/*!	\param pnt1 : point1
	\param pnt2 : point2
	\return point = point1 + point2
*/
Point operator+(const Point &pnt1, const Point &pnt2)
{
	Point res = pnt1;
	return res += pnt2;
}


/// Define the + for Point and Vector2D
/*!	Translate the point by vector.
	\param pnt : point
	\param vect : translation vector
	\return point = pnt + vect
*/
Point operator+(const Point &pnt, const Vector &vect)
{
	Point	res = pnt;
	return res += vect;
}


/// Define the + for Point and Vector2D
/*!	Translate the point by vector.
	\param pnt : point
	\param vect : translation vector
	\return point = pnt + vect
*/
Point operator-(const Point &pnt, const Vector &vect)
{
	Point	res = pnt;
	return res -= vect;
}


/// Define the - operator for Point
/*! \param pnt1 : point1
	\param pnt2 : point2
	\return point = pnt1 - pnt2
*/
Point operator-(const Point &pnt1, const Point &pnt2)
{
	Point res = pnt1;
	return res -= pnt2;
}


/// Define the opposite operator
/*!	Take the opposite of coordinates of point pnt.
	\param pnt : point
	\return point = - pnt
*/
Point operator-(const Point &pnt)
{
	Point res = pnt;
	return res *= -1;
}


/// Define the * operator for double and Point
/*!	Multiply point coordinates by k
	\param k : scalar
	\param pnt :  point
	\return point = k*pnt
*/
Point operator*(double k, const Point &pnt)
{
	Point res = pnt;
	return res *= k;
}


/// Define the / operator for Point and double
/*!	Divide point coordinates by k
	\param pnt : point
	\param k : scalar
	\return point = pnt/k
*/
Point operator/(const Point &pnt, double k)
{
	Point res = pnt;
	return res /= k;
}


/// Calculate the distance between two points
/*!	\param pnt1 : point1
	\param pnt2 : point2
	\return distance bewteen pnt1 and pnt2
*/
double dist(const Point &pnt1, const Point &pnt2)
{
	return pnt1.Dist(pnt2);
}


/// Calculate the squared distance between two points
/*!	\param pnt1 : point1
	\param pnt2 : point2
	\return squared distance bewteen pnt1 and pnt2
*/
double distsq(const Point &pnt1, const Point &pnt2)
{
	return pnt1.DistSquared(pnt2);
}


float4 make_float4(const Point &pt)
{
	return make_float4(float(pt(0)), float(pt(1)), float(pt(2)), float(pt(3)));
}

double4 make_double4(const Point &pt)
{
	return make_double4(pt(0), pt(1), pt(2), pt(3));
}

float3 make_float3(const Point &pt)
{
	return make_float3(float(pt(0)), float(pt(1)), float(pt(2)));
}

double3 make_double3(const Point &pt)
{
	return make_double3(pt(0), pt(1), pt(2));
}


void make_dvector3(const Point &pt, dVector3 vec)
{
	vec[0] = dReal(pt(0));
	vec[1] = dReal(pt(1));
	vec[2] = dReal(pt(2));
}


void make_dvector4(const Point &pt, dVector4 vec)
{
	vec[0] = dReal(pt(0));
	vec[1] = dReal(pt(1));
	vec[2] = dReal(pt(2));
	vec[3] = dReal(pt(3));
}

// DEBUG
#include <iostream>
void Point::print(void)
{
	std::cout << "Point (" << x[0] << ", " << x[1] << ", " << x[2] << ") mass = " << x[3] << "\n";
	return;
}
