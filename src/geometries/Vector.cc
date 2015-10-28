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


#include <cmath>
#include "Vector.h"
#include "Point.h"

/// Constructor
/*! Constructor from 2 points pnt1 and pnt2
	V = pnt1pnt2
	\param pnt1 : point1
	\param pnt2 : point2
*/
Vector::Vector(const Point &pnt1, const Point &pnt2)
{
	x[0] = pnt2(0) - pnt1(0);
	x[1] = pnt2(1) - pnt1(1);
	x[2] = pnt2(2) - pnt1(2);
}


/// Constructor
/*! Constructor from vector coordinates
	\param xx : x vector coordinate
	\param yy : y vector coordinate
*/
Vector::Vector(double xx, double yy, double zz)
{
	x[0] = xx;
	x[1] = yy;
	x[2] = zz;
}


/// Copy constructor
/*!	\param source : source data
*/
Vector::Vector(const Vector &source)
{
	x[0] = source.x[0];
	x[1] = source.x[1];
	x[2] = source.x[2];
}


/// Constructor from float3
/*!	\param pt : float3
  the fourth component (mass) is initialized to 0
*/
Vector::Vector(const float3 &v)
{
	x[0] = v.x;
	x[1] = v.y;
	x[2] = v.z;
	x[3] = 0;
}

/// Constructor from double3
/*!	\param pt : double3
  the fourth component (mass) is initialized to 0
*/
Vector::Vector(const double3 &v)
{
	x[0] = v.x;
	x[1] = v.y;
	x[2] = v.z;
	x[3] = 0;
}

/// Constructor from float4
/*!	\param pt : float4
*/
Vector::Vector(const float4 &v)
{
	x[0] = v.x;
	x[1] = v.y;
	x[2] = v.z;
	x[3] = v.w;
}

/// Constructor from double4
/*!	\param pt : double4
*/
Vector::Vector(const double4 &v)
{
	x[0] = v.x;
	x[1] = v.y;
	x[2] = v.z;
	x[3] = v.w;
}


/// Constructor from double coordinates array
/*!	\param xx : coordinates array
*/
Vector::Vector(const double *xx)
{
	x[0] = xx[0];
	x[1] = xx[1];
	x[2] = xx[2];
	x[3] = xx[3];
}


/// Constructor from float coordinates array
/*!	\param xx : coordinates array
*/
Vector::Vector(const float *xx)
{
	x[0] = xx[0];
	x[1] = xx[1];
	x[2] = xx[2];
	x[3] = xx[3];
}


Vector
Vector::Rot(const dMatrix3 rot)
{
	Vector res;
	res(0) = rot[0]*x[0] + rot[1]*x[1] + rot[2]*x[2];
	res(1) = rot[4]*x[0] + rot[5]*x[1] + rot[6]*x[2];
	res(2) = rot[8]*x[0] + rot[9]*x[1] + rot[10]*x[2];

	return res;
}


Vector
Vector::TransposeRot(const dMatrix3 rot)
{
	Vector res;
	res(0) = rot[0]*x[0] + rot[4]*x[1] + rot[8]*x[2];
	res(1) = rot[1]*x[0] + rot[5]*x[1] + rot[9]*x[2];
	res(2) = rot[2]*x[0] + rot[6]*x[1] + rot[10]*x[2];

	return res;
}


/// Return the norm of the vector
/*!	\return the norm of vector
*/
double
Vector::norm(void) const
{
	return sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
}


/// Normalize the vector
void
Vector::normalize(void)
{
	double n = norm();
	if (n) {
		x[0] /= n;
		x[1] /= n;
		x[2] /= n;
	}
}


double
Vector::normSquared(void) const
{
	return x[0]*x[0] + x[1]*x[1] + x[2]*x[2];
}


/// Return a normal vector of the vector
/*! \return a normal vector of the vector
*/
Vector
Vector::Normal(void) const
{
	Point p1, p2;
	Vector v;

	if (x[2] != 0) {
		v = Vector(-1.0, 1.0, -x[1]/x[2] + x[0]/x[2]);
		v.normalize();
	}
	else if (x[2] != 0) {
		v = Vector(0, 0, 1);
	}

	return v;
}


// cross product
Vector
Vector::cross(const Vector & v) const
{
	return Vector(x[1]*v(2) - x[2]*v(1), x[2]*v(0) - x[0]*v(2), x[0]*v(1) - x[1]*v(0));
}


Vector
Vector::rotated(const double &angle, const Vector &normal) const
{
	double vnorm = normal.norm();
	double ct = cos(angle);
	double st = sin(angle);
	double xx = normal(0),
		   yy = normal(1),
		   zz = normal(2);

	double a11, a12, a13, a21, a22, a23, a31, a32, a33;

	a11 = xx*xx + (yy*yy + zz*zz)*ct;
	a11 /= vnorm*vnorm;

	a22 = yy*yy + (xx*xx + zz*zz)*ct;
	a22 /= vnorm*vnorm;

	a33 = zz*zz + (xx*xx + yy*yy)*ct;
	a33 /= vnorm*vnorm;

	a12 = xx*yy*(1-ct)-zz*vnorm*st;
	a12 /= vnorm*vnorm;

	a21 = xx*yy*(1-ct)+zz*vnorm*st;
	a21 /= vnorm*vnorm;

	a13 = xx*zz*(1-ct)+yy*vnorm*st;
	a13 /= vnorm*vnorm;

	a31 = xx*zz*(1-ct)-yy*vnorm*st;
	a31 /= vnorm*vnorm;

	a23 = yy*zz*(1-ct)-xx*vnorm*st;
	a23 /= vnorm*vnorm;

	a32 = yy*zz*(1-ct)+xx*vnorm*st;
	a32 /= vnorm*vnorm;

	return Vector(
		a11*x[0]+a12*x[1]+a13*x[2],
		a21*x[0]+a22*x[1]+a23*x[2],
		a31*x[0]+a32*x[1]+a33*x[2]);
}


/// Affectation operator
/*! \param source : source vector
	\return this = source
*/
Vector &Vector::operator=(const Vector &source)
{
	x[0] = source.x[0];
	x[1] = source.x[1];
	x[2] = source.x[2];
	return *this;
}


/// Define the Vector2D+=Vector2D operator
/*!	\param vect : vector
	\return this = this + vect
*/
Vector &Vector::operator+=(const Vector &vect)
{
	x[0] += vect.x[0];
	x[1] += vect.x[1];
	x[2] += vect.x[2];
	return *this;
}


/// Define the Vector2D-=Vector2D operator
/*!	\param vect : vector
	\return this = this - vect
*/
Vector &Vector::operator-=(const Vector &vect)
{
	x[0] -= vect.x[0];
	x[1] -= vect.x[1];
	x[2] -= vect.x[2];
	return *this;
}


/// Define the Vector2D*=double operator
/*!	\param k : scalar
	\return vector = this*v
*/
Vector &Vector::operator*=(double k)
{
	x[0] *= k;
	x[1] *= k;
	x[2] *= k;
	return *this;
}


/// Define the Vector2D/=double operator
/*!	\param k : scalar
	\return vector = this/k
*/
Vector &Vector::operator/=(double k)
{
	x[0] /= k;
	x[1] /= k;
	x[2] /= k;
	return *this;
}


/// Return the value of coordinate i of vector
/*!	\param i : coordinate number
	\return reference to coordinate
*/
double &Vector::operator()(int i)
{
	return x[i];
}


/// Return the value of coordinate i of vector
/*!	\param i : coordinate number
	\return coordinate value
*/
double Vector::operator()(int i) const
{
	return x[i];
}


/// Define the + operator for Vector2D
/*!	\param vect1 : vector1
	\param vect2 : vector2
	\return vector = vect1 + vect2
*/
Vector operator+(const Vector &vect1, const Vector &vect2)
{
	Vector res = vect1;
	return res += vect2;
}


/// Define the - operator for Vector2D
/*!	\param vect1 : vector1
	\param vect2 : vector2
	\return vector = vect1 - vect2
*/
Vector operator-(const Vector &vect1, const Vector &vect2)
{
	Vector res = vect1;
	return res -= vect2;
}


/// Return opposite of vector
/*! \param vect : vector
	\return vector = -vect
*/
Vector operator-(const Vector &vect)
{
	Vector res = vect;
	return res *= -1;
}


/// Define the double*Vector2D operator
/*!	\param k : scalar
	\param vect : vector
	\return vector = k*vect
*/
Vector operator*(double k, const Vector &vect)
{
	Vector res = vect;
	return res *= k;
}


/// Define the Vector2D*double operator
/*!	\param k : scalar
	\param vect : vector
	\return vector = k*vect
*/
Vector operator*(const Vector &vect, double k)
{
	Vector res = vect;
	return res *= k;
}


/// Define the Vector2D*Vector2D (dot product operator)
/*!	\param vect1 : vector1
	\param vect2 : vector2
	\return vect1.vect2
*/
double operator*(const Vector &vect1, const Vector &vect2)
{
	return vect1.x[0]*vect2.x[0] + vect1.x[1]*vect2.x[1]  + vect1.x[2]*vect2.x[2];
}


/// Define the Vector2D/double operator
/*!	\param k : scalar
	\param vect : vector
	\return vector = vect/k
*/
Vector operator/(const Vector &vect, double k)
{
	Vector res = vect;
	return res /= k;
}


float3 make_float3(const Vector &v)
{
	return make_float3(float(v(0)), float(v(1)), float(v(2)));
}

double3 make_double3(const Vector &v)
{
	return make_double3(double(v(0)), double(v(1)), double(v(2)));
}

float4 make_float4(const Vector &v)
{
	return make_float4(float(v(0)), float(v(1)), float(v(2)), float(v(3)));
}

double4 make_double4(const Vector &v)
{
	return make_double4(double(v(0)), double(v(1)), double(v(2)), double(v(3)));
}


void make_dvector3(const Vector &v, dVector3 vec)
{
	vec[0] = dReal(v(0));
	vec[1] = dReal(v(1));
	vec[2] = dReal(v(2));
}


void make_dvector4(const Vector &v, dVector4 vec)
{
	vec[0] = dReal(v(0));
	vec[1] = dReal(v(1));
	vec[2] = dReal(v(2));
	vec[3] = dReal(v(3));
}


// DEBUG
#include <iostream>
void Vector::print(void)
{
	std::cout << "Vector (" << x[0] << ", " << x[1] << ", " << x[2] << ")\n";
	return;
}
