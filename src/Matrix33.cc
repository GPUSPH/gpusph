// Based on
// Geometric Tools, LLC
// Copyright (c) 1998-2011
// Distributed under the Boost Software License, Version 1.0.
// http://www.boost.org/LICENSE_1_0.txt
// http://www.geometrictools.com/License/Boost/LICENSE_1_0.txt
//
//----------------------------------------------------------------------------

#include "Matrix33.h"
#include <math.h>

Matrix33::Matrix33(void)
{
	MakeZero();
}


Matrix33::Matrix33(const Matrix33& mat)
{
	m_mat[0] = mat.m_mat[0];
	m_mat[1] = mat.m_mat[1];
	m_mat[2] = mat.m_mat[2];
	m_mat[3] = mat.m_mat[3];
	m_mat[4] = mat.m_mat[4];
	m_mat[5] = mat.m_mat[5];
	m_mat[6] = mat.m_mat[6];
	m_mat[7] = mat.m_mat[7];
	m_mat[8] = mat.m_mat[8];
}


Matrix33::Matrix33(double m00, double m01, double m02, double m10, double m11,
					double m12, double m20, double m21, double m22)
{
	m_mat[0] = m00;
	m_mat[1] = m01;
	m_mat[2] = m02;
	m_mat[3] = m10;
	m_mat[4] = m11;
	m_mat[5] = m12;
	m_mat[6] = m20;
	m_mat[7] = m21;
	m_mat[8] = m22;
}


Matrix33::Matrix33(double m00, double m11, double m22)
{
	MakeDiagonal(m00, m11, m22);
}


Matrix33::Matrix33(const Vector& axis, double angle)
{
	MakeRotation(axis, angle);
}


Matrix33& Matrix33::operator=(const Matrix33& mat)
{
	m_mat[0] = mat.m_mat[0];
	m_mat[1] = mat.m_mat[1];
	m_mat[2] = mat.m_mat[2];
	m_mat[3] = mat.m_mat[3];
	m_mat[4] = mat.m_mat[4];
	m_mat[5] = mat.m_mat[5];
	m_mat[6] = mat.m_mat[6];
	m_mat[7] = mat.m_mat[7];
	m_mat[8] = mat.m_mat[8];
	return *this;
}


Matrix33& Matrix33::MakeZero()
{
	m_mat[0] = 0.0;
	m_mat[1] = 0.0;
	m_mat[2] = 0.0;
	m_mat[3] = 0.0;
	m_mat[4] = 0.0;
	m_mat[5] = 0.0;
	m_mat[6] = 0.0;
	m_mat[7] = 0.0;
	m_mat[8] = 0.0;
    return *this;
}


Matrix33& Matrix33::MakeIdentity()
{
	m_mat[0] = 1.0;
	m_mat[1] = 0.0;
	m_mat[2] = 0.0;
	m_mat[3] = 0.0;
	m_mat[4] = 1.0;
	m_mat[5] = 0.0;
	m_mat[6] = 0.0;
	m_mat[7] = 0.0;
	m_mat[8] = 1.0;
    return *this;
}


Matrix33& Matrix33::MakeDiagonal(double m00, double m11, double m22)
{
	m_mat[0] = m00;
	m_mat[1] = 0.0;
	m_mat[2] = 0.0;
	m_mat[3] = 0.0;
	m_mat[4] = m11;
	m_mat[5] = 0.0;
	m_mat[6] = 0.0;
	m_mat[7] = 0.0;
	m_mat[8] = m22;
    return *this;
}


Matrix33& Matrix33::MakeRotation(const Vector& axis, double angle)
{
	double cs = cos(angle);
	double sn = sin(angle);
	double oneMinusCos = 1.0 - cs;
	double x2 = axis(0)*axis(1);
	double y2 = axis(1)*axis(1);
	double z2 = axis(2)*axis(2);
	double xym = axis(0)*axis(1)*oneMinusCos;
	double xzm = axis(0)*axis(2)*oneMinusCos;
	double yzm = axis(1)*axis(2)*oneMinusCos;
	double xSin = axis(0)*sn;
	double ySin = axis(1)*sn;
	double zSin = axis(2)*sn;

	m_mat[0] = x2*oneMinusCos + cs;
	m_mat[1] = xym - zSin;
	m_mat[2] = xzm + ySin;
	m_mat[3] = xym + zSin;
	m_mat[4] = y2*oneMinusCos + cs;
	m_mat[5] = yzm - xSin;
	m_mat[6] = xzm - ySin;
	m_mat[7] = yzm + xSin;
	m_mat[8] = z2*oneMinusCos + cs;
	return *this;
}


Matrix33 Matrix33::operator+ (const Matrix33& mat) const
{
	return Matrix33
	(
		m_mat[0] + mat.m_mat[0],
		m_mat[1] + mat.m_mat[1],
		m_mat[2] + mat.m_mat[2],
		m_mat[3] + mat.m_mat[3],
		m_mat[4] + mat.m_mat[4],
		m_mat[5] + mat.m_mat[5],
		m_mat[6] + mat.m_mat[6],
		m_mat[7] + mat.m_mat[7],
		m_mat[8] + mat.m_mat[8]
	);
}


Matrix33 Matrix33::operator- (const Matrix33& mat) const
{
	return Matrix33
	(
		m_mat[0] - mat.m_mat[0],
		m_mat[1] - mat.m_mat[1],
		m_mat[2] - mat.m_mat[2],
		m_mat[3] - mat.m_mat[3],
		m_mat[4] - mat.m_mat[4],
		m_mat[5] - mat.m_mat[5],
		m_mat[6] - mat.m_mat[6],
		m_mat[7] - mat.m_mat[7],
		m_mat[8] - mat.m_mat[8]
	);
}


Matrix33 Matrix33::operator* (double scalar) const
{
	return Matrix33
	(
		scalar*m_mat[0],
		scalar*m_mat[1],
		scalar*m_mat[2],
		scalar*m_mat[3],
		scalar*m_mat[4],
		scalar*m_mat[5],
		scalar*m_mat[6],
		scalar*m_mat[7],
		scalar*m_mat[8]
	);
}


inline Matrix33 operator* (double scalar, const Matrix33& mat)
{
	return mat*scalar;
}


Matrix33 Matrix33::operator/ (double scalar) const
{
	double invscalar = 1.0/scalar;

	return Matrix33
	(
		invscalar*m_mat[0],
		invscalar*m_mat[1],
		invscalar*m_mat[2],
		invscalar*m_mat[3],
		invscalar*m_mat[4],
		invscalar*m_mat[5],
		invscalar*m_mat[6],
		invscalar*m_mat[7],
		invscalar*m_mat[8]
	);
}


Matrix33 Matrix33::operator- () const
{
	return Matrix33
	(
		-m_mat[0],
		-m_mat[1],
		-m_mat[2],
		-m_mat[3],
		-m_mat[4],
		-m_mat[5],
		-m_mat[6],
		-m_mat[7],
		-m_mat[8]
	);
}


Matrix33& Matrix33::operator+= (const Matrix33& mat)
{
	m_mat[0] += mat.m_mat[0];
	m_mat[1] += mat.m_mat[1];
	m_mat[2] += mat.m_mat[2];
	m_mat[3] += mat.m_mat[3];
	m_mat[4] += mat.m_mat[4];
	m_mat[5] += mat.m_mat[5];
	m_mat[6] += mat.m_mat[6];
	m_mat[7] += mat.m_mat[7];
	m_mat[8] += mat.m_mat[8];
	return *this;
}


Matrix33& Matrix33::operator-= (const Matrix33& mat)
{
	m_mat[0] -= mat.m_mat[0];
	m_mat[1] -= mat.m_mat[1];
	m_mat[2] -= mat.m_mat[2];
	m_mat[3] -= mat.m_mat[3];
	m_mat[4] -= mat.m_mat[4];
	m_mat[5] -= mat.m_mat[5];
	m_mat[6] -= mat.m_mat[6];
	m_mat[7] -= mat.m_mat[7];
	m_mat[8] -= mat.m_mat[8];
	return *this;
}


Matrix33& Matrix33::operator*= (double scalar)
{
	m_mat[0] *= scalar;
	m_mat[1] *= scalar;
	m_mat[2] *= scalar;
	m_mat[3] *= scalar;
	m_mat[4] *= scalar;
	m_mat[5] *= scalar;
	m_mat[6] *= scalar;
	m_mat[7] *= scalar;
	m_mat[8] *= scalar;
	return *this;
}


Matrix33& Matrix33::operator/= (double scalar)
{
	double invScalar = 1.0/scalar;

	m_mat[0] *= invScalar;
	m_mat[1] *= invScalar;
	m_mat[2] *= invScalar;
	m_mat[3] *= invScalar;
	m_mat[4] *= invScalar;
	m_mat[5] *= invScalar;
	m_mat[6] *= invScalar;
	m_mat[7] *= invScalar;
	m_mat[8] *= invScalar;
	return *this;
}


Vector Matrix33::operator* (const Vector& vec) const
{
	return Vector
	(
		m_mat[0]*vec(0) + m_mat[1]*vec(1) + m_mat[2]*vec(2),
		m_mat[3]*vec(0) + m_mat[4]*vec(1) + m_mat[5]*vec(2),
		m_mat[6]*vec(0) + m_mat[7]*vec(1) + m_mat[8]*vec(2)
	);
}


inline Vector operator* (const Vector& vec, const Matrix33& mat)
{
    return Vector
    (
        mat.m_mat[0]*vec(0) + mat.m_mat[3]*vec(1) + mat.m_mat[6]*vec(2),
        mat.m_mat[1]*vec(0) + mat.m_mat[4]*vec(1) + mat.m_mat[7]*vec(2),
        mat.m_mat[2]*vec(0) + mat.m_mat[5]*vec(1) + mat.m_mat[8]*vec(2)
    );
}

Matrix33 Matrix33::Transpose() const
{
	return Matrix33
	(
        m_mat[0],
        m_mat[3],
        m_mat[6],
        m_mat[1],
        m_mat[4],
        m_mat[7],
        m_mat[2],
        m_mat[5],
        m_mat[8]
	);
}


Matrix33 Matrix33::operator* (const Matrix33& mat) const
{
	// A*B
	return Matrix33
	(
		m_mat[0]*mat.m_mat[0] +
		m_mat[1]*mat.m_mat[3] +
		m_mat[2]*mat.m_mat[6],

		m_mat[0]*mat.m_mat[1] +
		m_mat[1]*mat.m_mat[4] +
		m_mat[2]*mat.m_mat[7],

		m_mat[0]*mat.m_mat[2] +
		m_mat[1]*mat.m_mat[5] +
		m_mat[2]*mat.m_mat[8],

		m_mat[3]*mat.m_mat[0] +
		m_mat[4]*mat.m_mat[3] +
		m_mat[5]*mat.m_mat[6],

		m_mat[3]*mat.m_mat[1] +
		m_mat[4]*mat.m_mat[4] +
		m_mat[5]*mat.m_mat[7],

		m_mat[3]*mat.m_mat[2] +
		m_mat[4]*mat.m_mat[5] +
		m_mat[5]*mat.m_mat[8],

		m_mat[6]*mat.m_mat[0] +
		m_mat[7]*mat.m_mat[3] +
		m_mat[8]*mat.m_mat[6],

		m_mat[6]*mat.m_mat[1] +
		m_mat[7]*mat.m_mat[4] +
		m_mat[8]*mat.m_mat[7],

		m_mat[6]*mat.m_mat[2] +
		m_mat[7]*mat.m_mat[5] +
		m_mat[8]*mat.m_mat[8]
     );
}


Matrix33 Matrix33::TimesTranspose(const Matrix33 &mat) const
{
    // A*B^T
	return Matrix33
	(
		m_mat[0]*mat.m_mat[0] +
		m_mat[1]*mat.m_mat[1] +
		m_mat[2]*mat.m_mat[2],

		m_mat[0]*mat.m_mat[3] +
		m_mat[1]*mat.m_mat[4] +
		m_mat[2]*mat.m_mat[5],

		m_mat[0]*mat.m_mat[6] +
		m_mat[1]*mat.m_mat[7] +
		m_mat[2]*mat.m_mat[8],

		m_mat[3]*mat.m_mat[0] +
		m_mat[4]*mat.m_mat[1] +
		m_mat[5]*mat.m_mat[2],

		m_mat[3]*mat.m_mat[3] +
		m_mat[4]*mat.m_mat[4] +
		m_mat[5]*mat.m_mat[5],

		m_mat[3]*mat.m_mat[6] +
		m_mat[4]*mat.m_mat[7] +
		m_mat[5]*mat.m_mat[8],

		m_mat[6]*mat.m_mat[0] +
		m_mat[7]*mat.m_mat[1] +
		m_mat[8]*mat.m_mat[2],

		m_mat[6]*mat.m_mat[3] +
		m_mat[7]*mat.m_mat[4] +
		m_mat[8]*mat.m_mat[5],

		m_mat[6]*mat.m_mat[6] +
		m_mat[7]*mat.m_mat[7] +
		m_mat[8]*mat.m_mat[8]
	);
}


Matrix33 Matrix33::TransposeTimes(const Matrix33 &mat) const
{
    // A^T*B
	return Matrix33
	(
		m_mat[0]*mat.m_mat[0] +
		m_mat[3]*mat.m_mat[3] +
		m_mat[6]*mat.m_mat[6],

		m_mat[0]*mat.m_mat[1] +
		m_mat[3]*mat.m_mat[4] +
		m_mat[6]*mat.m_mat[7],

		m_mat[0]*mat.m_mat[2] +
		m_mat[3]*mat.m_mat[5] +
		m_mat[6]*mat.m_mat[8],

		m_mat[1]*mat.m_mat[0] +
		m_mat[4]*mat.m_mat[3] +
		m_mat[7]*mat.m_mat[6],

		m_mat[1]*mat.m_mat[1] +
		m_mat[4]*mat.m_mat[4] +
		m_mat[7]*mat.m_mat[7],

		m_mat[1]*mat.m_mat[2] +
		m_mat[4]*mat.m_mat[5] +
		m_mat[7]*mat.m_mat[8],

		m_mat[2]*mat.m_mat[0] +
		m_mat[5]*mat.m_mat[3] +
		m_mat[8]*mat.m_mat[6],

		m_mat[2]*mat.m_mat[1] +
		m_mat[5]*mat.m_mat[4] +
		m_mat[8]*mat.m_mat[7],

		m_mat[2]*mat.m_mat[2] +
		m_mat[5]*mat.m_mat[5] +
		m_mat[8]*mat.m_mat[8]
	);
}


Matrix33 Matrix33::Inverse(const double epsilon) const
{
	// Invert a 3x3 using cofactors.  This is faster than using a generic
	// Gaussian elimination because of the loop overhead of such a method.

	Matrix33 inverse;

	// Compute the adjoint.
	inverse.m_mat[0] = m_mat[4]*m_mat[8] - m_mat[5]*m_mat[7];
	inverse.m_mat[1] = m_mat[2]*m_mat[7] - m_mat[1]*m_mat[8];
	inverse.m_mat[2] = m_mat[1]*m_mat[5] - m_mat[2]*m_mat[4];
	inverse.m_mat[3] = m_mat[5]*m_mat[6] - m_mat[3]*m_mat[8];
	inverse.m_mat[4] = m_mat[0]*m_mat[8] - m_mat[2]*m_mat[6];
	inverse.m_mat[5] = m_mat[2]*m_mat[3] - m_mat[0]*m_mat[5];
	inverse.m_mat[6] = m_mat[3]*m_mat[7] - m_mat[4]*m_mat[6];
	inverse.m_mat[7] = m_mat[1]*m_mat[6] - m_mat[0]*m_mat[7];
	inverse.m_mat[8] = m_mat[0]*m_mat[4] - m_mat[1]*m_mat[3];

	double det = m_mat[0]*inverse.m_mat[0] + m_mat[1]*inverse.m_mat[3] +
	m_mat[2]*inverse.m_mat[6];

	if (abs(det) > epsilon)
	{
		double invDet = 1.0/det;
		inverse.m_mat[0] *= invDet;
		inverse.m_mat[1] *= invDet;
		inverse.m_mat[2] *= invDet;
		inverse.m_mat[3] *= invDet;
		inverse.m_mat[4] *= invDet;
		inverse.m_mat[5] *= invDet;
		inverse.m_mat[6] *= invDet;
		inverse.m_mat[7] *= invDet;
		inverse.m_mat[8] *= invDet;
		return inverse;
	}

	inverse.m_mat[0] = 0.0;
	inverse.m_mat[1] = 0.0;
	inverse.m_mat[2] = 0.0;
	inverse.m_mat[3] = 0.0;
	inverse.m_mat[4] = 0.0;
	inverse.m_mat[5] = 0.0;
	inverse.m_mat[6] = 0.0;
	inverse.m_mat[7] = 0.0;
	inverse.m_mat[8] = 0.0;
	return inverse;
}


double Matrix33::Determinant(void) const
{
	double co00 = m_mat[4]*m_mat[8] - m_mat[5]*m_mat[7];
	double co10 = m_mat[5]*m_mat[6] - m_mat[3]*m_mat[8];
	double co20 = m_mat[3]*m_mat[7] - m_mat[4]*m_mat[6];
	double det = m_mat[0]*co00 + m_mat[1]*co10 + m_mat[2]*co20;
	return det;
}


void Matrix33::ExtractAxisAngle(Vector& axis, double& angle) const
{
	// Let (x,y,z) be the unit-length axis and let A be an angle of rotation.
	// The rotation matrix is R = I + sin(A)*P + (1-cos(A))*P^2 where
	// I is the identity and
	//
	//       +-        -+
	//   P = |  0 -z +y |
	//       | +z  0 -x |
	//       | -y +x  0 |
	//       +-        -+
	//
	// If A > 0, R represents a counterclockwise rotation about the axis in
	// the sense of looking from the tip of the axis vector towards the
	// origin.  Some algebra will show that
	//
	//   cos(A) = (trace(R)-1)/2  and  R - R^t = 2*sin(A)*P
	//
	// In the event that A = pi, R-R^t = 0 which prevents us from extracting
	// the axis through P.  Instead note that R = I+2*P^2 when A = pi, so
	// P^2 = (R-I)/2.  The diagonal entries of P^2 are x^2-1, y^2-1, and
	// z^2-1.  We can solve these for axis (x,y,z).  Because the angle is pi,
	// it does not matter which sign you choose on the square roots.

	double trace = m_mat[0] + m_mat[4] + m_mat[8];
	double cs = 0.5*(trace - 1.0);
	angle = acos(cs);  // in [0,PI]

	if (angle > 0.0)
	{
		if (angle < M_PI)
		{
			axis(0) = m_mat[7] - m_mat[5];
			axis(1) = m_mat[2] - m_mat[6];
			axis(2) = m_mat[3] - m_mat[1];
			axis /= axis.norm();
		}
		else
		{
			// angle is PI
			double halfInverse;
			if (m_mat[0] >= m_mat[4])
			{
				// r00 >= r11
				if (m_mat[0] >= m_mat[8])
				{
					// r00 is maximum diagonal term
					axis(0) = 0.5*sqrt(1.0 + m_mat[0] - m_mat[4] - m_mat[8]);
					halfInverse = 0.5/axis(0);
					axis(1) = halfInverse*m_mat[1];
					axis(2) = halfInverse*m_mat[2];
				}
				else
				{
					// r22 is maximum diagonal term
					axis(2) = 0.5*sqrt(1.0 + m_mat[8] - m_mat[0] - m_mat[4]);
					halfInverse = 0.5/axis(2);
					axis(0) = halfInverse*m_mat[2];
					axis(1) = halfInverse*m_mat[5];
				}
			}
			else
			{
				// r11 > r00
				if (m_mat[4] >= m_mat[8])
				{
					// r11 is maximum diagonal term
					axis(1) = 0.5*sqrt(1.0 + + m_mat[4] - m_mat[0] - m_mat[8]);
					halfInverse  = 0.5/axis(1);
					axis(0) = halfInverse*m_mat[1];
					axis(2) = halfInverse*m_mat[5];
				}
				else
				{
					// r22 is maximum diagonal term
					axis(2) = 0.5*sqrt(1.0 + m_mat[8] - m_mat[0] - m_mat[4]);
					halfInverse = 0.5/axis(2);
					axis(0) = halfInverse*m_mat[2];
					axis(1) = halfInverse*m_mat[5];
				}
			}
		}
	}
	else
	{
		// The angle is 0 and the matrix is the identity.  Any axis will
		// work, so just use the x-axis.
		axis(0) = 1.0;
		axis(1) = 0.0;
		axis(2) = 0.0;
	}
}


Matrix33& Matrix33::MakeEulerZXZ(const double z0Angle, const double xAngle, const double z1Angle)
{
	double cs, sn;

	cs = cos(z0Angle);
	sn = sin(z0Angle);
	Matrix33 z0Mat(
		cs,		-sn,	0.0,
		sn,		cs,		0.0,
		0.0,	0.0,	1.0);

	cs = cos(xAngle);
	sn = sin(xAngle);
	Matrix33 xMat(
		1.0, 0.0, 0.0,
		0.0, cs, -sn,
		0.0, sn, cs);

	cs = cos(z1Angle);
	sn = sin(z1Angle);
	Matrix33 z1Mat(
		cs, -sn, 0.0,
		sn, cs, 0.0,
		0.0, 0.0, 1.0);

	*this = z0Mat*(xMat*z1Mat);

	return *this;
}


Matrix33::EulerResult Matrix33::ExtractEulerZXZ (
		double& z0Angle, double& xAngle, double& z1Angle) const
{
	// +-           -+   +-                                                -+
	// | r00 r01 r02 |   | cz0*cz1-cx*sz0*sz1  -cx*cz1*sz0-cz0*sz1   sx*sz0 |
	// | r10 r11 r12 | = | cz1*sz0+cx*cz0*sz1   cx*cz0*cz1-sz0*sz1  -sz*cz0 |
	// | r20 r21 r22 |   | sx*sz1               sx*cz1               cx     |
	// +-           -+   +-                                                -+

	if (m_mat[8] < 1.0)
	{
		if (m_mat[8] > -1.0)
		{
			// x_angle  = acos(r22)
			// z0_angle = atan2(r02,-r12)
			// z1_angle = atan2(r20,r21)
			xAngle = acos(m_mat[8]);
			z0Angle = atan2(m_mat[2], -m_mat[5]);
			z1Angle = atan2(m_mat[6], m_mat[7]);
			return EA_UNIQUE;
		}
		else
		{
			// Not a unique solution:  z1_angle - z0_angle = atan2(-r01,r00)
			xAngle = M_PI;
			z0Angle = -atan2(-m_mat[1], m_mat[0]);
			z1Angle = 0.0;
			return EA_NOT_UNIQUE_DIF;
		}
	}
	else
	{
		// Not a unique solution:  z1_angle + z0_angle = atan2(-r01,r00)
		xAngle = 0.0;
		z0Angle = atan2(-m_mat[1], m_mat[0]);
		z1Angle = 0.0;
		return EA_NOT_UNIQUE_SUM;
	}
}


inline double & Matrix33::operator() (int i, int j)
{
    return m_mat[3*i + j];
}


inline double Matrix33::operator() (int i, int j) const
{
    return m_mat[3*i + j];
}


inline double & Matrix33::operator() (int i)
{
    return m_mat[i];
}


inline double Matrix33::operator() (int i) const
{
    return m_mat[i];
}
