/*	Copyright 2011-2014 Alexis Herault, Giuseppe Bilotta,
	Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto Nazionale di Geofisica e Vulcanologia
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
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "EulerParameters.h"
#include <cmath>

/// Empty constructor
/*!	Set the Euler parameters to
 * identity (1, 0, 0, 0)
 */
EulerParameters::EulerParameters(void)
{
	m_ep[0] = 1;
	for (int i=1; i < 4; i++)
		m_ep[i] = 0;
}


/// Constructor form data array
/*! Constructor from array of values
 *	\param[in] ep : array containing the values of Euler parameters
 */
EulerParameters::EulerParameters(const double * ep)
{
	for (int i=0; i < 4; i++)
		m_ep[i] = ep[i];
}


/*!
 * \overload EulerParameters::EulerParameters(const float * ep)
 */
EulerParameters::EulerParameters(const float * ep)
{
	for (int i=0; i < 4; i++)
		m_ep[i] = double(ep[i]);
}


/// Constructor form for values
/*! Constructor from for double of values
 *	\param[in] e0 : 1st component
 *	\param[in] e1 : 2nd component
 *	\param[in] e2 : 3rd component
 *	\param[in] e3 : 4th component
 */
EulerParameters::EulerParameters(const double e0, const double e1, const double e2, const double e3)
{
	m_ep[0] = e0;
	m_ep[1] = e1;
	m_ep[2] = e2;
	m_ep[3] = e3;
}


/*!
 * \overload EulerParameters::EulerParameters(const double, cons double, const double, const double)
 */
EulerParameters::EulerParameters(const float e0, const float e1, const float e2, const float e3)
{
	EulerParameters((double) e0, (double) e1, (double) e2, (double) e3);
}


/// Copy constructor
/*!
 *	\param[in] source : source data
 */
EulerParameters::EulerParameters(const EulerParameters &source)
{
	m_ep[0] = source.m_ep[0];
	m_ep[1] = source.m_ep[1];
	m_ep[2] = source.m_ep[2];
	m_ep[3] = source.m_ep[3];

	for (int i = 0; i < 9; i++)
		m_rot[i] = source.m_rot[i];
}


/// Constructor from ODE quaternion
/*!
 *	\param[in] quat : ODE quaternion
 */
EulerParameters::EulerParameters(const dQuaternion &quat)
{
	for (int i = 0; i < 4; i++)
		m_ep[i] = quat[i];
}


/// Constructor from a vector
/*! All rotations can be done using only normalized quaternions:
 *  let be \f$ q \f$ a normalized quaternion and \f$ v \f$ the quaternion
 *  having a null real component and imaginary components equals to the
 *  components of vector \f$ \vec{c} \f$ i.e.  \f$ v = (0, v_x, v_y, v_z)\f$.
 *  The imaginary part of the quaternion \f$ qvq^{-1} \f$ is the vector \f$ \vec{c} \f$
 *  rotated by the rotation defined by \f$ q \f$.
 */
EulerParameters::EulerParameters(const double3 v)
{
	m_ep[0] = 0.;
	m_ep[1] = v.x;
	m_ep[2] = v.y;
	m_ep[3] = v.z;
}


EulerParameters::EulerParameters(const float3 v)
{
	m_ep[0] = 0.;
	m_ep[1] = (double) v.x;
	m_ep[2] = (double) v.y;
	m_ep[3] = (double) v.z;
}


/// Constructor from Euler angles
/*! Construct Euler parameters form a set of Euler angles \f$(\psi, \theta, \phi)\f$
 *  in zxz extrinsic convention.
 *  \htmlonly <img src="EulerZXZ.png" width="400px"></img> \endhtmlonly
 *	\param[in] z0Angle : \f$ \psi \f$
 *	\param[in] xAngle : \f$ \theta \f$
 *	\param[in] z1Angle : \f$ \phi \f$
 */
EulerParameters::EulerParameters(const double z0Angle, const double xAngle, const double z1Angle)
{
	double cx2 = cos(xAngle/2.0);
	double sx2 = sin(xAngle/2.0);

	m_ep[0] = cx2*cos((z0Angle + z1Angle)/2.0);
	m_ep[1] = sx2*cos((z0Angle - z1Angle)/2.0);
	m_ep[2] = sx2*sin((z0Angle - z1Angle)/2.0);
	m_ep[3] = cx2*sin((z0Angle + z1Angle)/2.0);
}


/// Constructor from vector and rotation angle
/*! Construct Euler parameters from a vector and a rotation angle
 * 	around this vector
 *	\param[in] dir : direction of the vector
 *	\param[in] angle : angle of rotation around dir
 */
EulerParameters::EulerParameters(const Vector & dir, const double angle)
{
	double angle_over2 = angle/2.0;
	m_ep[0] = cos(angle_over2);
	double temp = sin(angle_over2);
	double dnorm = dir.norm();
	if (dnorm) {
		m_ep[1] = dir(0)*temp/dnorm;
		m_ep[2] = dir(1)*temp/dnorm;
		m_ep[3] = dir(2)*temp/dnorm;
	}
	else {
		m_ep[0] = 1.0;
		m_ep[1] = m_ep[2] = m_ep[3] = 0.0;
	}

	Normalize();
}


/// Assignment operator
/*! Overload of the assignment operator = for Euler parameters
 * 	\param[in] source : value
 *	\return this = value
 */
EulerParameters&
EulerParameters::operator= (const EulerParameters& source)
{
	m_ep[0] = source.m_ep[0];
	m_ep[1] = source.m_ep[1];
	m_ep[2] = source.m_ep[2];
	m_ep[3] = source.m_ep[3];

	for (int i = 0; i < 9; i++)
		m_rot[i] = source.m_rot[i];

	return *this;
}


/// Normalize Euler parameters
/*! Divide Euler parameters by the norm of the associated quaternion:
 *		\f$\sqrt{q^2_0 + q^2_1 + q^2_2 + q^2_3}\f$
 */
void
EulerParameters::Normalize(void)
{
	double rnorm = 1.0/sqrt(m_ep[0]*m_ep[0] + m_ep[1]*m_ep[1] + m_ep[2]*m_ep[2] + m_ep[3]*m_ep[3]);

	m_ep[0] *= rnorm;
	m_ep[1] *= rnorm;
	m_ep[2] *= rnorm;
	m_ep[3] *= rnorm;
}


/// Euler angles computation
/*! Compute Euler angles \f$(\psi, \theta, \phi)\f$ in zxz
 *  intrinsic convention from Euler parameters.
 *  \htmlonly <img src="EulerZXZ.png" width="400px"></img> \endhtmlonly
 *	\param[out] psi : \f$ \psi \f$
 *	\param[out] theta : \f$ \theta \f$
 *	\param[out] phi : \f$ \phi \f$
 */
void
EulerParameters::ExtractEulerZXZ(double &psi, double &theta, double &phi) const
{
	phi = acos(2.0*(m_ep[0]*m_ep[0] + m_ep[3]*m_ep[3]) - 1.0);
	theta = atan2(m_ep[1]*m_ep[3] + m_ep[0]*m_ep[2], m_ep[0]*m_ep[1] - m_ep[2]*m_ep[3]);
	psi = atan2(m_ep[1]*m_ep[3] - m_ep[0]*m_ep[2], m_ep[2]*m_ep[3] + m_ep[0]*m_ep[1]);
}


/// Return associated ODE quaternion
/*!
 *	\param[in/out] quat : ODE quaternion
 */
void
EulerParameters::ToODEQuaternion(dQuaternion & quat) const
{
	for (int i = 0; i < 4; i++)
		quat[i] = m_ep[i];
}


/// Set Euler parameters to identity: (1, 0, 0, 0)
void
EulerParameters::ToIdentity(void)
{
	m_ep[0] = 1.0;
	for (int i = 1; i < 4; i++)
		m_ep[i] = 0.0;
}


/// Return the inverse of the parameter
/*! Compute the inverse of the Euler parameters. Given
 * 	\f$ q=(q_0, q_1, q_2, q_3)\f$ we have \f$ q^{-1}=(q_0, -q_1, -q_2, -q_3)\f$
 *
 *	\return the inverse
 */
EulerParameters
EulerParameters::Inverse(void)
{
	EulerParameters res = *this;
	res.m_ep[1] *= -1;
	res.m_ep[2] *= -1;
	res.m_ep[3] *= -1;

	return res;
}


/// Rotation matrix computation
/*! Compute the rotation matrix associated with the Euler parameters
 * 	according to:
 *
 *	\f$ R(q) = \begin{pmatrix}
 *  q^2_0 + q^2_1 - q^2_2 - q^2_3 & 2q_1q_0 - 2q_0q_3 & 2q_0q_2 + 2q_1q_3 \\
 *	2q_1q_0 - 2q_0q_3 & q^2_0 - q^2_1 + q^2_2 - q^2_3 & 2q_2q_3 - 2q_0q_1 \\
 *	2q_1q_3 - 2q_0q_2 & 2q_2q_3 - 2q_0q_1 & q^2_0 - q^2_1 - q^2_2 + q^2_3
 *	\end{pmatrix}
 *	\f$
 *
 *	and store it in m_rot array.
 *
 *	This function should be called before using Rot() or TransposeRot()
 */
void
EulerParameters::ComputeRot(void)
{
	/* For a normalized quaternion (q0, q1, q2, q3) the associated rotation
	 matrix R is :
		[0] q0^2+q1^-q2^2-q3^2		[1] 2q1q2-2q0q3				[2] 2q0q2+2q1q3
		[3] 2q0q3+2q1q2				[4] q0^2-q1^2+q2^2-q3^2		[5] 2q2q3-2q0q1
		[6] 2q1q3-2q0q2				[7] 2q0q1+2q2q3				[8] q0^2-q1^2-q2^2+q3^2

	 According to q0^2+q1^2+q2^2+q3^2=1 we can rewrite the diagonal terms:
		r11 = 1 - 2(q2^2 + q3^2)
		r22 = 1 - 2(q1^2 + q3^2)
		r33 = 1 - 2(q1^2 + q2^2)
	 */

	float temp = 2.0*m_ep[2]*m_ep[2];	// 2.0*q2^2
	m_rot[0] = 1.0 - temp;				// r0 = 1 - 2q2^2
	m_rot[8] = 1.0 - temp;				// r8 = 1 - 2q2^2
	temp = 2.0*m_ep[3]*m_ep[3];			// 2.0*q3^2
	m_rot[0] -= temp;					// r0 = 1 - 2(q2^2 + q3^2)
	m_rot[4] = 1.0 - temp;				// r4 = 1 - 2 q3^2
	temp = 2.0*m_ep[1]*m_ep[1];			// 2.0*q1^2
	m_rot[4] -= temp;					// r4 = 1 - 2(q1^2 + q3^2)
	m_rot[8] -= temp;					// r8 = 1 - 2(q1^2 + q2^2)

	temp = 2.0*m_ep[0]*m_ep[1];			// 2.0*q0q1
	m_rot[5] = - temp;					// r5 = - 2q0q1
	m_rot[7] = temp;					// r7 = 2p0p1
	temp = 2.0*m_ep[0]*m_ep[2];			// 2.0*q0q2
	m_rot[2] = temp;					// r2 = 2q0q2
	m_rot[6] = -temp;					// r6 = - 2q0q2
	temp = 2.0*m_ep[0]*m_ep[3];			// 2.0*q0q3
	m_rot[1] = - temp;					// r1 = - 2q0q3
	m_rot[3] = temp;					// r3 = 2q0q3
	temp = 2.0*m_ep[1]*m_ep[2];			// 2.0*q1q2
	m_rot[1] += temp;					// r1 =2q1q2 - 2q0q3
	m_rot[3] += temp;					// r3 = 2q0q3 + 2q1q2
	temp = 2.0*m_ep[1]*m_ep[3];			// 2.0*q1q3
	m_rot[2] += temp;					// r2 = 2q0q2 + 2q1q3
	m_rot[6] += temp;					// r6 = 2q1q3 - 2q0q2
	temp = 2.0*m_ep[2]*m_ep[3];			// 2.0*q2q3
	m_rot[5] += temp;					// r5 =  2q2q3 - 2q0q1
	m_rot[7] += temp;					// r7 = 2q2q3 + 2q0q1
}


/// Definition of *= operator for Euler parameters
/*!	Overload of the *= operator for Euler parameters. This operation
 * 	correspond to the composition of the rotations defined by the two
 * 	Euler parameters.
 * 	\param[in] val : EulerParameters
 *	\return this = this * val
 *
 *	For the mathematical definition see operator*()
 *
 *	Beware this operation is not commutative.
 */
EulerParameters &EulerParameters::operator*=(const EulerParameters &val)
{
	double temp[4];
	temp[0] = m_ep[0]*val.m_ep[0] - m_ep[1]*val.m_ep[1] - m_ep[2]*val.m_ep[2] - m_ep[3]*val.m_ep[3];
	temp[1] = m_ep[1]*val.m_ep[0] + m_ep[0]*val.m_ep[1] - m_ep[3]*val.m_ep[2] + m_ep[2]*val.m_ep[3];
	temp[2] = m_ep[2]*val.m_ep[0] + m_ep[3]*val.m_ep[1] + m_ep[0]*val.m_ep[2] - m_ep[1]*val.m_ep[3];
	temp[3] = m_ep[3]*val.m_ep[0] - m_ep[2]*val.m_ep[1] + m_ep[1]*val.m_ep[2] + m_ep[0]*val.m_ep[3];
	for (int i=0; i < 4; i++)
		m_ep[i] = temp[i];

	Normalize();

	return *this;
}


/*!	Define the + operation for EulerParmeters.
 * 	Overload of the + operator for Euler parameters.
 *
 *  Let be \f$ q=(q_0, q_1, q_2, q_3)\f$ and \f$ q'=(q'_0, q'_1, q'_2, q'_3)\f$ two set of Euler parameters
 *	we have :
 *  \f{eqnarray*}{ q*q' = & (q_0 + q'_0, q_1 + q'_1, q_2 + q'_2, q_3 + q'_3) \f}
 *
 *	\param[in] ep1 : Euler parameters
 *	\param[in] ep2 : Euler parameters
 *	\return ep1+ep2
 *
 *	Beware this operation is not commutative
 */
EulerParameters operator+(const EulerParameters &ep1, const EulerParameters &ep2)
{
	return EulerParameters(ep1.m_ep[0] + ep2.m_ep[0], ep1.m_ep[1] + ep2.m_ep[1],
				ep1.m_ep[2] + ep2.m_ep[2], ep1.m_ep[3] + ep2.m_ep[3]);
}


/*!	Define the * operation for EulerParmeters.
 * 	Overload of the * operator for Euler parameters. This operation corresponds to a rotation composition.
 *
 *  Let be \f$ q=(q_0, q_1, q_2, q_3)\f$ and \f$ q'=(q'_0, q'_1, q'_2, q'_3)\f$ two set of Euler parameters
 *	we have :
 *  \f{eqnarray*}{ q*q' = & (q_0q'_0 - q_1q'_1 - q_2q'_2 - q_3q'_3, q_1q'_0 + q_0q'_1 - q_3q'_2 + q_2q'_3, \\
 *	& q_2q'_0 + q_3q'_1 + q_0q'_2 - q_1q'_3, q_3q'_0 - q_2q'_1 + q_1q'_2 + q_0q'_3) \f}
 *
 *	\param[in] ep1 : Euler parameters
 *	\param[in] ep2 : Euler parameters
 *	\return ep1*ep2
 *
 *	Beware this operation is not commutative
 */
EulerParameters operator*(const EulerParameters &ep1, const EulerParameters &ep2)
{
	double temp[4];
	// Quaternion[a0, a1, a2, a3] * Quaternion[b0, b1, b2, b3] =
	// a0 b0 - a1 b1 - a2 b2 - a3 b3
	temp[0] = ep1.m_ep[0]*ep2.m_ep[0] - ep1.m_ep[1]*ep2.m_ep[1] - ep1.m_ep[2]*ep2.m_ep[2] - ep1.m_ep[3]*ep2.m_ep[3];
	// a1 b0 + a0 b1 - a3 b2 + a2 b3
	temp[1] = ep1.m_ep[1]*ep2.m_ep[0] + ep1.m_ep[0]*ep2.m_ep[1] - ep1.m_ep[3]*ep2.m_ep[2] + ep1.m_ep[2]*ep2.m_ep[3];
	// a2 b0 + a3 b1 + a0 b2 - a1 b3
	temp[2] = ep1.m_ep[2]*ep2.m_ep[0] + ep1.m_ep[3]*ep2.m_ep[1] + ep1.m_ep[0]*ep2.m_ep[2] - ep1.m_ep[1]*ep2.m_ep[3];
	// a3 b0 - a2 b1 + a1 b2 + a0 b3
	temp[3] = ep1.m_ep[3]*ep2.m_ep[0] - ep1.m_ep[2]*ep2.m_ep[1] + ep1.m_ep[1]*ep2.m_ep[2] + ep1.m_ep[0]*ep2.m_ep[3];

	EulerParameters res(temp);

	return res;
}


/*!
 * \overload EulerParameters operator*(const EulerParameters * ep1, const EulerParameters &ep2)
 */
EulerParameters operator*(const EulerParameters * ep1, const EulerParameters &ep2)
{
	double temp[4];
	temp[0] = ep1->m_ep[0]*ep2.m_ep[0] - ep1->m_ep[1]*ep2.m_ep[1] - ep1->m_ep[2]*ep2.m_ep[2] - ep1->m_ep[3]*ep2.m_ep[3];
	temp[1] = ep1->m_ep[1]*ep2.m_ep[0] + ep1->m_ep[0]*ep2.m_ep[1] - ep1->m_ep[3]*ep2.m_ep[2] + ep1->m_ep[2]*ep2.m_ep[3];
	temp[2] = ep1->m_ep[2]*ep2.m_ep[0] + ep1->m_ep[3]*ep2.m_ep[1] + ep1->m_ep[0]*ep2.m_ep[2] - ep1->m_ep[1]*ep2.m_ep[3];
	temp[3] = ep1->m_ep[3]*ep2.m_ep[0] - ep1->m_ep[2]*ep2.m_ep[1] + ep1->m_ep[1]*ep2.m_ep[2] + ep1->m_ep[0]*ep2.m_ep[3];

	EulerParameters res(temp);

	return res;
}


/*!	Define the * operation between double and EulerParmeters.
 * 	Overload of the * operator between double and Euler parameters.
 *
 *  Let be \f$ q=(q_0, q_1, q_2, q_3)\f$ an Euler parameters and \f$ a\f$ a real
 *	we have :
 *  \f$ a*q = & (a*q_0, a*q_1, a*q_2, a*q_3) \f$
 *
 *	\param[in] a : real
 *	\param[in] ep : Euler parameters
 *	\return a*ep
 *
 *	Beware this operation is not commutative
 */
EulerParameters operator*(const double a, const EulerParameters &ep)
{
	return EulerParameters(a*ep.m_ep[0], a*ep.m_ep[1], a*ep.m_ep[2], a*ep.m_ep[3]);
}


/// Copy rotation maxtrix in an array
/*!	Copy stored rotation matrix in an array of floatspointed by res.
 *	\param[out] res : pointer to rotation matrix
 *
 *  Beware: this method use the rotation matrix associated with each Euler parameters.
 *  Those matrix should be computed before calling the method.
 */
void EulerParameters::GetRotation(float *res) const
{
	for (int i = 0; i < 9; i++)
		res[i] = m_rot[i];
}


/// Relative rotation between two Euler parameters
/*!	Compute rotation matrix between the current object and another Euler parameters.
 *  The result \f$R(q).R(previous)^t\f$ is stored in the array pointed by res
 *	\param[in] previous : previous EulerParameters
 *	\param[out] res : pointer to rotation matrix
 *
 *  Beware: this method use the rotation matrix associated with each Euler parameters.
 *  Those matrix should be computed before calling the method.
 */
void EulerParameters::StepRotation(const EulerParameters & previous, float *res) const
{
	/*
	 | 0 1 2 | | 0 1 2 |t  | 0 1 2 | | 0 3 6 |
	 | 3 4 5 | | 3 4 5 | = | 3 4 5 | | 1 4 7 |
	 | 6 7 8 | | 6 7 8 |   | 6 7 8 | | 2 5 8 |
	 */
	res[0] = (float) (m_rot[0]*previous.m_rot[0] + m_rot[1]*previous.m_rot[1] + m_rot[2]*previous.m_rot[2]);
	res[1] = (float) (m_rot[0]*previous.m_rot[3] + m_rot[1]*previous.m_rot[4] + m_rot[2]*previous.m_rot[5]);
	res[2] = (float) (m_rot[0]*previous.m_rot[6] + m_rot[1]*previous.m_rot[7] + m_rot[2]*previous.m_rot[8]);

	res[3] = (float) (m_rot[3]*previous.m_rot[0] + m_rot[4]*previous.m_rot[1] + m_rot[5]*previous.m_rot[2]);
	res[4] = (float) (m_rot[3]*previous.m_rot[3] + m_rot[4]*previous.m_rot[4] + m_rot[5]*previous.m_rot[5]);
	res[5] = (float) (m_rot[3]*previous.m_rot[6] + m_rot[4]*previous.m_rot[7] + m_rot[5]*previous.m_rot[8]);

	res[6] = (float) (m_rot[6]*previous.m_rot[0] + m_rot[7]*previous.m_rot[1] + m_rot[8]*previous.m_rot[2]);
	res[7] = (float) (m_rot[6]*previous.m_rot[3] + m_rot[7]*previous.m_rot[4] + m_rot[8]*previous.m_rot[5]);
	res[8] = (float) (m_rot[6]*previous.m_rot[6] + m_rot[7]*previous.m_rot[7] + m_rot[8]*previous.m_rot[8]);
}


/// Apply the inverse of the rotation defined by the Euler parameters
/*!	Apply the inverse of rotation defined by the Euler parameters to input data.
 * 	This inverse rotation is computed by multiplying the input data by the the
 * 	transpose of the rotation matrix defined by the Euler parameters.
 *	\param[in] data : input data
 *	\return \f$R(q)^t*data\f$
 *
 *	The ComputeRot method should be called before calling this method.
 */
float3 EulerParameters::TransposeRot(const float3 &data) const
{
	float3 res;
	res.x = (float) (m_rot[0]*data.x + m_rot[3]*data.y + m_rot[6]*data.z);
	res.y = (float) (m_rot[1]*data.x + m_rot[4]*data.y + m_rot[7]*data.z);
	res.z = (float) (m_rot[2]*data.x + m_rot[5]*data.y + m_rot[8]*data.z);

	return res;
}


/*!
 * \overload Vector EulerParameters::TransposeRot(const Vector &data) const
 */
Vector EulerParameters::TransposeRot(const Vector &data) const
{
	Vector res;
	res(0) = m_rot[0]*data(0) + m_rot[3]*data(1) + m_rot[6]*data(2);
	res(1) = m_rot[1]*data(0) + m_rot[4]*data(1) + m_rot[7]*data(2);
	res(2) = m_rot[2]*data(0) + m_rot[5]*data(1) + m_rot[8]*data(2);

	return res;
}


/*!
 * \overload Point EulerParameters::TransposeRot(const Point &data) const
 */
Point EulerParameters::TransposeRot(const Point &data) const
{
	Point res;
	res(0) = m_rot[0]*data(0) + m_rot[3]*data(1) + m_rot[6]*data(2);
	res(1) = m_rot[1]*data(0) + m_rot[4]*data(1) + m_rot[7]*data(2);
	res(2) = m_rot[2]*data(0) + m_rot[5]*data(1) + m_rot[8]*data(2);

	return res;
}


/// Apply the rotation defined by the Euler parameters
/*!	Apply the rotation defined by the Euler parameters to input data.
 * 	This rotation is computed using the most efficient method aka using
 * 	directly the values of Rumer parameter.
 *
 *	\param[in] data : input data
 *	\return \f$R(q)*data\f$
 */
double3 EulerParameters::Rot(const double3 &data) const
{
	double3 res;
	double t2 = m_ep[0]*m_ep[1];
	double t3 = m_ep[0]*m_ep[2];
	double t4 = m_ep[0]*m_ep[3];
	double t5 = - m_ep[1]*m_ep[1];
	double t6 = m_ep[1]*m_ep[2];
	double t7 = m_ep[1]*m_ep[3];
	double t8 = -m_ep[2]*m_ep[2];
	double t9 = m_ep[2]*m_ep[3];
	double t10 = - m_ep[3]*m_ep[3];
	res.x = 2.*( (t8 + t10)*data.x + (t6 -  t4)*data.y + (t3 + t7)*data.z ) + data.x;
	res.y = 2.*( (t4 +  t6)*data.x + (t5 + t10)*data.y + (t9 - t2)*data.z ) + data.y;
	res.z = 2.*( (t7 -  t3)*data.x + (t2 +  t9)*data.y + (t5 + t8)*data.z ) + data.z;

	return res;
}


/*!
 * \overload Vector EulerParameters::Rot(const float3 &data) const
 */
float3 EulerParameters::Rot(const float3 &data) const
{
	float3 res;
	float t2 = m_ep[0]*m_ep[1];
	float t3 = m_ep[0]*m_ep[2];
	float t4 = m_ep[0]*m_ep[3];
	float t5 = - m_ep[1]*m_ep[1];
	float t6 = m_ep[1]*m_ep[2];
	float t7 = m_ep[1]*m_ep[3];
	float t8 = -m_ep[2]*m_ep[2];
	float t9 = m_ep[2]*m_ep[3];
	float t10 = - m_ep[3]*m_ep[3];
	res.x = 2.*( (t8 + t10)*data.x + (t6 -  t4)*data.y + (t3 + t7)*data.z ) + data.x;
	res.y = 2.*( (t4 +  t6)*data.x + (t5 + t10)*data.y + (t9 - t2)*data.z ) + data.y;
	res.z = 2.*( (t7 -  t3)*data.x + (t2 +  t9)*data.y + (t5 + t8)*data.z ) + data.z;

	return res;
}

/*!
 * \overload Vector EulerParameters::Rot(const Vector &data) const
 */
Vector EulerParameters::Rot(const Vector &data) const
{
	Vector res;
	res(0) = m_rot[0]*data(0) + m_rot[1]*data(1) + m_rot[2]*data(2);
	res(1) = m_rot[3]*data(0) + m_rot[4]*data(1) + m_rot[5]*data(2);
	res(2) = m_rot[6]*data(0) + m_rot[7]*data(1) + m_rot[8]*data(2);

	return res;
}


/*!
 * \overload Point EulerParameters::Rot(const Point &data) const
 */
Point EulerParameters::Rot(const Point &data) const
{
	Point res;
	res(0) = m_rot[0]*data(0) + m_rot[1]*data(1) + m_rot[2]*data(2);
	res(1) = m_rot[3]*data(0) + m_rot[4]*data(1) + m_rot[5]*data(2);
	res(2) = m_rot[6]*data(0) + m_rot[7]*data(1) + m_rot[8]*data(2);

	return res;
}


/// Return a reference to parameter i
/*!	\param[in] i : parameter number
 *	\return reference to parameter i
*/
double &
EulerParameters::operator()(int i)
{
	return m_ep[i];
}


/// Return the value of parameter i
/*!	\param[in] i : parameter number
 *	\return value of parameter i
*/
double
EulerParameters::operator()(int i) const
{
	return m_ep[i];
}

/// Return a double4 of the 4 parameters
double4
EulerParameters::params() const
{
	return make_double4(m_ep[0], m_ep[1], m_ep[2], m_ep[3]);
}


// DEBUG
#include <iostream>
void EulerParameters::print(void) const
{
	std::cout << "Ep (" << m_ep[0] << ", " << m_ep[1] << ", " << m_ep[2] <<", " << m_ep[3] << ")\n";
	return;
}

void EulerParameters::printrot(void) const
{
	std::cout << "Rotation matrix\n";
	for (int i = 0; i < 3; i++) {
		std::cout << "\t";
		for (int j = 0; j < 3; j ++)
			std::cout << m_rot[i*3 + j] << "\t";
		std::cout << "\n";
	}
	return;
}

std::ostream& operator<<(std::ostream& out, const EulerParameters& ep) // output
{
	out << "Ep (" << ep.m_ep[0] << ", " << ep.m_ep[1] << ", "
		<< ep.m_ep[2] <<", " << ep.m_ep[3] << ")";
	return out;
}
