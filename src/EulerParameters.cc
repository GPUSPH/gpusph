
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

#include "EulerParameters.h"
#include <cmath>


EulerParameters::EulerParameters(void)
{
	m_ep[0] = 1;
	for (int i=1; i < 4; i++)
		m_ep[i] = 0;
}


/*! Constructor from array of values
	\param val : array containing the values of Euler parameters
*/
EulerParameters::EulerParameters(const double * ep)
{
	for (int i=0; i < 4; i++)
		m_ep[i] = ep[i];
}


EulerParameters::EulerParameters(const float * ep)
{
	for (int i=0; i < 4; i++)
		m_ep[i] = double(ep[i]);
}


/*!	Copy constructor
	\param source : source data
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


/*! Constructor from Euler angles (theta, phi, psi) in zxz convention
	\param theta : theta
	\param phi : phi
	\param psi : psi
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


/*! Constructor from vector and rotation angle
	\param dir : direction of the vector
	\param angle : angle of rotation around dir
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


//! Assignment operator
/*! \param source : value 
	\return this = value
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


/// Normalization function
/*! Divide the Euler parameter by the norm of the associated quaternion:
		\f$\sqrt{q^2_0 + q^2_1 + q^2_2 + q^2_3}\f$
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
/*! Compute Euler angles (theta, phi, psi) in zxz convention from Euler parameters
	\param theta : theta
	\param phi : phi
	\param psi : psi
*/
void 
EulerParameters::ExtractEulerZXZ(double &theta, double &phi, double &psi) const
{
	phi = acos(2.0*(m_ep[0]*m_ep[0] + m_ep[3]*m_ep[3]) - 1.0);
	theta = atan2(m_ep[1]*m_ep[3] + m_ep[0]*m_ep[2], m_ep[0]*m_ep[1] - m_ep[2]*m_ep[3]);
	psi = atan2(m_ep[1]*m_ep[3] - m_ep[0]*m_ep[2], m_ep[2]*m_ep[3] + m_ep[0]*m_ep[1]);
}


/// Rotation matrix computation
/*! The rotation matrix is computed according to:
 * 
 *	\f$ R(q)=\begin{pmatrix}
 *  q^2_0 + q^2_1 - q^2_2 - q^2_3 & 2q_1q_0 - 2q_0q_3 & 2q_0q_2 + 2q_1q_3 \\ 
 *	2q_1q_0 - 2q_0q_3 & q^2_0 - q^2_1 + q^2_2 - q^2_3 & 2q_2q_3 - 2q_0q_1 \\
 *	2q_1q_3 - 2q_0q_2 & 2q_2q_3 - 2q_0q_1 & q^2_0 - q^2_1 - q^2_2 + q^2_3
 *	\end{pmatrix}\f$
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
		[0] q0^2+q1^-q2^2-q3^2		[1] 2q1q2-2q0q3			[2] 2q0q2+2q1q3
		[3] 2q0q3+2q1q2			[4] q0^2-q1^2+q2^2-q3^2		[5] 2q2q3-2q0q1
		[6] 2q1q3-2q0q2				[7] 2q0q1+2q2q3		[8] q0^2-q1^2-q2^2+q3^2
	 
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


/// Define the *= operator for Euler parameters
/*!	\param val : EulerParameters
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


/*!	Define the * operation for EulerParmeters.
 *  Let be \f$ q=(q_0, q_1, q_2, q_3)\f$ and \f$ q'=(q'_0, q'_1, q'_2, q'_3)\f$ two set of Euler parameters
 *	we have :
 * 
 *  \f{align*}{ q*q' =& (q_0q'_0 - q_1q'_1 - q_2q'_2 - q_3q'_3, q_1q'_0 + q_0q'_1 - q_3q'_2 + q_2q'_3, \\
 *	&q_2q'_0 + q_3q'_1 + q_0q'_2 - q_1q'_3, q_3q'_0 - q_2q'_1 + q_1q'_2 + q_0q'_3) \f}
 * 
 *	This operation corresponds to a rotation composition.
 *	\param ep1 : Euler parameters
 *	\param ep2 : Euler parameters
 *	\return ep1*ep2
 * 
 *	Beware this operation is not commutative
*/
EulerParameters operator*(const EulerParameters &ep1, const EulerParameters &ep2)
{
	double temp[4];
	// Quaternion[a0, a1, a2, a3] ** Quaternion[b0, b1, b2, b3] =
	// a0 b0 - a1 b1 - a2 b2 - a3 b3
	temp[0] = ep1.m_ep[0]*ep2.m_ep[0] - ep1.m_ep[1]*ep2.m_ep[1] - ep1.m_ep[2]*ep2.m_ep[2] - ep1.m_ep[3]*ep2.m_ep[3];
	// a1 b0 + a0 b1 - a3 b2 + a2 b3
	temp[1] = ep1.m_ep[1]*ep2.m_ep[0] + ep1.m_ep[0]*ep2.m_ep[1] - ep1.m_ep[3]*ep2.m_ep[2] + ep1.m_ep[2]*ep2.m_ep[3];
	// a2 b0 + a3 b1 + a0 b2 - a1 b3
	temp[2] = ep1.m_ep[2]*ep2.m_ep[0] + ep1.m_ep[3]*ep2.m_ep[1] + ep1.m_ep[0]*ep2.m_ep[2] - ep1.m_ep[1]*ep2.m_ep[3];
	// a3 b0 - a2 b1 + a1 b2 + a0 b3
	temp[3] = ep1.m_ep[3]*ep2.m_ep[0] - ep1.m_ep[2]*ep2.m_ep[1] + ep1.m_ep[1]*ep2.m_ep[2] + ep1.m_ep[0]*ep2.m_ep[3];

	EulerParameters res(temp);

	res.Normalize();

	return res;
}


EulerParameters operator*(const EulerParameters * ep1, const EulerParameters &ep2)
{
	double temp[4];
	temp[0] = ep1->m_ep[0]*ep2.m_ep[0] - ep1->m_ep[1]*ep2.m_ep[1] - ep1->m_ep[2]*ep2.m_ep[2] - ep1->m_ep[3]*ep2.m_ep[3];
	temp[1] = ep1->m_ep[1]*ep2.m_ep[0] + ep1->m_ep[0]*ep2.m_ep[1] - ep1->m_ep[3]*ep2.m_ep[2] + ep1->m_ep[2]*ep2.m_ep[3];
	temp[2] = ep1->m_ep[2]*ep2.m_ep[0] + ep1->m_ep[3]*ep2.m_ep[1] + ep1->m_ep[0]*ep2.m_ep[2] - ep1->m_ep[1]*ep2.m_ep[3];
	temp[3] = ep1->m_ep[3]*ep2.m_ep[0] - ep1->m_ep[2]*ep2.m_ep[1] + ep1->m_ep[1]*ep2.m_ep[2] + ep1->m_ep[0]*ep2.m_ep[3];

	EulerParameters res(temp);

	res.Normalize();

	return res;
}


/*!	Compute rotation matrix between the current object and another Euler paramters.
 *  The result \f$R(q).R(previous)^t\f$ is stored in the array pointed by res 
 *	\param previous : previous EulerParameters
 *	\param res : pointer to rotation matrix
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


/*!	Apply the inverse of rotation defined by the Euler parameters to input data.
 *	\param data : input data
 *	\return : \f$R(q)^t*data\f$
 */
float3 EulerParameters::TransposeRot(const float3 &data) const
{
	float3 res;
	res.x = (float) (m_rot[0]*data.x + m_rot[3]*data.y + m_rot[6]*data.z);
	res.y = (float) (m_rot[1]*data.x + m_rot[4]*data.y + m_rot[7]*data.z);
	res.z = (float) (m_rot[2]*data.x + m_rot[5]*data.y + m_rot[8]*data.z);

	return res;
}


Vector EulerParameters::TransposeRot(const Vector &data) const
{
	Vector res;
	res(0) = m_rot[0]*data(0) + m_rot[3]*data(1) + m_rot[6]*data(2);
	res(1) = m_rot[1]*data(0) + m_rot[4]*data(1) + m_rot[7]*data(2);
	res(2) = m_rot[2]*data(0) + m_rot[5]*data(1) + m_rot[8]*data(2);

	return res;
}


Point EulerParameters::TransposeRot(const Point &data) const
{
	Point res;
	res(0) = m_rot[0]*data(0) + m_rot[3]*data(1) + m_rot[6]*data(2);
	res(1) = m_rot[1]*data(0) + m_rot[4]*data(1) + m_rot[7]*data(2);
	res(2) = m_rot[2]*data(0) + m_rot[5]*data(1) + m_rot[8]*data(2);

	return res;
}


/*!	Apply the rotation defined by the Euler parameters to input data.
 *	\param data : input data
 *	\return : \f$R(q)*data\f$
*/
float3 EulerParameters::Rot(const float3 &data) const
{
	float3 res;
	res.x = (float) (m_rot[0]*data.x + m_rot[1]*data.y + m_rot[2]*data.z);
	res.y = (float) (m_rot[3]*data.x + m_rot[4]*data.y + m_rot[5]*data.z);
	res.z = (float) (m_rot[6]*data.x + m_rot[7]*data.y + m_rot[8]*data.z);

	return res;
}


Vector EulerParameters::Rot(const Vector &data) const
{
	Vector res;
	res(0) = m_rot[0]*data(0) + m_rot[1]*data(1) + m_rot[2]*data(2);
	res(1) = m_rot[3]*data(0) + m_rot[4]*data(1) + m_rot[5]*data(2);
	res(2) = m_rot[6]*data(0) + m_rot[7]*data(1) + m_rot[8]*data(2);

	return res;
}


Point EulerParameters::Rot(const Point &data) const
{
	Point res;
	res(0) = m_rot[0]*data(0) + m_rot[1]*data(1) + m_rot[2]*data(2);
	res(1) = m_rot[3]*data(0) + m_rot[4]*data(1) + m_rot[5]*data(2);
	res(2) = m_rot[6]*data(0) + m_rot[7]*data(1) + m_rot[8]*data(2);

	return res;
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
