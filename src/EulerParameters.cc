
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

#include "EulerParameters.h"
#include <math.h>

/// Constructor
EulerParameters::EulerParameters(void)
{
	m_ep[0] = 1;
	for (int i=1; i < 4; i++)
		m_ep[i] = 0;
}


/// Constructor
/*! Constructor from values array
	\param val : array containing the values of Euler parameters
*/
EulerParameters::EulerParameters(const double * ep)
{
	for (int i=0; i < 4; i++)
		m_ep[i] = ep[i];
}


/// Constructor
/*!	Copy constructor
	\param source : source data
*/
EulerParameters::EulerParameters(const EulerParameters &source)
{
	m_ep[0] = source.m_ep[0];
	m_ep[1] = source.m_ep[1];
	m_ep[2] = source.m_ep[2];
	m_ep[3] = source.m_ep[3];
}


/// Constructor
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


/// Constructor
/*! Constructor from vector an rotation angle
	\param dir : direction of the vector
	\param angle : angle of rotation around dir
*/
EulerParameters::EulerParameters(const Vector & dir, const double angle)
{
	double angle_over2 = angle/2.0;
	m_ep[0] = cos(angle_over2);
	float temp = sin(angle_over2);
	m_ep[1] = dir(0)*temp;
	m_ep[2] = dir(1)*temp;
	m_ep[3] = dir(2)*temp;
}


/// Assignement operator
EulerParameters& EulerParameters::operator= (const EulerParameters& source)
{
	m_ep[0] = source.m_ep[0];
	m_ep[1] = source.m_ep[1];
	m_ep[2] = source.m_ep[2];
	m_ep[3] = source.m_ep[3];
	
	return *this;
}


/// Normalization function
void EulerParameters::Normalize(void)
{
	double rnorm = 1.0/sqrt(m_ep[0]*m_ep[0] + m_ep[1]*m_ep[1] + m_ep[2]*m_ep[2] + m_ep[3]*m_ep[3]);

	m_ep[0] *= rnorm;
	m_ep[1] *= rnorm;
	m_ep[2] *= rnorm;
	m_ep[3] *= rnorm;
}


/// Compute Euler angles from Euler parameters
void EulerParameters::ExtractEulerZXZ(double &z0Angle, double &xAngle, double &z1Angle)
{
	xAngle = acos(2.0*(m_ep[0]*m_ep[0] + m_ep[3]*m_ep[3]) - 1.0);
	z0Angle = atan2(m_ep[1]*m_ep[3] + m_ep[0]*m_ep[2], m_ep[0]*m_ep[1] - m_ep[2]*m_ep[3]);
	z1Angle = atan2(m_ep[1]*m_ep[3] - m_ep[0]*m_ep[2], m_ep[2]*m_ep[3] + m_ep[0]*m_ep[1]);
}


/// Compute rotation matrix
void EulerParameters::ComputeRot(void)
{
	float temp = 2.0*m_ep[2]*m_ep[2];	// 2.0*p2^2
	m_rot[0] = 1.0 - temp;			// r11 = 1 - 2 p2^2
	m_rot[8] = 1 - temp;			// r33 = 1 - 2 p2^2
	temp = 2.0*m_ep[3]*m_ep[3];			// 2.0*p3^2
	m_rot[0] -= temp;				// r11 = 1 - 2 (p2^2 + p3^2)
	m_rot[4] = 1 - temp;			// r22 = 1 - 2 p3^2
	temp = 2.0*m_ep[1]*m_ep[1];			// 2.0*p1^2
	m_rot[4] -= temp;				// r22 = 1 - 2(p1^2 + p3^2)
	m_rot[8] -= temp;				// r33 = 1 - 2 (p1^2+ p2^2)

	temp = 2.0*m_ep[0]*m_ep[1];			// 2.0*p0p1
	m_rot[5] = - temp;				// r23 =  - 2p0p1
	m_rot[7] = temp;				// r32 = 2p0p1
	temp = 2.0*m_ep[0]*m_ep[2];			// 2.0*p0p2
	m_rot[2] = temp;				// r13 = 2 p0p2
	m_rot[6] = -temp;				// r31 = - 2 p0p2
	temp = 2.0*m_ep[0]*m_ep[3];			// 2.0*p0p3
	m_rot[1] = - temp;				// r12 = - 2 p0p3
	m_rot[3] = temp;				// r21 = 2 p0p3
	temp = 2.0*m_ep[1]*m_ep[2];			// 2.0*p1p2
	m_rot[1] += temp;				// r12 =2(p1p2 - p0p3)
	m_rot[3] += temp;				// r21 = 2(p1p2 + p0p3)
	temp = 2.0*m_ep[1]*m_ep[3];			// 2.0*p1p3
	m_rot[2] += temp;				// r13 = 2(p1p3 + p0p2)
	m_rot[6] += temp;				// r31 = 2(p1p3 - p0p2)
	temp = 2.0*m_ep[2]*m_ep[3];			// 2.0*p2p3
	m_rot[5] += temp;				// r23 =  2(p2p3 - p0p1)
	m_rot[7] += temp;				// r32 = 2(p2p3 + p0p1)
}


/// Return a refrence to parameters i
/*!	\param i : parameter number
	\return reference to parameter
*/
double & EulerParameters::operator()(int i)
{
	return m_ep[i];
}


/// Return the value of parameters i
/*!	\param i : parameter number
	\return value of parameter
*/
double EulerParameters::operator()(int i) const
{
	return m_ep[i];
}


/// Define the EulerParameters *= EulerParmeters operator
/*!	This operation corresponds to a rotation composition
	Beware this operation is not commutative
	\param val : EulerParameters
	\return this = this * val
*/
EulerParameters &EulerParameters::operator*=(const EulerParameters &val)
{
	double temp[4];
	temp[0] = m_ep[3]*val.m_ep[0] + m_ep[0]*val.m_ep[3] + m_ep[1]*val.m_ep[2] - m_ep[2]*val.m_ep[1];	// (Q1 * Q2).x = w1x2 + x1w2 + y1z2 - z1y2
	temp[1] = m_ep[3]*val.m_ep[1] - m_ep[0]*val.m_ep[2] + m_ep[1]*val.m_ep[3] + m_ep[2]*val.m_ep[0];	// (Q1 * Q2).y = w1y2 - x1z2 + y1w2 + z1x2
	temp[2] = m_ep[3]*val.m_ep[2] + m_ep[0]*val.m_ep[1] - m_ep[1]*val.m_ep[0] + m_ep[2]*val.m_ep[3];	// (Q1 * Q2).z = w1z2 + x1y2 - y1x2 + z1w2
	temp[3] = m_ep[3]*val.m_ep[3] - m_ep[0]*val.m_ep[0] - m_ep[1]*val.m_ep[1] - m_ep[2]*val.m_ep[2];	// (Q1 * Q2).w = w1w2 - x1x2 - y1y2 - z1z2

	for (int i=0; i < 4; i++)
		m_ep[i] = temp[i];

	Normalize();

	return *this;
}


/// Define the EulerParameters*EulerParameters operation (composition of rotations)
/*!	Beware this operation is not commutative
	\param ep1 : EulerParamters
	\param ep2 : EulerParamters
	\return ep1*ep2
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
	temp[3] = ep1.m_ep[3]*ep2.m_ep[0] - ep1.m_ep[2]*ep2.m_ep[1] + ep1.m_ep[1]*ep2.m_ep[2] + ep1.m_ep[0]*ep2.m_ep[2];

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
	temp[3] = ep1->m_ep[3]*ep2.m_ep[0] - ep1->m_ep[2]*ep2.m_ep[1] + ep1->m_ep[1]*ep2.m_ep[2] + ep1->m_ep[0]*ep2.m_ep[2];

	EulerParameters res(temp);

	res.Normalize();

	return res;
}


/// Compute rotation matriw between current value of Euler parameters and previous value
/*!	Beware this operation is not commutative
	\param previous : previous EulerParameters
	\param res : pointer to rotation matrix
	\return R(actual)*R(previous)^T
*/
void EulerParameters::StepRotation(const EulerParameters * previous, float *res)
{
	double temp_ep[4];

	// First we compute ep*(-previous)
	// note: -previous = (ep0, -ep1, -ep2, -ep3)

	// Assuming that: ep={a0, a1, a2, a3} and -previous={b0, -b1, -b2, -b3} we have
	// temp_ep[0] = a0 b0 + a1 b1 + a2 b2 + a3 b3
	temp_ep[0] = m_ep[0]*previous->m_ep[0] + m_ep[1]*previous->m_ep[1] + m_ep[2]*previous->m_ep[2] + m_ep[3]*previous->m_ep[3];
	// temp_ep[1] = a1 b0 - a0 b1 + a3 b2 - a2 b3
	temp_ep[1] = m_ep[1]*previous->m_ep[0] - m_ep[0]*previous->m_ep[1] + m_ep[3]*previous->m_ep[2] - m_ep[2]*previous->m_ep[3];	
	// temp_ep[2] = a2 b0 - a3 b1 - a0 b2 + a1 b3
	temp_ep[2] = m_ep[2]*previous->m_ep[0] - m_ep[3]*previous->m_ep[1] - m_ep[0]*previous->m_ep[2] + m_ep[1]*previous->m_ep[3];
	// temp_ep[3] = a3 b0 + a2 b1 - a1 b2 - a0 b3	
	temp_ep[3] = m_ep[3]*previous->m_ep[0] + m_ep[2]*previous->m_ep[1] - m_ep[1]*previous->m_ep[2] - m_ep[0]*previous->m_ep[3];

	
	// Then we compute the associated rotation matrix
	double temp = 2.0*temp_ep[2]*temp_ep[2];		// 2.0*p2^2
	res[0] = (float) (1.0 - temp);					// r00 = 1 - 2 p2^2
	res[8] = (float) (1 - temp);					// r33 = 1 - 2 p2^2
	temp = 2.0*temp_ep[3]*temp_ep[3];				// 2.0*p3^2
	res[0] -= (float) temp;							// r11 = 1 - 2 (p2^2 + p3^2)
	res[4] = (float) (1 - temp);					// r22 = 1 - 2 p3^2
	temp = 2.0*temp_ep[1]*temp_ep[1];				// 2.0*p1^2
	res[4] -= (float) temp;							// r22 = 1 - 2(p1^2 + p3^2)
	res[8] -= (float) temp;							// r33 = 1 - 2 (p1^2+ p2^2)

	temp = 2.0*temp_ep[0]*temp_ep[1];			// 2.0*p0p1
	res[5] = (float) (- temp);							// r23 =  - 2p0p1
	res[7] = (float) temp;								// r32 = 2p0p1
	temp = 2.0*temp_ep[0]*temp_ep[2];			// 2.0*p0p2
	res[2] = (float) temp;								// r13 = 2 p0p2
	res[6] = (float) (- temp);								// r31 = - 2 p0p2
	temp = 2.0*temp_ep[0]*temp_ep[3];			// 2.0*p0p3
	res[1] = (float) (- temp);							// r12 = - 2 p0p3
	res[3] = (float) temp;								// r21 = 2 p0p3
	temp = 2.0*temp_ep[1]*temp_ep[2];			// 2.0*p1p2
	res[1] += (float) temp;								// r12 =2(p1p2 - p0p3)
	res[3] += (float) temp;								// r21 = 2(p1p2 + p0p3)
	temp = 2.0*temp_ep[1]*temp_ep[3];			// 2.0*p1p3
	res[2] += (float) temp;								// r13 = 2(p1p3 + p0p2)
	res[6] += (float) temp;								// r31 = 2(p1p3 - p0p2)
	temp = 2.0*temp_ep[2]*temp_ep[3];			// 2.0*p2p3
	res[5] += (float) temp;								// r23 =  2(p2p3 - p0p1)
	res[7] += (float) temp;								// r32 = 2(p2p3 + p0p1)
}


void EulerParameters::StepRotation(const EulerParameters & previous, float *res)
{
	double temp_ep[4];

	// First we compute ep*(-previous)
	// note: -previous = (ep0, -ep1, -ep2, -ep3)
	temp_ep[0] = m_ep[0]*previous.m_ep[0] + m_ep[1]*previous.m_ep[1] + m_ep[2]*previous.m_ep[2] + m_ep[3]*previous.m_ep[3];
	temp_ep[1] = m_ep[1]*previous.m_ep[0] - m_ep[0]*previous.m_ep[1] + m_ep[3]*previous.m_ep[2] - m_ep[2]*previous.m_ep[3];	
	temp_ep[2] = m_ep[2]*previous.m_ep[0] - m_ep[3]*previous.m_ep[1] - m_ep[0]*previous.m_ep[2] + m_ep[1]*previous.m_ep[3];
	temp_ep[3] = m_ep[3]*previous.m_ep[0] + m_ep[2]*previous.m_ep[1] - m_ep[1]*previous.m_ep[2] - m_ep[0]*previous.m_ep[3];
	
	// Then we compute the associated rotation matrix
	double temp = 2.0*temp_ep[2]*temp_ep[2];		// 2.0*p2^2
	res[0] = (float) (1.0 - temp);					// r11 = 1 - 2 p2^2
	res[8] = (float) (1 - temp);					// r33 = 1 - 2 p2^2
	temp = 2.0*temp_ep[3]*temp_ep[3];				// 2.0*p3^2
	res[0] -= (float) temp;							// r11 = 1 - 2 (p2^2 + p3^2)
	res[4] = (float) (1 - temp);					// r22 = 1 - 2 p3^2
	temp = 2.0*temp_ep[1]*temp_ep[1];				// 2.0*p1^2
	res[4] -= (float) temp;							// r22 = 1 - 2(p1^2 + p3^2)
	res[8] -= (float) temp;							// r33 = 1 - 2 (p1^2+ p2^2)

	temp = 2.0*temp_ep[0]*temp_ep[1];				// 2.0*p0p1
	res[5] = (float) (- temp);						// r23 =  - 2p0p1
	res[7] = (float) temp;							// r32 = 2p0p1
	temp = 2.0*temp_ep[0]*temp_ep[2];				// 2.0*p0p2
	res[2] = (float) temp;							// r13 = 2 p0p2
	res[6] = (float) (- temp);						// r31 = - 2 p0p2
	temp = 2.0*temp_ep[0]*temp_ep[3];				// 2.0*p0p3
	res[1] = (float) (- temp);						// r12 = - 2 p0p3
	res[3] = (float) temp;							// r21 = 2 p0p3
	temp = 2.0*temp_ep[1]*temp_ep[2];				// 2.0*p1p2
	res[1] += (float) temp;							// r12 =2(p1p2 - p0p3)
	res[3] += (float) temp;							// r21 = 2(p1p2 + p0p3)
	temp = 2.0*temp_ep[1]*temp_ep[3];				// 2.0*p1p3
	res[2] += (float) temp;							// r13 = 2(p1p3 + p0p2)
	res[6] += (float) temp;							// r31 = 2(p1p3 - p0p2)
	temp = 2.0*temp_ep[2]*temp_ep[3];				// 2.0*p2p3
	res[5] += (float) temp;							// r23 =  2(p2p3 - p0p1)
	res[7] += (float) temp;							// r32 = 2(p2p3 + p0p1)
}


float3 EulerParameters::TransposeRot(const float3 &vec)
{
	float3 res;
	res.x = (float) (m_rot[0]*vec.x + m_rot[3]*vec.y + m_rot[6]*vec.z);
	res.y = (float) (m_rot[1]*vec.x + m_rot[4]*vec.y + m_rot[7]*vec.z);
	res.z = (float) (m_rot[2]*vec.x + m_rot[5]*vec.y + m_rot[8]*vec.z);

	return res;
}