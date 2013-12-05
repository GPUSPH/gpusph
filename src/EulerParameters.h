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

#ifndef EULERPARAMETERS_H
#define	EULERPARAMETERS_H

#include "Point.h"
#include "Vector.h"

//! Euler parameters class
/*! 
 *	Euler parameters class provide:
 *		- basic operations with Euler parameters
 *		- rotation matrix (and inverse rotation) from Euler parameters
 *		- access to parameters values
 *
 *	Euler parameters are normalized quaternions and then can represent any 
 *	arbitrary rotation in space.
 * 
 *	In the following documentation \f$ q=(q_0, q_1, q_2, q_3)\f$ will denotes 
 *	a set of Euler parameters (i.e. with \f$ q^2_0 + q^2_1 + q^2_2 + q^2_3 = 1\f$).
*/


class EulerParameters {
	private:
		double		m_ep[4];			///< Values of Euler parameters 
		double		m_rot[9];			///< Associated rotation matrix
		
	public:
		EulerParameters(void);
		EulerParameters(const double *);
		EulerParameters(const float *);
		EulerParameters(const double, const double, const double);
		EulerParameters(const Vector &, const double);
		EulerParameters(const EulerParameters &);
		~EulerParameters(void) {};

		EulerParameters& operator = (const EulerParameters&);

		void Normalize(void);

		void ExtractEulerZXZ(double &, double &, double &) const;

		void ComputeRot(void);
		
		/// \name Rotation 
		//@{
		float3 Rot(const float3 &) const;
		Point Rot(const Point &) const;
		Vector Rot(const Vector &) const;
		//@}
		
		/// \name Inverse rotation
		//@{
		float3 TransposeRot(const float3 &) const;
		Vector TransposeRot(const Vector &) const;
		Point TransposeRot(const Point &) const;
		//@}
		
		/// \name Rotation matrix between two Euler parameters
		//@{
		void StepRotation(const EulerParameters &, float *) const;
		//@}
		
		/** \name Access operators */
		//@{
		double & operator()(int);
		double operator()(int) const;
		//@}

		EulerParameters &operator*=(const EulerParameters &);
		
		/** \name Overloaded friends operators */
		//@{
		friend EulerParameters operator*(const EulerParameters &, const EulerParameters &);
		friend EulerParameters operator*(const EulerParameters *, const EulerParameters &);
		//@}
		
		/** \name Printing functions */
		//@{
		void print(void) const;
		void printrot(void) const;
		//@}
};


/// Return a refrence to parameter i
/*!	\param i : parameter number
 *	\return reference to parameter i
*/ 
inline double & 
EulerParameters::operator()(int i)
{
	return m_ep[i];
}


/// Return the value of parameter i
/*!	\param i : parameter number
 *	\return value of parameter i
*/
inline double 
EulerParameters::operator()(int i) const
{
	return m_ep[i];
}
#endif	/* EULERPARAMETERS_H */
