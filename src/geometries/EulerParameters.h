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

#ifndef EULERPARAMETERS_H
#define	EULERPARAMETERS_H

#include <iostream>

#include "Point.h"
#include "Vector.h"
#include "ode/ode.h"

/// Euler parameters class
/*!
 *	Euler parameters class provide:
 *		- basic operations with Euler parameters
 *		- rotation matrix (and inverse rotation) from Euler parameters
 *		- access to parameters values
 *
 *	Euler parameters are normalized quaternions and then can represent any
 *	arbitrary rotation in space.
 *
 *	Associating a rotation matrix to Eulers parameters needs some computation
 *	so a precomputed rotation matrix is stored with the Euler parameters.
 *	Considering that choice the user must call the method computing the rotation
 *	matrix before making any rotation or inverse rotation.
 *
 *	There is a one-to-one correspondence between Euler parameters and Euler angles
 *	\f$(\psi, \theta, \phi)\f$.
 *	The Euler angles are used to represent the relative orientation between two
 *	coordinate system :
 *		- the rotation of angle \f$ \psi \f$ around (Oz) transforms (O,xyz) in (O, uvz).
 *		- the rotation of angle \f$ \theta \f$ around (Ou) transforms (O,uvz) in (O, uwz').
 *		- the rotation of angle \f$ \phi \f$ around (Oz) transforms (O,uwz') in (O, x'y'z').
 *
 * 	\htmlonly
 * 	<table border="0">
 * 	<tr>
 *  	<td><img src="EulerAngles.png" width="300px"></img></td>
 *	</tr>
 *  	<td align="center">Euler angles</td>
 *	</tr>
 * 	</table>
 *	\endhtmlonly
 *
 *	The three elemental rotations may occur either about the axes (xyz) of the original
 *	coordinate system, which is assumed to remain motionless (extrinsic rotations), or
 *	about the axes of the rotating coordinate system, which changes its orientation
 *	after each elemental rotation (intrinsic rotations). There are 6 possible choice
 *	and for sake of simplicity all Euler angles are defined according to the (zxz)
 *	intrinsic choice.
 *	The succession of rotations from coordinate system (xyz) to (x'y'z') in the (zxz)
 *	intrinsic convention is depicted in the animation below.
 *
 * 	\htmlonly
 * 	<table border="0">
 * 	<tr>
 *  	<td><img src="EulerZXZ.gif" width="380px"></img></td>
 *	</tr>
 *  	<td align="center">Coordinate change</td>
 *	</tr>
 * 	</table>
 *	\endhtmlonly
 *
 *	In the following documentation \f$ q=(q_0, q_1, q_2, q_3)\f$ will denotes
 *	a set of Euler parameters (i.e. with \f$ q^2_0 + q^2_1 + q^2_2 + q^2_3 = 1\f$).
*/
class EulerParameters {
	private:
		double		m_ep[4];			///< Values of Euler parameters
		double		m_rot[9];			///< Associated rotation matrix

	public:
		/// \name Constructors and destructor
		//@{
		EulerParameters(void);
		EulerParameters(const double *);
		EulerParameters(const float *);
		EulerParameters(const double, const double, const double);
		EulerParameters(const double, const double, const double, const double);
		EulerParameters(const float, const float, const float, const float);
		EulerParameters(const double3);
		EulerParameters(const float3);
		EulerParameters(const Vector &, const double);
		EulerParameters(const EulerParameters &);
		EulerParameters(const dQuaternion &);
		~EulerParameters(void) {};
		//@}

		static inline EulerParameters Identity(void) { return EulerParameters(); }

		/// \name Rotation related methods
		//@{
		EulerParameters Inverse(void);
		void ComputeRot(void);
		double3 Rot(const double3 &) const;
		float3 Rot(const float3 &) const;
		Point Rot(const Point &) const;
		Vector Rot(const Vector &) const;
		float3 TransposeRot(const float3 &) const;
		Vector TransposeRot(const Vector &) const;
		Point TransposeRot(const Point &) const;
		void GetRotation(float *) const;
		void StepRotation(const EulerParameters &, float *) const;
		//@}

		/// \name Utility methods
		//@{
		void Normalize(void);
		void ExtractEulerZXZ(double &, double &, double &) const;
		void ToODEQuaternion(dQuaternion &) const;
		void ToIdentity(void);
		//@}

		/** \name Access operators */
		//@{
		double & operator()(int);
		double operator()(int) const;
		double4 params() const;
		//@}

		/** \name Overloaded operators */
		//@{
		EulerParameters& operator = (const EulerParameters&);
		EulerParameters &operator *= (const EulerParameters &);
		//@}

		/** \name Overloaded friends operators */
		//@{
		friend EulerParameters operator+(const EulerParameters &, const EulerParameters &);
		friend EulerParameters operator*(const EulerParameters &, const EulerParameters &);
		friend EulerParameters operator*(const EulerParameters *, const EulerParameters &);
		friend EulerParameters operator*(const double, const EulerParameters &);
		//@}

		/** \name Debug printing functions */
		//@{
		void print(void) const;
		void printrot(void) const;
		friend std::ostream& operator<<(std::ostream&, const EulerParameters&);
		//@}
};
#endif	/* EULERPARAMETERS_H */
