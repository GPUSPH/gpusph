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

#include "Matrix33.h"
#include "Vector.h"

/// Euler parameters class
/*! Euler parameters class provide:
	- basic operations with Eulers parameters
	- rotation matrix (and inverse rotation) from Euler parameters
	- access to parameters values
*/
class EulerParameters {
	private:
		double		m_ep[4];			///< Euler parameters
		double		m_rot[9];			///< Rotation matrix

	public:
		EulerParameters(void);
		EulerParameters(const double *);
		EulerParameters(const double, const double, const double);
		EulerParameters(const Vector &, const double);
		EulerParameters(const EulerParameters &);
		~EulerParameters(void) {};

		/*! Assignment operator */
		EulerParameters& operator= (const EulerParameters& ep);

		/*! Normalize Euler parameters */
		void Normalize(void);
		/* Compute and return rotation matrix */
		//const float * GetRotation(void) const;

		/* Compute EUler angles from parameters */
		void ExtractEulerZXZ(double &, double &, double &);

		/*! Compute roation between actual and previous parameters */
		void StepRotation(const EulerParameters *, float *);
		void StepRotation(const EulerParameters &, float *);

		float3 TransposeRot(const float3 &);

		/*! \name
			Overloaded operators
		*/
		//\{
		double & operator()(int);
		double operator()(int) const;
		
		EulerParameters &operator*=(const EulerParameters &);
		//\}

		/*! \name
			Overloaded friend operators
		*/
		//\{
		friend inline EulerParameters operator*(const EulerParameters &, const EulerParameters &);
		friend inline EulerParameters operator*(const EulerParameters *, const EulerParameters &);
		//\}

		void ComputeRot(void);
};
#endif	/* EULERPARAMETERS_H */

