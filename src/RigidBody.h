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

#ifndef RIGIDBODY_H
#define	RIGIDBODY_H

#include <cstdio>

#include "EulerParameters.h"
#include "Point.h"
#include "Vector.h"
#include "Object.h"


/// Rigid body class
/*! Rigid body class provide:
	- definition of the rigid body
	- utilty functions (rotation, tranlation , ...)
	- integration of Euler equations
*/
class RigidBody {
	private:
		static int			m_bodies_number;	///< Total number of bodies
		int					m_body_number;		///< Number of body
		double				m_mass;				///< Mass of the body
		double				m_inertia[3];		///< Prinipal moments of SetInertia
		PointVect			m_parts;			///< Particles belonging to the rigid body
		EulerParameters*	m_ep;				///< Euler parameters
		double*				m_cg;				///< Center of gravity
//		double*				m_current_cg;		///< Current position of center of gravity
//		EulerParameters*	m_current_ep;		///< Current value of Euler parameters
		double*				m_vel;				///< Velovity of center of gravity
		double*				m_omega;			///< Angular velocity
		Point				m_current_cg;
		EulerParameters*	m_current_ep;
	public:
		Object*				m_object;			///< Pointer to object (used if rigid body is attached to an object)


	public:
		RigidBody(void);
		~RigidBody(void);

		/*! Adding particles to rigid body */
		void AddParts(const PointVect&);
		/*! Translate rigid body points */
		void Translate(const Vector&);
		/*! Rotate parts around a givent point */
		void Rotate(const Point&, const EulerParameters&);

		/* Setting SetInertia frame data */
		void SetInertialFrameData(const Point&, const double*, const double, const EulerParameters&);
		void AttachObject(Object*);

		/*! Setting initial values for integration */
		void SetInitialValues(const Vector&, const Vector&);

		/*! Return a reference to parts vect*/
		PointVect& GetParts(void);

		/*! Return center of gravity */
		void GetCG(float3&) const;
		const Point& GetCG(void) const;
		const EulerParameters& GetEulerParameters(void) const;


		/*! Perform an integration time step */
		void TimeStep(const float3&, const float3&, const float3&, const int, const double, float3*, float3*, float*);

		void Write(const float, FILE*) const;
};

#endif	/* RIGIDBODY_H */

