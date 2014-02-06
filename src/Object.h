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

#ifndef OBJECT_H
#define	OBJECT_H

#include <stdexcept>

#include "Point.h"
#include "EulerParameters.h"
#include "ode/ode.h"

//! Object container class
/*!
 *	The Object class is a container class for geomtrical objects.
 *	It contains several utility function for drawing, filling, unfilling
 *  and setting or accesing object data along with pure virtual function
 *  that need to be implemented by his childs.
 *
 *
 *  For simplicity and efficiency reasons we use, often as possible, the
 *  principal axis of inertia frame as intermediate referential system
 *  and use Euler parameters to specify its relative situation respect
 *  to the global frame.
*/


class Object {
	protected:
		EulerParameters		m_ep;					///< Euler parameters associated with the object
		dMatrix3 			m_ODERot;				///< ODE rotation matrix associated to the object
		Point				m_center;				///< Coordinates of center of gravity
		double				m_inertia[3];			///< Inertia matrix in the principal axes of inertia frame
		double				m_mass;					///< Mass of the object
		PointVect			m_parts;				///< Particles belonging to the object
	public:
		dBodyID				m_ODEBody;		///< ODE body ID associated with the object
		dGeomID				m_ODEGeom;		///< ODE geometry ID assicuated with the object
		dMass				m_ODEMass;		///< ODE iniertial parameters of the object

		Object(void) {
			m_ODEBody = 0;
			m_ODEGeom = 0;
			dRSetIdentity (m_ODERot);
			m_center = Point(0,0,0);
		};

		virtual ~Object(void) {};

		/// \name Mass related functions
		//@{
		virtual double SetPartMass(const double, const double);
		virtual void SetPartMass(const double);
		/// Return the volume of an object
		/*! Given a particle spacing, this function should return
		 *	the volume of the object.
		 *	In the SPH framework it's normal that the volume of objects
		 *	depends on particle spacing: for example if we fill a cube of side
		 *	l with particles spaced by dx, the volume occupied by the particles
		 *	is \f$(l + dx)^3 \f$.
		 *	\param dx : particle spacing
		 *	\return volume of the object
		 *
		 *	This function is pure virtual and then has to be defined at child level
		 */
		virtual double Volume(const double dx) const = 0;
		virtual double SetMass(const double, const double);
		virtual void SetMass(const double);
		//@}

		/// \name Inertia related functions
		//@{
		/// Compute the matrix of inertia
		/*! This function compute the matrix of inertia of the obkect in the inertial
		 *  frame (i.e. the 3 diagonal components) and store it in the m_inertia array.
		 *	For the same reasons as volume, the inertia depends on particle spacing.
		 *	\param dx : particle spacing
		 *
		 *	This function is pure virtual and then has to be defined at child level
		 */
		virtual void SetInertia(const double dx) = 0;
		virtual void SetInertia(const double*);
		virtual void GetInertialFrameData(double*, double&, double*, EulerParameters&) const;
		//@}

		/// Returns the particle vector associatet with the object
		PointVect& GetParts(void);

		/// \name ODE related functions
		/* These are not pure virtual to allow new GPUSPH Objects to be defined without
		 * needing an ODE counterpart, but the default implementation will just throw
		 * an exception
		 */
		//@{
		/// Create an ODE body in the specified ODE world and space
		virtual void ODEBodyCreate(dWorldID, const double, dSpaceID ODESpace = 0)
		{ throw std::runtime_error("ODEBodyCreate called but not defined!"); }
		/// Create an ODE geometry in the specified ODE space
		virtual void ODEGeomCreate(dSpaceID, const double)
		{ throw std::runtime_error("ODEGeomCreate called but not defined!"); }
		//@}


		/// \name Filling functions
		//@{
		int FillDisk(PointVect&, const EulerParameters&, const Point&, const double,
					const double, const double, const bool fill = true) const;
		int FillDisk(PointVect&, const EulerParameters&, const Point&, const double,
					const double, const double, const double, const bool fill = true) const;
		int FillDiskBorder(PointVect&, const EulerParameters&, const Point&, const double,
					const double, const double, const double, const bool fill = true) const;
		/// Fill object surface with particles
		/*!	Fill the object surface with particle at a given particle spacing and add
		 *	the particles to the given particle vector
		 *  the number of particles added.
		 *	\param points : particle vector to add particles to
		 *  \param dx : particle spacing
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual void FillBorder(PointVect& points, const double dx) = 0;
		/// Fill object with particles
		/*!	Fill the whole object with particle at a given particle spacing and return
		 *  the needed number of particles.
		 *
		 *  If the fill parameter is set to false the function just count the number of
		 *  particles needed otherwise the particles are added to the particle vector.
		 *
		 *	\param points : particle vector to add particles to
		 *	\param dx : particle spacing
		 *  \param fill : fill flag (true particles are generated and added to parts, false no particle is generated)
		 *	\return number of particles needed to fill the object
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual int Fill(PointVect& points, const double dx, const bool fill) = 0;
		virtual void Fill(PointVect& , const double);
		void Unfill(PointVect&, const double) const;
		//@}

		/// Detect if a particle is inside an object
		/*!	Detect if a perticle is located inside the object or at a distance inferior
		 *  to a threshold value.
		 *	\param dx : threshold value
		 *	\return true if particle is inside the object or closer than dx
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual bool IsInside(const Point& p, const double dx) const = 0;
};
#endif	/* OBJECT_H */

