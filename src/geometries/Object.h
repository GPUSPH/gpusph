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
 *	The Object class is a container class for geometrical objects.
 *	It contains several utility function for drawing, filling, unfilling
 *  and setting or accessing object data along with pure virtual function
 *  that need to be implemented by his children.
 *
 *
 *  For simplicity and efficiency reasons we use, often as possible, the
 *  principal axis of inertia frame as intermediate referential system
 *  and use Euler parameters to specify its relative situation respect
 *  to the global frame.
*/
class Object {
	protected:
		EulerParameters		m_ep;			///< Euler parameters associated with the object
		dMatrix3			m_ODERot;		///< ODE rotation matrix associated to the object
		Point				m_center;		///< Coordinates of center of gravity in the global reference frame
		double				m_inertia[3];	///< Inertia matrix in the principal axes of inertia frame
		double				m_mass;			///< Mass of the object
		PointVect			m_parts;		///< Particles belonging to the object
		uint				m_numParts;		///< Number of particles belonging to the object

		// auxiliary function for computing the bounding box
		void getBoundingBoxOfCube(Point &out_min, Point &out_max,
			Point &origin, Vector v1, Vector v2, Vector v3);
	public:
		dBodyID				m_ODEBody;		///< ODE body ID associated with the object
		dGeomID				m_ODEGeom;		///< ODE geometry ID associated with the object
		dMass				m_ODEMass;		///< ODE inertial parameters of the object

		Object(void) {
			m_ODEBody = 0;
			m_ODEGeom = 0;
			m_mass = 0.0;
			dRSetIdentity (m_ODERot);
			m_center = Point(0,0,0);
			m_numParts = 0;
			m_inertia[0] = NAN;
			m_inertia[1] = NAN;
			m_inertia[2] = NAN;
		};

		virtual ~Object(void) {};

		/// \name Mass related functions
		//@{
		virtual double SetPartMass(const double, const double);
		virtual void SetPartMass(const double);
		double GetPartMass();
		virtual double SetMass(const double, const double);
		virtual void SetMass(const double);
		double GetMass();
		virtual double Volume(const double dx) const = 0;
		//@}

		/// \name Inertia related functions
		//@{
		/// Compute the matrix of inertia
		/*! This function compute the matrix of inertia of the object in the inertial
		 *  frame (i.e. the 3 diagonal components) and store it in the m_inertia array.
		 *	For the same reasons as volume, the inertia depends on particle spacing.
		 *	\param dx : particle spacing
		 *
		 *	This function is pure virtual and then has to be defined at child level
		 */
		virtual void SetInertia(const double dx) = 0;
		void SetInertia(const double*);
		void SetCenterOfGravity(const double*);
		double3 GetCenterOfGravity(void) const;
		EulerParameters GetOrientation(void) const;
		virtual void GetInertialFrameData(double*, double&, double*, EulerParameters&) const;
		//@}

		/// Returns the particle vector associated with the object
		PointVect& GetParts(void);

		/// Sets the number of particles associated with an object
		void SetNumParts(const int numParts);
		/// Gets the number of particles associated with an object
		/*! This function either returns the set number of particles which is used
		 *  in case of a loaded STL mesh or the number of particles set in m_parts
		 */
		uint GetNumParts();

		/// \name ODE related functions
		/* These are not pure virtual to allow new GPUSPH Objects to be defined without
		 * needing an ODE counterpart, but the default implementation will just throw
		 * an exception
		 */
		//@{
		/// Create an ODE body in the specified ODE world and space
		/*! \throws std::runtime_error if the method is not implemented
		 */
		virtual void ODEBodyCreate(dWorldID, const double, dSpaceID ODESpace = 0)
		{ throw std::runtime_error("ODEBodyCreate called but not defined!"); }
		/// Create an ODE geometry in the specified ODE space
		/*! \throws std::runtime_error if the method is not implemented
		 */
		virtual void ODEGeomCreate(dSpaceID, const double)
		{ throw std::runtime_error("ODEGeomCreate called but not defined!"); }
		/// Return the ODE body ID associated with the Object
		/*! \return the body ID associated with the object
		 *	\throws std::runtime_error if the object has no associated ODE body
		 */
		dBodyID ODEGetBody(void)
		{	if (!m_ODEBody)
				throw std::runtime_error("ODEGetBody called but object is not associated with an ODE body !");
			return m_ODEBody; }
		/// Print ODE-related information such as position, CG, geometry bounding box (if any), etc.
		void ODEPrintInformation(const bool print_geom = true);
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
		 *	\param dx : particle spacing
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
		 *	\param points : particle vector to add particles to
		 *	\param dx : particle spacing
		 *	\param fill : fill flag (true particles are generated and added to parts, false no particle is generated)
		 *	\return number of particles needed to fill the object
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual int Fill(PointVect& points, const double dx, const bool fill = true) = 0;
		/// Fill object with a specified number of particles layer
		/*!	Fill multiple layers of particles starting from the object surface
		 *
		 *	\param points : particle vector to add particles to
		 *	\param dx : particle spacing
		 *	\param layers : number of layers
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual void FillIn(PointVect& points, const double dx, const int layers) = 0;
		void Unfill(PointVect&, const double) const;
		void Intersect(PointVect&, const double) const;
		//@}

		/// Detect if a particle is inside an object
		/*!	Detect if a particle is located inside the object or at a distance inferior
		 *  to a threshold value.
		 *	\param dx : threshold value
		 *	\return true if particle is inside the object or closer than dx
		 *
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual bool IsInside(const Point& p, const double dx) const = 0;

		/// \name Other functions
		//@{
		/// Set the EulerParameters
		/*! This function sets the EulerParameters and updateds the object accordingly
		 *	\param ep : new EulerParameters
		 *
		 *	This function is pure virtual and then has to be defined at child level
		 */
		virtual void setEulerParameters(const EulerParameters &ep) = 0;

		/// Get the EulerParameters
		/*! This function returns the EulerParameters
		 *	\return EulerParameters
		 *
		 *	This function is pure virtual and then has to be defined at child level
		 */
		const EulerParameters* getEulerParameters() {return &m_ep; }

		/// Update the ODE rotation matrix according to the EulerParameters
		void updateODERotMatrix();

		/// Get the bounding box
		/*! This function writes the bounding box of the object in the given parameters,
		 *  taking into account also the object rotation
		 *  \param min : minimum coodinates
		 *  \param min : maximum coodinates
		 *
		 *  This function is pure virtual and then has to be defined at child level.
		 */
		virtual void getBoundingBox(Point &output_min, Point &output_max) = 0;

		/// Shift the object (center, origin, etc.) with the given offset
		/*! This function shifts the object with the given offset. The object
		 *  internally updates everything necessary.
		 *  \param double3 : offset
		 *
		 *  This function is pure virtual and then has to be defined at child level.
		 */
		virtual void shift(const double3 &offset) = 0;
};
#endif	/* OBJECT_H */

