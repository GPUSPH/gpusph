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

#ifndef OBJECT_H
#define	OBJECT_H

#include "Point.h"
#include "EulerParameters.h"

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
		EulerParameters		m_ep;			///< Euler parameters associated with the object
		Point				m_center;		///< Coordinates of center of gravity
		double				m_inertia[3];	///< Inertia matrix in the principal axes of inertia frame
		double				m_mass;			///< Mass of the object
		
	public:
		Object(void) {};
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
		 *  This function is pure virtual and then as to be defined at child level
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
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual void SetInertia(const double dx) = 0;
		virtual void SetInertia(const double*);
		virtual void GetInertialFrameData(double*, double&, double*, EulerParameters&) const;
		//@}
		
		/// \name Drawing functions
		//@{
		void GLDrawQuad(const Point&, const Point&, const Point&, const Point&) const;
		void GLDrawQuad(const EulerParameters&, const Point&, const Point&, const Point&, const Point&, const Point&) const;
		void GLDrawQuad(const EulerParameters&, const Point&, const Point&, const Point&, const Point&, const Vector&) const;
		void GLDrawLine(const Point &, const Point&) const;
		void GLDrawCircle(const EulerParameters&, const Point&, const double, const double) const;
		/// Draw a wireframe representation of the object
		/*!
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual void GLDraw(void) const = 0;
		/// Draw a rotated representation of the objct
		/*!	Given a position of the center of gravity and a rotation around it
		 *  defined by its Euler parameters, this function draw a wireframe
		 *	representation of the object after applying a rotation around 
		 *	the center of gravity of the given Euler parameters.
		 *	Typically this function is called when the object is a floating body.
		 *  \param ep : Euler parameters defining the rotation around center of gravity
		 *	\param cg : position of center of gravity
		 * 
		 *  This function is pure virtual and then as to be defined at child level
		 */
		virtual void GLDraw(const EulerParameters& ep, const Point& cg) const = 0;
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

