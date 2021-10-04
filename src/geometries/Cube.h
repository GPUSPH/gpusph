/*  Copyright (c) 2011-2017 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

/*
 * File:   Cube.h
 * Author: alexis
 *
 * Created on 14 juin 2008, 18:04
 */

#ifndef _CUBE_H
#define	_CUBE_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"
#include <ostream>

#include "deprecation.h"

//! Cube object class
/*!
 *	The cube class defines the cube object and implements all the
 *	methods necessary for using the object within a simulation as
 *	a fluid container or as a floating body.
 *
 * 	<TABLE border="0">
 * 	<TR>
 *  	<TD>\image html cube1.png</TD>
 * 		<TD>\image html cube2.png</TD>
 *	</TR>
 * 	</TABLE>
 *
 * The body frame (x', y', z') is centered at the center of gravity
 * of the cube and coincide with the principal axis of inertia frame.
 *
 * The cube is defined with an origin (i.e. the bottom left corner
 * in the body frame) expressed in the global frame (x, y, z), three
 * length (the length along x', y' and z') and an orientation ( the
 * relative orientation of the body frame respect to the global one).
*/
class Cube: public Object {
	private:
		Point	m_origin;	///< origin of the cube (bottom left corner in the body frame) expressed in the global reference frame
		Vector	m_vx;		///< vector representing the edge along x'
		Vector	m_vy;		///< vector representing the edge along y'
		Vector	m_vz;		///< vector representing the edge along z'
		double	m_lx;		///< length along x' axis
		double	m_ly;		///< length along y' axis
		double	m_lz;		///< length along z' axis

		uint3	m_nels;		///< number of fea elements in the three directins

	public:
		/// \name Constructors and destructor
		//@{
		Cube(void);
		Cube(const Point&, const double, const double, const double, int nelsx, int nelsy, int nelsz, const EulerParameters& = EulerParameters());
		Cube(const Point& o, const double lx, const double ly, const double lz, const EulerParameters& ep = EulerParameters()) :
			Cube(o, lx, ly, lz, 1, 1, 1, ep) {}
		Cube(const Point&, const Vector&, const Vector&, const Vector&) DEPRECATED_MSG("rotate a Cube(double, double, double) instead");
		virtual ~Cube(void) {};
		//@}

		/// \name Filling functions
		//@{
		void FillBorder(PointVect&, PointVect&, PointVect&, std::vector<uint4>&, const double, const bool);
		void FillBorder(PointVect&, const double, const int, const bool*);
		void FillBorder(PointVect&, const double, const bool);
		/// Fill the whole surface of the cube with particles
		/* 	Fill the whole surface of the cube with particles with a
		 * 	given particle spacing.
		 * 	\param points : vector where the particles will be added
		 * 	\param dx : particle spacing
		 */
		void FillBorder(PointVect& points, const double dx)
		{ FillBorder(points, dx, true);}
		int Fill(PointVect&, const double, const bool, const bool);
		/// Fill the cube with particles
		/* Fill the whole cube (including faces) with particles with a given
		 * particle spacing.
		 * 	\param points : vector where the particles will be added
		 * 	\param dx : particle spacing
		 * 	\param fill : if true add the particles to points otherwise just
		 * 				count the number of particles
		 * 	\return the number of particles used in the fill
		 */
		int Fill(PointVect& points, const double dx, const bool fill = true)
		{ return Fill(points, dx, true, fill);}
		void InnerFill(PointVect&, const double);
		void FillOut(PointVect&, const double, const int, const bool);
		void FillIn(PointVect&, const double, const int, const bool);
		void FillIn(PointVect&, const double, const int);
		//@}

		double Volume(const double) const;
		void SetInertia(const double);
		bool IsInside(const Point&, const double) const;

		void setEulerParameters(const EulerParameters &ep);
		void getBoundingBox(Point &output_min, Point &output_max);
		void shift(const double3 &offset);

		float4 getNaturalCoords(const double4 abs_coords);
		int4 getOwningNodes(const double4 abs_coords);

		/// \name Chrono related  functions
		//@{
#if USE_CHRONO == 1
		void BodyCreate(::chrono::ChSystem * bodies_physical_system, const double dx, const bool collide,
			const EulerParameters & orientation_diff);
		void CreateFemMesh(::chrono::ChSystem *fea_system);
#endif
		//@}
		friend std::ostream& operator<<(std::ostream&, const Cube&);
};
#endif	/* _CUBE_H */
