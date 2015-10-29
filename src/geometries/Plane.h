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

#ifndef _PLANE_H
#define _PLANE_H

#include "Object.h"

//! Plane object class
class Plane: public Object {
	private:
		double m_a;		// a coefficient of implicit equation
		double m_b;		// b coefficient of implicit equation
		double m_c;		// c coefficient of implicit equation
		double m_d;		// d coefficient of implicit equation
		double m_norm;	// norm

	public:
		/// \name Constructors and destructor
		//@{
		Plane(const double a, const double b, const double c, const double d);
		~Plane() {};
		//@}

		double Volume(const double dx) const { return 0.0; }
		void SetInertia(const double dx);

		/// \name Filling functions
		//@{
		void FillBorder(PointVect& points, const double dx);
		int Fill(PointVect& points, const double dx, const bool fill);
		void FillIn(PointVect& points, const double dx, const int layers);

		bool IsInside(const Point& p, const double dx) const;
		void setEulerParameters(const EulerParameters &ep);
		void getBoundingBox(Point &output_min, Point &output_max);
		void shift(const double3 &offset);

		// getters
		double getA() { return m_a; }
		double getB() { return m_b; }
		double getC() { return m_c; }
		double getD() { return m_d; }
		double getNorm() { return m_norm; }

		/// \name ODE related  functions
		//@{
		void ODEBodyCreate(dWorldID world, const double dx, dSpaceID ODESpace);
		void ODEGeomCreate(dSpaceID space, const double dx);
		//@}
};

#endif /* _PLANE_H */

