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

#ifndef _CYLINDER_H
#define	_CYLINDER_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"


class Cylinder: public Object {
	private:
		Point	m_origin;
		double	m_r;
		double	m_h;

	public:
		Cylinder(void);
		Cylinder(const Point&, const double, const Vector&);
		Cylinder(const Point&, const Vector&, const Vector&);
		Cylinder(const Point&, const double, const double, const EulerParameters&);
		~Cylinder(void) {};
		
		double Volume(const double) const;
		void Inertia(const double);

		void FillBorder(PointVect&, const double, const bool, const bool);
		void FillBorder(PointVect& points, const double dx) 
		{
			FillBorder(points, dx, true, true);
		}
		
		int Fill(PointVect&, const double, const bool fill = true);

		void GLDraw(void) const;
		void GLDraw(const EulerParameters&, const Point&) const;
		
		bool IsInside(const Point&, const double) const;
};

#endif	/* _CYLINDER_H */

