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


class Cube: public Object {
	private:
		Point	m_origin;
		Vector	m_vx, m_vy, m_vz;
		double	m_lx, m_ly, m_lz;

	public:
		Cube(void);
		Cube(const Point&, const double, const double, const double, const EulerParameters&);
		Cube(const Point&, const Vector&, const Vector&, const Vector&);
		~Cube(void) {};

		double Volume(const double) const;
		void SetInertia(const double);
		
		void FillBorder(PointVect&, const double, const int, const bool*);
		void FillBorder(PointVect&, const double, const bool);
		void FillBorder(PointVect& points, const double dx)
		{
			FillBorder(points, dx, true);
		}
		
		int Fill(PointVect&, const double, const bool, const bool);
		int Fill(PointVect& points, const double dx, const bool fill = true)
		{
			return Fill(points, dx, true, fill);
		}
		
		void InnerFill(PointVect&, const double);
		
		void GLDraw(void) const;
		void GLDraw(const EulerParameters&, const Point&) const;
		
		bool IsInside(const Point&, const double) const;
};

#endif	/* _CUBE_H */

