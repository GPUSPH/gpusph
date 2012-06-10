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
 * File:   TopoCube.h
 * Author: alexis
 *
 * Created on 14 juin 2008, 18:04
 */

#ifndef _TOPOCUBE_H
#define	_TOPOCUBE_H

#include "Object.h"
#include "Point.h"
#include "Vector.h"


class TopoCube: public Object {
	private:
		Point	m_origin;
		Vector	m_vx, m_vy, m_vz;
		float*	m_dem;
		int		m_ncols, m_nrows;
		double	m_nsres, m_ewres;
		bool	m_interpol;
		double	m_H;

	public:
		TopoCube(void);
		virtual ~TopoCube(void);

		void SetCubeDem(const double, const float*, const int, const int, const double, const double, const bool);

		double SetPartMass(const double, const double);
		double Volume(const double dx) const
		{
			return 0.0;
		}
		void SetInertia(const double dx)
		{
			m_inertia[0] = 0.0;
			m_inertia[1] = 0.0;
			m_inertia[2] = 0.0;
		}
		
		void FillBorder(PointVect&, const double, const int, const bool);
		void FillBorder(PointVect& points, const double dx)
		{
			FillBorder(points, dx, 0, true);
			FillBorder(points, dx, 1, false);
			FillBorder(points, dx, 2, true);
			FillBorder(points, dx, 3, false);
		}
		
		void FillDem(PointVect&, const double);
		double DemInterpol(const double, const double);
		double DemDist(const double, const double, const double, const double);
		
		int Fill(PointVect&, const double, const double, const bool, const bool);
		int Fill(PointVect& points, const double H, const double dx, const bool faces_filled)
		{
			return Fill(points, H, dx, faces_filled, true);
		}
		int Fill(PointVect& points, const double dx, const bool fill = true)
		{
			return Fill(points, m_H, dx, false, fill);
		}
		
		void GLDraw(void) const;
		void GLDraw(const EulerParameters&, const Point&) const;
		
		bool IsInside(const Point&, const double) const;
};

#endif	/* _CUBE_H */

