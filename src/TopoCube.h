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

#include "Point.h"
#include "Vector.h"
#include "Object.h"

class TopoCube: public Object {
	private:
		Point	origin;
		Vector	vx, vy, vz;
		float*	m_dem;
		int		m_ncols, m_nrows;
		float	m_nsres, m_ewres;
		bool	m_interpol;
		float	m_H;

	public:
		TopoCube(void);
		~TopoCube(void);

		void SetCubeDem(float H, float *dem, int ncols, int nrows, float nsres, float ewres, bool interpol);

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);
		
		void FillBorder(PointVect& points, double dx, int face_num, bool fill_edges);
		void FillBorder(PointVect& points, double dx)
		{
			FillBorder(points, dx, 0, true);
			FillBorder(points, dx, 1, false);
			FillBorder(points, dx, 2, true);
			FillBorder(points, dx, 3, false);
		}
		
		void FillDem(PointVect& points, double dx);
		float DemInterpol(float x, float y);
		float DemDist(float x, float y, float z, float dx);
		
		void Fill(PointVect& points, double H, double dx, bool faces_filled);
		void Fill(PointVect& points, double dx)
		{
			Fill(points, m_H, dx, false);
		}
		
		void GLDraw(void);
};

#endif	/* _CUBE_H */

