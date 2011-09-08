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
 * File:   Rect.h
 * Author: alexis
 *
 * Created on 14 juin 2008, 17:04
 */

#ifndef _RECT_H
#define	_RECT_H

#include "Point.h"
#include "Vector.h"
#include "Object.h"

class Rect: public Object {
	private:
		Point   origin;
		Vector  vx, vy;

	public:
		Rect(void);
		Rect(const Point& p, const Vector& v1, const Vector& v2);
		~Rect(void) {};

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);
		
		void FillBorder(PointVect& points, double dx, bool fill_top);
		void FillBorder(PointVect& points, double dx, bool populate_first,
				bool populate_last, int edge_num);
		void FillBorder(PointVect& points, double dx)
		{
			FillBorder(points, dx, true);
		}
		
		void Fill(PointVect& points, double dx, bool fill_edges);
		void Fill(PointVect& points, double dx, bool* edges_to_fill);
		void Fill(PointVect& points, double dx)
		{
			Fill(points, dx, true);
		}
				
		void GLDraw(void);
};
#endif	/* _RECT_H */
