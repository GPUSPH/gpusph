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

#ifndef _CIRCLE_H
#define	_CIRCLE_H

#include "Point.h"
#include "Vector.h"
#include "Object.h"


class Circle: public Object {
	private:
		Point	center;
		Vector	radius;
		Vector	normal;

	public:
		Circle(void);
		Circle(const Point &p, const Vector &r, const Vector &u);
		~Circle(void) {};

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);
		
		void FillBorder(PointVect& points, double dx);
		
		void Fill(PointVect& points, double dx, bool fill_edge);
		void Fill(PointVect& points, double dx)
		{
			Fill(points, dx, true);
		}
		
		void GLDraw(void);

};
#endif	/* _CIRCLE_H */


