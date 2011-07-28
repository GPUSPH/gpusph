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

#include "Point.h"
#include "Vector.h"

class Cylinder {
	private:
		Point	center;
		Vector	radius, height;

	public:
		Cylinder(void);
		Cylinder(const Point &center, const Vector &radius, const Vector &height);
		~Cylinder(void) {};

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);

		void FillBorder(PointVect& points, double dx, bool bottom, bool top);
		void Fill(PointVect& points, double dx);

		void GLDraw(void);
};

#endif	/* _CYLINDER_H */

