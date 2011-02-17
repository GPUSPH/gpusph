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
 *  Vector.h
 *  NNS
 *
 *  Created by Alexis Herault on 27/07/06.
 *  Copyright 2006 __MyCompanyName__. All rights reserved.
 *
 */

#ifndef _GEOMVECTOR_H
#define _GEOMVECTOR_H

#include "Point.h"

class Point;

/// 3D vector class
/*! 3D vector class provide :
	- standard operators for vectors
	- norm of vector
	- normal vector
	- access to coordinates values
*/
class Vector {
	private:
		double	x[3];	///< coordinates of vector

	public:
		Vector(const Point &, const Point &);
		Vector(const Vector &);
		Vector(double xx = 0, double yy = 0, double zz = 0);
		~Vector(void) {};

		/*! Return the norm of vector */
		double norm(void) const;
		double normSquared(void) const;
		/*! Return a normal vector of vector */
		Vector Normal(void);
		/*! Return the vector rotated by the given angle (in radians)
			around the given vector.
		 */
		Vector rotated(const double &angle, const Vector &normal) const;

		/*! \name
			Overloaded operators
		*/
		//\{
		Vector &operator+=(const Vector &);
		Vector &operator-=(const Vector &);
		Vector &operator*=(double);
		Vector &operator/=(double);
		Vector &operator=(const Vector &);
		double &operator()(int);
		double operator()(int) const;
		//\}

		// DEBUG
		void print(void);

		/*! \name
			Overloaded friend operators
		*/
		//\{
		friend Vector operator+(const Vector &, const Vector &);
		friend Vector operator-(const Vector &, const Vector &);
		friend Vector operator*(double, const Vector &);
		friend Vector operator*(const Vector &, double);
		friend double operator*(const Vector &, const Vector &);
		friend Vector operator/(const Vector &, double);
		friend Vector operator-(const Vector &);
		//\}
};
#endif
