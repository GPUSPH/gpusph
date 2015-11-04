/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/* Common host-side utility functions and macros */

#ifndef _UTILS_H
#define _UTILS_H

#include <cmath>
#include <cfloat>

// use STR(SOMEMACRO) to turn the content of SOMEMACRO into a string
#define _STR(x) #x
#define STR(x) _STR(x)

// Compute a/b rounding up instead of down. The type T is supposed to be an
// integer, since the code behaves as expected only for integer division.
// Commonly used e.g. to compute the number of blocks to launch in a kernel.
template<typename T>
inline
T div_up(T a, T b) {
	return (a + b - 1)/b;
}

// for non-integral types:
inline
float div_up(float a, float b) {
	return std::ceil(a/b);
}
inline
double div_up(double a, double b) {
	return std::ceil(a/b);
}

// Round a up to the next multiple of b.
template<typename T>
T round_up(T a, T b) {
	return div_up(a, b)*b;
}

// check if a is a multiple of b
inline
bool is_multiple(double a, double b) {
	double div = a/b;
	int i_div = int(div);
	double bmul = i_div*b;
	return fabs(bmul - a) < FLT_EPSILON*fabs(bmul+a);
}

#endif
