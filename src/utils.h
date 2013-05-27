/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto Nazionale di Geofisica e Vulcanologia
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

/* Common host-side utility functions */

#ifndef _UTILS_H
#define _UTILS_H

// Compute a/b rounding up instead of down. The type T is supposed to be an
// integer, since the code behaves as expected only for integer division.
// Commonly used e.g. to compute the number of blocks to launch in a kernel.
template<typename T>
T div_up(T a, T b) {
	return (a + b - 1)/b;
}

// Round a up to the next multiple of b.
template<typename T>
T round_up(T a, T b) {
	return div_up(a, b)*b;
}

#endif
