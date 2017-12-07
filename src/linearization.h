/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A.
 	Dalrymple, Eugenio Rustico, Ciro Del Negro

	Conservatoire National des Arts et Metiers, Paris, France

	Istituto Nazionale di Geofisica e Vulcanologia,
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


/* The following defines constitute an abstraction of the cartesian coordinates to easily change
 * the order they are used in cell linearization in the whole GPUSPH. This enables fine tuning
 * of the linearization function for performance improvements. In particular, MULTI_DEVICE
 * simulations will benefit of it when the major split axis is COORD3: this means that all the
 * particles in an edging slice (orthogonal to COORD3 axis) will be consecutive in memory and
 * thus eligible for a single burst transfer.
 * Cells with consecutive COORD1 are consecutive in their linearized index. */

#include "linearization_select.opt"
