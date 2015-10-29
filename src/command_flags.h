/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _COMMAND_FLAGS_H
#define _COMMAND_FLAGS_H

#include "common_types.h"

// 0 reserved as "no flags"
#define NO_FLAGS	((flag_t)0)

// flags for kernels that process arguments differently depending on which
// step of the simulation we are at
// (e.g. forces, euler)
// these grow from the bottom
#define INITIALIZATION_STEP	((flag_t)1)
#define INTEGRATOR_STEP_1	(INITIALIZATION_STEP << 1)
#define INTEGRATOR_STEP_2	(INTEGRATOR_STEP_1 << 1)
#define	LAST_DEFINED_STEP	INTEGRATOR_STEP_2
// if new steps are added after INTEGRATOR_STEP_2, remember to update LAST_DEFINED_STEP

// flags to select which buffer to access, in case of double-buffered arrays
// these grow from the top
#define DBLBUFFER_WRITE		((flag_t)1 << (sizeof(flag_t)*8 - 1)) // last bit of the type
#define DBLBUFFER_READ		(DBLBUFFER_WRITE >> 1)

// now, flags used to specify the buffers to access for swaps, uploads, updates, etc.
// these start from the next available bit from the bottom and SHOULD NOT get past the highest bit available
// at the top

// start with a generic define that can be used to iterate over all buffers
#define FIRST_DEFINED_BUFFER	(LAST_DEFINED_STEP << 1)

#endif

