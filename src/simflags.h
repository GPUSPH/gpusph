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

/* Set of boolean aspects of the simulation, to determine if
 * any of the features is enabled (XSPH, adaptive timestep, moving
 * boundaries, inlet/outlet, DEM, Ferrari correction, etc)
 */

#ifndef _SIMFLAGS_H
#define _SIMFLAGS_H

#include "common_types.h"

// TODO macros to test presence of flag

// no options
#define ENABLE_NONE			0UL
#define DISABLE_ALL_SIMFLAGS	(ENABLE_NONE)

// adaptive timestepping
#define ENABLE_DTADAPT			1UL

// XSPH
#define ENABLE_XSPH				(ENABLE_DTADAPT << 1)

// planes
#define ENABLE_PLANES			(ENABLE_XSPH << 1)
// DEM
#define ENABLE_DEM				(ENABLE_PLANES << 1)

// moving boundaries and rigid bodies
#define ENABLE_MOVING_BODIES	(ENABLE_DEM << 1)

// inlet/outlet
#define ENABLE_INLET_OUTLET		(ENABLE_MOVING_BODIES << 1)

// water depth computation
#define ENABLE_WATER_DEPTH		(ENABLE_INLET_OUTLET << 1)

// Ferrari correction
#define ENABLE_FERRARI			(ENABLE_WATER_DEPTH << 1)

// Density diffusion (Molteni & Colagrossi 2009)
#define ENABLE_DENSITY_DIFFUSION (ENABLE_FERRARI << 1)

// Summation density
#define ENABLE_DENSITY_SUM		(ENABLE_DENSITY_DIFFUSION << 1)

// Compute gamma through Gauss quadrature forumla
#define ENABLE_GAMMA_QUADRATURE		(ENABLE_DENSITY_SUM << 1)

#define LAST_SIMFLAG		ENABLE_GAMMA_QUADRATURE

// since flags are a bitmap, LAST_SIMFLAG - 1 sets all bits before
// the LAST_SIMFLAG bit, and OR-ing with LAST_SIMFLAG gives us
// all flags. This is slightly safer than using ((LAST_SIMFLAG << 1) - 1)
// in case LAST_SIMFLAG is already the last bit
#define ENABLE_ALL_SIMFLAGS		(LAST_SIMFLAG | (LAST_SIMFLAG-1))

/// General query that identifies whether the flags in field are set, true only if all of them are
#define QUERY_ALL_FLAGS(field, flags)	(((field) & (flags)) == (flags))
/// General query that identifies whether at least one flag in field is set
#define QUERY_ANY_FLAGS(field, flags)	((field) & (flags))

#endif
