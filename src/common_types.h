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

/* Common types used throughout GPUSPH */

#ifndef _COMMON_TYPES_H
#define _COMMON_TYPES_H

// uint64_t et similia
#include <stdint.h>
// size_t
#include <stddef.h>

// define uint, uchar, ulong
typedef unsigned long ulong; // only used in timing
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

// neighbor data
typedef unsigned short neibdata;

// type for index that iterates on the neighbor list
typedef size_t idx_t;

// particleinfo cum suis
#include "particleinfo.h"

// hashKey cum suis
#include "hashkey.h"

// flags type
// could be made an uint_fast64_t if we were concerned about performance,
typedef uint64_t flag_t;
#define FLAG_MAX UINT64_MAX

#endif
