/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

/*! \file
 * Common types used throughout GPUSPH
 */

#ifndef _COMMON_TYPES_H
#define _COMMON_TYPES_H

/* We need to include stdint for uint64_t and similia,
 * stddef for size_t. When using C++11 or higher, we also
 * want to avoid global pollution from the std namespace,
 * due to our intention to switch to Chrono, whose choice
 * of namespace conflicts with std::chrono, so we want
 * to use cstdint, which is not available in C++98:
 */
#if __cplusplus < 201103L
#include <stdint.h>
#else
#include <cstdint>
#endif

// size_t
#include <cstddef>

// define uint, uchar, ulong
typedef unsigned long ulong; // only used in timing
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;

// neighbor data
typedef unsigned short neibdata;

/* The neighbor cell num ranges from 1 to 27 (included), so it fits in
 * 5 bits, which we put in the upper 5 bits of the neibdata, which is
 * 16-bit wide.
 * TODO actually compute this from sizeof(neibdata)
 */
#define CELLNUM_SHIFT	11
#define CELLNUM_ENCODED	(1U<<CELLNUM_SHIFT)
#define NEIBINDEX_MASK	(CELLNUM_ENCODED-1)
#define ENCODE_CELL(cell) ((cell + 1) << CELLNUM_SHIFT)
#define DECODE_CELL(data) ((data >> CELLNUM_SHIFT) - 1)

#define NEIBS_END	USHRT_MAX
#define CELL_EMPTY	UINT_MAX

// type for index that iterates on the neighbor list
typedef size_t idx_t;

// particleinfo cum suis
#include "particleinfo.h"

// hashKey cum suis
#include "hashkey.h"

//! Type used to hold flags
//! We could make this into an uint_fast64_t if we were concerned about performance
typedef uint64_t flag_t;
//! Value 0 reserved as "no flags"
#define NO_FLAGS	(UINT64_C(0))
//! Maximum value representable by a flag_t
#define FLAG_MAX	UINT64_MAX
#define ALL_FLAGS	FLAG_MAX


#endif
