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

#ifndef _HASHKEY_H
#define _HASHKEY_H

// for CELLTYPE_BITMASK
#include "multi_gpu_defines.h"
// for HAS_KEY_SIZE
#include "hash_key_size_select.opt"

/*
   Particle sorting relies on a particle hash that is built from the particle
   position relative to a regular cartesian grid (gridHash).
   The gridHash is an unsigned int (32-bit), so the particle hash key should
   be at least as big, but in theory it could be bigger (if sorting should be
   done using additional information, such as the particle id, too).
   We therefore make the hash key size configurable, with HASH_KEY_SIZE
   bits in the key.
 */

#ifndef HASH_KEY_SIZE
#error "undefined hash key size"
#endif

#if HASH_KEY_SIZE < 32
#error "Hash keys should be at least 32-bit wide"
#elif HASH_KEY_SIZE == 32
typedef unsigned int hashKey;
#elif HASH_KEY_SIZE == 64
typedef unsigned long hashKey;
#else
#error "unmanaged hash key size"
#endif

/*
   The particle hash should always have the grid hash in the upper 32 bits,
   so a GRIDHASH_BITSHIFT is defined, counting the number of bits the grid
   hash should be shifted when inserted in the particle hash key.
 */
#define GRIDHASH_BITSHIFT (HASH_KEY_SIZE - 32)

// now follow a few utility functions to convert between cellHash <-> particleHash, defined with the same compiler directives ___spec
#define __spec static inline __host__ __device__

// In multi-device simulations the 2 high bits of the long particle hash are used to store the cell type
// (internal/external, edge); they are reset by default, allowing for using the hash as an index for cell-based
// arrays. Set preserveHighbits to true to preserve them instead.
// FIXME TODO: check that single-device simulations with 32 or 64 bits hashes allow to use them for the actual hash
// without resetting them
__spec
unsigned int cellHashFromParticleHash(const hashKey &partHash, bool preserveHighbits = false) {
	uint cellHash = (partHash >> GRIDHASH_BITSHIFT);
	return (preserveHighbits ? cellHash : (cellHash & CELLTYPE_BITMASK) );
}

// if HASH_KEY_SIZE is 32 bits wide, this just returns the cellHash; otherwise, the extended particle hash is computed
__spec
hashKey makeParticleHash(const unsigned int &cellHash, const particleinfo& info) {
#if HASH_KEY_SIZE == 32
	return cellHash;
#else
	return ((hashKey)cellHash << GRIDHASH_BITSHIFT) & id(info);
#endif
	// alternatively, to avoid conditionals one can use the more compact but less readable:
	// return ((hashKey)cellHash << GRIDHASH_BITSHIFT) | (id(info) & (EMPTY_CELL >> (32 - GRIDHASH_BITSHIFT) ));
}

#undef __spec

#endif
