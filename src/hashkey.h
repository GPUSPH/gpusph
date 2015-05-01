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

#ifndef _HASHKEY_H
#define _HASHKEY_H

// For CELLTYPE_BITMASK
#include "multi_gpu_defines.h"
// For HAS_KEY_SIZE
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

// CELL_HASH_MAX replaces HASH_KEY_MAX. It is always 32 bits long since it is used only as a cellHash.
#define CELL_HASH_MAX	UINT_MAX

// Now follow a few utility functions to convert between cellHash <-> particleHash.
// In multi-device simulations the 2 high bits of the long particle hash are used to store the cell type
// (internal/external, edge); they are reset by default, allowing for using the hash as an index for cell-based
// arrays. Set preserveHighbits to true to preserve them instead.
// Note the explicit uint cast: we know that the shifted partHash will fit in a uint, but the compiler doesn't,
// and would warn if -Wconversion is enabled; the explicit cast silences the warning.
// FIXME TODO: check that single-device simulations with 32 or 64 bits hashes allow to use them for the actual hash
// without resetting them

/// Compute cell hash from particle hash
/*! Compute the cell hash value from particle hash according to the chosen
 * 	key size.
 *
 *	\param[in] partHash : particle hash
 *
 *	\return cell hash value
 */
static __host__ __device__ __forceinline__
unsigned int cellHashFromParticleHash(const hashKey &partHash, bool preserveHighbits = false) {
	uint cellHash = uint(partHash >> GRIDHASH_BITSHIFT);
	return (preserveHighbits ? cellHash : (cellHash & CELLTYPE_BITMASK) );
}


/// Compute particle hash from cell hash and particle info
/*! Compute particle hash from cell hash and particle info
 * 	according to the chosen key size.
 * 	If HASH_KEY_SIZE is 32 bits wide, this just returns the
 * 	cellHash; otherwise, the extended particle hash is computed
 *
 *	\param[in] celHash : cell hash
 *	\param[in] info : particle info
 *
 *	\return cell hash value
 */
static __host__ __device__ __forceinline__
hashKey makeParticleHash(const unsigned int &cellHash, const particleinfo& info) {
#if HASH_KEY_SIZE == 32
	return cellHash;
#else
	return ((hashKey)cellHash << GRIDHASH_BITSHIFT) | id(info);
#endif
	// Alternatively, to avoid conditionals one can use the more compact but less readable:
	// return ((hashKey)cellHash << GRIDHASH_BITSHIFT) | (id(info) & (EMPTY_CELL >> (32 - GRIDHASH_BITSHIFT) ));
}

#endif
