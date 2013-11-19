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
#define HASH_KEY_SIZE 32
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

#endif
