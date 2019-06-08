/*  Copyright (c) 2013-2018 INGV, EDF, UniCT, JHU

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
 * Hashkey-related definitions
 */

#ifndef _HASHKEY_H
#define _HASHKEY_H

// For CELLTYPE_BITMASK
#include "multi_gpu_defines.h"

/*! Particle sorting relies on a particle hash that is built from the particle
 * position relative to a regular cartesian grid (gridHash), which is an
 * unsigned int.
 */
typedef unsigned int hashKey;

//! Maximum value for the cell hash
#define CELL_HASH_MAX	UINT_MAX

/// Compute cell hash from particle hash
/*! Compute the cell hash value from particle hash according to the chosen
 * 	key size.
 *
 *	\param[in] partHash : particle hash
 *	\param[in] preserveHighbits : in multi-device simulations the 2 high bits
 *	of the long particle hash are used to store the cell type (internal/external/edge);
 *	they are reset by default, allowing for using the hash as an index for
 *	cell-based arrays. Set preserveHighbits to true to preserve them instead.
 *
 *	\return cell hash value
 */
static __host__ __device__ __forceinline__
hashKey cellHashFromParticleHash(const hashKey &partHash, bool preserveHighbits = false) {
	return (preserveHighbits ? partHash : (partHash & CELLTYPE_BITMASK) );
}

#endif
