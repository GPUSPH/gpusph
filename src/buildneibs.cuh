/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
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

#ifndef _BUILDNEIBS_CUH_
#define _BUILDNEIBS_CUH_

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "timing.h"

#include "vector_math.h"

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
#define HASH_KEY_SIZE 64
#endif

/* Either the hash is 32 or 64 bits long, in reorder kernel we work with the "short hash"
 * (i.e. only the cell hash part, without the particle id) and we need to check it to detect
 * if the current particle is inactive or not. In the calchash, however, we need a long hash
 * if the key is 64 bits wide. So we define the 32 bit HASH_KEY_MAX anyway, and the 64 bits
 * one only if HASH_KEY_SIZE == 64
 */
#ifndef HASH_KEY_MAX
#define HASH_KEY_MAX_32 UINT_MAX
#endif

#define HASH_KEY_MAX UINT_MAX

#if HASH_KEY_SIZE < 32
#error "Hash keys should be at least 32-bit wide"
#elif HASH_KEY_SIZE == 32
typedef unsigned int hashKey;
#elif HASH_KEY_SIZE == 64
typedef unsigned long hashKey;
#define HASH_KEY_MAX_64 ULONG_MAX
#else
#error "unmanaged hash key size"
#endif

/*
   The particle hash should always have the grid hash in the upper 32 bits,
   so a GRIDHASH_BITSHIFT is defined, counting the number of bits the grid
   hash should be shifted when inserted in the particle hash key.
 */
#define GRIDHASH_BITSHIFT (HASH_KEY_SIZE - 32)

/* Important notes on block sizes:
	- all kernels accessing the neighbor list MUST HAVE A BLOCK
	MULTIPLE OF NEIBINDEX_INTERLEAVE
	- a parallel reduction for max neibs number is done inside neiblist, block
	size for neiblist MUST BE A POWER OF 2
 */
#if (__COMPUTE__ >= 20)
	#define BLOCK_SIZE_CALCHASH		256
	#define MIN_BLOCKS_CALCHASH		6
	#define BLOCK_SIZE_REORDERDATA	256
	#define MIN_BLOCKS_REORDERDATA	6
	#define BLOCK_SIZE_BUILDNEIBS	256
	#define MIN_BLOCKS_BUILDNEIBS	5
#else
	#define BLOCK_SIZE_CALCHASH		256
	#define MIN_BLOCKS_CALCHASH		1
	#define BLOCK_SIZE_REORDERDATA	256
	#define MIN_BLOCKS_REORDERDATA	1
	#define BLOCK_SIZE_BUILDNEIBS	256
	#define MIN_BLOCKS_BUILDNEIBS	1
#endif


extern "C"
{
void
setneibsconstants(const SimParams *simparams, const PhysParams *physparams);

void
getneibsconstants(SimParams *simparams, PhysParams *physparams);

void
resetneibsinfo(void);

void
getneibsinfo(TimingInfo & timingInfo);

void
calcHash(float4*	pos,
#if HASH_KEY_SIZE >= 64
		 particleinfo* pinfo,
		 uint* compactDeviceMap,
#endif
		 hashKey*	particleHash,
		 uint*		particleIndex,
		 uint3		gridSize,
		 float3		cellSize,
		 float3		worldOrigin,
		 uint		numParticles);

void
reorderDataAndFindCellStart(uint*			cellStart,		// output: cell start index
							uint*			cellEnd,		// output: cell end index
							float4*			newPos,			// output: sorted positions
							float4*			newVel,			// output: sorted velocities
							particleinfo*	newInfo,		// output: sorted info
							hashKey*		particleHash,   // input: sorted grid hashes
							uint*			particleIndex,	// input: sorted particle indices
							float4*			oldPos,			// input: sorted position array
							float4*			oldVel,			// input: sorted velocity array
							particleinfo*	oldInfo,		// input: sorted info array
#if HASH_KEY_SIZE >= 64
							uint*			segmentStart,
#endif
							uint			numParticles,
							uint*			newNumParticles,	// output: number of active particles found
							uint			numGridCells);

void
buildNeibsList( uint*				neibsList,
				const float4*		pos,
				const particleinfo*	info,
				const hashKey*		particleHash,
				const uint*			cellStart,
				const uint*			cellEnd,
				const uint3			gridSize,
				const float3		cellSize,
				const float3		worldOrigin,
				const uint			numParticles,
				const uint			particleRangeEnd,
				const uint			gridCells,
				const float			sqinfluenceradius,
				const bool			periodicbound);

void
buildNeibsList2( uint*			neibsList,
				float4*			pos,
				particleinfo*	info,
				hashKey*		particleHash,
				uint*			cellStart,
				uint*			cellEnd,
				uint3			gridSize,
				float3			cellSize,
				float3			worldOrigin,
				uint			numParticles,
				uint			gridCells,
				float			sqinfluenceradius,
				bool			periodicbound);

void
buildNeibsList4( uint*				neibsList,
				const float4*		pos,
				const particleinfo*	info,
				const hashKey*		particleHash,
				const uint*			cellStart,
				const uint*			cellEnd,
				const uint3			gridSize,
				const float3		cellSize,
				const float3		worldOrigin,
				const uint			numParticles,
				const uint			gridCells,
				const float			sqinfluenceradius,
				const bool			periodicbound);

void
sort(	hashKey*	particleHash,
		uint*	particleIndex,
		uint	numParticles
		);
}
#endif
