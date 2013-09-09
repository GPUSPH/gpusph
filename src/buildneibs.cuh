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
#include "vector_math.h"

/* Important notes on block sizes:
	- all kernels accessing the neighbor list MUST HAVE A BLOCK
	MULTIPLE OF NEIBINDEX_INTERLEAVE
	- a parallel reduction for max neibs number is done inside neiblist, block
	size for neiblist MUST BE A POWER OF 2
 */
#if (__COMPUTE__ >= 30)
	#define BLOCK_SIZE_CALCHASH		256
	#define MIN_BLOCKS_CALCHASH		8
	#define BLOCK_SIZE_REORDERDATA	256
	#define MIN_BLOCKS_REORDERDATA	8
	#define BLOCK_SIZE_BUILDNEIBS	256
	#define MIN_BLOCKS_BUILDNEIBS	8
#elif (__COMPUTE__ == 20 || __COMPUTE__ == 21)
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
setneibsconstants(const SimParams & simparams, const PhysParams & physparams);

void
getneibsconstants(SimParams & simparams, PhysParams & physparams);

void
resetneibsinfo(void);

void
getneibsinfo(TimingInfo & timingInfo);

void
calcHash(float4*	pos,
		 uint*		particleHash,
		 uint*		particleIndex,
		 const particleinfo* particleInfo,
		 const uint3	gridSize,
		 const float3	cellSize,
		 const float3	worldOrigin,
		 const uint		numParticles,
		 const int		periodicbound);

void
reorderDataAndFindCellStart(uint*			cellStart,		// output: cell start index
							uint*			cellEnd,		// output: cell end index
							float4*			newPos,			// output: sorted positions
							float4*			newVel,			// output: sorted velocities
							particleinfo*	newInfo,		// output: sorted info
							const uint*		particleHash,   // input: sorted grid hashes
							const uint*		particleIndex,  // input: sorted particle indices
							const float4*	oldPos,			// input: unsorted positions
							const float4*	oldVel,			// input: unsorted velocities
							const particleinfo*	oldInfo,		// input: unsorted info
							const uint		numParticles,
							const uint		numGridCells);

void
buildNeibsList( neibdata*			neibsList,
				const float4*		pos,
				const particleinfo*	info,
				const uint*			particleHash,
				const uint*			cellStart,
				const uint*			cellEnd,
				const uint3			gridSize,
				const float3		cellSize,
				const float3		worldOrigin,
				const uint			numParticles,
				const uint			gridCells,
				const float			sqinfluenceradius,
				const int			periodicbound);

void
sort(	uint*	particleHash,
		uint*	particleIndex,
		uint	numParticles
		);
}
#endif
