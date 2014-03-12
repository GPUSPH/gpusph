/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _BUILDNEIBS_CUH_
#define _BUILDNEIBS_CUH_

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "timing.h"

#include "vector_math.h"

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
setneibsconstants(const SimParams *simparams, const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
	idx_t const& allocatedParticles);

void
getneibsconstants(SimParams *simparams, PhysParams *physparams);

void
resetneibsinfo(void);

void
getneibsinfo(TimingInfo & timingInfo);

void
calcHash(float4*	pos,
		 hashKey*	particleHash,
		 uint*		particleIndex,
		 const particleinfo* particleInfo,
#if HASH_KEY_SIZE >= 64
		 uint*		compactDeviceMap,
#endif
		 const uint		numParticles,
		 const int		periodicbound);

void
inverseParticleIndex (	uint*	particleIndex,
			uint*	inversedParticleIndex,
			uint	numParticles);

void reorderDataAndFindCellStart(	uint*				cellStart,			// output: cell start index
									uint*				cellEnd,			// output: cell end index
#if HASH_KEY_SIZE >= 64
									uint*			segmentStart,
#endif
									float4*				newPos,				// output: sorted positions
									float4*				newVel,				// output: sorted velocities
									particleinfo*		newInfo,			// output: sorted info
									float4*				newBoundElement,	// output: sorted boundary elements
									float4*				newGradGamma,		// output: sorted gradient gamma
									vertexinfo*			newVertices,		// output: sorted vertices
									float*				newTKE,				// output: k for k-e model
									float*				newEps,				// output: e for k-e model
									float*				newTurbVisc,		// output: eddy viscosity
									const hashKey*		particleHash,		// input: sorted grid hashes
									const uint*			particleIndex,		// input: sorted particle indices
									const float4*		oldPos,				// input: unsorted positions
									const float4*		oldVel,				// input: unsorted velocities
									const particleinfo*	oldInfo,			// input: unsorted info
									const float4*		oldBoundElement,	// input: sorted boundary elements
									const float4*		oldGradGamma,		// input: sorted gradient gamma
									const vertexinfo*	oldVertices,		// input: sorted vertices
									const float*		oldTKE,				// input: k for k-e model
									const float*		oldEps,				// input: e for k-e model
									const float*		oldTurbVisc,		// input: eddy viscosity
									const uint			numParticles,
									const uint			numGridCells,
									uint*				inversedParticleIndex);

void
buildNeibsList(	neibdata*			neibsList,
				const float4*		pos,
				const particleinfo*	info,
				vertexinfo*			vertices,
				const float4		*boundelem,
				float2*				vertPos[],
				const hashKey*		particleHash,
				const uint*			cellStart,
				const uint*			cellEnd,
				const uint			numParticles,
				const uint			particleRangeEnd,
				const uint			gridCells,
				const float			sqinfluenceradius,
				const float			sqdpo2,
				const int			periodicbound);

void
sort(	hashKey*	particleHash,
		uint*	particleIndex,
		uint	numParticles
		);
}
#endif
