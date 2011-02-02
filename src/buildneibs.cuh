#ifndef _BUILDNEIBS_CUH_
#define _BUILDNEIBS_CUH_

#define BLOCK_SIZE_CALCHASH		256
#define BLOCK_SIZE_REORDERDATA	256
#define BLOCK_SIZE_BUILDNEIBS	256

#include "vector_math.h"

extern "C"
{

void
calcHash(float4*	pos,
		 uint*		particleHash,
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
							uint*			particleHash,   // input: sorted grid hashes
							uint*			particleIndex,	// input: sorted particle indices
							float4*			oldPos,			// input: sorted position array
							float4*			oldVel,			// input: sorted velocity array
							particleinfo*	oldInfo,		// input: sorted info array
							uint			numParticles,
							uint			numGridCells);

void
buildNeibsList( uint*			neibsList,
				float4*			pos,
				particleinfo*	info,
				uint*			particleHash,
				uint*			cellStart,
				uint*			cellEnd,
				uint3			gridSize,
				float3			cellSize,
				float3			worldOrigin,
				uint			numParticles,
				uint			gridCells,
				float			influenceradius,
				bool			periodicbound);
}
#endif
