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
#include <stdio.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "textures.cuh"
#include "buildneibs.cuh"
#include "buildneibs_kernel.cu"

extern "C"
{

void
calcHash(float4*	pos,
		 uint*		particleHash,
		 uint*		particleIndex,
		 uint3		gridSize,
		 float3		cellSize,
		 float3		worldOrigin,
		 uint		numParticles)
{
	int numThreads = min(BLOCK_SIZE_CALCHASH, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	calcHashDevice<<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
										   gridSize, cellSize, worldOrigin, numParticles);

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Kernel execution failed");
}


void reorderDataAndFindCellStart(	uint*			cellStart,		// output: cell start index
									uint*			cellEnd,		// output: cell end index
									float4*			newPos,			// output: sorted positions
									float4*			newVel,			// output: sorted velocities
									particleinfo*	newInfo,		// output: sorted info
									uint*			particleHash,   // input: sorted grid hashes
									uint*			particleIndex,  // input: sorted particle indices
									float4*			oldPos,			// input: sorted position array
									float4*			oldVel,			// input: sorted velocity array
									particleinfo*	oldInfo,		// input: sorted info array
									uint			numParticles,
									uint			numGridCells)
{
	int numThreads = min(BLOCK_SIZE_REORDERDATA, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaMemset(cellStart, 0xffffffff, numGridCells*sizeof(uint)));

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, oldPos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, oldInfo, numParticles*sizeof(particleinfo)));

	uint smemSize = sizeof(uint)*(numThreads+1);
	reorderDataAndFindCellStartDevice<<< numBlocks, numThreads, smemSize >>>(cellStart, cellEnd, newPos,
													newVel, newInfo, particleHash, particleIndex, numParticles);

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Kernel execution failed");
}


void
buildNeibsList(	uint*			neibsList,
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
				bool			periodicbound)
{
	int numThreads = min(BLOCK_SIZE_BUILDNEIBS, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaMemset(neibsList, 0xffffffff, numParticles*MAXNEIBSNUM*sizeof(uint)));

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellStartTex, cellStart, gridCells*sizeof(uint)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellEndTex, cellEnd, gridCells*sizeof(uint)));

	if (periodicbound)
		buildNeibsListDevice<true><<< numBlocks, numThreads >>>(neibsList, gridSize,
				cellSize, worldOrigin, numParticles, influenceradius);
	else
		buildNeibsListDevice<false><<< numBlocks, numThreads >>>(neibsList, gridSize,
				cellSize, worldOrigin, numParticles, influenceradius);

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cellStartTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cellEndTex));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Kernel execution failed");
}

void
sort(uint*	particleHash, uint*	particleIndex, uint	numParticles)
{
	thrust::device_ptr<uint> particleHash_devptr = thrust::device_pointer_cast(particleHash);
	thrust::device_ptr<uint> particleIndex_devptr = thrust::device_pointer_cast(particleIndex);
	
	 thrust::sort_by_key(particleHash_devptr, particleHash_devptr + numParticles, particleIndex_devptr);

}
}
