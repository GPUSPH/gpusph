#include <stdio.h>
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

	// Setting 48KB cache for Fermi
	// Note: contrary as stated in the CUDA documentation this setting is program wide
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(calcHashDevice, cudaFuncCachePreferL1));

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
}
