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
setneibsconstants(const SimParams *simparams, const PhysParams *physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_dispvect, &physparams->dispvect, sizeof(float3)));
	uint maxneibs_time_neibinterleave = simparams->maxneibsnum*NEIBINDEX_INTERLEAVE;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxneibsnum, &simparams->maxneibsnum, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxneibsnum_time_neibindexinterleave, &maxneibs_time_neibinterleave, sizeof(uint)));
}


void
getneibsconstants(SimParams *simparams, PhysParams *physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->dispvect, cuneibs::d_dispvect, sizeof(float3), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&simparams->maxneibsnum, cuneibs::d_maxneibsnum, sizeof(uint), 0));
}


void
resetneibsinfo(void)
{
	uint temp = 0;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_numInteractions, &temp, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxNeibs, &temp, sizeof(int)));
}


void
getneibsinfo(TimingInfo & timingInfo)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&timingInfo.numInteractions, cuneibs::d_numInteractions, sizeof(int), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&timingInfo.maxNeibs, cuneibs::d_maxNeibs, sizeof(int), 0));
}


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

	cuneibs::calcHashDevice<<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
										   gridSize, cellSize, worldOrigin, numParticles);
	
	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("CalcHash kernel execution failed");
}

void
inverseParticleIndex (	uint*	particleIndex,
			uint*	inversedParticleIndex,
			uint	numParticles)
{
	int numThreads = min(BLOCK_SIZE_REORDERDATA, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	cuneibs::inverseParticleIndexDevice<<< numBlocks, numThreads >>>(particleIndex, inversedParticleIndex, numParticles);
	
	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("InverseParticleIndex kernel execution failed");	
}


void reorderDataAndFindCellStart(	uint*			cellStart,		// output: cell start index
									uint*			cellEnd,		// output: cell end index
									float4*			newPos,			// output: sorted positions
									float4*			newVel,			// output: sorted velocities
									particleinfo*		newInfo,		// output: sorted info
									float4*			newBoundElement,	// output: sorted boundary elements
									float4*			newGradGamma,		// output: sorted gradient gamma
									vertexinfo*		newVertices,		// output: sorted vertices
									float*			newPressure,		// output: sorted pressure
									uint*			particleHash,		// input: sorted grid hashes
									uint*			particleIndex,		// input: sorted particle indices
									float4*			oldPos,			// input: sorted position array
									float4*			oldVel,			// input: sorted velocity array
									particleinfo*		oldInfo,		// input: sorted info array
									float4*			oldBoundElement,	// input: sorted boundary elements
									float4*			oldGradGamma,		// input: sorted gradient gamma
									vertexinfo*		oldVertices,		// input: sorted vertices
									float*			oldPressure,		// input: sorted pressure
									uint			numParticles,
									uint			numGridCells,
									uint*			inversedParticleIndex)
{
	int numThreads = min(BLOCK_SIZE_REORDERDATA, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);
	
	CUDA_SAFE_CALL(cudaMemset(cellStart, 0xffffffff, numGridCells*sizeof(uint)));

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, oldPos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, oldInfo, numParticles*sizeof(particleinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, oldBoundElement, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, gamTex, oldGradGamma, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, vertTex, oldVertices, numParticles*sizeof(vertexinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, presTex, oldPressure, numParticles*sizeof(float)));
	

	uint smemSize = sizeof(uint)*(numThreads+1);
	cuneibs::reorderDataAndFindCellStartDevice<<< numBlocks, numThreads, smemSize >>>(cellStart, cellEnd, newPos,
												newVel, newInfo, newBoundElement, newGradGamma, newVertices, newPressure,
												particleHash, particleIndex, numParticles, inversedParticleIndex);
	
	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("ReorderDataAndFindCellStart kernel execution failed");

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(gamTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(vertTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(presTex));
}


void
buildNeibsList(	uint*				neibsList,
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
				const bool			periodicbound)
{
	const int numThreads = min(BLOCK_SIZE_BUILDNEIBS, numParticles);
	const int numBlocks = (int) ceil(numParticles / (float) numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellStartTex, cellStart, gridCells*sizeof(uint)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellEndTex, cellEnd, gridCells*sizeof(uint)));
	
	if (periodicbound)
		cuneibs::buildNeibsListDevice<true, true><<< numBlocks, numThreads >>>(
			#if (__COMPUTE__ >= 20)			
			pos, 
			#endif
			neibsList, gridSize, cellSize, worldOrigin, numParticles, sqinfluenceradius);
	else
		cuneibs::buildNeibsListDevice<false, true><<< numBlocks, numThreads >>>(
			#if (__COMPUTE__ >= 20)			
			pos, 
			# endif
			neibsList, gridSize, cellSize, worldOrigin, numParticles, sqinfluenceradius);
		
	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("BuildNeibs kernel execution failed");
	
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cellStartTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cellEndTex));
}

void
sort(uint*	particleHash, uint*	particleIndex, uint	numParticles)
{
	thrust::device_ptr<uint> particleHash_devptr = thrust::device_pointer_cast(particleHash);
	thrust::device_ptr<uint> particleIndex_devptr = thrust::device_pointer_cast(particleIndex);
	
	thrust::sort_by_key(particleHash_devptr, particleHash_devptr + numParticles, particleIndex_devptr);

}
}
