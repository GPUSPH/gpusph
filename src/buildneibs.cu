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
#include <stdio.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "textures.cuh"
#include "buildneibs.cuh"
#include "buildneibs_kernel.cu"

#include "utils.h"

extern "C"
{

void
setneibsconstants(const SimParams *simparams, const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxneibsnum, &simparams->maxneibsnum, sizeof(uint)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_gridSize, &gridSize, sizeof(uint3)));
}


void
getneibsconstants(SimParams *simparams, PhysParams *physparams)
{
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
		 hashKey*	particleHash,
		 uint*		particleIndex,
		 const particleinfo* particleInfo,
		 const uint		numParticles,
		 const int		periodicbound)
{
	uint numThreads = min(BLOCK_SIZE_CALCHASH, numParticles);
	uint numBlocks = div_up(numParticles, numThreads);

	//TODO: implement other peridodicty than XPERIODIC
	switch (periodicbound) {
		case 0:
			cuneibs::calcHashDevice<0><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
					   particleInfo, numParticles);
			break;

		default:
			cuneibs::calcHashDevice<XPERIODIC><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
								   particleInfo, numParticles);
			break;
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("CalcHash kernel execution failed");
}


void reorderDataAndFindCellStart(	uint*			cellStart,		// output: cell start index
									uint*			cellEnd,		// output: cell end index
									float4*			newPos,			// output: sorted positions
									float4*			newVel,			// output: sorted velocities
									particleinfo*	newInfo,		// output: sorted info
									const hashKey*	particleHash,   // input: sorted grid hashes
									const uint*		particleIndex,  // input: sorted particle indices
									const float4*	oldPos,			// input: unsorted positions
									const float4*	oldVel,			// input: unsorted velocities
									const particleinfo*	oldInfo,	// input: unsorted info
									const uint		numParticles,
									const uint		numGridCells)
{
	uint numThreads = min(BLOCK_SIZE_REORDERDATA, numParticles);
	uint numBlocks = div_up(numParticles, numThreads);
	
	CUDA_SAFE_CALL(cudaMemset(cellStart, 0xffffffff, numGridCells*sizeof(uint)));

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, oldPos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, oldInfo, numParticles*sizeof(particleinfo)));

	uint smemSize = sizeof(uint)*(numThreads+1);
	cuneibs::reorderDataAndFindCellStartDevice<<< numBlocks, numThreads, smemSize >>>(cellStart, cellEnd, newPos,
													newVel, newInfo, particleHash, particleIndex, numParticles);
	
	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("ReorderDataAndFindCellStart kernel execution failed");
	
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
}


void
buildNeibsList(	neibdata*			neibsList,
				const float4*		pos,
				const particleinfo*	info,
				const hashKey*		particleHash,
				const uint*			cellStart,
				const uint*			cellEnd,
				const uint			numParticles,
				const uint			gridCells,
				const float			sqinfluenceradius,
				const int			periodicbound)
{
	const uint numThreads = min(BLOCK_SIZE_BUILDNEIBS, numParticles);
	const uint numBlocks = div_up(numParticles, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellStartTex, cellStart, gridCells*sizeof(uint)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellEndTex, cellEnd, gridCells*sizeof(uint)));

	switch (periodicbound) {
		case 0:
			cuneibs::buildNeibsListDevice<0, true><<< numBlocks, numThreads >>>(
					#if (__COMPUTE__ >= 20)
					pos,
					#endif
					particleHash,neibsList, numParticles, sqinfluenceradius);
		break;

		case 1:
				cuneibs::buildNeibsListDevice<1, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						particleHash,neibsList, numParticles, sqinfluenceradius);
				break;

		case 2:
				cuneibs::buildNeibsListDevice<2, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						particleHash,neibsList, numParticles, sqinfluenceradius);
				break;

		case 3:
				cuneibs::buildNeibsListDevice<3, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						particleHash,neibsList, numParticles, sqinfluenceradius);
				break;

		case 4:
				cuneibs::buildNeibsListDevice<4, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						particleHash,neibsList, numParticles, sqinfluenceradius);
				break;

		case 5:
				cuneibs::buildNeibsListDevice<5, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						particleHash,neibsList, numParticles, sqinfluenceradius);
				break;

		case 6:
				cuneibs::buildNeibsListDevice<6, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						particleHash,neibsList, numParticles, sqinfluenceradius);
				break;

		case 7:
				cuneibs::buildNeibsListDevice<7, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						particleHash,neibsList, numParticles, sqinfluenceradius);
				break;
	}
		
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
sort(hashKey*	particleHash, uint*	particleIndex, uint	numParticles)
{
	thrust::device_ptr<hashKey> particleHash_devptr = thrust::device_pointer_cast(particleHash);
	thrust::device_ptr<uint> particleIndex_devptr = thrust::device_pointer_cast(particleIndex);
	
	thrust::sort_by_key(particleHash_devptr, particleHash_devptr + numParticles, particleIndex_devptr);

}
}
