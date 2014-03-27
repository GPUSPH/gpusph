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
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
	idx_t const& allocatedParticles)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxneibsnum, &simparams->maxneibsnum, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_neiblist_stride, &allocatedParticles, sizeof(idx_t)));


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
#if HASH_KEY_SIZE >= 64
		 uint*		compactDeviceMap,
#endif
		 const uint		numParticles,
		 const Periodicity	periodicbound)
{
	uint numThreads = min(BLOCK_SIZE_CALCHASH, numParticles);
	uint numBlocks = div_up(numParticles, numThreads);

	switch (periodicbound) {
		case PERIODIC_NONE:
			cuneibs::calcHashDevice<PERIODIC_NONE><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
					   particleInfo,
#if HASH_KEY_SIZE >= 64
					   compactDeviceMap,
#endif
					   numParticles);
			break;

		case PERIODIC_X:
			cuneibs::calcHashDevice<PERIODIC_X><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
					   particleInfo,
#if HASH_KEY_SIZE >= 64
					   compactDeviceMap,
#endif
					   numParticles);
			break;

		case PERIODIC_Y:
			cuneibs::calcHashDevice<PERIODIC_Y><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
					   particleInfo,
#if HASH_KEY_SIZE >= 64
					   compactDeviceMap,
#endif
					   numParticles);
			break;

		case PERIODIC_XY:
			cuneibs::calcHashDevice<PERIODIC_XY><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
					   particleInfo,
#if HASH_KEY_SIZE >= 64
					   compactDeviceMap,
#endif
					   numParticles);
			break;

		case PERIODIC_Z:
			cuneibs::calcHashDevice<PERIODIC_Z><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
					   particleInfo,
#if HASH_KEY_SIZE >= 64
					   compactDeviceMap,
#endif
					   numParticles);
			break;

		case PERIODIC_XZ:
			cuneibs::calcHashDevice<PERIODIC_XZ><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
					   particleInfo,
#if HASH_KEY_SIZE >= 64
					   compactDeviceMap,
#endif
					   numParticles);
			break;

		case PERIODIC_YZ:
			cuneibs::calcHashDevice<PERIODIC_YZ><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
					   particleInfo,
#if HASH_KEY_SIZE >= 64
					   compactDeviceMap,
#endif
					   numParticles);
			break;

		case PERIODIC_XYZ:
			cuneibs::calcHashDevice<PERIODIC_XYZ><<< numBlocks, numThreads >>>(pos, particleHash, particleIndex,
					   particleInfo,
#if HASH_KEY_SIZE >= 64
					   compactDeviceMap,
#endif
					   numParticles);
			break;

		default:
			throw std::runtime_error("Incorrect value of periodicbound!");
	}

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
									uint*				inversedParticleIndex)
{
	uint numThreads = min(BLOCK_SIZE_REORDERDATA, numParticles);
	uint numBlocks = div_up(numParticles, numThreads);

	// now in a separate function
	// CUDA_SAFE_CALL(cudaMemset(cellStart, 0xffffffff, numGridCells*sizeof(uint)));

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, oldPos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, oldInfo, numParticles*sizeof(particleinfo)));

	// TODO reduce these conditionals

	if (oldBoundElement)
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, oldBoundElement, numParticles*sizeof(float4)));
	if (oldGradGamma)
		CUDA_SAFE_CALL(cudaBindTexture(0, gamTex, oldGradGamma, numParticles*sizeof(float4)));
	if (oldVertices)
		CUDA_SAFE_CALL(cudaBindTexture(0, vertTex, oldVertices, numParticles*sizeof(vertexinfo)));

	if (oldTKE)
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_kTex, oldTKE, numParticles*sizeof(float)));
	if (oldEps)
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_eTex, oldEps, numParticles*sizeof(float)));
	if (oldTurbVisc)
		CUDA_SAFE_CALL(cudaBindTexture(0, tviscTex, oldTurbVisc, numParticles*sizeof(float)));

	uint smemSize = sizeof(uint)*(numThreads+1);
	cuneibs::reorderDataAndFindCellStartDevice<<< numBlocks, numThreads, smemSize >>>(cellStart, cellEnd,
#if HASH_KEY_SIZE >= 64
													segmentStart,
#endif
		newPos, newVel, newInfo, newBoundElement, newGradGamma, newVertices, newTKE, newEps, newTurbVisc,
												particleHash, particleIndex, numParticles, inversedParticleIndex);

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("ReorderDataAndFindCellStart kernel execution failed");

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	if (oldBoundElement)
		CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
	if (oldGradGamma)
		CUDA_SAFE_CALL(cudaUnbindTexture(gamTex));
	if (oldVertices)
		CUDA_SAFE_CALL(cudaUnbindTexture(vertTex));

	if (oldTKE)
		CUDA_SAFE_CALL(cudaUnbindTexture(keps_kTex));
	if (oldEps)
		CUDA_SAFE_CALL(cudaUnbindTexture(keps_eTex));
	if (oldTurbVisc)
		CUDA_SAFE_CALL(cudaUnbindTexture(tviscTex));
}


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
				const Periodicity	periodicbound)
{
	const uint numThreads = min(BLOCK_SIZE_BUILDNEIBS, particleRangeEnd);
	const uint numBlocks = div_up(particleRangeEnd, numThreads);

	// bind textures to read all particles, not only internal ones
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellStartTex, cellStart, gridCells*sizeof(uint)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellEndTex, cellEnd, gridCells*sizeof(uint)));
	if (vertices)
		CUDA_SAFE_CALL(cudaBindTexture(0, vertTex, vertices, numParticles*sizeof(vertexinfo)));
	if (boundelem)
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelem, numParticles*sizeof(float4)));

	float2 *vertPos0, *vertPos1, *vertPos2;
	if (vertPos) {
		vertPos0 = vertPos[0];
		vertPos1 = vertPos[1];
		vertPos2 = vertPos[2];
	} else
		vertPos0 = vertPos1 = vertPos2 = NULL;


	switch (periodicbound) {
		case PERIODIC_NONE:
			cuneibs::buildNeibsListDevice<PERIODIC_NONE, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						vertPos0,
						vertPos1,
						vertPos2,
						particleHash,
						neibsList,
						particleRangeEnd,
						sqinfluenceradius,
						sqdpo2);
		break;

		case PERIODIC_X:
				cuneibs::buildNeibsListDevice<PERIODIC_X, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						vertPos0,
						vertPos1,
						vertPos2,
						particleHash,
						neibsList,
						particleRangeEnd,
						sqinfluenceradius,
						sqdpo2);
				break;

		case PERIODIC_Y:
				cuneibs::buildNeibsListDevice<PERIODIC_Y, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						vertPos0,
						vertPos1,
						vertPos2,
						particleHash,
						neibsList,
						particleRangeEnd,
						sqinfluenceradius,
						sqdpo2);
				break;

		case PERIODIC_XY:
				cuneibs::buildNeibsListDevice<PERIODIC_XY, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						vertPos0,
						vertPos1,
						vertPos2,
						particleHash,
						neibsList,
						particleRangeEnd,
						sqinfluenceradius,
						sqdpo2);
				break;

		case PERIODIC_Z:
				cuneibs::buildNeibsListDevice<PERIODIC_Z, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						vertPos0,
						vertPos1,
						vertPos2,
						particleHash,
						neibsList,
						particleRangeEnd,
						sqinfluenceradius,
						sqdpo2);
				break;

		case PERIODIC_XZ:
				cuneibs::buildNeibsListDevice<PERIODIC_XZ, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						vertPos0,
						vertPos1,
						vertPos2,
						particleHash,
						neibsList,
						particleRangeEnd,
						sqinfluenceradius,
						sqdpo2);
				break;

		case PERIODIC_YZ:
				cuneibs::buildNeibsListDevice<PERIODIC_YZ, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						vertPos0,
						vertPos1,
						vertPos2,
						particleHash,
						neibsList,
						particleRangeEnd,
						sqinfluenceradius,
						sqdpo2);
				break;

		case PERIODIC_XYZ:
				cuneibs::buildNeibsListDevice<PERIODIC_XYZ, true><<< numBlocks, numThreads >>>(
						#if (__COMPUTE__ >= 20)
						pos,
						#endif
						vertPos0,
						vertPos1,
						vertPos2,
						particleHash,
						neibsList,
						particleRangeEnd,
						sqinfluenceradius,
						sqdpo2);
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
	if (vertices)
		CUDA_SAFE_CALL(cudaUnbindTexture(vertTex));
	if (boundelem)
		CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
}

void
sort(hashKey*	particleHash, uint*	particleIndex, uint	numParticles)
{
	thrust::device_ptr<hashKey> particleHash_devptr = thrust::device_pointer_cast(particleHash);
	thrust::device_ptr<uint> particleIndex_devptr = thrust::device_pointer_cast(particleIndex);

	thrust::sort_by_key(particleHash_devptr, particleHash_devptr + numParticles, particleIndex_devptr);

	CUT_CHECK_ERROR("thrust sort failed");

}
}
