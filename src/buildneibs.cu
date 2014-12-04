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

#include <stdexcept>

#include <stdio.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>

#include "textures.cuh"
#include "buildneibs.cuh"

#include "buildneibs_params.h"
#include "buildneibs_kernel.cu"

#include "vector_math.h"

#include "utils.h"


template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
setconstants(const SimParams *simparams, const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
	idx_t const& allocatedParticles)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxneibsnum, &simparams->maxneibsnum, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_neiblist_stride, &allocatedParticles, sizeof(idx_t)));


	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_gridSize, &gridSize, sizeof(uint3)));
}


template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
getconstants(SimParams *simparams, PhysParams *physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&simparams->maxneibsnum, cuneibs::d_maxneibsnum, sizeof(uint), 0));
}


template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
resetinfo(void)
{
	uint temp = 0;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_numInteractions, &temp, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxNeibs, &temp, sizeof(int)));
}


template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
getinfo(TimingInfo & timingInfo)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&timingInfo.numInteractions, cuneibs::d_numInteractions, sizeof(int), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&timingInfo.maxNeibs, cuneibs::d_maxNeibs, sizeof(int), 0));
}


template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
calcHash(float4		*pos,
		hashKey		*particleHash,
		uint		*particleIndex,
const	particleinfo	*particleInfo,
		uint		*compactDeviceMap,
const	uint		numParticles)
{
	uint numThreads = min(BLOCK_SIZE_CALCHASH, numParticles);
	uint numBlocks = div_up(numParticles, numThreads);

	cuneibs::calcHashDevice<periodicbound><<< numBlocks, numThreads >>>
		(pos, particleHash, particleIndex, particleInfo, compactDeviceMap, numParticles);

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("CalcHash kernel execution failed");
}

template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
fixHash(hashKey	*particleHash,
		uint	*particleIndex,
const	particleinfo* particleInfo,
		uint	*compactDeviceMap,
const	uint	numParticles)
{
	uint numThreads = min(BLOCK_SIZE_CALCHASH, numParticles);
	uint numBlocks = div_up(numParticles, numThreads);

	cuneibs::fixHashDevice<<< numBlocks, numThreads >>>(particleHash, particleIndex,
				particleInfo, compactDeviceMap, numParticles);

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("FixHash kernel execution failed");
}

template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
reorderDataAndFindCellStart(
		uint		*cellStart,			// output: cell start index
		uint		*cellEnd,			// output: cell end index
		uint		*segmentStart,
		float4		*newPos,			// output: sorted positions
		float4		*newVel,			// output: sorted velocities
		particleinfo	*newInfo,		// output: sorted info
		float4		*newBoundElement,	// output: sorted boundary elements
		float4		*newGradGamma,		// output: sorted gradient gamma
		vertexinfo	*newVertices,		// output: sorted vertices
		float		*newTKE,			// output: k for k-e model
		float		*newEps,			// output: e for k-e model
		float		*newTurbVisc,		// output: eddy viscosity
		float4		*newEulerVel,		// output: eulerian velocity
const	hashKey		*particleHash,		// input: sorted grid hashes
const	uint		*particleIndex,		// input: sorted particle indices
const	float4		*oldPos,			// input: unsorted positions
const	float4		*oldVel,			// input: unsorted velocities
const	particleinfo	*oldInfo,		// input: unsorted info
const	float4		*oldBoundElement,	// input: sorted boundary elements
const	float4		*oldGradGamma,		// input: sorted gradient gamma
const	vertexinfo	*oldVertices,		// input: sorted vertices
const	float		*oldTKE,			// input: k for k-e model
const	float		*oldEps,			// input: e for k-e model
const	float		*oldTurbVisc,		// input: eddy viscosity
const	float4		*oldEulerVel,		// input: eulerian velocity
const	uint		numParticles,
		uint		*newNumParticles)	// output: number of active particles found
{
	uint numThreads = min(BLOCK_SIZE_REORDERDATA, numParticles);
	uint numBlocks = div_up(numParticles, numThreads);

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
	if (oldEulerVel)
		CUDA_SAFE_CALL(cudaBindTexture(0, eulerVelTex, oldEulerVel, numParticles*sizeof(float4)));

	uint smemSize = sizeof(uint)*(numThreads+1);
	cuneibs::reorderDataAndFindCellStartDevice<<< numBlocks, numThreads, smemSize >>>(cellStart, cellEnd, segmentStart,
		newPos, newVel, newInfo, newBoundElement, newGradGamma, newVertices, newTKE, newEps, newTurbVisc,
		newEulerVel, particleHash, particleIndex, numParticles, newNumParticles);

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

	if (oldEulerVel)
		CUDA_SAFE_CALL(cudaUnbindTexture(eulerVelTex));
}

template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
updateVertIDToIndex(
	particleinfo	*particleInfo,
			uint	*vertIDToIndex,
	const	uint	numParticles)
{
	uint numThreads = min(BLOCK_SIZE_REORDERDATA, numParticles);
	uint numBlocks = div_up(numParticles, numThreads);

	cuneibs::updateVertIDToIndexDevice<<< numBlocks, numThreads>>>(particleInfo, vertIDToIndex, numParticles);
}

template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
buildNeibsList(
		neibdata	*neibsList,
const	float4		*pos,
const	particleinfo*info,
		vertexinfo	*vertices,
const	float4		*boundelem,
		float2		*vertPos[],
const	uint		*vertIDToIndex,
const	hashKey		*particleHash,
const	uint		*cellStart,
const	uint		*cellEnd,
const	uint		numParticles,
const	uint		particleRangeEnd,
const	uint		gridCells,
const	float		sqinfluenceradius,
const	float		boundNlSqInflRad)
{
	// vertices, boundeleme and vertPos must be either all NULL or all not-NULL.
	// throw otherwise
	if (vertices || boundelem || vertPos) {
		if (!vertices || !boundelem || ! vertPos) {
			fprintf(stderr, "%p vs %p vs %p\n", vertices, boundelem, vertPos);
			throw std::invalid_argument("inconsistent params to buildNeibsList");
		}
	}

	if (boundarytype == SA_BOUNDARY && !vertices) {
		fprintf(stderr, "%s boundary type selected, but no vertices!\n",
			BoundaryName[boundarytype]);
		throw std::invalid_argument("missing data");
	}

	const uint numThreads = min(BLOCK_SIZE_BUILDNEIBS, particleRangeEnd);
	const uint numBlocks = div_up(particleRangeEnd, numThreads);

	// bind textures to read all particles, not only internal ones
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellStartTex, cellStart, gridCells*sizeof(uint)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellEndTex, cellEnd, gridCells*sizeof(uint)));

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaBindTexture(0, vertTex, vertices, numParticles*sizeof(vertexinfo)));
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelem, numParticles*sizeof(float4)));
	}

	buildneibs_params<boundarytype == SA_BOUNDARY> params(neibsList, pos, particleHash, particleRangeEnd, sqinfluenceradius,
			vertPos, vertIDToIndex, boundNlSqInflRad);

	cuneibs::buildNeibsListDevice<boundarytype, periodicbound, neibcount><<<numBlocks, numThreads>>>(params);

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("BuildNeibs kernel execution failed");

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaUnbindTexture(vertTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
	}

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cellStartTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cellEndTex));
}

template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
void
CUDANeibsEngine<boundarytype, periodicbound, neibcount>::
sort(hashKey*	particleHash, uint*	particleIndex, uint	numParticles)
{
	thrust::device_ptr<hashKey> particleHash_devptr = thrust::device_pointer_cast(particleHash);
	thrust::device_ptr<uint> particleIndex_devptr = thrust::device_pointer_cast(particleIndex);

	thrust::sort_by_key(particleHash_devptr, particleHash_devptr + numParticles, particleIndex_devptr);

	CUT_CHECK_ERROR("thrust sort failed");
}

// Force the instantiation of all instances
// TODO this is until the engines are turned into header-only classes

#define DECLARE_NEIBSENGINE_PERIODIC(btype) \
	template class CUDANeibsEngine<btype, PERIODIC_NONE, true>; \
	template class CUDANeibsEngine<btype, PERIODIC_X, true>; \
	template class CUDANeibsEngine<btype, PERIODIC_Y, true>; \
	template class CUDANeibsEngine<btype, PERIODIC_XY, true>; \
	template class CUDANeibsEngine<btype, PERIODIC_Z, true>; \
	template class CUDANeibsEngine<btype, PERIODIC_XZ, true>; \
	template class CUDANeibsEngine<btype, PERIODIC_YZ, true>; \
	template class CUDANeibsEngine<btype, PERIODIC_XYZ, true>;

#define DECLARE_NEIBSENGINE \
	DECLARE_NEIBSENGINE_PERIODIC(LJ_BOUNDARY) \
	DECLARE_NEIBSENGINE_PERIODIC(MK_BOUNDARY) \
	DECLARE_NEIBSENGINE_PERIODIC(SA_BOUNDARY) \
	DECLARE_NEIBSENGINE_PERIODIC(DYN_BOUNDARY)

DECLARE_NEIBSENGINE
