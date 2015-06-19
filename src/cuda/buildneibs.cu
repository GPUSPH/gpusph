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

#include "define_buffers.h"
#include "engine_neibs.h"
#include "utils.h"

/* Important notes on block sizes:
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

#include "textures.cuh"

#include "buildneibs_params.h"
#include "buildneibs_kernel.cu"

#include "vector_math.h"


/// Neighbor engine class
/*!	CUDANeibsEngine is an implementation of the abstract class AbstractNeibsEngine
 *	and is providing :
 *		- device constants upload to the device
 *		- device variables upload/download to/from the device
 *		- launch of sorting and reordering kernels
 *		- launch of neighbor list construction kernels
 *
 *	It is templatizd by:
 *	\tparam boundarytype : type of boundary
 *	\tparam periodicbound : type of periodic boundaries (0 ... 7)
 *	\tparam neibcount : true if we want to compute actual neighbors number

 *	\ingroup neibs
*/
template<SPHFormulation sph_formulation, BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
class CUDANeibsEngine : public AbstractNeibsEngine
{
public:

/** \name Constants upload/download and timing related function
 *  @{ */

/// Upload constants on the device
/*! This function upload neighbors search related constants on the device.
 * 	\param[in] simparams : pointer to simulation parameters structure
 * 	\param[in] physparams : pointer to physical parameters structure
 * 	\param[in] worldOrigin : origin of the simulation domain
 * 	\param[in] gridSize : size of computational domain in grid cells
 * 	\param[in] cellSize : size of each cell
 * 	\param[in] allocatedParticles : number of allocated particles
 */
void
setconstants(	const SimParams *simparams,		// pointer to simulation parameters structure (in)
				const PhysParams *physparams,		// pointer to physical parameters structure (in)
				float3 const& worldOrigin,			// origin of the simulation domain (in)
				uint3 const& gridSize,				// size of computational domain in grid cells (in)
				float3 const& cellSize,				// size of each cell (in)
				idx_t const& allocatedParticles)	// number of allocated particles (in)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxneibsnum, &simparams->maxneibsnum, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_neiblist_stride, &allocatedParticles, sizeof(idx_t)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_gridSize, &gridSize, sizeof(uint3)));
}

/// Download maximum number of neighbors
/*! Download from device the maximum number of neighbors per particle
 *  computed by buildNeibsDevice kernel.
 *  \param[in] simparams : pointer to simulation parameters structure
 *  \param[in] physparams : pointer to physical parameters structure
 */
void
getconstants(	SimParams *simparams,	// pointer to simulation parameters structure (in)
				PhysParams *physparams)	// pointer to physical parameters structure (in)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&simparams->maxneibsnum, cuneibs::d_maxneibsnum, sizeof(uint), 0));
}


/// Reset number of neighbors and interaction
/*! Reset number of neighbors and number of interactions stored
 * 	into GPU constant memory.
 */
void
resetinfo(void)
{
	uint temp = 0;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_numInteractions, &temp, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxNeibs, &temp, sizeof(int)));
}


/// Download number of neighbors and interactions
/*!	Download from GPU the maximum number of neighbors along with the
 * 	total number of interactions. Those data will be used to update a
 * 	TimingInfo structure.
 * 	\param[in, out] timingInfo : timing info struct where number of interactions and max
 * 	neighbors number will be updated
 */
void
getinfo(TimingInfo & timingInfo)	// timing info (in, out)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&timingInfo.numInteractions, cuneibs::d_numInteractions, sizeof(int), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&timingInfo.maxNeibs, cuneibs::d_maxNeibs, sizeof(int), 0));
}

/** @} */

/** \name Reordering and sort related function
 *  @{ */

/// Launch the compute hash kernel
/*!	CPU part responsible of launching the compute hash kernel
 * 	(cuneibs::calcHashDevice) on the device.
 * 	\param[in,out] posArray : particle's positions
 *	\param[in,out] particleHash : particle's hashes
 *	\param[out] particleIndex : particle's indexes
 *	\param[in] particleInfo : particle's information
 *	\param[out] compactDeviceMap : TODO
 *	\param[in] numParticles : total number of particles
 */
void
calcHash(float4		*pos,					// particle's positions (in, out)
		hashKey		*particleHash,			// particle's hashes (in, out)
		uint		*particleIndex,			// particle's indexes (out)
		const particleinfo	*particleInfo,	// particle's information (in)
		uint		*compactDeviceMap,		// TODO
		const uint	numParticles)			// total number of particles
{
	uint numThreads = BLOCK_SIZE_CALCHASH;
	uint numBlocks = div_up(numParticles, numThreads);

	cuneibs::calcHashDevice<periodicbound><<< numBlocks, numThreads >>>
		(pos, particleHash, particleIndex, particleInfo, compactDeviceMap, numParticles);

	// Check if kernel invocation generated an error
	CUT_CHECK_ERROR("CalcHash kernel execution failed");
}


/// Launch the fix hash kernel
/*!	CPU part responsible of launching the fix hash kernel
 * 	(cuneibs::fixHashDevice) on the device.
 * 	\param[in,out] particleHash : particle's hashes
 * 	\param[out] particleIndex : particle's indexes
 * 	\param[in] particleInfo : particle's informations
 * 	\param[out] compactDeviceMap : ???
 * 	\param[in] numParticles : total number of particles
 */
void
fixHash(hashKey	*particleHash,				// particle's hashes (in, out)
		uint	*particleIndex,				// particle's indexes (out)
		const particleinfo* particleInfo,	// particle's information (in)
		uint	*compactDeviceMap,			// TODO
		const uint	numParticles)			// total number of particles
{
	uint numThreads = BLOCK_SIZE_CALCHASH;
	uint numBlocks = div_up(numParticles, numThreads);

	cuneibs::fixHashDevice<<< numBlocks, numThreads >>>(particleHash, particleIndex,
				particleInfo, compactDeviceMap, numParticles);

	// Check if kernel invocation generated an error
	CUT_CHECK_ERROR("FixHash kernel execution failed");
}


/// Launch the reorder kernel
/*!	CPU part responsible of launching the reorder kernel
 * 	(cuneibs::reorderDataAndFindCellStartDevice) on the device.
 * 	\param[out] cellStart : index of cells first particle
 * 	\param[out] cellEnd : index of cells last particle
 * 	\param[out] segmentStart : TODO
 * 	\param[in] particleHash : sorted particle hashes
 * 	\param[in] particleIndex : sorted particle indices
 * 	\param[in] numParticles : total number of particles in input buffers
 * 	\param[out] newNumParticles : device pointer to number of active particles found
 */
void
reorderDataAndFindCellStart(
		uint*				cellStart,			// index of cells first particle (out)
		uint*				cellEnd,			// index of cells last particle (out)
		uint*				segmentStart,		// TODO
		const hashKey*		particleHash,		// sorted particle hashes (in)
		const uint*			particleIndex,		// sorted particle indices (in)
		MultiBufferList::iterator sorted_buffers,			// list of sorted buffers (out)
		MultiBufferList::const_iterator unsorted_buffers,	// list of buffers to sort (in)
		const uint			numParticles,		// total number of particles in input buffers (in)
		uint*				newNumParticles)	// device pointer to number of active particles found (out)
{
	const uint numThreads = BLOCK_SIZE_REORDERDATA;
	const uint numBlocks = div_up(numParticles, numThreads);

	// TODO find a smarter way to do this
	const float4 *oldPos = unsorted_buffers->getData<BUFFER_POS>();
	float4 *newPos = sorted_buffers->getData<BUFFER_POS>();
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, oldPos, numParticles*sizeof(float4)));

	const float4 *oldVel = unsorted_buffers->getData<BUFFER_VEL>();
	float4 *newVel = sorted_buffers->getData<BUFFER_VEL>();
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));

	const float4 *oldVol = unsorted_buffers->getData<BUFFER_VOLUME>();
	float4 *newVol = sorted_buffers->getData<BUFFER_VOLUME>();
	if (oldVol)
		CUDA_SAFE_CALL(cudaBindTexture(0, volTex, oldVol, numParticles*sizeof(float4)));

	const particleinfo *oldInfo = unsorted_buffers->getData<BUFFER_INFO>();
	particleinfo *newInfo = sorted_buffers->getData<BUFFER_INFO>();
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, oldInfo, numParticles*sizeof(particleinfo)));

	const float4 *oldBoundElement = unsorted_buffers->getData<BUFFER_BOUNDELEMENTS>();
	float4 *newBoundElement = sorted_buffers->getData<BUFFER_BOUNDELEMENTS>();
	if (oldBoundElement)
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, oldBoundElement, numParticles*sizeof(float4)));

	const float4 *oldGradGamma = unsorted_buffers->getData<BUFFER_GRADGAMMA>();
	float4 *newGradGamma = sorted_buffers->getData<BUFFER_GRADGAMMA>();
	if (oldGradGamma)
		CUDA_SAFE_CALL(cudaBindTexture(0, gamTex, oldGradGamma, numParticles*sizeof(float4)));

	const vertexinfo *oldVertices = unsorted_buffers->getData<BUFFER_VERTICES>();
	vertexinfo *newVertices = sorted_buffers->getData<BUFFER_VERTICES>();
	if (oldVertices)
		CUDA_SAFE_CALL(cudaBindTexture(0, vertTex, oldVertices, numParticles*sizeof(vertexinfo)));

	const float *oldTKE = unsorted_buffers->getData<BUFFER_TKE>();
	float *newTKE = sorted_buffers->getData<BUFFER_TKE>();
	if (oldTKE)
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_kTex, oldTKE, numParticles*sizeof(float)));

	const float *oldEps = unsorted_buffers->getData<BUFFER_EPSILON>();
	float *newEps = sorted_buffers->getData<BUFFER_EPSILON>();
	if (oldEps)
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_eTex, oldEps, numParticles*sizeof(float)));

	const float *oldTurbVisc = unsorted_buffers->getData<BUFFER_TURBVISC>();
	float *newTurbVisc = sorted_buffers->getData<BUFFER_TURBVISC>();
	if (oldTurbVisc)
		CUDA_SAFE_CALL(cudaBindTexture(0, tviscTex, oldTurbVisc, numParticles*sizeof(float)));

	const float4 *oldEulerVel = unsorted_buffers->getData<BUFFER_EULERVEL>();
	float4 *newEulerVel = sorted_buffers->getData<BUFFER_EULERVEL>();
	if (oldEulerVel)
		CUDA_SAFE_CALL(cudaBindTexture(0, eulerVelTex, oldEulerVel, numParticles*sizeof(float4)));

	uint smemSize = sizeof(uint)*(numThreads+1);
	cuneibs::reorderDataAndFindCellStartDevice<<< numBlocks, numThreads, smemSize >>>(cellStart, cellEnd, segmentStart,
		newPos, newVel, newVol, newInfo, newBoundElement, newGradGamma, newVertices, newTKE, newEps, newTurbVisc,
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


/// Update vertex ID
void
updateVertIDToIndex(
	const particleinfo	*particleInfo,
			uint	*vertIDToIndex,
	const	uint	numParticles)
{
	uint numThreads = BLOCK_SIZE_REORDERDATA;
	uint numBlocks = div_up(numParticles, numThreads);

	cuneibs::updateVertIDToIndexDevice<<< numBlocks, numThreads>>>(particleInfo, vertIDToIndex, numParticles);
}

void
sort(hashKey*	particleHash, uint*	particleIndex, uint	numParticles)
{
	thrust::device_ptr<hashKey> particleHash_devptr = thrust::device_pointer_cast(particleHash);
	thrust::device_ptr<uint> particleIndex_devptr = thrust::device_pointer_cast(particleIndex);

	thrust::sort_by_key(particleHash_devptr, particleHash_devptr + numParticles, particleIndex_devptr);

	CUT_CHECK_ERROR("thrust sort failed");
}


/** @} */

/** \name Neighbors list building
 *  @{ */

/// Build neibs list
void
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

	const uint numThreads = BLOCK_SIZE_BUILDNEIBS;
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

	cuneibs::buildNeibsListDevice<sph_formulation, boundarytype, periodicbound, neibcount><<<numBlocks, numThreads>>>(params);

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

/** @} */

};

