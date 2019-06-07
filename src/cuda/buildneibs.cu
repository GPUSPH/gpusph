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

/*! \file
 * Template implementation of the NeibsEngine in CUDA
 */

#include <stdexcept>

#include <stdio.h>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

#include "define_buffers.h"
#include "engine_neibs.h"
#include "utils.h"

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
template<SPHFormulation sph_formulation, typename ViscSpec, BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
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
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_neibboundpos, &simparams->neibboundpos, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_neiblistsize, &simparams->neiblistsize, sizeof(uint)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_neiblist_stride, &allocatedParticles, sizeof(idx_t)));
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
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&simparams->neibboundpos, cuneibs::d_neibboundpos, sizeof(uint), 0));
}


/// Reset number of neighbors and interaction
/*! Reset number of neighbors and number of interactions stored
 * 	into GPU constant memory.
 */
void
resetinfo(void)
{
	int temp = 0;

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_numInteractions, &temp, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxFluidBoundaryNeibs, &temp, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_maxVertexNeibs, &temp, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_hasMaxNeibs, &temp, sizeof(int)));
	temp = -1;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_hasTooManyNeibs, &temp, sizeof(int)));
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
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&timingInfo.maxFluidBoundaryNeibs, cuneibs::d_maxFluidBoundaryNeibs, sizeof(int), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&timingInfo.maxVertexNeibs, cuneibs::d_maxVertexNeibs, sizeof(int), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&timingInfo.hasTooManyNeibs, cuneibs::d_hasTooManyNeibs, sizeof(int), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(timingInfo.hasMaxNeibs, cuneibs::d_hasMaxNeibs, sizeof(int)*PT_TESTPOINT, 0));
}

/** @} */

/** \name Reordering and sort related function
 *  @{ */

/// Launch the compute hash kernel
/*!	Update the particle position and cell hash,
 * compute the particle index for sorting,
 * update the compact device map
 */
void
calcHash(	const BufferList& bufread, ///< input buffers (INFO, COMPACT_DEV_MAP)
			BufferList& bufwrite, ///< output buffers: HASH, POS (updated in place), PARTINDEX
			const uint	numParticles)			///< total number of particles
{
	uint numThreads = BLOCK_SIZE_CALCHASH;
	uint numBlocks = div_up(numParticles, numThreads);

	cuneibs::calcHashDevice<periodicbound><<< numBlocks, numThreads >>>
		(bufwrite.getData<BUFFER_POS>(),
		 bufwrite.getData<BUFFER_HASH>(),
		 bufwrite.getData<BUFFER_PARTINDEX>(),
		 bufread.getData<BUFFER_INFO>(),
		 bufread.getData<BUFFER_COMPACT_DEV_MAP>(),
		 numParticles);

	// Check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}


/// Launch the fix hash kernel
/*!	Restricted version of \seealso calcHash, assuming the hash was already computed on host
 * and only needs a fixup to include the cell type specified in the COMPACT_DEV_MAP
 */
void
fixHash(	const BufferList& bufread, ///< input buffers (INFO, COMPACT_DEV_MAP)
			BufferList& bufwrite, ///< output buffers: HASH (updated in place), PARTINDEX
			const uint	numParticles)			///< total number of particles
{
	uint numThreads = BLOCK_SIZE_CALCHASH;
	uint numBlocks = div_up(numParticles, numThreads);

	cuneibs::fixHashDevice<<< numBlocks, numThreads >>>(
		bufwrite.getData<BUFFER_HASH>(),
		bufwrite.getData<BUFFER_PARTINDEX>(),
		bufread.getData<BUFFER_INFO>(),
		bufread.getData<BUFFER_COMPACT_DEV_MAP>(),
		numParticles);

	// Check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
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
		uint*				segmentStart,		// TODO
		BufferList& sorted_buffers,			// list of sorted buffers (out)
		BufferList const& unsorted_buffers,	// list of buffers to sort (in)
		const uint			numParticles,		// total number of particles in input buffers (in)
		uint*				newNumParticles)	// device pointer to number of active particles found (out)
{

#if 0
#define MUST_HAVE(ar) \
	if (!ar) throw std::invalid_argument(#ar " is null")
#else
#define MUST_HAVE(ar) do { /* nothing */ } while (0)
#endif

#define BIND_CHECK(old_, new_, tex_) \
	if (old_) { \
		CUDA_SAFE_CALL(cudaBindTexture(0, tex_, old_, numParticles*sizeof(*old_))); \
		MUST_HAVE(new_); \
	} \
	if (new_) MUST_HAVE(old_)

	const uint numThreads = BLOCK_SIZE_REORDERDATA;
	const uint numBlocks = div_up(numParticles, numThreads);

	const hashKey *particleHash = sorted_buffers.getConstData<BUFFER_HASH>();
	const uint *particleIndex = sorted_buffers.getConstData<BUFFER_PARTINDEX>();

	MUST_HAVE(particleHash);
	MUST_HAVE(particleIndex);

	// index of cells first and last particles (computed by the kernel)
	uint *cellStart = sorted_buffers.getData<BUFFER_CELLSTART>();
	uint *cellEnd = sorted_buffers.getData<BUFFER_CELLEND>();

	MUST_HAVE(cellStart);
	MUST_HAVE(cellEnd);

	// TODO find a smarter way to do this
	const float4 *oldPos = unsorted_buffers.getData<BUFFER_POS>();
	float4 *newPos = sorted_buffers.getData<BUFFER_POS>();
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, oldPos, numParticles*sizeof(float4)));
	MUST_HAVE(newPos);

	const float4 *oldVel = unsorted_buffers.getData<BUFFER_VEL>();
	float4 *newVel = sorted_buffers.getData<BUFFER_VEL>();
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	MUST_HAVE(newVel);

	const float4 *oldVol = unsorted_buffers.getData<BUFFER_VOLUME>();
	float4 *newVol = sorted_buffers.getData<BUFFER_VOLUME>();
	BIND_CHECK(oldVol, newVol, volTex);

	const float *oldEnergy = unsorted_buffers.getData<BUFFER_INTERNAL_ENERGY>();
	float *newEnergy = sorted_buffers.getData<BUFFER_INTERNAL_ENERGY>();
	BIND_CHECK(oldEnergy, newEnergy, energyTex);

	// sorted already
	const particleinfo *particleInfo = sorted_buffers.getConstData<BUFFER_INFO>();
	MUST_HAVE(particleInfo);

	const float4 *oldBoundElement = unsorted_buffers.getData<BUFFER_BOUNDELEMENTS>();
	float4 *newBoundElement = sorted_buffers.getData<BUFFER_BOUNDELEMENTS>();
	BIND_CHECK(oldBoundElement, newBoundElement, boundTex);

	const float4 *oldGradGamma = unsorted_buffers.getData<BUFFER_GRADGAMMA>();
	float4 *newGradGamma = sorted_buffers.getData<BUFFER_GRADGAMMA>();
	BIND_CHECK(oldGradGamma, newGradGamma, gamTex);

	const vertexinfo *oldVertices = unsorted_buffers.getData<BUFFER_VERTICES>();
	vertexinfo *newVertices = sorted_buffers.getData<BUFFER_VERTICES>();
	BIND_CHECK(oldVertices, newVertices, vertTex);

	const float *oldTKE = unsorted_buffers.getData<BUFFER_TKE>();
	float *newTKE = sorted_buffers.getData<BUFFER_TKE>();
	BIND_CHECK(oldTKE, newTKE, keps_kTex);

	const float *oldEps = unsorted_buffers.getData<BUFFER_EPSILON>();
	float *newEps = sorted_buffers.getData<BUFFER_EPSILON>();
	BIND_CHECK(oldEps, newEps, keps_eTex);

	const float *oldTurbVisc = unsorted_buffers.getData<BUFFER_TURBVISC>();
	float *newTurbVisc = sorted_buffers.getData<BUFFER_TURBVISC>();
	BIND_CHECK(oldTurbVisc, newTurbVisc, tviscTex);

	const float *oldEffPres = unsorted_buffers.getData<BUFFER_EFFPRES>();
	float *newEffPres = sorted_buffers.getData<BUFFER_EFFPRES>();
	if (oldEffPres)
		CUDA_SAFE_CALL(cudaBindTexture(0, effpresTex, oldEffPres, numParticles*sizeof(float)));

	const float4 *oldEulerVel = unsorted_buffers.getData<BUFFER_EULERVEL>();
	float4 *newEulerVel = sorted_buffers.getData<BUFFER_EULERVEL>();
	BIND_CHECK(oldEulerVel, newEulerVel, eulerVelTex);

	const uint *oldNextIDs = unsorted_buffers.getData<BUFFER_NEXTID>();
	uint *newNextIDs = sorted_buffers.getData<BUFFER_NEXTID>();
	if (oldNextIDs && !newNextIDs)
		throw std::invalid_argument("newNextIDs is null");

	uint smemSize = sizeof(uint)*(numThreads+1);
	cuneibs::reorderDataAndFindCellStartDevice<<< numBlocks, numThreads, smemSize >>>(cellStart, cellEnd, segmentStart,
		newPos, newVel, newVol, newEnergy, newBoundElement, newGradGamma, newVertices, newTKE, newEps, newTurbVisc,
		newEffPres,
		newEulerVel,
		oldNextIDs, newNextIDs,
		particleInfo, particleHash, particleIndex, numParticles, newNumParticles);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));

	if (oldVol)
		CUDA_SAFE_CALL(cudaUnbindTexture(volTex));
	if (oldEnergy)
		CUDA_SAFE_CALL(cudaUnbindTexture(energyTex));

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

	if (oldEffPres)
		CUDA_SAFE_CALL(cudaUnbindTexture(effpresTex));

	if (oldEulerVel)
		CUDA_SAFE_CALL(cudaUnbindTexture(eulerVelTex));

#undef BIND_CHECK
#undef MUST_HAVE
}

/// Functor to sort particles by hash (cell), and
/// by fluid number within the cell
struct ptype_hash_compare :
	public thrust::binary_function<
		thrust::tuple<hashKey, particleinfo>,
		thrust::tuple<hashKey, particleinfo>,
		bool>
{
	typedef thrust::tuple<hashKey, particleinfo> value_type;

	__host__ __device__
	bool operator()(const value_type& a, const value_type& b)
	{
		const hashKey ha(cellHashFromParticleHash(thrust::get<0>(a), true)),
				hb(cellHashFromParticleHash(thrust::get<0>(b), true));
		const particleinfo pa(thrust::get<1>(a)),
					 pb(thrust::get<1>(b));

		if (ha == hb) {
			const ParticleType pta = PART_TYPE(pa),
				ptb = PART_TYPE(pb);
			if (pta == ptb)
				return id(pa) < id(pb);
			return (pta < ptb);
		}
		return (ha < hb);
	}
};

void
sort(	BufferList const& bufread,
		BufferList& bufwrite,
		uint	numParticles)
{
	thrust::device_ptr<particleinfo> particleInfo =
		thrust::device_pointer_cast(bufwrite.getData<BUFFER_INFO>());
	thrust::device_ptr<hashKey> particleHash =
		thrust::device_pointer_cast(bufwrite.getData<BUFFER_HASH>());
	thrust::device_ptr<uint> particleIndex =
		thrust::device_pointer_cast(bufwrite.getData<BUFFER_PARTINDEX>());

	ptype_hash_compare comp;

	if (numParticles > 0) {
		// Sort of the particle indices by cell, fluid number, id and
		// particle type (PT_FLUID < PT_BOUNDARY < PT_VERTEX)
		// There is no need for a stable sort due to the id sort
		thrust::sort_by_key(
			thrust::make_zip_iterator(thrust::make_tuple(particleHash, particleInfo)),
			thrust::make_zip_iterator(thrust::make_tuple(
				particleHash + numParticles,
				particleInfo + numParticles)),
			particleIndex, comp);
	}

	KERNEL_CHECK_ERROR;
}


/** @} */

/** \name Neighbors list building
 *  @{ */

/// Build neibs list
void
buildNeibsList(
const	BufferList&	bufread,
		BufferList&	bufwrite,
const	uint		numParticles,
const	uint		particleRangeEnd,
const	uint		gridCells,
const	float		sqinfluenceradius,
const	float		boundNlSqInflRad)
{
	const float4 *pos = bufread.getData<BUFFER_POS>();
	const particleinfo *info = bufread.getData<BUFFER_INFO>();
	const vertexinfo *vertices = bufread.getData<BUFFER_VERTICES>();
	const float4 *boundelem = bufread.getData<BUFFER_BOUNDELEMENTS>();
	const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
	const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
	const uint *cellEnd = bufread.getData<BUFFER_CELLEND>();

	neibdata	*neibsList = bufwrite.getData<BUFFER_NEIBSLIST>();
	float2		**vertPos  = bufwrite.getRawPtr<BUFFER_VERTPOS>();

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
	#if !PREFER_L1
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellStartTex, cellStart, gridCells*sizeof(uint)));
	CUDA_SAFE_CALL(cudaBindTexture(0, cellEndTex, cellEnd, gridCells*sizeof(uint)));

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaBindTexture(0, vertTex, vertices, numParticles*sizeof(vertexinfo)));
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelem, numParticles*sizeof(float4)));
	}

	buildneibs_params<boundarytype> params(neibsList, pos, particleHash, particleRangeEnd, sqinfluenceradius,
			vertPos, boundNlSqInflRad);

	cuneibs::buildNeibsListDevice<sph_formulation, ViscSpec, boundarytype, periodicbound, neibcount><<<numBlocks, numThreads>>>(params);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaUnbindTexture(vertTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
	}

	#if !PREFER_L1
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cellStartTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(cellEndTex));
}

/** @} */

};

