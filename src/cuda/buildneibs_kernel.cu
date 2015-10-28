/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A.
 	Dalrymple, Eugenio Rustico, Ciro Del Negro

	Conservatoire National des Arts et Metiers, Paris, France

	Istituto Nazionale di Geofisica e Vulcanologia,
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

/*
 * Device code.
 */


// TODO :
// We can also plan to have separate arrays for boundary parts
// one for the fixed boundary that is sorted only one time in the simulation
// an other one for moving boundary that will be sort with fluid particle
// and a last one for fluid particles. In this way we will compute interactions
// only on fluid particles.

#ifndef _BUILDNEIBS_KERNEL_
#define _BUILDNEIBS_KERNEL_

#include "particledefine.h"
#include "textures.cuh"
#include "vector_math.h"

// TODO : what was CELLTYPE_MASK_* supposed to be ? Can we delete ?
// CELLTYPE_MASK_*
#include "multi_gpu_defines.h"


/** \namespace cuneibs
 *  \brief Contains all device functions/kernels/variables used for neighbor list construction
 *
 *  The namespace cuneibs contains all the device part of neighbor list construction :
 *  	- device constants/variables
 *  	- device functions
 *  	- kernels
 *
 *  \ingroup neibs
 */
namespace cuneibs {
/** \name Device constants
 *  @{ */
__constant__ uint d_maxneibsnum;		///< Maximum allowed number of neighbors per particle
__constant__ idx_t d_neiblist_stride;	///< Stride dimension
/** @} */
/** \name Device variables
 *  @{ */
__device__ int d_numInteractions;		///< Total number of interactions
__device__ int d_maxNeibs;				///< Computed maximum number of neighbors per particle
/** @} */

using namespace cubounds;

/** \name Device functions
 *  @{ */
/// Clamp grid position to edges according to periodicity
/*! This function clamp grid position to edges according to the chosen
 * 	periodicity, returns the new grid position and update the grid offset.
 *
 *	\tparam periodicbound : type of periodic boundaries (0 ... 7)
 *
 *	\param[in] gridPos : grid position to be clamped
 *	\param[in] gridOffset : grid offset
 *	\param[out] toofar : has the gridPos been clamped when the offset was of more than 1 cell ?
 *
 *	\return new grid position
 */
// TODO: verify periodicity along multiple axis
template <Periodicity periodicbound>
__device__ __forceinline__ int3
clampGridPos(const int3& gridPos, int3& gridOffset, bool *toofar)
{
	int3 newGridPos = gridPos + gridOffset;
	// For the axis involved in periodicity the new grid position reflects
	// the periodicity and should not be clamped and the grid offset remains
	// unchanged.
	// For the axis not involved in periodicity the new grid position
	// is equal to the clamped old one and the grid offset is updated.

	// Periodicity in x
	if (periodicbound & PERIODIC_X) {
		if (newGridPos.x < 0) newGridPos.x += d_gridSize.x;
		if (newGridPos.x >= d_gridSize.x) newGridPos.x -= d_gridSize.x;
	} else {
		newGridPos.x = min(max(0, newGridPos.x), d_gridSize.x-1);
		if (abs(gridOffset.x) > 1 && newGridPos.x == gridPos.x)
			*toofar = true;
		gridOffset.x = newGridPos.x - gridPos.x;
	}

	// Periodicity in y
	if (periodicbound & PERIODIC_Y) {
		if (newGridPos.y < 0) newGridPos.y += d_gridSize.y;
		if (newGridPos.y >= d_gridSize.y) newGridPos.y -= d_gridSize.y;
	} else {
		newGridPos.y = min(max(0, newGridPos.y), d_gridSize.y-1);
		if (abs(gridOffset.y) > 1 && newGridPos.y == gridPos.y)
			*toofar = true;
		gridOffset.y = newGridPos.y - gridPos.y;
	}

	// Periodicity in z
	if (periodicbound & PERIODIC_Z) {
		if (newGridPos.z < 0) newGridPos.z += d_gridSize.z;
		if (newGridPos.z >= d_gridSize.z) newGridPos.z -= d_gridSize.z;
	} else {
		newGridPos.z = min(max(0, newGridPos.z), d_gridSize.z-1);
		if (abs(gridOffset.z) > 1 && newGridPos.z == gridPos.z)
			*toofar = true;
		gridOffset.z = newGridPos.z - gridPos.z;
	}

	return newGridPos;
}


/// clampGridPos Specialization without any periodicity
/*! @see clampGridPos
 */
template <>
__device__ __forceinline__ int3
clampGridPos<PERIODIC_NONE>(const int3& gridPos, int3& gridOffset, bool *toofar)
{
	int3 newGridPos = gridPos + gridOffset;

	// Without periodicity the new grid position is clamped to edges
	newGridPos.x = min(max(0, newGridPos.x), d_gridSize.x-1);
	newGridPos.y = min(max(0, newGridPos.y), d_gridSize.y-1);
	newGridPos.z = min(max(0, newGridPos.z), d_gridSize.z-1);
	if ((abs(gridOffset.x) > 1 && newGridPos.x == gridPos.x) ||
		(abs(gridOffset.y) > 1 && newGridPos.y == gridPos.y) ||
		(abs(gridOffset.z) > 1 && newGridPos.z == gridPos.z))
		*toofar = true;

	// In case of change in grid position the grid offset is updated
	gridOffset = newGridPos - gridPos;

	return newGridPos;
}
/** @} */


/** \name Kernels
 *  @{ */
/// Updates particles hash value of particles and prepare the index table
/*! This kernel should be called before the sort. It
 * 	- updates hash values and relative positions for fluid and
 * 	object particles
 * 	- fill the particle's indexes array with current index
 *	\tparam periodicbound : type of periodic boundaries (0 ... 7)
 *	\param[in,out] posArray : particle's positions
 *	\param[in,out] particleHash : particle's hashes
 *	\param[out] particleIndex : particle's indexes
 *	\param[in] particleInfo : particle's informations
 *	\param[in] numParticles : total number of particles
 */
// TODO: document compactMapDevice (Alexis).
template <Periodicity periodicbound>
__global__ void
/*! \cond */
__launch_bounds__(BLOCK_SIZE_CALCHASH, MIN_BLOCKS_CALCHASH)
/*! \endcond */
calcHashDevice(float4*			posArray,			// particle's positions (in, out)
			   hashKey*			particleHash,		// particle's hashes (in, out)
			   uint*			particleIndex,		// particle's indexes (out)
			   const particleinfo*	particelInfo,	// particle's informations (in)
			   uint				*compactDeviceMap,	// TODO
			   const uint		numParticles)		// total number of particles
{
	const uint index = INTMUL(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	const particleinfo info = particelInfo[index];

	// Get the old grid hash
	uint gridHash = cellHashFromParticleHash( particleHash[index] );

	// We compute new hash only for fluid and moving not fluid particles (object, moving boundaries)
	if (FLUID(info) || MOVING(info)) {
		// Getting new pos relative to old cell
		float4 pos = posArray[index];

		// Getting grid address of old cell (computed from old hash)
		const int3 gridPos = calcGridPosFromCellHash(gridHash);

		// Computing grid offset from new pos relative to old hash
		int3 gridOffset = make_int3(floor((as_float3(pos) + 0.5f*d_cellSize)/d_cellSize));

		// Has the particle flown out of the domain by more than a cell? Clamping
		// its position will set this to true if necessary
		bool toofar = false;
		// Compute new grid pos relative to cell, adjust grid offset and compute new cell hash
		gridHash = calcGridHash(clampGridPos<periodicbound>(gridPos, gridOffset, &toofar));

		// Adjust position
		as_float3(pos) -= gridOffset*d_cellSize;

		// If the particle would have flown out of the domain by more than a cell, disable it
		if (toofar)
			disable_particle(pos);

		// Mark with special hash if inactive.
		// NOTE: it could have been marked as inactive outside this kernel.
		if (INACTIVE(pos))
			gridHash = CELL_HASH_MAX;

		posArray[index] = pos;
	}

	// Mark the cell as inner/outer and/or edge by setting the high bits
	// the value in the compact device map is a CELLTYPE_*_SHIFTED, so 32 bit with high bits set.
	// See multi_gpu_defines.h for the definition of these macros.
	if (compactDeviceMap && gridHash != CELL_HASH_MAX)
		gridHash |= compactDeviceMap[gridHash];

	// Store grid hash
	particleHash[index] = gridHash;

	// Preparing particle index array for the sort phase
	particleIndex[index] = index;
}


/// Updates high bits of cell hash with compact device map
/*! This kernel is specific for MULTI_DEVICE simulations
 * 	and should be called at the 1st iteration.
 * 	He computes the high bits of particle hash according to the
 * 	compact device map. Also initialize particleIndex.
 * 	\tparam periodicbound : type of periodic boundaries (0 ... 7)
 * 	\param[in,out] particleHash : particle's hashes
 * 	\param[out] particleIndex : particle's indexes
 * 	\param[in] particleInfo : particle's informations
 * 	\param[out] compactDeviceMap : ???
 * 	\param[in] numParticles : total number of particles
 */
__global__ void
/*! \cond */
__launch_bounds__(BLOCK_SIZE_CALCHASH, MIN_BLOCKS_CALCHASH)
/*! \endcond */
fixHashDevice(hashKey*			particleHash,		// particle's hashes (in, out)
			   uint*			particleIndex,		// particle's indexes (out)
			   const particleinfo*	particelInfo,	// particle's informations (in)
			   uint				*compactDeviceMap,	// TODO
			   const uint		numParticles)		// total number of particles
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	if (particleHash) {

		uint gridHash = cellHashFromParticleHash( particleHash[index] );

		// Mark the cell as inner/outer and/or edge by setting the high bits
		// the value in the compact device map is a CELLTYPE_*_SHIFTED, so 32 bit with high bits set
		if (compactDeviceMap)
			particleHash[index] = particleHash[index] | compactDeviceMap[gridHash];
	}

	// Preparing particle index array for the sort phase
	particleIndex[index] = index;
}


/// Reorders particles data after the sort and updates cells informations
/*! This kernel should be called after the sort. It
 * 		- computes the index of the first and last particle of
 * 		each grid cell
 * 		- reorders the particle's data (position, velocity, ...)
 * 		according to particles index that have been previously
 * 		sorted during the sort phase
 *
 *	\param[out] cellStart : index of cells first particle
 *	\param[out] cellEnd : index of cells last particle
 *	\param[out] segmentStart : TODO
 *	\param[out] sortedPos : new sorted particle's positions
 *	\param[out] sortedVel : new sorted particle's velocities
 *	\param[out] sortedInfo : new sorted particle's informations
 *	\param[out] sortedBoundElements : new sorted boundary elements
 *	\param[out] sortedGradGamma : new sorted gradient of gamma
 *	\param[out] sortedVertices : new sorted vertices
 *	\param[out] sortedTKE : new sorted k
 *	\param[out] sortedEps : new sorted e
 *	\param[out] sortedTurbVisc : new sorted eddy viscosity
 *	\param[out] sortedEulerVel : new sorted eulerian velocity (used in SA only)
 *	\param[in] particleHash : previously sorted particle's hashes
 *	\param[in] particleIndex : previously sorted particle's indexes
 *	\param[in] numParticles : total number of particles
 *	\param[out] numParticles : device pointer to new number of active particles
 *
 * In order to avoid WAR issues we use double buffering : the unsorted data
 * are read trough texture fetches and the sorted one written in a coalesced
 * way in global memory.
 */
// FIXME: we cannot avoid WAR, instead we need to be prepared to WAR ....
// TODO: should be templatized according to boundary type. (Alexis)
// TODO: k goes with e, make it a float2. (Alexis).
// TODO: document segmentStart (Alexis).
__global__
/*! \cond */
__launch_bounds__(BLOCK_SIZE_REORDERDATA, MIN_BLOCKS_REORDERDATA)
/*! \endcond */
void reorderDataAndFindCellStartDevice( uint*			cellStart,			// index of cells first particle (out)
										uint*			cellEnd,			// index of cells last particle (out)
										uint*			segmentStart,		// TODO
										float4*			sortedPos,			// new sorted particle's positions (out)
										float4*			sortedVel,			// new sorted particle's velocities (out)
										float4*			sortedVol,			// new sorted particle's volumes (out)
										float4*			sortedBoundElements,// new sorted boundary elements (out)
										float4*			sortedGradGamma,	// new sorted gradient gamma (out)
										vertexinfo*		sortedVertices,		// new sorted vertices (out)
										float*			sortedTKE,			// new sorted k for k-e model (out)
										float*			sortedEps,			// new sorted e for k-e model (out)
										float*			sortedTurbVisc,		// new sorted eddy viscosity (out)
										float4*			sortedEulerVel,		// new sorted eulerian velocity (out)
										const particleinfo*	particleInfo,	// previously sorted particle's informations (in)
										const hashKey*	particleHash,		// previously sorted particle's hashes (in)
										const uint*		particleIndex,		// previously sorted particle's hashes (in)
										const uint		numParticles,		// total number of particles (in)
										uint*			newNumParticles)	// device pointer to new number of active particles (out)
{
	// Shared hash array of dimension blockSize + 1
	extern __shared__ uint sharedHash[];

	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	// Initialize segmentStarts
	if (segmentStart && index < 4) segmentStart[index] = EMPTY_SEGMENT;

	uint cellHash;
	// Handle the case when number of particles is not multiple of block size
	if (index < numParticles) {
		// To find where cells start/end we only need the cell part of the hash.
		// Note: we do not reset the high bits since we need them to find the segments
		// (aka where the outer particles begin)
		cellHash = cellHashFromParticleHash(particleHash[index], true);

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = cellHash;

		if (index > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
			sharedHash[0] = cellHashFromParticleHash(particleHash[index - 1], true);
		}
	}

	__syncthreads();

	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell
		// or the first inactive particle.
		// Store the index of this particle as the new cell start and as
		// the previous cell end

		// Note: we need to reset the high bits of the cell hash if the particle hash is 64 bits wide
		// every time we use a cell hash to access an element of CellStart or CellEnd

		if (index == 0 || cellHash != sharedHash[threadIdx.x]) {

			// New cell, otherwise, it's the number of active particles (short hash: compare with 32 bits max)
			if (cellHash != CELL_HASH_MAX)
				// If it isn't an inactive particle, it is also the start of the cell
				cellStart[cellHash & CELLTYPE_BITMASK] = index;
			else
				*newNumParticles = index;

			// If it isn't the first particle, it must also be the end of the previous cell
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x] & CELLTYPE_BITMASK] = index;
		}

		// If we are an inactive particle, we're done (short hash: compare with 32 bits max)
		if (cellHash == CELL_HASH_MAX)
			return;

		if (index == numParticles - 1) {
			// Ditto
			cellEnd[cellHash & CELLTYPE_BITMASK] = index + 1;
			*newNumParticles = numParticles;
		}

		if (segmentStart) {
			// If segment start is given, hash key size is 64 and we detect the segments
			uchar curr_type = cellHash >> 30;
			uchar prev_type = sharedHash[threadIdx.x] >> 30;
			if (index == 0 || curr_type != prev_type)
				segmentStart[curr_type] = index;
		}

		// Now use the sorted index to reorder particle's data
		const uint sortedIndex = particleIndex[index];
		const float4 pos = tex1Dfetch(posTex, sortedIndex);
		const float4 vel = tex1Dfetch(velTex, sortedIndex);

		sortedPos[index] = pos;
		sortedVel[index] = vel;

		if (sortedVol) {
			sortedVol[index] = tex1Dfetch(volTex, sortedIndex);
		}

		if (sortedBoundElements) {
			sortedBoundElements[index] = tex1Dfetch(boundTex, sortedIndex);
		}

		if (sortedGradGamma) {
			sortedGradGamma[index] = tex1Dfetch(gamTex, sortedIndex);
		}

		if (sortedVertices) {
			if (BOUNDARY(particleInfo[index])) {
				const vertexinfo vertices = tex1Dfetch(vertTex, sortedIndex);
				sortedVertices[index] = vertices;
			}
			else
				sortedVertices[index] = make_vertexinfo(0, 0, 0, 0);
		}

		if (sortedTKE) {
			sortedTKE[index] = tex1Dfetch(keps_kTex, sortedIndex);
		}

		if (sortedEps) {
			sortedEps[index] = tex1Dfetch(keps_eTex, sortedIndex);
		}

		if (sortedTurbVisc) {
			sortedTurbVisc[index] = tex1Dfetch(tviscTex, sortedIndex);
		}

		if (sortedEulerVel) {
			sortedEulerVel[index] = tex1Dfetch(eulerVelTex, sortedIndex);
		}

	}
}


/// Update ID-to-particleIndex lookup table (BUFFER_VERTIDINDEX)
/*! Update ID-to-particleIndex lookup table. This kernel should be
 * 	called after the reorder.
 * 	\param[in] particleInfo : particle's informations
 * 	\param[out] vertIDToIndex : ID-to-particle index lookup table, overwritten
 * 	\param[in] numParticles : total number of particles
 */
__global__
/*! \cond */
__launch_bounds__(BLOCK_SIZE_REORDERDATA, MIN_BLOCKS_REORDERDATA)
/*! \endcond */
void updateVertIDToIndexDevice(	const particleinfo*	particleInfo,	///< particle's informations (in)
								uint*			vertIDToIndex,		///< vertex ID to index array (out)
								const uint		numParticles)		///< total number of particles (in)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	// Handle the case when number of particles is not multiple of block size
	if (index >= numParticles)
		return;

	// Assuming vertIDToIndex is allocated, since this kernel is called only with SA bounds
	particleinfo info = particleInfo[index];

	// Only vertex particles need to have this information, it should not be done
	// for fluid particles as their id's can grow and cause buffer overflows
	if(VERTEX(info))
		// As the vertex particles never change their id (which is <= than the initial
		// particle count), this buffer does not overflow
		vertIDToIndex[ id(info) ] = index;
}
/** @} */


/** \name Device functions
 *  @{ */
/// Compute the grid position for a neighboring cell
/*! This function computes the grid position for a neighboring cell,
 * 	according to the given offset and periodicity.
 *
 *	\param[in, out] gridPos : current grid position (in) and neighbor grid position (out)
 *	\param[in] gridOffset : offset from current grid position
 *
 *	\tparam
 *
 *	\return true if the new cell is in the domain, false otherwise.
 */
template <Periodicity periodicbound>
__device__ __forceinline__ bool
calcNeibCell(
		int3 &gridPos, 			///< current grid position
		int3 const& gridOffset) ///< cell offset from current grid position
{
	// Compute the grid position of the current cell
	gridPos += gridOffset;

	// With periodic boundary when the neighboring cell grid position lies
	// outside the domain size: we wrap it to the d_gridSize or 0 according
	// with the chosen periodicity
	// TODO: verify periodicity along multiple axis
	if (periodicbound) {
		// Periodicity along x axis
		if (gridPos.x < 0) {
			if (periodicbound & PERIODIC_X)
				gridPos.x = d_gridSize.x - 1;
			else
				return false;
		}
		else if (gridPos.x >= d_gridSize.x) {
			if (periodicbound & PERIODIC_X)
				gridPos.x = 0;
			else
				return false;
		}

		// Periodicity along y axis
		if (gridPos.y < 0) {
			if (periodicbound & PERIODIC_Y)
				gridPos.y = d_gridSize.y - 1;
			else
				return false;
		}
		else if (gridPos.y >= d_gridSize.y) {
			if (periodicbound & PERIODIC_Y)
				gridPos.y = 0;
			else
				return false;
		}

		// Periodicity along z axis
		if (gridPos.z < 0) {
			if (periodicbound & PERIODIC_Z)
				gridPos.z = d_gridSize.z - 1;
			else
				return false;
		}
		else if (gridPos.z >= d_gridSize.z) {
			if (periodicbound & PERIODIC_Z)
				gridPos.z = 0;
			else
				return false;
		}
	}
	// Without periodic boundary when the neighboring cell grid position lies
	// outside the domain size there is nothing to do
	else {
		if ((gridPos.x < 0) || (gridPos.x >= d_gridSize.x) ||
			(gridPos.y < 0) || (gridPos.y >= d_gridSize.y) ||
			(gridPos.z < 0) || (gridPos.z >= d_gridSize.z))
				return false;
	}

	// If we get here, the new gridPos was computed correctly, we are
	// still in the domain
	return true;

}
/** @} */

//TODO: Giuseppe write a REAL documentation COMPLYING with Doxygen
// standards and with OUR DOCUMENTING CONVENTIONS !!!! (Alexis).
/// variables found in all specializations of neibsInCell
/** \name Data structures
 *  @{ */
/*!	\struct common_niC_vars
 * 	\brief Common parameters used in neibsInCell device function
 *
 * 	Parameters passed to neibsInCell device function depends on the type of
 * 	of boundary used. This structure contains the parameters common to all
 * 	boundary types.
 */
struct common_niC_vars
{
	const	uint	gridHash;		///< Hash value of grid position
	const	uint	bucketStart;	///< Index of first particle in cell
	const	uint	bucketEnd;		///< Index of last particle in cell

	/// Constructor
	/*!	Computes struct member values according to the grid position.
	 * 	\param[in] gridPos : position in the grid
	 */
	__device__ __forceinline__
	common_niC_vars(int3 const& gridPos) :
		gridHash(calcGridHash(gridPos)),
		bucketStart(tex1Dfetch(cellStartTex, gridHash)),
		bucketEnd(tex1Dfetch(cellEndTex, gridHash))
	{}
};


/*!	\struct sa_boundary_niC_vars
 * 	\brief Specific parameters used in neibsInCell with SA boundary
 *
 * 	This structure contains specific parameters to be passed in plus of the
 * 	common one to neibsInCell.
 */
struct sa_boundary_niC_vars
{
	vertexinfo			vertices; 		///< TODO
	const	float4		boundElement; 	///< TODO
	const	uint		j; 				///< TODO
	const	float4		coord1; 		///< TODO
	const	float4		coord2; 		///< TODO

	/// Constructor
	/*!	Computes struct variable values according to particle's index
	 * 	and parameters passed to buildNeibsKernnel
	 * 	\param[in] index : particle index
	 * 	\param[in] bparams : TODO
	 */
	__device__ __forceinline__
	sa_boundary_niC_vars(const uint index, buildneibs_params<SA_BOUNDARY> const& bparams) :
		vertices(tex1Dfetch(vertTex, index)),
		boundElement(tex1Dfetch(boundTex, index)),
		// j is 0, 1 or 2 depending on which is smaller (in magnitude) between
		// boundElement.{x,y,z}
		j(
			(fabs(boundElement.z) < fabs(boundElement.y) &&
			fabs(boundElement.z) < fabs(boundElement.x)) ? 2 :
			(fabs(boundElement.y) < fabs(boundElement.x) ? 1 : 0)
		 ),
		// Compute the first coordinate which is a 2-D rotated version of the normal with the j-th coordinate set to 0
		coord1(
			normalize(make_float4(
			// switch over j to give: 0 -> (0, z, -y); 1 -> (-z, 0, x); 2 -> (y, -x, 0)
			-((j==1)*boundElement.z) +  (j == 2)*boundElement.y , // -z if j == 1, y if j == 2
			  (j==0)*boundElement.z  - ((j == 2)*boundElement.x), // z if j == 0, -x if j == 2
			-((j==0)*boundElement.y) +  (j == 1)*boundElement.x , // -y if j == 0, x if j == 1
			0))
			),
		// The second coordinate is the cross product between the normal and the first coordinate
		coord2( cross3(boundElement, coord1) )
		{
			// Here local copy of part IDs of vertices are replaced by the correspondent part indices
			vertices.x = bparams.vertIDToIndex[vertices.x];
			vertices.y = bparams.vertIDToIndex[vertices.y];
			vertices.z = bparams.vertIDToIndex[vertices.z];
		}
};


/*!	\struct niC_vars
 * 	\brief Parameters used in neibsInCell device function
 *
 * 	This structure contains all the parameters needed by neibsInCell.
 * 	The parameters automatically adjust them self in case of use of SA
 * 	boundary type.
 * 	\tparam boundarytype : the boundary model used
 */
template<BoundaryType boundarytype>
struct niC_vars :
	common_niC_vars,
	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_niC_vars)
{
	/// Constructor
	/*!	Computes struct member values according to particle's index
	 * 	and parameters passed to buildNeibsKernnel
	 * 	\param[in] index : particle index
	 * 	\param[in] bparams : TODO
	 */
	__device__ __forceinline__
	niC_vars(int3 const& gridPos, const uint index, buildneibs_params<boundarytype> const& bparams) :
		common_niC_vars(gridPos),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_niC_vars)(index, bparams)
	{}
};
/** @} */


/** \name Device functions
 *  @{ */
/// Check if a particle is close enough to be considered for neibslist inclusion
/*! Compares the squared distance between two particles to the squared influence
 * 	radius.
 *
 * 	\param[in] relPos : relative position vector
 * 	\return : true if the distance is < to the squared influence radius, false otherwise
 * 	\tparam boundarytype : the boundary model used
 */
template<BoundaryType boundarytype>
__device__ __forceinline__
bool isCloseEnough(float3 const& relPos, particleinfo const& neib_info,
	buildneibs_params<boundarytype> const& params)
{
	// Default : check against the influence radius
	return sqlength(relPos) < params.sqinfluenceradius;
}

/// Specialization of isCloseEnough for SA boundaries
/// \see isCloseEnough
template<>
__device__ __forceinline__
bool isCloseEnough<SA_BOUNDARY>(float3 const& relPos, particleinfo const& neib_info,
	buildneibs_params<SA_BOUNDARY> const& params)
{
	const float rp2(sqlength(relPos));
	// Include boundary neighbors which are a little further than sqinfluenceradius
	return (rp2 < params.sqinfluenceradius ||
		(rp2 < params.boundNlSqInflRad && BOUNDARY(neib_info)));
}


/// Process SA segments in neibsInCell
/*! Do special treatment for segments when using SA boundaries. Obviously
 * 	don't do anything at all in the standard case..
 *
 * 	\param[in] index : particle index
 * 	\param[in] neib_index : neighbor index
 * 	\param[in] relPos : relative position vector
 * 	\param[in] params : build neibs parameters
 * 	\param[in] vars : neib in cell variables
 * 	\return : true if the distance is < to the squared influence radius, false otherwise
 * 	\tparam boundarytype : the boundary model used
 */
template<BoundaryType boundarytype>
__device__ __forceinline__
void process_niC_segment(const uint index, const uint neib_index, float3 const& relPos,
	buildneibs_params<boundarytype> const& params,
	niC_vars<boundarytype> const& var)
{ /* Do nothing by default */ }


/// Specialization of process_niC_segment for SA boundaries
/// \see process_niC_segment
template<>
__device__ __forceinline__
void process_niC_segment<SA_BOUNDARY>(const uint index, const uint neib_index, float3 const& relPos,
	buildneibs_params<SA_BOUNDARY> const& params,
	niC_vars<SA_BOUNDARY> const& var)
{
	int i = -1;
	if (neib_index == var.vertices.x)
		i = 0;
	else if (neib_index == var.vertices.y)
		i = 1;
	else if (neib_index == var.vertices.z)
		i = 2;
	if (i>-1) {
		// relPosProj is the projected relative position of the vertex to the segment.
		// the first coordinate system is given by the following two vectors:
		// 1. set one coordinate to 0 and rotate the remaining 2-D vector
		// 2. cross product between coord1 and the normal of the boundary element
		float2 relPosProj = make_float2(0.0);
		relPosProj.x = dot(relPos, as_float3(var.coord1));
		relPosProj.y = dot(relPos, as_float3(var.coord2));
		// save relPosProj in vertPos buffer
		if (i==0)
			params.vertPos0[index] = relPosProj;
		else if (i==1)
			params.vertPos1[index] = relPosProj;
		else
			params.vertPos2[index] = relPosProj;
	}
}


/// Find neighbors in a given cell
/*! This function look for neighbors of the current particle in
 * 	a given cell.
 * 	The parameter params is built on specialized version of
 * 	build_neibs_params according to template values.
 * 	If the current particle belongs to a segment, the function
 * 	will also look for the tree associated vertices.
 *
 *	\param[in, out] buildneibs_params : build neibs parameters
 *	\param[in] gridPos : current particle grid position
 *	\param[in] gridOffset : cell offset from current particle cell
 *	\param[in] cell : cell number
 *	\param[in] index : index of the current particle
 *	\param[in] pos : position of the current particle
 *	\param[in, out] neibs_num : current number of neighbors found for current particle
 *	\param[in] segment : true if the current particle belongs to a segment
 *
 *	\tparam boundarytype : the boundary model used
 *	\tparam periodicbound : type of periodic boundaries (0 ... 7)
 *
 * First and last particle index for grid cells and particle's informations
 * are read through texture fetches.
 */
template <SPHFormulation sph_formulation, BoundaryType boundarytype, Periodicity periodicbound>
__device__ __forceinline__ void
neibsInCell(
			buildneibs_params<boundarytype>
				const& params,			// build neibs parameters
			int3			gridPos,	// current particle grid position
			const int3		gridOffset,	// cell offset from current particle grid position
			const uchar		cell,		// cell number (0 ... 26)
			const uint		index,		// current particle index
			float3			pos,		// current particle position
			uint&			neibs_num,	// number of neighbors for the current particle
			const bool		segment,	// true if the current particle belongs to a segment
			const bool		boundary)	// true if the current particle is a boundary particle
{
	// Compute the grid position of the current cell, and return if it's
	// outside the domain
	if (!calcNeibCell<periodicbound>(gridPos, gridOffset))
		return;

	// Internal variables used by neibsInCell. Structure built on
	// specialized template of niC_vars.
	niC_vars<boundarytype> var(gridPos, index, params);

	// Return if the cell is empty
	if (var.bucketStart == 0xffffffff)
		return;

	// Substract gridOffset*cellsize to pos so we don't need to do it each time
	// we compute relPos respect to potential neighbor
	pos -= gridOffset*d_cellSize;

	// Iterate over all particles in the cell
	bool encode_cell = true;

	for (uint neib_index = var.bucketStart; neib_index < var.bucketEnd; neib_index++) {

		// Prevent self-interaction
		if (neib_index == index)
			continue;

		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// Testpoints have a neighbor list, but are not considered in the neighbor list
		// of other points
		if (TESTPOINT(neib_info))
			continue;

		// With dynamic boundaries, boundary parts don't interact with other boundary parts
		// except for Grenier's formulation, where the sigma computation needs all neighbors
		// to be enumerated
		if (boundarytype == DYN_BOUNDARY && sph_formulation != SPH_GRENIER) {
			if (boundary && BOUNDARY(neib_info))
				continue;
		}

		// Compute relative position between particle and potential neighbor
		// NOTE: using as_float3 instead of make_float3 result in a 25% performance loss
		#if (__COMPUTE__ >= 20)
		const float4 neib_pos = params.posArray[neib_index];
		#else
		const float4 neib_pos = tex1Dfetch(posTex, neib_index);
		#endif

		// Skip inactive particles
		if (INACTIVE(neib_pos))
			continue;

		const float3 relPos = pos - make_float3(neib_pos);

		// Check if the squared distance is smaller than the squared influence radius
		// used for neighbor list construction
		bool close_enough = isCloseEnough(relPos, neib_info, params);

		if (close_enough) {
			if (neibs_num < d_maxneibsnum) {
				params.neibsList[neibs_num*d_neiblist_stride + index] =
						neib_index - var.bucketStart + ((encode_cell) ? ENCODE_CELL(cell) : 0);
				encode_cell = false;
			}
			neibs_num++;
		}
		if (segment) {
			process_niC_segment(index, neib_index, relPos, params, var);
		}

	}

	return;
}
/**  @} */

/** \name Kernels
 *  @{ */
/// Builds particles neighbors list
/*! This kernel builds the neighbor's indexes of all particles. The
 * 	parameter params is built on specialized version of
 * 	build_neibs_params according to template values.
 *
 *	\param[in, out] params: build neibs parameters
 *	\tparam boundarytype : boundary type (determines which particles have a neib list)
 *	\tparam periodicbound : type periodic boundaries (0 ... 7)
 *	\tparam neibcount : if true we compute maximum neighbor number
 *
 *	First and last particle index for grid cells and particle's informations
 *	are read through texture fetches.
 */
template<SPHFormulation sph_formulation, BoundaryType boundarytype, Periodicity periodicbound,
	bool neibcount>
__global__ void
/*! \cond */
__launch_bounds__( BLOCK_SIZE_BUILDNEIBS, MIN_BLOCKS_BUILDNEIBS)
/*! \endcond */
buildNeibsListDevice(buildneibs_params<boundarytype> params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	// Number of neighbors for the current particle
	uint neibs_num = 0;

	// Rather than nesting if's, use a do { } while (0) loop with breaks
	// for early bail outs
	do {
		if (index >= params.numParticles)
			break;

		// Read particle info from texture
		const particleinfo info = tex1Dfetch(infoTex, index);

		// The way the neighbor's list it's construct depends on
		// the boundary type used in the simulation.
		// 	* For Lennard-Johnes boundaries :
		//		we construct a neighbor's list for fluid, test points
		//		and particles belonging to a floating body or a moving
		//		boundary on which we want to compute forces.
		//	* For SA boundaries :
		//		same as Lennard-Johnes plus vertice and boundary particles
		//	* For dynamic boundaries :
		//		we construct a neighbor's for all particles.
		//TODO: optimze test. (Alexis).
		bool build_nl = FLUID(info) || TESTPOINT(info) || FLOATING(info) || COMPUTE_FORCE(info);
		if (boundarytype == SA_BOUNDARY)
			build_nl = build_nl || VERTEX(info) || BOUNDARY(info);
		if (boundarytype == DYN_BOUNDARY)
			build_nl = true;

		// Exit if we have nothing to do
		if (!build_nl)
			break;

		// Get particle position
		#if (__COMPUTE__ >= 20)
		const float4 pos = params.posArray[index];
		#else
		const float4 pos = tex1Dfetch(posTex, index);
		#endif

		// If the particle is inactive we have nothing to do
		if (INACTIVE(pos))
			break;

		const float3 pos3 = make_float3(pos);

		// Get particle grid position computed from particle hash
		const int3 gridPos = calcGridPosFromParticleHash(params.particleHash[index]);

		for(int z=-1; z<=1; z++) {
			for(int y=-1; y<=1; y++) {
				for(int x=-1; x<=1; x++) {
					neibsInCell<sph_formulation, boundarytype, periodicbound>(params,
						gridPos,
						make_int3(x, y, z),
						(x + 1) + (y + 1)*3 + (z + 1)*9,
						index,
						pos3,
						neibs_num,
						BOUNDARY(info),
						BOUNDARY(info));
				}
			}
		}
	} while (0);

	// Setting the end marker. Must be done here so that
	// particles for which the neighbor list is not built actually
	// have an empty neighbor list. Otherwise, particles which are
	// marked inactive will keep their old neighbor list.
	if (index < params.numParticles && neibs_num < d_maxneibsnum) {
		params.neibsList[neibs_num*d_neiblist_stride + index] = 0xffff;
	}

	if (neibcount) {
		// Shared memory reduction of per block maximum number of neighbors
		__shared__ volatile uint sm_neibs_num[BLOCK_SIZE_BUILDNEIBS];
		__shared__ volatile uint sm_neibs_max[BLOCK_SIZE_BUILDNEIBS];

		sm_neibs_num[threadIdx.x] = neibs_num;
		sm_neibs_max[threadIdx.x] = neibs_num;
		__syncthreads();

		uint i = blockDim.x/2;
		while (i != 0) {
			if (threadIdx.x < i) {
				sm_neibs_num[threadIdx.x] += sm_neibs_num[threadIdx.x + i];
				const float n1 = sm_neibs_max[threadIdx.x];
				const float n2 = sm_neibs_max[threadIdx.x + i];
				if (n2 > n1)
					sm_neibs_max[threadIdx.x] = n2;
			}
			__syncthreads();
			i /= 2;
		}

		if (!threadIdx.x) {
			atomicAdd(&d_numInteractions, sm_neibs_num[0]);
			atomicMax(&d_maxNeibs, sm_neibs_max[0]);
		}
	}
	return;
}
/**  @} */

}
#endif
