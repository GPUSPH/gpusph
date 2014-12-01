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
// CELLTYPE_MASK_*
#include "multi_gpu_defines.h"

namespace cuneibs {
__constant__ uint d_maxneibsnum;
__constant__ idx_t d_neiblist_stride;
__device__ int d_numInteractions;
__device__ int d_maxNeibs;

#include "cellgrid.h"

/// Clamp grid position to edges according to periodicity
/*! This function clamp grid position to edges according to the chosen
 * periodicity, returns the new grid position and update the grid offset.
 *
 *	\param[in] gridPos : grid position to be clamped
 *	\param[in] gridOffset : grid offset
 *	\param[out] toofar : has the gridPos been clamped when the offset was of more than 1 cell?
 *
 * 	\pparam periodicbound : use periodic boundaries (0 ... 7)
 *
 * 	\return : new grid position
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

	// periodicity in x
	if (periodicbound & PERIODIC_X) {
		if (newGridPos.x < 0) newGridPos.x += d_gridSize.x;
		if (newGridPos.x >= d_gridSize.x) newGridPos.x -= d_gridSize.x;
	} else {
		newGridPos.x = min(max(0, newGridPos.x), d_gridSize.x-1);
		if (abs(gridOffset.x) > 1 && newGridPos.x == gridPos.x)
			*toofar = true;
		gridOffset.x = newGridPos.x - gridPos.x;
	}

	// periodicity in y
	if (periodicbound & PERIODIC_Y) {
		if (newGridPos.y < 0) newGridPos.y += d_gridSize.y;
		if (newGridPos.y >= d_gridSize.y) newGridPos.y -= d_gridSize.y;
	} else {
		newGridPos.y = min(max(0, newGridPos.y), d_gridSize.y-1);
		if (abs(gridOffset.y) > 1 && newGridPos.y == gridPos.y)
			*toofar = true;
		gridOffset.y = newGridPos.y - gridPos.y;
	}

	// periodicity in z
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

/// Clamp grid position to edges without periodicity
/*! This function clamp grid position to edges according and
 * returns the new grid position and an updated grid offset.
 *
 *	\param[in] gridPos : grid position to be clamped
 *	\param[in/out] gridOffset : grid offset
 *	\param[out] toofar : has the gridPos been clamped when the offset was of more than 1 cell?
 *
 * 	\return : new grid position
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

/// Updates particles hash value of particles and prepare the index table
/*! This kernel should be called before the sort. It
 * 		- updates hash values and relative positions for fluid and
 * 		object particles
 * 		- fill the particle's indexes array with current index
 *
 *	\param[in,out] posArray : particle's positions
 *	\param[in,out] particleHash : particle's hashes
 *	\param[out] particleIndex : particle's indexes
 *	\param[in] particleInfo : particle's informations
 *	\param[in] numParticles : total number of particles
 *
 *	\pparam periodicbound : use periodic boundaries (0 ... 7)
 */
#define MOVINGNOTFLUID (PISTONPART | PADDLEPART | GATEPART | OBJECTPART | VERTEXPART) //TODO-AM the *PART defines are not flags
template <Periodicity periodicbound>
__global__ void
__launch_bounds__(BLOCK_SIZE_CALCHASH, MIN_BLOCKS_CALCHASH)
calcHashDevice(float4*			posArray,		///< particle's positions (in, out)
			   hashKey*			particleHash,	///< particle's hashes (in, out)
			   uint*			particleIndex,	///< particle's indexes (out)
			   const particleinfo*	particelInfo,	///< particle's informations (in)
			   uint				*compactDeviceMap,
			   const uint		numParticles)	///< total number of particles
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// Getting new pos relative to old cell
	float4 pos = posArray[index];
	const particleinfo info = particelInfo[index];

	// we compute new hash only for fluid and moving not fluid particles (object, moving boundaries)
	if ((FLUID(info) || (type(info) & MOVINGNOTFLUID))) {
		// Getting the old grid hash
		uint gridHash = cellHashFromParticleHash( particleHash[index] );

		// Getting grid address of old cell (computed from old hash)
		const int3 gridPos = calcGridPosFromCellHash(gridHash);

		// Computing grid offset from new pos relative to old hash
		int3 gridOffset = make_int3(floor((as_float3(pos) + 0.5f*d_cellSize)/d_cellSize));

		// has the particle flown out of the domain by more than a cell? clamping
		// its position will set this to true if necessary
		bool toofar = false;
		// Compute new grid pos relative to cell, adjust grid offset and compute new cell hash
		gridHash = calcGridHash(clampGridPos<periodicbound>(gridPos, gridOffset, &toofar));

		// mark the cell as inner/outer and/or edge by setting the high bits
		// the value in the compact device map is a CELLTYPE_*_SHIFTED, so 32 bit with high bits set
		if (compactDeviceMap)
			gridHash |= compactDeviceMap[gridHash];

		// Adjust position
		as_float3(pos) -= gridOffset*d_cellSize;
		// if the particle would have flown out of the domain by more than a cell, disable it
		if (toofar)
			disable_particle(pos);

		// mark with special hash if inactive
		if (INACTIVE(pos))
			gridHash = CELL_HASH_MAX;

		// Store grid hash, particle index and position relative to cell
		particleHash[index] = makeParticleHash(gridHash, info);
		posArray[index] = pos;
	}



	// Preparing particle index array for the sort phase
	particleIndex[index] = index;
}

// Similar to calcHash but specific for 1st iteration in MULTI_DEVICE simulations: does not change the cellHash,
// but only sets the high bits according to the compact device map. also, initializes particleIndex
__global__ void
__launch_bounds__(BLOCK_SIZE_CALCHASH, MIN_BLOCKS_CALCHASH)
fixHashDevice(hashKey*			particleHash,	///< particle's hashes (in, out)
			   uint*			particleIndex,	///< particle's indexes (out)
			   const particleinfo*	particelInfo,	///< particle's informations (in)
			   uint				*compactDeviceMap,
			   const uint		numParticles)	///< total number of particles
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	const particleinfo info = particelInfo[index];

	// We compute new hash only for fluid and moving not fluid particles (object, moving boundaries).
	// Also, if particleHash is NULL we just want to set particleIndex (see comment in GPUWorker::kernel_calcHash())
	if ((FLUID(info) || (type(info) & MOVINGNOTFLUID)) && particleHash) {

		uint gridHash = cellHashFromParticleHash( particleHash[index] );

		// mark the cell as inner/outer and/or edge by setting the high bits
		// the value in the compact device map is a CELLTYPE_*_SHIFTED, so 32 bit with high bits set
		if (compactDeviceMap)
			particleHash[index] = particleHash[index] | ((hashKey)compactDeviceMap[gridHash] << 32);
	}

	// Preparing particle index array for the sort phase
	particleIndex[index] = index;
}

#undef MOVINGNOTFLUID

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
 *	\param[out] sortedPos : new sorted particle's positions
 *	\param[out] sortedVel : new sorted particle's velocities
 *	\param[out] sortedInfo : new sorted particle's informations
 *	\param[in] particleHash : previously sorted particle's hashes
 *	\param[in] particleIndex : previously sorted particle's indexes
 *	\param[in] numParticles : total number of particles
 *
 * In order to avoid WAR issues we use double buffering : the unsorted data
 * are read trough texture fetches and the sorted one written in a coalesced
 * way in global memory.
 */
__global__
__launch_bounds__(BLOCK_SIZE_REORDERDATA, MIN_BLOCKS_REORDERDATA)
void reorderDataAndFindCellStartDevice( uint*			cellStart,		///< index of cells first particle (out)
										uint*			cellEnd,		///< index of cells last particle (out)
										uint*			segmentStart,
										float4*			sortedPos,		///< new sorted particle's positions (out)
										float4*			sortedVel,		///< new sorted particle's velocities (out)
										particleinfo*	sortedInfo,		///< new sorted particle's informations (out)
										float4*			sortedBoundElements,	// output: sorted boundary elements
										float4*			sortedGradGamma,	// output: sorted gradient gamma
										vertexinfo*		sortedVertices,		// output: sorted vertices
										float*			sortedTKE,			// output: k for k-e model
										float*			sortedEps,			// output: e for k-e model
										float*			sortedTurbVisc,		// output: eddy viscosity
										float4*			sortedEulerVel,		// output: sorted euler vel
										const hashKey*	particleHash,	///< previously sorted particle's hashes (in)
										const uint*		particleIndex,	///< previously sorted particle's hashes (in)
										const uint		numParticles,	///< total number of particles
										uint*			newNumParticles)	// output: number of active particles
{
	// Shared hash array of dimension blockSize + 1
	extern __shared__ uint sharedHash[];

	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	// initialize segmentStarts
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
		// everytime we use a cell hash to access an element of CellStart or CellEnd

		if (index == 0 || cellHash != sharedHash[threadIdx.x]) {

			// new cell, otherwise, it's the number of active particles (short hash: compare with 32 bits max)
			if (cellHash != CELL_HASH_MAX)
				// if it isn't an inactive particle, it is also the start of the cell
				cellStart[cellHash & CELLTYPE_BITMASK] = index;
			else
				*newNumParticles = index;

			// If it isn't the first particle, it must also be the end of the previous cell
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x] & CELLTYPE_BITMASK] = index;
		}

		// if we are an inactive particle, we're done (short hash: compare with 32 bits max)
		if (cellHash == CELL_HASH_MAX)
			return;

		if (index == numParticles - 1) {
			// ditto
			cellEnd[cellHash & CELLTYPE_BITMASK] = index + 1;
			*newNumParticles = numParticles;
		}

		if (segmentStart) {
			// if segment start is given, hash key size is 64 and we detect the segments
			uchar curr_type = cellHash >> 30;
			uchar prev_type = sharedHash[threadIdx.x] >> 30;
			if (index == 0 || curr_type != prev_type)
				segmentStart[curr_type] = index;
		}

		// Now use the sorted index to reorder particle's data
		const uint sortedIndex = particleIndex[index];
		const float4 pos = tex1Dfetch(posTex, sortedIndex);
		const float4 vel = tex1Dfetch(velTex, sortedIndex);
		const particleinfo info = tex1Dfetch(infoTex, sortedIndex);

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedInfo[index] = info;

		if (sortedBoundElements) {
			sortedBoundElements[index] = tex1Dfetch(boundTex, sortedIndex);
		}

		if (sortedGradGamma) {
			sortedGradGamma[index] = tex1Dfetch(gamTex, sortedIndex);
		}

		if (sortedVertices) {
			if (BOUNDARY(info)) {
				const vertexinfo vertices = tex1Dfetch(vertTex, sortedIndex);
				sortedVertices[index] = make_vertexinfo(
					vertices.x,
					vertices.y,
					vertices.z,
					vertices.w);
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
/*! This kernel should be called after the reorder.
 *
 *	\param[in] particleInfo : particleInfo
 *	\param[out] vertIDToIndex : ID-to-particleIndex lookup table, overwritten
 *	\param[in] numParticles : total number of particles
 */
__global__
__launch_bounds__(BLOCK_SIZE_REORDERDATA, MIN_BLOCKS_REORDERDATA)
void updateVertIDToIndexDevice(	particleinfo*	particleInfo,	///< particle's informations
								uint*			vertIDToIndex,	///< vertIDToIndex array (out)
								const uint		numParticles)	///< total number of particles
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	// Handle the case when number of particles is not multiple of block size
	if (index >= numParticles)
		return;

	// assuming vertIDToIndex is allocated, since this kernel is called only with SA bounds
	particleinfo info = particleInfo[index];

	// only vertex particles need to have this information, it should not be done
	// fluid particles as their ids can grow and cause buffer overflows
	if(VERTEX(info))
		// as the vertex particles never change their id (which is <= than the initial
		// particle count, this buffer does not overflow
		vertIDToIndex[ id(info) ] = index;
}

/// Compute the grid position for a neighbor cell
/*! This function computes the grid position for a neighbor cell,
 * according to periodicity.
 *
 * Returns true if the new cell is in the domain, false otherwise.
 */
template <Periodicity periodicbound>
__device__ __forceinline__ bool
calcNeibCell(
		int3 &gridPos, ///< current grid position
		int3 const& gridOffset) ///< cell offset from current grid position
{
	// Compute the grid position of the current cell
	gridPos += gridOffset;

	// With periodic boundary when the neighboring cell grid position lies
	// outside the domain size we wrap it to the d_gridSize or 0 according
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
	// if we get here, the new gridPos was computed correctly, we are
	// still in the domain
	return true;

}

/// variables found in all specializations of neibsInCell
struct common_niC_vars
{
	const	uint	gridHash;		// hash value of grid position
	const	uint	bucketStart;	// index of first particle in cell
	const	uint	bucketEnd;		// index of last particle in cell

	__device__ __forceinline__
	common_niC_vars(int3 const& gridPos) :
		gridHash(calcGridHash(gridPos)),
		bucketStart(tex1Dfetch(cellStartTex, gridHash)),
		bucketEnd(tex1Dfetch(cellEndTex, gridHash))
	{}
};

/// variables found in use_sa_boundary specialization of neibsInCell
struct sa_boundary_niC_vars
{
	vertexinfo	vertices;
	const	float4		boundElement;
	const	uint		j;
	const	float4		coord2;

	__device__ __forceinline__
	sa_boundary_niC_vars(const uint index, buildneibs_params<true> const& bparams) :
		vertices(tex1Dfetch(vertTex, index)),
		boundElement(tex1Dfetch(boundTex, index)),
		// j is 0, 1 or 2 depending on which is smaller (in magnitude) between
		// boundElement.{x,y,z}
		j(
			(fabs(boundElement.z) < fabs(boundElement.y) &&
			fabs(boundElement.z) < fabs(boundElement.x)) ? 2 :
			(fabs(boundElement.y) < fabs(boundElement.x) ? 1 : 0)
		 ),
		// compute second coordinate which is equal to n_s x e_j
		coord2(
			j == 0 ?
			make_float4(0.0f, boundElement.z, -boundElement.y, 0.0f) :
			j == 1 ?
			make_float4(-boundElement.z, 0.0f, boundElement.x, 0.0f) :
			// j == 2
			make_float4(boundElement.y, -boundElement.x, 0.0f, 0.0f)
			)
		{
			// here local copy of part IDs of vertices are replaced by the correspondent part indices
			vertices.x = bparams.vertIDToIndex[vertices.x];
			vertices.y = bparams.vertIDToIndex[vertices.y];
			vertices.z = bparams.vertIDToIndex[vertices.z];
		}
};

/// all neibsInCell variables
template<bool use_sa_boundary>
struct niC_vars :
	common_niC_vars,
	COND_STRUCT(use_sa_boundary, sa_boundary_niC_vars)
{
	__device__ __forceinline__
	niC_vars(int3 const& gridPos, const uint index, buildneibs_params<use_sa_boundary> const& bparams) :
		common_niC_vars(gridPos),
		COND_STRUCT(use_sa_boundary, sa_boundary_niC_vars)(index, bparams)
	{}
};

/// check if a particle at distance relPos is close enough to be considered for neibslist inclusion
template<bool use_sa_boundary>
__device__ __forceinline__
bool isCloseEnough(float3 const& relPos, particleinfo const& neibInfo,
	buildneibs_params<use_sa_boundary> params)
{
	return sqlength(relPos) < params.sqinfluenceradius; // default check: against the influence radius
}

/// SA_BOUNDARY specialization
template<>
__device__ __forceinline__
bool isCloseEnough<true>(float3 const& relPos, particleinfo const& neibInfo,
	buildneibs_params<true> params)
{
	const float rp2(sqlength(relPos));
	// include BOUNDARY neighbors which are a little further than sqinfluenceradius
	return (rp2 < params.sqinfluenceradius ||
		(rp2 < params.boundNlSqInflRad && BOUNDARY(neibInfo)));
}

/// process SA_BOUNDARY segments in neibsInCell
template<bool use_sa_boundary>
__device__ __forceinline__
void process_niC_segment(const uint index, const uint neib_index, float3 const& relPos,
	buildneibs_params<use_sa_boundary> const& params,
	niC_vars<use_sa_boundary> const& var)
{ /* do nothing by default */ }

template<>
__device__ __forceinline__
void process_niC_segment<true>(const uint index, const uint neib_index, float3 const& relPos,
	buildneibs_params<true> const& params,
	niC_vars<true> const& var)
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
		// 1. The unit vector e_j, where j is the coordinate for which n_s is minimal
		// 2. The cross product between n_s and e_j
		float2 relPosProj = make_float2(0.0);
		// relPosProj.x = relPos . e_j
		relPosProj.x = var.j==0 ? relPos.x : (var.j==1 ? relPos.y : relPos.z);
		// relPosProj.y = relPos . (n_s x e_j)
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
 * a given cell
 *
 *	\param[in] buildneibs_params : parameters to buildneibs
 *	\param[in] gridPos : current particle grid position
 *	\param[in] gridOffset : cell offset from current particle cell
 *	\param[in] cell : cell number
 *	\param[in] index : index of the current particle
 *	\param[in] pos : position of the current particle
 *	\param[in, out] neibs_num : current number of neighbors found for current particle
 *
 *	\pparam use_sa_boundary : use SA_BOUNDARY
 *	\pparam periodicbound : use periodic boundaries (0 ... 7)
 *
 * First and last particle index for grid cells and particle's informations
 * are read through texture fetches.
 */
template <bool use_sa_boundary, Periodicity periodicbound>
__device__ __forceinline__ void
neibsInCell(
			buildneibs_params<use_sa_boundary>
				const& params,	///< buildneibs params
			int3			gridPos,	///< current particle grid position
			const int3		gridOffset,	///< cell offset from current particle grid position
			const uchar		cell,		///< cell number (0 ... 26)
			const uint		index,		///< current particle index
			float3			pos,		///< current particle position
			uint&			neibs_num,	///< number of neighbors for the current particle
			const bool		segment)	///< if a segment is searching we are also looking for the three vertices
{
	// Compute the grid position of the current cell, and return if it's
	// outside the domain
	if (!calcNeibCell<periodicbound>(gridPos, gridOffset))
		return;

	niC_vars<use_sa_boundary> var(gridPos, index, params);

	// Return if the cell is empty
	if (var.bucketStart == 0xffffffff)
		return;

	// Substract gridOffset*cellsize to pos so we don't need to do it each time
	// we compute relPos respect to potential neighbor
	pos -= gridOffset*d_cellSize;

	// Iterate over all particles in the cell
	bool encode_cell = true;

	for (uint neib_index = var.bucketStart; neib_index < var.bucketEnd; neib_index++) {

		// no self-interaction
		if (neib_index == index)
			continue;

		const particleinfo neibInfo = tex1Dfetch(infoTex, neib_index);

		// testpoints have a neibs list, but are not considered in the neibs list of other
		// points
		if (TESTPOINTS(neibInfo))
			continue;

		// Compute relative position between particle and potential neighbor
		// NOTE: using as_float3 instead of make_float3 result in a 25% performance loss
		#if (__COMPUTE__ >= 20)
		const float4 neib_pos = params.posArray[neib_index];
		#else
		const float4 neib_pos = tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(neib_pos))
			continue;

		const float3 relPos = pos - make_float3(neib_pos);

		// Check if the squared distance is smaller than the squared influence radius
		// used for neighbor list construction
		bool close_enough = isCloseEnough(relPos, neibInfo, params);

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


/// Builds particles neighbors list
/*! This kernel computes the neighbor's indexes of all particles.
 *
 *	\pparam boundarytype : the boundary type (determines which particles have a neib list)
 *	\pparam periodicbound : use periodic boundaries (0 ... 7)
 *	\pparam neibcount : compute maximum neighbor number (0, 1)
 *
 * First and last particle index for grid cells and particle's informations
 * are read through texture fetches.
 */
template<BoundaryType boundarytype, Periodicity periodicbound, bool neibcount>
__global__ void
__launch_bounds__( BLOCK_SIZE_BUILDNEIBS, MIN_BLOCKS_BUILDNEIBS)
buildNeibsListDevice(buildneibs_params<boundarytype == SA_BOUNDARY> params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	uint neibs_num = 0;		// Number of neighbors for the current particle

	// rather than nesting if's, use a do { } while (0) loop with breaks for early bailouts
	do {
		if (index >= params.numParticles)
			break;

		// Read particle info from texture
		const particleinfo info = tex1Dfetch(infoTex, index);

		// the neighbor list is only constructed for fluid, testpoint, and object particles.
		// if we use SA_BOUNDARY, also for vertex and boundary particles
		bool build_nl = FLUID(info) || TESTPOINTS(info) || OBJECT(info);
		if (boundarytype == DYN_BOUNDARY)
			build_nl = build_nl || BOUNDARY(info);
		if (boundarytype == SA_BOUNDARY)
			build_nl = build_nl || VERTEX(info) || BOUNDARY(info);
		if (!build_nl)
			break; // nothing to do for other particles

		// Get particle position
		#if (__COMPUTE__ >= 20)
		const float4 pos = params.posArray[index];
		#else
		const float4 pos = tex1Dfetch(posTex, index);
		#endif

		if (INACTIVE(pos))
			break; // no NL for inactive particles

		const float3 pos3 = make_float3(pos);

		// Get particle grid position computed from particle hash
		const int3 gridPos = calcGridPosFromParticleHash(params.particleHash[index]);

		for(int z=-1; z<=1; z++) {
			for(int y=-1; y<=1; y++) {
				for(int x=-1; x<=1; x++) {
					neibsInCell<boundarytype == SA_BOUNDARY, periodicbound>(params,
						gridPos,
						make_int3(x, y, z),
						(x + 1) + (y + 1)*3 + (z + 1)*9,
						index,
						pos3,
						neibs_num,
						BOUNDARY(info));
				}
			}
		}
	} while (0);

	// Setting the end marker. Must be done here so that
	// particles for which the neighbor list is not built actually
	// have an empty neib list. Otherwise, particles which are
	// marked inactive will keep their old neiblist.
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
}
#endif
