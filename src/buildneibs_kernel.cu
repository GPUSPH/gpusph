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

namespace cuneibs {
__constant__ uint d_maxneibsnum;
__constant__ uint d_maxneibsnum_time_neibindexinterleave;
__device__ int d_numInteractions;
__device__ int d_maxNeibs;
__constant__ float3 d_dispvect;

// calculate position in uniform grid
__device__ __forceinline__ int3
calcGridPos(float3			pos,
			const float3	worldOrigin,
			const float3	cellSize)
{
	int3 gridPos;
	gridPos.x = floor((pos.x - worldOrigin.x) / cellSize.x);
	gridPos.y = floor((pos.y - worldOrigin.y) / cellSize.y);
	gridPos.z = floor((pos.z - worldOrigin.z) / cellSize.z);

	return gridPos;
}


// calculate address in grid from position (clamping to edges)
__device__ __forceinline__ uint
calcGridHash(int3			gridPos,
			 const uint3	gridSize)
{
	gridPos.x = max(0, min(gridPos.x, gridSize.x-1));
	gridPos.y = max(0, min(gridPos.y, gridSize.y-1));
	gridPos.z = max(0, min(gridPos.z, gridSize.z-1));
	return INTMUL(INTMUL(gridPos.z, gridSize.y), gridSize.x) + INTMUL(gridPos.y, gridSize.x) + gridPos.x;
}


/// Compute grid position from hash value
/*! Compute the grid position corresponding to the given hash. The position
 * 	should be in the range [0, gridSize.x - 1]x[0, gridSize.y - 1]x[0, gridSize.z - 1].
 *
 *	\param[in] gridHash : hash value
 *	\param[in] gridSize : grid size
 *
 *	\return grid position
 *
 *	Beware : no test is done by this function to ensure that hash value is valid.
 */
__device__ __forceinline__ int3
calcGridPosFromHash(const uint gridHash, const uint3 gridSize)
{
	int3 gridPos;
	int temp = INTMUL(gridSize.y, gridSize.x);
	gridPos.z = gridHash/temp;
	temp = gridHash - gridPos.z*temp;
	gridPos.y = temp/gridSize.x;
	gridPos.x = temp - gridPos.y*gridSize.x;

	return gridPos;
}


/// Clamp grid position to edges
/*! Clamp grid position to [0, gridSize.x - 1]x[0, gridSize.y - 1]x[0, gridSize.z - 1].
 *
 *	\param[in,out] gridPos : gridPos to be clamped
 *	\param[in] gridSize : grid size
 */
__device__ __forceinline__ void
clampGridPos(int3& gridPos, const uint3 gridSize)
{
	gridPos.x = max(0, min(gridPos.x, gridSize.x-1));
	gridPos.y = max(0, min(gridPos.y, gridSize.y-1));
	gridPos.z = max(0, min(gridPos.z, gridSize.z-1));
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
 *	\param[in] gridSize : grid size
 *	\param[in] cellSize : cell size
 *	\param[in] numParticles : total number of particles
 */
__global__ void
__launch_bounds__(BLOCK_SIZE_CALCHASH, MIN_BLOCKS_CALCHASH)
calcHashDevice(float4*			posArray,		///< particle's positions (in, out)
			   uint*			particleHash,	///< particle's hashes (in, out)
			   uint*			particleIndex,	///< particle's indexes (out)
			   particleinfo*	particelInfo,	///< particle's informations (in)
			   const uint3		gridSize,		///< grid size (in)
			   const float3		cellSize,		///< cell size (in)
			   const uint		numParticles)	///< total number of particles (in)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// Getting new pos relative to old cell
	float4 pos = posArray[index];
	const particleinfo info = particelInfo[index];

	// We compute new hash only for fluid or object particles
	if (FLUID(info) || OBJECT(info)) {
		// Getting the old grid hash
		uint gridHash = particleHash[index];

		// Getting grid address of old cell (computed from old hash)
		const int3 gridPos = calcGridPosFromHash(gridHash, gridSize);

		// Computing grid offset from new pos relative to old hash
		int3 gridOffset = make_int3(make_float3(pos)/cellSize);

		// Compute new grid pos relative to cell and new cell hash
		int3 newGridPos = gridPos + gridOffset;
		clampGridPos(newGridPos, gridSize);
		gridOffset = newGridPos - gridPos;

		pos.x -= (float) gridOffset.x*cellSize.x;
		pos.y -= (float) gridOffset.y*cellSize.y;
		pos.z -= (float) gridOffset.z*cellSize.z;

		// Compute new hash
		gridHash = calcGridHash(newGridPos, gridSize);

		// Store grid hash, particle index and position relative to cell
		particleHash[index] = gridHash;
		posArray[index] = pos;
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
										float4*			sortedPos,		///< new sorted particle's positions (out)
										float4*			sortedVel,		///< new sorted particle's velocities (out)
										particleinfo*	sortedInfo,		///< new sorted particle's informations (out)
										uint*			particleHash,	///< previously sorted particle's hashes (in)
										uint*			particleIndex,	///< previously sorted particle's hashes (in)
										uint			numParticles)
{
	// Shared hash array of dimension blockSize + 1
	extern __shared__ uint sharedHash[];

	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	uint hash;
	// Handle the case when number of particles is not multiple of block size
	if (index < numParticles) {
		hash = particleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0) {
			// First thread in block must load neighbor particle hash
			sharedHash[0] = particleHash[index-1];
			}
	}

	__syncthreads();

	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell,
		// so store the index of this particle in the cell.
		// As it isn't the first particle, it must also be the cell end of
		// the previous particle's cell

		if (index == 0 || hash != sharedHash[threadIdx.x]) {
			cellStart[hash] = index;
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
			}

		if (index == numParticles - 1) {
			cellEnd[hash] = index + 1;
			}

		// Now use the sorted index to reorder particle's data
		const uint sortedIndex = particleIndex[index];
		const float4 pos = tex1Dfetch(posTex, sortedIndex);
		const float4 vel = tex1Dfetch(velTex, sortedIndex);
		const particleinfo info = tex1Dfetch(infoTex, sortedIndex);

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedInfo[index] = info;
		}
}


/// Find neighbors in a given cell
/*! This function look for neighbors of the current particle in
 * a given cell
 *
 *	\param[in] posArray : particle's positions
 *	\param[in] gridPos : current particle grid position
 *	\param[in] gridOffset : cell offset from current particle cell
 *	\param[in] cell : cell number
 *	\param[in] index : index of the current particle
 *	\param[in] pos : position of the current particle
 *	\param[in] gridSize : grid size
 *	\param[in] cellSize : cell size
 *	\param[in] numParticles : total number of particles
 *	\param[in] sqinfluenceradius : squared value of the influence radius
 *	\param[out] neibList : neighbor's list
 *	\param[in, out] neibs_num : current number of neighbors found for current particle
 *	\param[in] lane : lane for write interleaving
 *	\param[in] offset : pffset for write interleaving
 *
 *	\pparam periodicbound : use periodic boundaries (0, 1)
 *
 * First and last particle index for grid cells and particle's informations
 * are read trough texture fetches.
 */
template <bool periodicbound>
__device__ __forceinline__ void
neibsInCell(
			#if (__COMPUTE__ >= 20)			
			const float4*	posArray,
			#endif
			int3			gridPos,
			const int3		gridOffset,
			const uchar		cell,
			const uint		index,
			const float3	pos,
			const uint3		gridSize,
			const float3	cellSize,
			const uint		numParticles,
			const float		sqinfluenceradius,
			neibdata*		neibsList,
			uint&			neibs_num,
			const uint		lane,
			const uint		offset)
{
	// Compute the grid position of the current cell
	gridPos += gridOffset;

	// Deal with periodic boundaries
	// TODO: fix
	int3 periodic = make_int3(0);
	if (periodicbound) {
		if (gridPos.x < 0) {
			if (d_dispvect.x) {
				gridPos.x = gridSize.x;
				periodic.x = 1;
			} else
				return;
		} else if (gridPos.x >= gridSize.x) {
			if (d_dispvect.x) {
				gridPos.x = 0;
				periodic.x = -1;
			} else
				return;
		}
		if (gridPos.y < 0) {
			if (d_dispvect.y) {
				gridPos.y = gridSize.y;
				periodic.y = 1;
			} else
				return;
		} else if (gridPos.y >= gridSize.y) {
			if (d_dispvect.y) {
				gridPos.y = 0;
				periodic.y = -1;
			} else
				return;
		}
		if (gridPos.z < 0) {
			if (d_dispvect.z) {
				gridPos.z = gridSize.z;
				periodic.z = 1;
			} else
				return;
		} else if (gridPos.z >= gridSize.z) {
			if (d_dispvect.z) {
				gridPos.z = 0;
				periodic.z = -1;
			} else
				return;
		}
	} else {
		if ((gridPos.x < 0) || (gridPos.x >= gridSize.x) ||
			(gridPos.y < 0) || (gridPos.y >= gridSize.y) ||
			(gridPos.z < 0) || (gridPos.z >= gridSize.z))
				return;
	}

	// Get hash value from grid position
	const uint gridHash = calcGridHash(gridPos, gridSize);

	// Get the first particle index of the cell
	const uint bucketStart = tex1Dfetch(cellStartTex, gridHash);

	// Return if the cell is empty
	if (bucketStart == 0xffffffff)
		return;

	// Get the last particle index of the cell
	const uint bucketEnd = tex1Dfetch(cellEndTex, gridHash);
	// Iterate over all particles in the cell
	bool encode_cell = true;
	for(uint neib_index = bucketStart; neib_index < bucketEnd; neib_index++) {

		// Testpoints are not considered in neighboring list of other particles since they are imaginary particles.
    	const particleinfo info = tex1Dfetch(infoTex, neib_index);
        if (!TESTPOINTS (info)) {
        	// Check for self interaction
			if (neib_index != index) {
				// Compute relative position between particle and potential neighbor
				#if (__COMPUTE__ >= 20)			
				float3 relPos = pos - make_float3(posArray[neib_index]);
				#else
				float3 relPos = pos - make_float3(tex1Dfetch(posTex, neib_index));
				#endif
				relPos -= gridOffset*cellSize;

				// Deal with periodic boundaries
				// TODO: fix
				if (periodicbound)
					relPos += periodic*d_dispvect;

				// Check if the squared distance is smaller than the squared influence radius
				// used for neighbor list construction
				if (sqlength(relPos) < sqinfluenceradius) {

					// Deal with periodic boundaries
					// TODO: fix
					if (periodicbound) {
						uint mod_index = neib_index;
						if (periodic.x == 1)
							mod_index |= WARPXPLUS;
						else if (periodic.x == -1)
							mod_index |= WARPXMINUS;
						if (periodic.y == 1)
							mod_index |= WARPYPLUS;
						else if (periodic.y == -1)
							mod_index |= WARPYMINUS;
						if (periodic.z == 1)
							mod_index |= WARPZPLUS;
						else if (periodic.z == -1)
							mod_index |= WARPZMINUS;
					}

					if (neibs_num < d_maxneibsnum) {
						neibsList[d_maxneibsnum_time_neibindexinterleave*lane + neibs_num*NEIBINDEX_INTERLEAVE + offset] =
								neib_index - bucketStart + ((encode_cell) ? ((cell + 1) << 11) : 0);
						encode_cell = false;
					}
					neibs_num++;
				}

			}
		} //If  not Testpoints
	}

	return;
}


/// Builds particles neighbors list
/*! This kernel computes the neighbor's indexes of all particles.
 * In order to have best performance across different compute capabilities
 * particle's positions are read from global memory for compute capability
 * greather or equal to 2.0 and from texture otherwise.
 *
 *	\param[in] posArray : particle's positions
 *	\param[in] particleHash : particle's hashes
 *	\param[out] neibList : neighbor's list
 *	\param[in] gridSize : grid size
 *	\param[in] cellSize : cell size
 *	\param[in] numParticles : total number of particles
 *	\param[in] sqinfluenceradius : squared value of the influence radius
 *
 *	\pparam periodicbound : use periodic boundaries (0, 1)
 *	\pparam neibcount : compute maximum neighbor number (0, 1)
 *
 * First and last particle index for grid cells and particle's informations
 * are read trough texture fetches.
 */
template<bool periodicbound, bool neibcount>
__global__ void
__launch_bounds__( BLOCK_SIZE_BUILDNEIBS, MIN_BLOCKS_BUILDNEIBS)
buildNeibsListDevice(   
						#if (__COMPUTE__ >= 20)			
						const float4*	posArray,				///< particle's positions (in)
						#endif
						const uint*		particleHash,			///< particle's hashes (in)
						neibdata*		neibsList,				///< neighbor's list (out)
						const uint3		gridSize,				///< grid size (in)
						const float3	cellSize,				///< cell size (in)
						const uint		numParticles,			///< total number of particles (in)
						const float		sqinfluenceradius)		///< squared influence radius (in)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint tid = threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = tid & (NEIBINDEX_INTERLEAVE - 1);

	uint neibs_num = 0;		// Number of neighbors for the current particle

	if (index < numParticles) {
		// Read particle info from texture
    	const particleinfo info = tex1Dfetch(infoTex, index);

		// Only fluid particle needs to have a boundary list
		// TODO: this is not true with dynamic boundary particles
		// so change that when implementing dynamics boundary parts

		// Neighbor list is build for fluid, test and object particle's
		if (FLUID(info) || TESTPOINTS (info) || OBJECT(info)) {
			// Get particle position
			#if (__COMPUTE__ >= 20)
			const float3 pos = make_float3(posArray[index]);
			#else
			const float3 pos = make_float3(tex1Dfetch(posTex, index));
			#endif

			// Get particle grid position computed from particle hash
			const int3 gridPos = calcGridPosFromHash(particleHash[index], gridSize);

			// Look trough the 26 neighboring cells and the current particle cell
			for(int z=-1; z<=1; z++) {
				for(int y=-1; y<=1; y++) {
					for(int x=-1; x<=1; x++) {
						neibsInCell<periodicbound>(
							#if (__COMPUTE__ >= 20)
							posArray, 
							#endif
							gridPos, make_int3(x, y, z), (x + 1) + (y + 1)*3 + (z + 1)*9, index, pos, gridSize, cellSize,
							numParticles, sqinfluenceradius, neibsList, neibs_num, lane, offset);
					}
				}
			}
		}
		
		// Setting the end marker
		if (neibs_num < d_maxneibsnum)
			neibsList[d_maxneibsnum_time_neibindexinterleave*lane + neibs_num*NEIBINDEX_INTERLEAVE + offset] = 0xffff;
	}
	
	if (neibcount) {
		// Shared memory reduction of per block maximum number of neighbors
		__shared__ volatile uint sm_neibs_num[BLOCK_SIZE_BUILDNEIBS];
		__shared__ volatile uint sm_neibs_max[BLOCK_SIZE_BUILDNEIBS];

		sm_neibs_num[tid] = neibs_num;	
		sm_neibs_max[tid] = neibs_num;
		__syncthreads();

		uint i = blockDim.x/2;
		while (i != 0) {
			if (tid < i) {
				sm_neibs_num[tid] += sm_neibs_num[tid + i];
				const float n1 = sm_neibs_max[tid];
				const float n2 = sm_neibs_max[tid + i];
				if (n2 > n1)
					sm_neibs_max[tid] = n2;
			}
			__syncthreads();
			i /= 2;
		}

		if (!tid) {
			atomicAdd(&d_numInteractions, sm_neibs_num[0]);
			atomicMax(&d_maxNeibs, sm_neibs_max[0]);
		}
	}
	return;
}
}
#endif
