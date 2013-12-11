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

namespace cuneibs {
__constant__ uint d_maxneibsnum;
__device__ int d_numInteractions;
__device__ int d_maxNeibs;

#include "cellgrid.h"

/// Clamp grid position to edges according to periodicity
/*! This function clamp grid position to edges according to the chosen
 * periodicity, returns the new grid position and update the grid offset.
 *
 *	\param[in] gridPos : grid position to be clamped
 *	\param[in] gridOffset : grid offset
 *
 * 	\pparam periodicbound : use periodic boundaries (0 ... 7)
 *
 * 	\return : new grid position
 */
// TODO: verify periodicity along multiple axis
template <int periodicbound>
__device__ __forceinline__ int3
clampGridPos(const int3& gridPos, int3& gridOffset)
{
	int3 newGridPos = gridPos + gridOffset;
	// For the axis involved in periodicity the new grid position reflects
	// the periodicity and should not be clamped and the grid offset remains
	// unchanged.
	// For the axis not involved in periodicity the new grid position
	// is equal to the clamped old one and the grid offset is updated.

	// periodicity in x
	if (periodicbound & XPERIODIC) {
		if (newGridPos.x < 0) newGridPos.x += d_gridSize.x;
		if (newGridPos.x >= d_gridSize.x) newGridPos.x -= d_gridSize.x;
	} else {
		newGridPos.x = max(0, min(gridPos.x, d_gridSize.x-1));
		gridOffset.x = newGridPos.x - gridPos.x;
	}

	// periodicity in y
	if (periodicbound & YPERIODIC) {
		if (newGridPos.y < 0) newGridPos.y += d_gridSize.y;
		if (newGridPos.y >= d_gridSize.y) newGridPos.y -= d_gridSize.y;
	} else {
		newGridPos.y = max(0, min(gridPos.y, d_gridSize.y-1));
		gridOffset.y = newGridPos.y - gridPos.y;
	}

	// periodicity in z
	if (periodicbound & ZPERIODIC) {
		if (newGridPos.z < 0) newGridPos.z += d_gridSize.z;
		if (newGridPos.z >= d_gridSize.z) newGridPos.z -= d_gridSize.z;
	} else {
		newGridPos.z = max(0, min(gridPos.z, d_gridSize.z-1));
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
 *
 * 	\return : new grid position
 */
template <>
__device__ __forceinline__ int3
clampGridPos<0>(const int3& gridPos, int3& gridOffset)
{
	int3 newGridPos = gridPos + gridOffset;

	// Without periodicity the new grid position is clamped to edges
	newGridPos.x = max(0, min(newGridPos.x, d_gridSize.x-1));
	newGridPos.y = max(0, min(newGridPos.y, d_gridSize.y-1));
	newGridPos.z = max(0, min(newGridPos.z, d_gridSize.z-1));

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
template <int periodicbound>
__global__ void
__launch_bounds__(BLOCK_SIZE_CALCHASH, MIN_BLOCKS_CALCHASH)
calcHashDevice(float4*			posArray,		///< particle's positions (in, out)
			   hashKey*			particleHash,	///< particle's hashes (in, out)
			   uint*			particleIndex,	///< particle's indexes (out)
			   const particleinfo*	particelInfo,	///< particle's informations (in)
			   const uint		numParticles)	///< total number of particles
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// Getting new pos relative to old cell
	float4 pos = posArray[index];
	const particleinfo info = particelInfo[index];

	// We compute new hash only for fluid and moving not fluid particles (object, moving boundaries)
	if (FLUID(info) || (type(info) & MOVINGNOTFLUID)) {
	//if (true) {
		// Getting the old grid hash
		hashKey gridHash = particleHash[index];

		// Getting grid address of old cell (computed from old hash)
		const int3 gridPos = calcGridPosFromHash(gridHash);

		// Computing grid offset from new pos relative to old hash
		int3 gridOffset = make_int3(floor((as_float3(pos) + 0.5f*d_cellSize)/d_cellSize));

		// Compute new grid pos relative to cell, adjust grid offset and compute new cell hash
		gridHash = calcGridHash(clampGridPos<periodicbound>(gridPos, gridOffset));
#if HASH_KEY_SIZE >= 64
		gridHash <<= GRIDHASH_BITSHIFT
		gridHash |= id(pinfo[index]);
#endif

		// Adjust position
		as_float3(pos) -= gridOffset*d_cellSize;

		// Store grid hash, particle index and position relative to cell
		particleHash[index] = gridHash;
		posArray[index] = pos;
	}

	// Preparing particle index array for the sort phase
	particleIndex[index] = index;
}
#undef MOVINGNOTFLUID

__global__
__launch_bounds__(BLOCK_SIZE_REORDERDATA, MIN_BLOCKS_REORDERDATA)
void inverseParticleIndexDevice (   uint*   particleIndex,
                    uint*   inversedParticleIndex,
                    uint    numParticles)
{
    const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

    if (index < numParticles) {
        int oldindex = particleIndex[index];
        inversedParticleIndex[oldindex] = index;
    }
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
										float4*			sortedBoundElements,	// output: sorted boundary elements
										float4*			sortedGradGamma,	// output: sorted gradient gamma
										vertexinfo*		sortedVertices,		// output: sorted vertices
										float*			sortedPressure,		// output: sorted pressure
										float*			sortedTKE,			// output: k for k-e model
										float*			sortedEps,			// output: e for k-e model
										float*			sortedTurbVisc,		// output: eddy viscosity
										float*			sortedStrainRate,	// output: strain rate
										const hashKey*	particleHash,	///< previously sorted particle's hashes (in)
										const uint*		particleIndex,	///< previously sorted particle's hashes (in)
										const uint		numParticles,	///< total number of particles
										const uint*		inversedParticleIndex)
{
	// Shared hash array of dimension blockSize + 1
	extern __shared__ uint sharedHash[];

	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	uint hash;
	// Handle the case when number of particles is not multiple of block size
	if (index < numParticles) {
		hash = (uint)(particleHash[index] >> GRIDHASH_BITSHIFT);

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
			sharedHash[0] = (uint)(particleHash[index-1] >> GRIDHASH_BITSHIFT);
			}
	}

	__syncthreads();

	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell
		// or the first inactive particle.
		// Store the index of this particle as the new cell start and as
		// the previous cell end

		if (index == 0 || hash != sharedHash[threadIdx.x]) {
			cellStart[hash] = index;
			// If it isn't the first particle, it must also be the cell end of
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x]] = index;
		}

		if (index == numParticles - 1)
			cellEnd[hash] = index + 1;

		// Now use the sorted index to reorder particle's data
		const uint sortedIndex = particleIndex[index];
		const float4 pos = tex1Dfetch(posTex, sortedIndex);
		const float4 vel = tex1Dfetch(velTex, sortedIndex);
		const particleinfo info = tex1Dfetch(infoTex, sortedIndex);
		const float4 boundelement = tex1Dfetch(boundTex, sortedIndex);
		const float4 gradgamma = tex1Dfetch(gamTex, sortedIndex);
		const vertexinfo vertices = tex1Dfetch(vertTex, sortedIndex);
		const float pressure = tex1Dfetch(presTex, sortedIndex);

		const float keps_k = tex1Dfetch(keps_kTex, sortedIndex);
		const float keps_e = tex1Dfetch(keps_eTex, sortedIndex);
		const float tvisc = tex1Dfetch(tviscTex, sortedIndex);
		const float strainrate = tex1Dfetch(strainTex, sortedIndex);

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedInfo[index] = info;
		sortedBoundElements[index] = boundelement;
		sortedGradGamma[index] = gradgamma;
		sortedPressure[index] = pressure;
		
		sortedVertices[index].x = inversedParticleIndex[vertices.x];
		sortedVertices[index].y = inversedParticleIndex[vertices.y];
		sortedVertices[index].z = inversedParticleIndex[vertices.z];

		sortedTKE[index] = keps_k;
		sortedEps[index] = keps_e;
		sortedTurbVisc[index] = tvisc;
		sortedStrainRate[index] = strainrate;
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
 *	\param[in] numParticles : total number of particles
 *	\param[in] sqinfluenceradius : squared value of the influence radius
 *	\param[out] neibList : neighbor's list
 *	\param[in, out] neibs_num : current number of neighbors found for current particle
 *
 *	\pparam periodicbound : use periodic boundaries (0 ... 7)
 *
 * First and last particle index for grid cells and particle's informations
 * are read trough texture fetches.
 */
template <int periodicbound>
__device__ __forceinline__ void
neibsInCell(
			#if (__COMPUTE__ >= 20)			
			const float4*	posArray,	///< particle's positions (in)
			#endif
			int3			gridPos,	///< current particle grid position
			const int3		gridOffset,	///< cell offset from current particle grid position
			const uchar		cell,		///< cell number (0 ... 26)
			const uint		index,		///< current particle index
			float3			pos,		///< current particle position
			const uint		numParticles,	///< total number of particles
			const float		sqinfluenceradius,	///< squared value of influence radius
			neibdata*		neibsList,	///< neighbor's list (out)
			uint&			neibs_num)	///< number of neighbors for the current particle
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
			if (periodicbound & XPERIODIC)
				gridPos.x = d_gridSize.x - 1;
			else
				return;
		}
		else if (gridPos.x >= d_gridSize.x) {
			if (periodicbound & XPERIODIC)
				gridPos.x = 0;
			else
				return;
		}

		// Periodicity along y axis
		if (gridPos.y < 0) {
			if (periodicbound & YPERIODIC)
				gridPos.y = d_gridSize.y - 1;
			else
				return;
		}
		else if (gridPos.y >= d_gridSize.y) {
			if (periodicbound & YPERIODIC)
				gridPos.y = 0;
			else
				return;
		}

		// Periodicity along z axis
		if (gridPos.z < 0) {
			if (periodicbound & ZPERIODIC)
				gridPos.z = d_gridSize.z - 1;
			else
				return;
		}
		else if (gridPos.z >= d_gridSize.z) {
			if (periodicbound & ZPERIODIC)
				gridPos.z = 0;
			else
				return;
		}
	}
	// Without periodic boundary when the neighboring cell grid position lies
	// outside the domain size there is nothing to do
	else {
		if ((gridPos.x < 0) || (gridPos.x >= d_gridSize.x) ||
			(gridPos.y < 0) || (gridPos.y >= d_gridSize.y) ||
			(gridPos.z < 0) || (gridPos.z >= d_gridSize.z))
				return;
	}

	// Get hash value from grid position
	const uint gridHash = calcGridHash(gridPos);

	// Get the first particle index of the cell
	const uint bucketStart = tex1Dfetch(cellStartTex, gridHash);

	// Return if the cell is empty
	if (bucketStart == 0xffffffff)
		return;

	// Substract gridOffset*cellsize to pos so we don't need to do it each time
	// we compute relPos respect to potential neighbor
	pos -= gridOffset*d_cellSize;

	// Get the last particle index of the cell
	const uint bucketEnd = tex1Dfetch(cellEndTex, gridHash);
	// Iterate over all particles in the cell
	bool encode_cell = true;
	for(uint neib_index = bucketStart; neib_index < bucketEnd; neib_index++) {

		// Test and probe points are not considered in neighboring list of other particles since they are imaginary particles.
    	const particleinfo info = tex1Dfetch(infoTex, neib_index);
        if (!TESTPOINTS (info) && !PROBE(info)) {
        	// Check for self interaction
			if (neib_index != index) {
				// Compute relative position between particle and potential neighbor
				// NOTE: using as_float3 instead of make_float3 result in a 25% performance loss
				#if (__COMPUTE__ >= 20)			
				const float3 relPos = pos - make_float3(posArray[neib_index]);
				#else
				const float3 relPos = pos - make_float3(tex1Dfetch(posTex, neib_index));
				#endif

				// Check if the squared distance is smaller than the squared influence radius
				// used for neighbor list construction
				if (sqlength(relPos) < sqinfluenceradius) {
					if (neibs_num < d_maxneibsnum) {
						neibsList[neibs_num*numParticles + index] =
								neib_index - bucketStart + ((encode_cell) ? ENCODE_CELL(cell) : 0);
						encode_cell = false;
					}
					neibs_num++;
				}

			}
		} // if not Testpoints
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
 *	\param[in] numParticles : total number of particles
 *	\param[in] sqinfluenceradius : squared value of the influence radius
 *
 *	\pparam periodicbound : use periodic boundaries (0 ... 7)
 *	\pparam neibcount : compute maximum neighbor number (0, 1)
 *
 * First and last particle index for grid cells and particle's informations
 * are read trough texture fetches.
 */
template<int periodicbound, bool neibcount>
__global__ void
__launch_bounds__( BLOCK_SIZE_BUILDNEIBS, MIN_BLOCKS_BUILDNEIBS)
buildNeibsListDevice(   
						#if (__COMPUTE__ >= 20)			
						const float4*	posArray,				///< particle's positions (in)
						#endif
						const hashKey*	particleHash,			///< particle's hashes (in)
						neibdata*		neibsList,				///< neighbor's list (out)
						const uint		numParticles,			///< total number of particles
						const float		sqinfluenceradius)		///< squared influence radius
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	uint neibs_num = 0;		// Number of neighbors for the current particle

	if (index < numParticles) {
		// Read particle info from texture
    	const particleinfo info = tex1Dfetch(infoTex, index);

		// Only fluid particle needs to have a boundary list
		// TODO: this is not true with dynamic boundary particles
		// so change that when implementing dynamics boundary parts
		// This is also not true for "Ferrand et al." boundary model,
		// where vertex particles also need to have a list of neighbours

		// Neighbor list is build for fluid, object, vertex and probe particles
		if (FLUID(info) || TESTPOINTS (info) || OBJECT(info) || VERTEX(info) || PROBE(info)/*TODO: || BOUNDARY(info)*/) {
			// Get particle position
			#if (__COMPUTE__ >= 20)
			const float3 pos = make_float3(posArray[index]);
			#else
			const float3 pos = make_float3(tex1Dfetch(posTex, index));
			#endif

			// Get particle grid position computed from particle hash
			const int3 gridPos = calcGridPosFromHash(particleHash[index]);

			// Look trough the 26 neighboring cells and the current particle cell
			for(int z=-1; z<=1; z++) {
				for(int y=-1; y<=1; y++) {
					for(int x=-1; x<=1; x++) {
						neibsInCell<periodicbound>(
							#if (__COMPUTE__ >= 20)
							posArray, 
							#endif
							gridPos, make_int3(x, y, z), (x + 1) + (y + 1)*3 + (z + 1)*9, index, pos,
							numParticles, sqinfluenceradius, neibsList, neibs_num);
					}
				}
			}
		}
		
		// Setting the end marker
		if (neibs_num < d_maxneibsnum) {
			neibsList[neibs_num*numParticles + index] = 0xffff;
		}
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
