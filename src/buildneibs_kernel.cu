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


// calculate grid hash value for each particle
__global__ void
__launch_bounds__(BLOCK_SIZE_CALCHASH, MIN_BLOCKS_CALCHASH)
calcHashDevice(const float4*	posArray,
			   uint*			particleHash,
			   uint*			particleIndex,
			   const uint3		gridSize,
			   const float3		cellSize,
			   const float3		worldOrigin,
			   const uint		numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	const float4 pos = posArray[index];

	// get address in grid
	const int3 gridPos = calcGridPos(make_float3(pos), worldOrigin, cellSize);
	const uint gridHash = calcGridHash(gridPos, gridSize);

	// store grid hash and particle index
	particleHash[index] = gridHash;
	particleIndex[index] = index;
}

__global__
__launch_bounds__(BLOCK_SIZE_REORDERDATA, MIN_BLOCKS_REORDERDATA)
void inverseParticleIndexDevice (	uint*	particleIndex,
					uint*	inversedParticleIndex,
					uint	numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	
	if (index < numParticles) {
		int oldindex = particleIndex[index];
		inversedParticleIndex[oldindex] = index;
	}
}
__global__
__launch_bounds__(BLOCK_SIZE_REORDERDATA, MIN_BLOCKS_REORDERDATA)
void reorderDataAndFindCellStartDevice( uint*			cellStart,		// output: cell start index
										uint*			cellEnd,		// output: cell end index
										float4*			sortedPos,		// output: sorted positions
										float4*			sortedVel,		// output: sorted velocities
										particleinfo*		sortedInfo,		// output: sorted info
										float4*			sortedBoundElements,	// output: sorted boundary elements
										float4*			sortedGradGamma,	// output: sorted gradient gamma
										vertexinfo*		sortedVertices,		// output: sorted vertices
										uint*			particleHash,	// input: sorted grid hashes
										uint*			particleIndex,	// input: sorted particle indices
										uint			numParticles,
										uint*			inversedParticleIndex)
{
	extern __shared__ uint sharedHash[];	// blockSize + 1 elements

	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	uint hash;
	// handle case when no. of particles not multiple of block size
	if (index < numParticles) {
		hash = particleHash[index];

		// Load hash data into shared memory so that we can look
		// at neighboring particle's hash value without loading
		// two hash values per thread
		sharedHash[threadIdx.x + 1] = hash;

		if (index > 0 && threadIdx.x == 0) {
			// first thread in block must load neighbor particle hash
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

		// Now use the sorted index to reorder the pos and vel data
		uint sortedIndex = particleIndex[index];
		float4 pos = tex1Dfetch(posTex, sortedIndex);	   // macro does either global read or texture fetch
		float4 vel = tex1Dfetch(velTex, sortedIndex);	   // see particles_kernel.cuh
		particleinfo info = tex1Dfetch(infoTex, sortedIndex);
		float4 boundelement = tex1Dfetch(boundTex, sortedIndex);
		float4 gradgamma = tex1Dfetch(gamTex, sortedIndex);
		vertexinfo vertices = tex1Dfetch(vertTex, sortedIndex);

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedInfo[index] = info;
		sortedBoundElements[index] = boundelement;
		sortedGradGamma[index] = gradgamma;
		
		sortedVertices[index].x = inversedParticleIndex[vertices.x];
		sortedVertices[index].y = inversedParticleIndex[vertices.y];
		sortedVertices[index].z = inversedParticleIndex[vertices.z];
	}
}


template <bool periodicbound>
__device__ __forceinline__ void
neibsInCell(
			#if (__COMPUTE__ >= 20)			
			const float4*	posArray,
			#endif
			int3			gridPos,
			const uint		index,
			const float3	pos,
			const uint3		gridSize,
			const uint		numParticles,
			const float		sqinfluenceradius,
			uint*			neibsList,
			uint&			neibs_num,
			const uint		lane,
			const uint		offset)
{
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

	// get hash value of grid position
	const uint gridHash = calcGridHash(gridPos, gridSize);

	// get start of bucket for this cell
	const uint bucketStart = tex1Dfetch(cellStartTex, gridHash);

	if (bucketStart == 0xffffffff)
		return;   // cell empty

	// iterate over particles in this cell
	const uint bucketEnd = tex1Dfetch(cellEndTex, gridHash);
	for(uint neib_index = bucketStart; neib_index < bucketEnd; neib_index++) {

		//Testpoints ( Testpoints are not considered in neighboring list of other particles since they are imaginary particles)
    	const particleinfo info = tex1Dfetch(infoTex, neib_index);
        if (!TESTPOINTS (info)) {
			if (neib_index != index) {			  // check not interacting with self
				#if (__COMPUTE__ >= 20)			
				float3 relPos = pos - make_float3(posArray[neib_index]);
				#else
				float3 relPos = pos - make_float3(tex1Dfetch(posTex, neib_index));
				#endif
				if (periodicbound)
					relPos += periodic*d_dispvect;

				uint mod_index = neib_index;
				if (sqlength(relPos) < sqinfluenceradius) {
					if (periodicbound) {
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

					if (neibs_num < d_maxneibsnum)
						neibsList[d_maxneibsnum_time_neibindexinterleave*lane + neibs_num*NEIBINDEX_INTERLEAVE + offset] = mod_index;
					neibs_num++;
				}

			}
		} //If  not Testpoints
	}

	return;
}


template<bool periodicbound, bool neibcount>
__global__ void
__launch_bounds__( BLOCK_SIZE_BUILDNEIBS, MIN_BLOCKS_BUILDNEIBS)
buildNeibsListDevice(   
						#if (__COMPUTE__ >= 20)			
						const float4*	posArray,
						#endif
						uint*			neibsList,
						const uint3		gridSize,
						const float3	cellSize,
						const float3	worldOrigin,
						const uint		numParticles,
						const float		sqinfluenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint tid = threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = tid & (NEIBINDEX_INTERLEAVE - 1);

	uint neibs_num = 0;

	if (index < numParticles) {
		// read particle info from texture
    	const particleinfo info = tex1Dfetch(infoTex, index);

		// Only fluid particle needs to have a boundary list
		// TODO: this is not true with dynamic boundary particles
		// so change that when implementing dynamics boundary parts
		// This is also not true for "Ferrand et al." boundary model,
		// where vertex particles also need to have a list of neighbours

		// Neighboring list is calculated for testpoints and object points)
		if (FLUID(info) || TESTPOINTS (info) || OBJECT(info)/*TODO: || VERTEX(info) || BOUNDARY(info)*/) {
			// read particle position from global memory or texture according to architecture
			#if (__COMPUTE__ >= 20)
			const float3 pos = make_float3(posArray[index]);
			#else
			const float3 pos = make_float3(tex1Dfetch(posTex, index));
			#endif

			// get address in grid
			const int3 gridPos = calcGridPos(pos, worldOrigin, cellSize);

			// examine only neighboring cells
			for(int z=-1; z<=1; z++) {
				for(int y=-1; y<=1; y++) {
					for(int x=-1; x<=1; x++)
						neibsInCell<periodicbound>(
							#if (__COMPUTE__ >= 20)
							posArray, 
							#endif
							gridPos + make_int3(x, y, z), index, pos, gridSize, numParticles, 
							sqinfluenceradius, neibsList, neibs_num, lane, offset);
				}
			}
		}
		
		if (neibs_num < d_maxneibsnum)
			neibsList[d_maxneibsnum_time_neibindexinterleave*lane + neibs_num*NEIBINDEX_INTERLEAVE + offset] = 0xffffffff;
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
