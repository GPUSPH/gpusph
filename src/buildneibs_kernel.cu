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
// Is not necessary to build a neib list for a repulsive part !
// Pass partinfo to bluidneibs to do that
// We can also plan to have separate arrays for boundary parts
// one for the fixed boundary that is sorted only one time in the simulation
// an other one for moving boundary that will be sort with fluid particle
// and a last one for fluid particles. In this way we will compute interactions
// only on fluid particles.

#ifndef _BUILDNEIBS_KERNEL_
#define _BUILDNEIBS_KERNEL_

#include "particledefine.h"
#include "textures.cuh"

__device__ int d_numInteractions;
__device__ int d_maxNeibs;
__constant__ float3 d_dispvect1;

// calculate position in uniform grid
__device__ __forceinline__ int3
calcGridPos(float3	pos,
			float3	worldOrigin,
			float3	cellSize)
{
	int3 gridPos;
	gridPos.x = floor((pos.x - worldOrigin.x) / cellSize.x);
	gridPos.y = floor((pos.y - worldOrigin.y) / cellSize.y);
	gridPos.z = floor((pos.z - worldOrigin.z) / cellSize.z);

	return gridPos;
}


// calculate address in grid from position (clamping to edges)
__device__ __forceinline__ uint
calcGridHash(int3	gridPos,
			 uint3	gridSize)
{
	gridPos.x = max(0, min(gridPos.x, gridSize.x-1));
	gridPos.y = max(0, min(gridPos.y, gridSize.y-1));
	gridPos.z = max(0, min(gridPos.z, gridSize.z-1));
	return __mul24(__mul24(gridPos.z, gridSize.y), gridSize.x) + __mul24(gridPos.y, gridSize.x) + gridPos.x;
}


// calculate grid hash value for each particle
__global__ void
calcHashDevice(float4*	pos,
			   uint*	particleHash,
			   uint*	particleIndex,
			   uint3	gridSize,
			   float3	cellSize,
			   float3	worldOrigin,
			   uint		numParticles)
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	float4 p = pos[index];

	// get address in grid
	int3 gridPos = calcGridPos(make_float3(p.x, p.y, p.z), worldOrigin, cellSize);
	uint gridHash = calcGridHash(gridPos, gridSize);

	// store grid hash and particle index
	particleHash[index] = gridHash;
	particleIndex[index] = index;
}


__global__
void reorderDataAndFindCellStartDevice( uint*			cellStart,		// output: cell start index
										uint*			cellEnd,		// output: cell end index
										float4*			sortedPos,		// output: sorted positions
										float4*			sortedVel,		// output: sorted velocities
										particleinfo*	sortedInfo,		// output: sorted info
										uint*			particleHash,	// input: sorted grid hashes
										uint*			particleIndex,	// input: sorted particle indices
										uint			numParticles)
{
	extern __shared__ uint sharedHash[];	// blockSize + 1 elements

	uint index = __umul24(blockIdx.x,blockDim.x) + threadIdx.x;

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

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedInfo[index] = info;
		}
}


template <bool periodicbound>
__device__ __forceinline__ void
neibsInCell(int3	gridPos,
			uint	index,
			float3	pos,
			uint3	gridSize,
			uint	numParticles,
			float	influenceradius,
			uint	*neibsList,
			uint	*neibs_num)
{
	int3 periodic = make_int3(0);
	if (periodicbound) {
		if (gridPos.x < 0) {
			if (d_dispvect1.x) {
				gridPos.x = gridSize.x;
				periodic.x = 1;
			} else
				return;
		} else if (gridPos.x >= gridSize.x) {
			if (d_dispvect1.x) {
				gridPos.x = 0;
				periodic.x = -1;
			} else
				return;
		}
		if (gridPos.y < 0) {
			if (d_dispvect1.y) {
				gridPos.y = gridSize.y;
				periodic.y = 1;
			} else
				return;
		} else if (gridPos.y >= gridSize.y) {
			if (d_dispvect1.y) {
				gridPos.y = 0;
				periodic.y = -1;
			} else
				return;
		}
		if (gridPos.z < 0) {
			if (d_dispvect1.z) {
				gridPos.z = gridSize.z;
				periodic.z = 1;
			} else
				return;
		} else if (gridPos.z >= gridSize.z) {
			if (d_dispvect1.z) {
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
	uint gridHash = calcGridHash(gridPos, gridSize);

	// get start of bucket for this cell
	uint bucketStart = tex1Dfetch(cellStartTex, gridHash);

	if (bucketStart == 0xffffffff)
		return;   // cell empty

	// iterate over particles in this cell
	uint bucketEnd = tex1Dfetch(cellEndTex, gridHash);
	for(uint neib_index = bucketStart; neib_index < bucketEnd; neib_index++) {

		//Testpoints ( Testpoints are not considered in neighboring list of other particles since they are imaginary particles)
    	particleinfo info = tex1Dfetch(infoTex, neib_index);
        if (!TESTPOINTS (info)) {
			if (neib_index != index) {			  // check not interacting with self
				float3 neibPos = make_float3(tex1Dfetch(posTex, neib_index));
				float3 relPos = pos - neibPos;

				if (periodicbound)
					relPos += periodic*d_dispvect1;

				uint mod_index = neib_index;
				if (length(relPos) < influenceradius) {
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

					if (*neibs_num < MAXNEIBSNUM)
						neibsList[MAXNEIBSNUM*index + *neibs_num] = mod_index;
					(*neibs_num)++;
				}

			}
		} //If  not Testpoints
	}

	return;
}


template<bool periodicbound>
__global__ void
buildNeibsListDevice(   uint*	neibsList,
						uint3	gridSize,
						float3	cellSize,
						float3	worldOrigin,
						uint	numParticles,
						float	influenceradius)
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;
	int tid = threadIdx.x;

	// total number of neibs for this particle
	__shared__ uint sm_neibs_num[BLOCK_SIZE_BUILDNEIBS];

	sm_neibs_num[tid] = 0;

	if (index < numParticles) {
		// read particle info from texture
    	particleinfo info = tex1Dfetch(infoTex, index);

		// Only fluid particle needs to have a boundary list
		// TODO: this is not true with dynamic boundary particles
		// so change that when implementing dynamics boundary parts

		// Neibouring list is calculated for testpoints and object points)
		if (FLUID(info) || TESTPOINTS (info) || OBJECT(info)) {
			
			// read particle position from texture
			float3 pos = make_float3(tex1Dfetch(posTex, index));

			// get address in grid
			int3 gridPos = calcGridPos(pos, worldOrigin, cellSize);

			// examine only neighbouring cells
			#pragma unroll
			for(int z=-1; z<=1; z++) {
				#pragma unroll
				for(int y=-1; y<=1; y++) {
					#pragma unroll
					for(int x=-1; x<=1; x++)
					neibsInCell<periodicbound>(gridPos + make_int3(x, y, z), index, pos,
							gridSize, numParticles, influenceradius, neibsList, &sm_neibs_num[tid]);
				}
			}
		}
	}

	// Shared memory reduction of per block maximum number of neighbours
	__syncthreads();
	if (tid == 0) {
	  	uint num_interactions = 0;
	  	uint max = 0;
	  	for(int i=0; i< BLOCK_SIZE_BUILDNEIBS; i++) {
	  		num_interactions += sm_neibs_num[i];
	  		max = (max > sm_neibs_num[i]) ? max : sm_neibs_num[i];
	  	}

		atomicAdd(&d_numInteractions, (uint) num_interactions);
		atomicMax(&d_maxNeibs, (uint) max);
	}
	return;
}
#endif
