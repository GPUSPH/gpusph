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

__device__ int d_numInteractions;
__device__ int d_maxNeibs;
__constant__ float3 d_dispvect1;
__constant__ int d_cellofsets[27];

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

		sortedPos[index] = pos;
		sortedVel[index] = vel;
		sortedInfo[index] = info;
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
					relPos += periodic*d_dispvect1;

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

					if (neibs_num < MAXNEIBSNUM)
						neibsList[MAXNEIBSNUM*NEIBINDEX_INTERLEAVE*lane + neibs_num*NEIBINDEX_INTERLEAVE + offset] = mod_index;
					neibs_num++;
				}

			}
		} //If  not Testpoints
	}

	return;
}


template<bool periodicbound>
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
	
	// total number of neibs for this particle
	__shared__ volatile uint sm_neibs_num[BLOCK_SIZE_BUILDNEIBS];
	__shared__ volatile uint sm_neibs_max[BLOCK_SIZE_BUILDNEIBS];

	uint neibs_num = 0;

	if (index < numParticles) {
		// read particle info from texture
    	const particleinfo info = tex1Dfetch(infoTex, index);

		// Only fluid particle needs to have a boundary list
		// TODO: this is not true with dynamic boundary particles
		// so change that when implementing dynamics boundary parts

		// Neighboring list is calculated for testpoints and object points)
		if (FLUID(info) || TESTPOINTS (info) || OBJECT(info)) {
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
		
		if (neibs_num < MAXNEIBSNUM)
			neibsList[MAXNEIBSNUM*NEIBINDEX_INTERLEAVE*lane + neibs_num*NEIBINDEX_INTERLEAVE + offset] = 0xffffffff;
	}
	
	// Shared memory reduction of per block maximum number of neighbors
	sm_neibs_num[tid] = neibs_num;	
	sm_neibs_max[tid] = neibs_num;
	__syncthreads();
	
	uint i = blockDim.x/2;
	while (i != 0) {
		if (tid < i) {
			sm_neibs_num[tid] += sm_neibs_num[tid + 1];
	  		sm_neibs_max[tid] = (sm_neibs_max[tid] > sm_neibs_max[tid + 1]) ? sm_neibs_max[tid] : sm_neibs_max[tid + 1];
		}
		__syncthreads();
		i /= 2;
	}
	
	if (tid == 0) {
		atomicAdd(&d_numInteractions, sm_neibs_num[0]);
		atomicMax(&d_maxNeibs, sm_neibs_max[0]);
	}
	
//	if (tid == 0) {
//	  	uint num_interactions = 0;
//	  	uint max = 0;
//	  	for(int i=0; i< BLOCK_SIZE_BUILDNEIBS; i++) {
//	  		num_interactions += sm_neibs_num[i];
//	  		max = (max > sm_neibs_num[i]) ? max : sm_neibs_num[i];
//	  	}
//
//		atomicAdd(&d_numInteractions, num_interactions);
//		atomicMax(&d_maxNeibs, max);
//	}
	return;
}

#define MAX_PARTS_IN_CELL	48
__global__ void
buildNeibsListDevice4(  const float4* posArray,
						uint*	neibsList,
						uint3	gridSize,
						float3	cellSize,
						float3	worldOrigin,
						uint	numParticles,
						uint	numCells,
						float	sqinfluenceradius)
{
	const uint tid = threadIdx.x;
	
	// total number of neibs for this particle
	__shared__ float3 sm_posi[BLOCK_SIZE_BUILDNEIBS4];
	__shared__ ushort sm_flagi[BLOCK_SIZE_BUILDNEIBS4];
	__shared__ ushort sm_neibsnumi[BLOCK_SIZE_BUILDNEIBS4];
	__shared__ uint sm_neibsindexi[BLOCK_SIZE_BUILDNEIBS4][BLOCK_SIZE_BUILDNEIBS4];
	
	__shared__ uint sm_basecellstart;
	__shared__ uint sm_basecellend;
	__shared__ uint sm_basecellsize;
	__shared__ ushort sm_notonlybound;

	__shared__ volatile ushort sm_nothingtodo;
	
	// Thread 0 grab the first particle of block belonging to a new cell
	if (tid == 0) {
		sm_notonlybound = 0;
		sm_nothingtodo = 0;
		uint first_index_incell = INTMUL(blockIdx.x,blockDim.x);
		uint basecell_index = 0xffffffff;
		if (first_index_incell) {
			float3 pos = make_float3(tex1Dfetch(posTex, first_index_incell - 1));
			const uint last_cell_done = calcGridHash(calcGridPos(pos, worldOrigin, cellSize), gridSize);
			while (true) {
				float3 pos = make_float3(tex1Dfetch(posTex, first_index_incell));
				basecell_index = calcGridHash(calcGridPos(pos, worldOrigin, cellSize), gridSize);
				if (basecell_index != last_cell_done)
					break;
				first_index_incell++;
				if (first_index_incell >= numParticles)
					sm_nothingtodo = 1;
			}
			sm_basecellstart = tex1Dfetch(cellStartTex, basecell_index);
			sm_basecellend = tex1Dfetch(cellEndTex, basecell_index);
		}
		else {
			float3 pos = make_float3(tex1Dfetch(posTex, 0));
			basecell_index = calcGridHash(calcGridPos(pos, worldOrigin, cellSize), gridSize);
			sm_basecellstart = tex1Dfetch(cellStartTex, basecell_index);
			sm_basecellend = tex1Dfetch(cellEndTex, basecell_index);
		}
		
		if (sm_basecellend - sm_basecellstart >  MAX_PARTS_IN_CELL)
			 printf("block = %d, cellstart=%d, cellend = %d\n", basecell_index, sm_basecellstart);
	}
	__syncthreads();
	
	if (sm_nothingtodo)
		return;
	
	// Each thread load data from particle number base_cellstart + tid in shared memory
	// We are using texture fectches, so no problems if index is out of bounds
	uint indexi = sm_basecellstart + tid;	
	if (indexi < sm_basecellend) {
		const ushort type = tex1Dfetch(infoTex, indexi).x;
		const ushort flagi = !(type & 0xf0) || type == TESTPOINTSPART || type == OBJECTPART;
		if (flagi)
			sm_notonlybound = 1;
		sm_flagi[tid] = flagi;
		sm_posi[tid] = make_float3(posArray[indexi]);
	}
	__syncthreads();
		
	// If cell contains only boundary parts the whole block can quit
	if (!sm_notonlybound)
		return;  
	
	return;
}
	
	
__global__ void
buildNeibsListDevice2(  const float4* posArray,
						uint*	neibsList,
						uint3	gridSize,
						float3	cellSize,
						float3	worldOrigin,
						uint	numParticles,
						uint	numCells,
						float	sqinfluenceradius)
{
	//const uint basecell_index = blockIdx.x + gridSize.x*blockIdx.y + gridSize.x*gridSize.y*blockIdx.z;
	const uint basecell_index = INTMUL(INTMUL(blockIdx.z, gridSize.y), gridSize.x) + INTMUL(blockIdx.y, gridSize.x) + blockIdx.x;
	const uint tid = threadIdx.x;
	
	if (basecell_index >= numCells)
		return;
	
	// total number of neibs for this particle
	__shared__ uint sm_neibs_num[BLOCK_SIZE_BUILDNEIBS2];
	__shared__ uint sm_neibs_max[BLOCK_SIZE_BUILDNEIBS];
	__shared__ float3 sm_pos[BLOCK_SIZE_BUILDNEIBS2];
	__shared__ ushort sm_type[BLOCK_SIZE_BUILDNEIBS2];
	__shared__ uint sm_basecellstart;
	__shared__ uint sm_basecellend;
	__shared__ uint sm_basecellsize;

	uint neibs_num = 0;

	// Thread 0 load base_cellstart, base_cellend and base_cellsize in shared memory
	// Base_cell is the cell blockIdx.x
	if (tid == 0) {
		sm_basecellstart = tex1Dfetch(cellStartTex, basecell_index);
		sm_basecellend = tex1Dfetch(cellEndTex, basecell_index);
		sm_basecellsize = sm_basecellend - sm_basecellstart;
		if (sm_basecellsize > BLOCK_SIZE_BUILDNEIBS2 && sm_basecellstart != 0xffffffff)
			 printf("block = %d, cellstart=%d, cellend = %d, cellsize=%d\n", basecell_index, sm_basecellstart, sm_basecellend, sm_basecellsize);
		sm_basecellsize = (sm_basecellsize > BLOCK_SIZE_BUILDNEIBS2) ? BLOCK_SIZE_BUILDNEIBS2 : sm_basecellsize;
//		__threadfence_block();
		}
	__syncthreads();
		
	// If cell is empty the whole block can quit
	if (sm_basecellstart == 0xffffffff)
		return;  
	
	// Each thread load data from particle number base_cellstart + tid in shared memory
	// We are using texture fectches, so no problems if index is out of bounds
	const uint pos_index = sm_basecellstart + tid;	
	uint type = 0;
	float3 pos = make_float3(0.0f);
	if (pos_index < sm_basecellend) {
		type = tex1Dfetch(infoTex, pos_index).x;
		pos = make_float3(posArray[pos_index]);
		sm_type[tid] = type;
		sm_pos[tid] = pos;
	}
	__syncthreads();
	// Now all data from cell basecell_index are loaded in shared memory
	
	// Each thread will do test if particle at pos has neighbors in 
	// in shared memory
	if (pos_index < sm_basecellend) {
		// Neibouring list is calculated for testpoints and object points)
		if (!(type & 0xf0) || type == TESTPOINTSPART || type == OBJECTPART) {
			for (int i = 0; i < sm_basecellsize; i++) {
				// We avoid self interaction
				if (i != tid) {
					ushort neibType = sm_type[i];
					if (neibType != TESTPOINTSPART) {
						float3 neibPos = sm_pos[i];
						float3 relPos = pos - neibPos;
//						printf("block = %d, tid=%d, i=%d, r=%f\n", basecell_index, tid, i, length(relPos));
						if (sqlength(relPos) < sqinfluenceradius) {
							const uint neib_index = sm_basecellstart + i;
							if (neibs_num < MAXNEIBSNUM)
								neibsList[MAXNEIBSNUM*pos_index + neibs_num] = neib_index;
							neibs_num++;
						}
					}
				}
			}
		}
	}
		
	// Now we explore the remaining 26 cells
	for(int z=-1; z<=1; z++) {
		for(int y=-1; y<=1; y++) {
			for(int x=-1; x<=1; x++) {
				// Base cell is done
				if (x == 0 && y == 0 && z == 0)
					break;
				
				const int3 gridPos = make_int3(x,y,z) + make_int3(blockIdx.x, blockIdx.y, blockIdx.z);
				if ((gridPos.x < 0) || (gridPos.x >= gridSize.x) ||
					(gridPos.y < 0) || (gridPos.y >= gridSize.y) ||
					(gridPos.z < 0) || (gridPos.z >= gridSize.z))
				return;
				
				const int cell_index = INTMUL(INTMUL(gridPos.z, gridSize.y), gridSize.x) + INTMUL(gridPos.y, gridSize.x) + gridPos.x;
				
				// Check if out of bounds
				if (cell_index < 0 || cell_index >= numCells)
					break;
				
				__shared__  uint sm_cellstart;
				__shared__  uint sm_cellend;
				__shared__  uint sm_cellsize;
	
				// Thread 0 load cellstart, cellend and cellsize of cell number cell_index in shared memory
				if (tid == 0) {
					sm_cellstart = tex1Dfetch(cellStartTex, cell_index);
					sm_cellend = tex1Dfetch(cellEndTex, cell_index);
					sm_cellsize = sm_cellend - sm_cellstart;				
					sm_cellsize = (sm_cellsize > BLOCK_SIZE_BUILDNEIBS2) ? BLOCK_SIZE_BUILDNEIBS2 : sm_cellsize;
					}
				__syncthreads();

				if (sm_cellstart == 0xffffffff)
					break;   // cell empty
				
				// Each thread load data from particle number cellstart + tid in shared memory
				// We are using texture fectches, so no problems if index is out of bounds
				const uint index = sm_cellstart + tid;
				if (index < sm_cellend) {
					sm_type[tid] = tex1Dfetch(infoTex, index).x;
					sm_pos[tid] = make_float3(posArray[index]);
					}
				__syncthreads();
				// Now all data from cell cell_index are loaded in shared memory
				
				if (pos_index < sm_basecellend && index < sm_cellend && (!(type & 0xf0) || type == TESTPOINTSPART || type == OBJECTPART)) {
					// Checking for neibs in data loaded in shared memory
					for (int i = 0; i < sm_cellsize; i++) {
						ushort neibType = sm_type[i];
						if (neibType != TESTPOINTSPART) {
							float3 neibPos = sm_pos[i];
							float3 relPos = pos - neibPos;
							if (sqlength(relPos) < sqinfluenceradius) {
								const uint neib_index = sm_cellstart + i;
								if (neibs_num < MAXNEIBSNUM)
									neibsList[MAXNEIBSNUM*pos_index + neibs_num] = neib_index;
								neibs_num++;
							}
						}
					}
				}
			}
		}
	}
	
	// Shared memory reduction of per block maximum number of neighbors
	sm_neibs_num[tid] = neibs_num;
	sm_neibs_max[tid] = neibs_num;	
	__syncthreads();
	if (tid == 0) {
	  	uint num_interactions = 0;
	  	uint max = 0;
	  	for(int i=0; i< BLOCK_SIZE_BUILDNEIBS2; i++) {
	  		num_interactions += sm_neibs_num[i];
	  		max = (max > sm_neibs_num[i]) ? max : sm_neibs_num[i];
	  	}

		atomicAdd(&d_numInteractions, num_interactions);
		atomicMax(&d_maxNeibs, max);
	}
	
//	sm_neibs_num[tid] = neibs_num;	
//	sm_neibs_max[tid] = neibs_num;
//	__syncthreads();
//	
//	uint i = blockDim.x/2;
//	while (i != 0) {
//		if (tid < i) {
//			sm_neibs_num[tid] += sm_neibs_num[tid + 1];
//	  		sm_neibs_max[tid] = (sm_neibs_max[tid] > sm_neibs_max[tid + 1]) ? sm_neibs_max[tid] : sm_neibs_max[tid + 1];
//		}
//		__syncthreads();
//		i /= 2;
//	}
//	
//	if (tid == 0) {
//		atomicAdd(&d_numInteractions, sm_neibs_num[0]);
//		atomicMax(&d_maxNeibs, sm_neibs_max[0]);
//	}
//	return;
}

#define MAX_PARTS_IN_CELL	48
__global__ void
buildNeibsListDevice3(  const float4* posArray,
						uint*	neibsList,
						uint3	gridSize,
						float3	cellSize,
						float3	worldOrigin,
						uint	numParticles,
						uint	numCells,
						float	sqinfluenceradius)
{
	//const uint basecell_index = blockIdx.x + gridSize.x*blockIdx.y + gridSize.x*gridSize.y*blockIdx.z;
	const uint basecell_index = INTMUL(INTMUL(blockIdx.z, gridSize.y), gridSize.x) + INTMUL(blockIdx.y, gridSize.x) + blockIdx.x;
	const uint tid = threadIdx.x;
	
	if (basecell_index >= numCells)
		return;
	
	// total number of neibs for this particle
	__shared__ float3 sm_posi[MAX_PARTS_IN_CELL];
	__shared__ ushort sm_flagi[MAX_PARTS_IN_CELL];
	__shared__ ushort sm_neibsnumi[MAX_PARTS_IN_CELL];
	__shared__ uint sm_neibsindexi[MAX_PARTS_IN_CELL][BLOCK_SIZE_BUILDNEIBS2];
	
	__shared__ uint sm_basecellstart;
	__shared__ uint sm_basecellend;
	__shared__ uint sm_basecellsize;
	
	__shared__ ushort sm_notonlybound;

	uint neibs_num = 0;

	// Thread 0 load base_cellstart, base_cellend and base_cellsize in shared memory
	// Base_cell is the cell blockIdx.x
	if (tid == 0) {
		sm_basecellstart = tex1Dfetch(cellStartTex, basecell_index);
		sm_basecellend = tex1Dfetch(cellEndTex, basecell_index);
		sm_basecellsize = sm_basecellend - sm_basecellstart + 1;
		if (sm_basecellsize > MAX_PARTS_IN_CELL && sm_basecellstart != 0xffffffff)
			 printf("block = %d, cellstart=%d, cellend = %d, cellsize=%d\n", basecell_index, sm_basecellstart, sm_basecellend, sm_basecellsize);
		sm_basecellsize = (sm_basecellsize > MAX_PARTS_IN_CELL) ? MAX_PARTS_IN_CELL : sm_basecellsize;
		sm_notonlybound = 0;
		}
	__syncthreads();
		
	// If cell is empty the whole block can quit
	if (sm_basecellstart == 0xffffffff)
		return;  
	
	// Each thread load data from particle number base_cellstart + tid in shared memory
	// We are using texture fectches, so no problems if index is out of bounds
	uint indexi = sm_basecellstart + tid;	
	if (indexi <= sm_basecellend) {
		const ushort type = tex1Dfetch(infoTex, indexi).x;
		const ushort flagi = !(type & 0xf0) || type == TESTPOINTSPART || type == OBJECTPART;
		if (flagi)
			sm_notonlybound = 1;
		sm_flagi[tid] = flagi;
		sm_posi[tid] = make_float3(posArray[indexi]);
	}
	const uint lindex = tid + BLOCK_SIZE_BUILDNEIBS2;
	indexi = sm_basecellstart + lindex;
	if (indexi <= sm_basecellend) {
		const ushort type = tex1Dfetch(infoTex, indexi).x;
		const ushort flagi = !(type & 0xf0) || type == TESTPOINTSPART || type == OBJECTPART;
		if (flagi)
			sm_notonlybound = 1;
		sm_flagi[lindex] = flagi;
		sm_posi[lindex] = make_float3(posArray[indexi]);
	}
	__syncthreads();
	// Now all data from cell basecell_index are loaded in shared memory
	
	// If cell contains only boundary parts the whole block can quit
	if (!sm_notonlybound)
		return;  
	
	return;
}

#endif
