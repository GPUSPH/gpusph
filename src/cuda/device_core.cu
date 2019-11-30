/*  Copyright (c) 2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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
 * Common device functions of general interests (reductions, etc)
 */

#ifndef DEVICE_CORE_CU
#define DEVICE_CORE_CU

//! Reductions are achieved by doing an in-block reduction first,
//! followed by a global reduction after the kernel.
//! On GPU, the in-block reduction is achieved by storing per-thread data
//! in sm_max, and the calling maxBlockReduce.
//! On CPU, threads don't communicate with each other, so sm_max can have size 1.
//! With OpenMP enabled, we let each thread store data to global memory separately,
//! and still do the final reduction after the kernel
#if CPU_BACKEND_ENABLED
#define SHMEM_SIZE(block_size) 1
#define SHMEM_IDX 0
#else // CUDA_BACKEND_ENABLED
#define SHMEM_SIZE(block_size) (block_size)
#define SHMEM_IDX threadIdx.x
#endif

//! Index of the block (for GPU) or the OpenMP thread (CPU + OpenMP)
//! Returns 0 for single-threaded CPU.
__device__ __forceinline__
uint block_idx() {
#if CPU_BACKEND_ENABLED && USE_OPENMP
	return omp_get_thread_num();
#elif CPU_BACKEND_ENABLED
	return 0;
#else // CUDA_BACKEND_ENABLED
	return blockIdx.x;
#endif
}

//! Find the maximum of an array in shared memory, and store it to global memory
/*! This function takes as input an array in shared memory, initialized
 * with one value per thread, and finds the maximum for the block,
 * storing the maximum (one per block) in global memory.
 */
__device__ __forceinline__ void
maxBlockReduce(
	float*	sm_max, ///< shared memory array to be maximized
	float*	cfl, ///< global memory storage location
	uint	cflOffset ///< offset to index the cfl array
)
{
	const uint block_offset = cflOffset + block_idx();
#if CPU_BACKEND_ENABLED
	cfl[block_offset] = max(cfl[block_offset], sm_max[0]);
#else
	// CUDA_BACKEND_ENABLED
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1)
	{
		__syncthreads();
		if (threadIdx.x < s)
		{
			sm_max[threadIdx.x] = max(sm_max[threadIdx.x + s], sm_max[threadIdx.x]);
		}
	}

	// write result for this block to global mem
	if (!threadIdx.x)
		cfl[block_offset] = sm_max[0];
#endif
}

#endif

