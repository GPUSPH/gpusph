/*  Copyright 2019 Giuseppe Bilotta, Alexis Hérault, Robert A. Dalrymple, Ciro Del Negro

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
 * Common device functions of general interests (reductions, etc)
 */

#ifndef DEVICE_CORE_CU
#define DEVICE_CORE_CU

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
		cfl[cflOffset + blockIdx.x] = sm_max[0];
}

#endif

