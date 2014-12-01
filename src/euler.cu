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

#include <stdio.h>

#include "euler.cuh"
#include "euler_kernel.cu"

#include "utils.h"

extern "C"
{
void
seteulerconstants(const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_epsxsph, &physparams->epsxsph, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_gridSize, &gridSize, sizeof(uint3)));
}


void
geteulerconstants(PhysParams *physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->epsxsph, cueuler::d_epsxsph, sizeof(float), 0));
}


void
setmbdata(const float4* MbData, uint size)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_mbdata, MbData, size));
}


void
seteulerrbcg(const float3* cg, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbcg, cg, numbodies*sizeof(float3)));
}


void
seteulerrbtrans(const float3* trans, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbtrans, trans, numbodies*sizeof(float3)));
}


void
seteulerrblinearvel(const float3* linearvel, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rblinearvel, linearvel, numbodies*sizeof(float3)));
	//printf("Upload linear vel: %e %e %e\n", linearvel[0].x, linearvel[0].y, linearvel[0].z);
}


void
seteulerrbangularvel(const float3* angularvel, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbangularvel, angularvel, numbodies*sizeof(float3)));
	//printf("Upload angular vel: %e %e %e\n", angularvel[0].x, angularvel[0].y, angularvel[0].z);
}


void
seteulerrbsteprot(const float* rot, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbsteprot, rot, 9*numbodies*sizeof(float)));
}


void
euler(	const float4*		oldPos,
		const hashKey*		particleHash,
		const float4*		oldVel,
		const float*		oldTKE,
		const float*		oldEps,
		const particleinfo* info,
		const float4*		forces,
		float2*				keps_dkde,
		const float4*		xsph,
		float4*				newPos,
		float4*				newVel,
		float*				newTKE,
		float*				newEps,
		const uint			numParticles,
		const uint			particleRangeEnd,
		const float			dt,
		const float			dt2,
		const int			step,
		const float			t,
		const bool			xsphcorr,
		BoundaryType		boundarytype)
{
	// thread per particle
	uint numThreads = min(BLOCK_SIZE_INTEGRATE, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

#define ARGS oldPos, particleHash, oldVel, oldTKE, oldEps, \
	info, forces, keps_dkde, xsph, newPos, newVel, newTKE, newEps, particleRangeEnd, dt, dt2, t

	// execute the kernel
	if (boundarytype == DYN_BOUNDARY) {
		if (step == 1) {
			if (xsphcorr)
				cueuler::eulerDevice<1, true, true><<< numBlocks, numThreads >>>(ARGS);
			else
				cueuler::eulerDevice<1, false, true><<< numBlocks, numThreads >>>(ARGS);
		} else if (step == 2) {
			if (xsphcorr)
				cueuler::eulerDevice<2, true, true><<< numBlocks, numThreads >>>(ARGS);
			else
				cueuler::eulerDevice<2, false, true><<< numBlocks, numThreads >>>(ARGS);
		}
	} else {
		if (step == 1) {
			if (xsphcorr)
				cueuler::eulerDevice<1, true, false><<< numBlocks, numThreads >>>(ARGS);
			else
				cueuler::eulerDevice<1, false, false><<< numBlocks, numThreads >>>(ARGS);
		} else if (step == 2) {
			if (xsphcorr)
				cueuler::eulerDevice<2, true, false><<< numBlocks, numThreads >>>(ARGS);
			else
				cueuler::eulerDevice<2, false, false><<< numBlocks, numThreads >>>(ARGS);
		}
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Euler kernel execution failed");
}
}
