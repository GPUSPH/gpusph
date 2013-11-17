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

#include <stdio.h>

#include "euler.cuh"
#include "euler_kernel.cu"

#include "utils.h"

extern "C"
{
void
seteulerconstants(const PhysParams *physparams, const uint3 gridSize, const float3 cellSize)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_epsxsph, &physparams->epsxsph, sizeof(float)));

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
seteulerrbsteprot(const float* rot, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbsteprot, rot, 9*numbodies*sizeof(float)));
}


void
euler(	const float4*		oldPos,
		const hashKey*		particleHash,
		const float4*		oldVel,
		const particleinfo* info,
		const float4*		forces,
		const float4*		xsph,
		float4*				newPos,
		float4*				newVel,
		const uint			numParticles,
		const float			dt,
		const float			dt2,
		const int			step,
		const float			t,
		const bool			xsphcorr)
{
	// thread per particle
	uint numThreads = min(BLOCK_SIZE_INTEGRATE, numParticles);
	uint numBlocks = div_up(numParticles, numThreads);

	// execute the kernel
	if (step == 1) {
		if (xsphcorr)
			cueuler::eulerXsphDevice<1><<< numBlocks, numThreads >>>(oldPos, particleHash, oldVel,
								info, forces, xsph, newPos, newVel, numParticles, dt2, dt2, t);
		else
			cueuler::eulerDevice<1><<< numBlocks, numThreads >>>(oldPos, particleHash, oldVel,
								info, forces, xsph, newPos, newVel, numParticles, dt2, dt2, t);
	} else if (step == 2) {
		if (xsphcorr)
			cueuler::eulerXsphDevice<2><<< numBlocks, numThreads >>>(oldPos, particleHash, oldVel,
								info, forces, xsph, newPos, newVel, numParticles, dt, dt2, t);
		else
			cueuler::eulerDevice<2><<< numBlocks, numThreads >>>(oldPos, particleHash, oldVel,
								info, forces, xsph, newPos, newVel, numParticles, dt, dt2, t);
	} // if (step == 2)

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Euler kernel execution failed");
}
}
