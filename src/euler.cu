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

// Creates a kernel name based on whether XSPH is used or not
#define _EULER_KERNEL_NAME(xsph) cueuler::euler##xsph##Device

// Run the Euler kernel defined by EULER_KERNEL_NAME with the appropriate
// template and launch grid parameters, passing the arguments defined in the
// EULER_KERNEL_ARGS() macro. This macro takes one parameter, which is the
// timestep passed to the kernel (dt on the first step, dt2 on the second)
#define EULER_STEP_BOUNDARY_SWITCH \
	do { \
		if (step == 1) { \
			if (periodicbound) \
				EULER_KERNEL_NAME<1, true><<< numBlocks, numThreads >>>(EULER_KERNEL_ARGS(dt2)); \
			else \
				EULER_KERNEL_NAME<1, false><<< numBlocks, numThreads >>>(EULER_KERNEL_ARGS(dt2)); \
		} else if (step == 2) { \
			if (periodicbound) \
				EULER_KERNEL_NAME<2, true><<< numBlocks, numThreads >>>(EULER_KERNEL_ARGS(dt)); \
			else \
				EULER_KERNEL_NAME<2, false><<< numBlocks, numThreads >>>(EULER_KERNEL_ARGS(dt)); \
		} \
	} while (0)

#undef EULER_KERNEL_NAME
#undef EULER_KERNEL_ARGS

extern "C"
{
void
seteulerconstants(const PhysParams *physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_epsxsph, &physparams->epsxsph, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_dispvect, &physparams->dispvect, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_dispOffset, &physparams->dispOffset, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_minlimit, &physparams->minlimit, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_maxlimit, &physparams->maxlimit, sizeof(float3)));
}


void
geteulerconstants(PhysParams *physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->epsxsph, cueuler::d_epsxsph, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->maxlimit, cueuler::d_maxlimit, sizeof(float3), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->minlimit, cueuler::d_minlimit, sizeof(float3), 0));
}


void
setinleteuler(int numInlets, const float4* inletMin, const float4* inletMax, const float4* inletDisp, const float4 *inletVel)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_inlets, &numInlets, sizeof(numInlets)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_inlet_min, inletMin, numInlets*sizeof(float4)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_inlet_max, inletMax, numInlets*sizeof(float4)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_inlet_disp, inletDisp, numInlets*sizeof(float4)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_inlet_vel, inletVel, numInlets*sizeof(float4)));
}

void
setoutleteuler(const PhysParams *phys)
{
	uint numOutlets = phys->outlets;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_outlets, &numOutlets, sizeof(numOutlets)));
#define COPY_UP(field) \
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_##field, phys->field, numOutlets*sizeof(float4)))

	COPY_UP(outlet_min);
	COPY_UP(outlet_max);
	COPY_UP(outlet_disp);
	COPY_UP(outlet_plane);
#undef COPY_UP
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
euler(	const float4	*oldPos,
		const float4	*oldVel,
		particleinfo	*info,
		const float4	*forces,
		const float4	*xsph,
		float4		*newPos,
		float4		*newVel,
		uint		numParticles,
		uint		*newNumParts,
		uint		maxParticles,
		uint		particleRangeEnd,
		float		dt,
		float		dt2,
		int			step,
		float		t,
		bool		xsphcorr,
		bool		periodicbound)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_INTEGRATE, particleRangeEnd);
	int numBlocks = (int) ceil(particleRangeEnd / (float) numThreads);

	// execute the kernel
	if (xsphcorr) {
#define EULER_KERNEL_NAME _EULER_KERNEL_NAME(Xsph)
#define EULER_KERNEL_ARGS(dt) \
					oldPos, oldVel, info, \
					forces, xsph, \
					newPos, newVel, \
					newNumParts, numParticles, maxParticles, \
					dt, dt2, t
		EULER_STEP_BOUNDARY_SWITCH;
#undef EULER_KERNEL_NAME
#undef EULER_KERNEL_ARGS
	} else {
#define EULER_KERNEL_NAME _EULER_KERNEL_NAME()
#define EULER_KERNEL_ARGS(dt) \
					oldPos, oldVel, info, \
					forces, \
					newPos, newVel, \
					newNumParts, numParticles, maxParticles, \
					dt, dt2, t
		EULER_STEP_BOUNDARY_SWITCH;
#undef EULER_KERNEL_NAME
#undef EULER_KERNEL_ARGS
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Euler kernel execution failed");
}
}
