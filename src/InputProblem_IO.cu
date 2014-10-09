#ifndef PROBLEM_IO_CU
#define PROBLEM_IO_CU

#include <math.h>
#include <string>
#include <iostream>

#include "InputProblem.h"
#include "GlobalData.h"
#include "textures.cuh"
#include "utils.h"
#include "Problem.h"

namespace cuInputProblem
{
#include "cellgrid.h"
// Core SPH functions
#include "sph_core_utils.cuh"

__device__
void
InputProblem_imposeOpenBoundaryCondition(
	const	particleinfo	info,
	const	float3			absPos,
			float			waterdepth,
			float4&			eulerVel,
			float&			tke,
			float&			eps)
{
	eulerVel = make_float4(0.0f);
	tke = 0.0f;
	eps = 0.0f;

	if (VEL_IO(info)) {
#if SPECIFIC_PROBLEM == SmallChannelFlowIO
			// third order approximation to the flow in a rectangular duct
			const float y2 = absPos.y*absPos.y;
			const float z2 = absPos.z*absPos.z;
			const float y4 = y2*y2;
			const float z4 = z2*z2;
			const float y6 = y2*y4;
			const float z6 = z2*z4;
			const float y8 = y4*y4;
			const float z8 = z4*z4;
			eulerVel.x = (461.0f+y8-392.0f*z2-28.0f*y6*z2-70.0f*z4+z8+70.0f*y4*(z4-1.0f)-28.0f*y2*(14.0f-15.0f*z2+z6))/461.0f;
			eulerVel.x = fmax(eulerVel.x, 0.0f);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
			eulerVel.x = 1.0f;
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPer
			eulerVel.x = 1.0f-absPos.z*absPos.z;
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
			// the 0.025 is deltap*0.5 = 0.05*0.5
			eulerVel.x = log(fmax(1.0f-fabs(absPos.z), 0.025f)/0.0015625f)/0.41f+5.2f;
#else
			eulerVel.x = 0.0f;
#endif
	}
	else {
#if SPECIFIC_PROBLEM == LaPalisseSmallTest
		if (INFLOW(info))
			waterdepth = 0.21; // set inflow waterdepth to 0.21 (with respect to world_origin)
		const float localdepth = fmax(waterdepth - absPos.z, 0.0f);
		const float pressure = 9.81e3f*localdepth;
		eulerVel.w = RHO(pressure, PART_FLUID_NUM(info));
#else
		eulerVel.w = 1000.0f;
#endif
	}

	// impose tangential velocity
	if (INFLOW(info)) {
		eulerVel.y = 0.0f;
		eulerVel.z = 0.0f;
#if SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
		// k and eps based on Versteeg & Malalasekera (2001)
		// turbulent intensity (between 1% and 6%)
		const float Ti = 0.01f;
		// in case of a pressure inlet eulerVel.x = 0 so we set u to 1 to multiply it later once
		// we know the correct velocity
		const float u = eulerVel.x > 1e-6f ? eulerVel.x : 1.0f;
		tke = 3.0f/2.0f*(u*Ti)*(u*Ti);
		tke = 3.33333f;
		// length scale of the flow
		const float L = 1.0f;
		// constant is C_\mu^(3/4)/0.07*sqrt(3/2)
		// formula is epsilon = C_\mu^(3/4) k^(3/2)/(0.07 L)
		eps = 2.874944542f*tke*u*Ti/L;
		eps = 1.0f/0.41f/fmax(1.0f-fabs(absPos.z),0.025f);
#endif
	}
}

__global__ void
InputProblem_imposeOpenBoundaryConditionDevice(
			float4*		newEulerVel,
			float*		newTke,
			float*		newEpsilon,
	const	float4*		oldPos,
	const	uint*		IOwaterdepth,
	const	uint		numParticles,
	const	hashKey*	particleHash)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	float4 eulerVel = make_float4(0.0f); // imposed velocity/pressure
	float tke = 0.0f;
	float eps = 0.0f;
	if(index < numParticles) {
		const particleinfo info = tex1Dfetch(infoTex, index);
		if (VERTEX(info) && IO_BOUNDARY(info)) {
			// open boundaries
			const float3 absPos = d_worldOrigin + as_float3(oldPos[index])
									+ calcGridPosFromParticleHash(particleHash[index])*d_cellSize
									+ 0.5f*d_cellSize;
			// when pressure outlets require the water depth compute it from the IOwaterdepth integer
			float waterdepth = 0.0f;
			if (!VEL_IO(info) && !INFLOW(info)) {
				waterdepth = ((float)IOwaterdepth[object(info)-1])/((float)UINT_MAX); // now between 0 and 1
				waterdepth *= d_cellSize.z*d_gridSize.z; // now between 0 and world size
				waterdepth += d_worldOrigin.z; // now absolute z position
			}
			// this now calls the virtual function that is problem specific
			InputProblem_imposeOpenBoundaryCondition(info, absPos, waterdepth, eulerVel, tke, eps);
			// copy values to arrays
			newEulerVel[index] = eulerVel;
			if(newTke)
				newTke[index] = tke;
			if(newEpsilon)
				newEpsilon[index] = eps;
		}
	}
}

} // end of cuInputProblem namespace

extern "C"
{

void
InputProblem::setioboundconstants(
	const	PhysParams	*physparams,
	float3	const&		worldOrigin,
	uint3	const&		gridSize,
	float3	const&		cellSize)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_gridSize, &gridSize, sizeof(uint3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_rho0, &physparams->rho0, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_bcoeff, &physparams->bcoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_gammacoeff, &physparams->gammacoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_sscoeff, &physparams->sscoeff, MAX_FLUID_TYPES*sizeof(float)));

}

}

void
InputProblem::imposeOpenBoundaryConditionHost(
			float4*			newEulerVel,
			float*			newTke,
			float*			newEpsilon,
	const	particleinfo*	info,
	const	float4*			oldPos,
			uint			*IOwaterdepth,
	const	uint			numParticles,
	const	uint			numObjects,
	const	uint			particleRangeEnd,
	const	hashKey*		particleHash)
{
	uint numThreads = min(BLOCK_SIZE_IOBOUND, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	cuInputProblem::InputProblem_imposeOpenBoundaryConditionDevice<<< numBlocks, numThreads, dummy_shared >>>
		(newEulerVel, newTke, newEpsilon, oldPos, IOwaterdepth, numParticles, particleHash);

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// reset waterdepth calculation
	uint h_IOwaterdepth[numObjects];
	for (uint i=0; i<numObjects; i++)
		h_IOwaterdepth[i] = 0;
	CUDA_SAFE_CALL(cudaMemcpy(IOwaterdepth, h_IOwaterdepth, numObjects*sizeof(int), cudaMemcpyHostToDevice));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("imposeOpenBoundaryCondition kernel execution failed");
}

#endif
