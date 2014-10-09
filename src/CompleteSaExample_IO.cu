#ifndef PROBLEM_IO_CU
#define PROBLEM_IO_CU

#include <math.h>
#include <string>
#include <iostream>

#include "CompleteSaExample.h"
#include "GlobalData.h"
#include "textures.cuh"
#include "utils.h"
#include "Problem.h"

namespace cuCompleteSaExample
{
#include "cellgrid.h"
// Core SPH functions
#include "sph_core_utils.cuh"

__device__
void
CompleteSaExample_imposeOpenBoundaryCondition(
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

	if (INFLOW(info) && !VEL_IO(info)) {
		waterdepth = 0.9f; // set inflow waterdepth to 0.9 (with respect to world_origin)
		const float localdepth = fmax(waterdepth - absPos.z, 0.0f);
		const float pressure = 9.81e3f*localdepth;
		eulerVel.w = RHO(pressure, PART_FLUID_NUM(info));
	}

	// impose tangential velocity
	if (INFLOW(info)) {
		eulerVel.y = 0.0f;
		eulerVel.z = 0.0f;
		// k and eps based on Versteeg & Malalasekera (2001)
		// turbulent intensity (between 1% and 6%)
		const float Ti = 0.01f;
		// in case of a pressure inlet eulerVel.x = 0 so we set u to 1 to multiply it later once
		// we know the correct velocity
		const float u = eulerVel.x > 1e-6f ? eulerVel.x : 1.0f;
		tke = 3.33333f;
		// length scale of the flow
		const float L = 1.0f;
		// constant is C_\mu^(3/4)/0.07*sqrt(3/2)
		// formula is epsilon = C_\mu^(3/4) k^(3/2)/(0.07 L)
		eps = 2.874944542f*tke*u*Ti/L;
	}
}

__global__ void
CompleteSaExample_imposeOpenBoundaryConditionDevice(
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
			/*
			if (!VEL_IO(info) && !INFLOW(info)) {
				waterdepth = ((float)IOwaterdepth[object(info)-1])/((float)UINT_MAX); // now between 0 and 1
				waterdepth *= d_cellSize.z*d_gridSize.z; // now between 0 and world size
				waterdepth += d_worldOrigin.z; // now absolute z position
			}
			*/
			// this now calls the virtual function that is problem specific
			CompleteSaExample_imposeOpenBoundaryCondition(info, absPos, waterdepth, eulerVel, tke, eps);
			// copy values to arrays
			newEulerVel[index] = eulerVel;
			if(newTke)
				newTke[index] = tke;
			if(newEpsilon)
				newEpsilon[index] = eps;
		}
	}
}

} // end of cuCompleteSaExample namespace

extern "C"
{

void
CompleteSaExample::setioboundconstants(
	const	PhysParams	*physparams,
	float3	const&		worldOrigin,
	uint3	const&		gridSize,
	float3	const&		cellSize)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuCompleteSaExample::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuCompleteSaExample::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuCompleteSaExample::d_gridSize, &gridSize, sizeof(uint3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuCompleteSaExample::d_rho0, &physparams->rho0, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuCompleteSaExample::d_bcoeff, &physparams->bcoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuCompleteSaExample::d_gammacoeff, &physparams->gammacoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuCompleteSaExample::d_sscoeff, &physparams->sscoeff, MAX_FLUID_TYPES*sizeof(float)));

}

}

void
CompleteSaExample::imposeOpenBoundaryConditionHost(
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

	cuCompleteSaExample::CompleteSaExample_imposeOpenBoundaryConditionDevice<<< numBlocks, numThreads, dummy_shared >>>
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
