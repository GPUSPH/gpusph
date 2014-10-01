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

namespace cuIO
{
#include "cellgrid.h"

__device__
float4
InputProblem_imposeOpenBoundaryCondition(
	const	particleinfo	info,
	const	float3			absPos)
{
	float4 eulerVel = make_float4(0.0f);

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
#else
			eulerVel.x = 0.0f;
#endif
	}
	else {
		eulerVel.w = 1000.0f;
	}

	// impose tangential velocity
	if (INFLOW(info)) {
		eulerVel.y = 0.0f;
		eulerVel.z = 0.0f;
	}

	return eulerVel;
}

__global__ void
InputProblem_imposeOpenBoundaryConditionDevice(
			float4*		newEulerVel,
	const	float4*		oldPos,
	const	uint		numParticles,
	const	hashKey*	particleHash)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	float4 eulerVel = make_float4(0.0f); // imposed velocity/pressure
	if(index < numParticles) {
		const particleinfo info = tex1Dfetch(infoTex, index);
		if (VERTEX(info) && IO_BOUNDARY(info)) {
			// open boundaries
			const float3 absPos = d_worldOrigin + as_float3(oldPos[index])
									+ calcGridPosFromParticleHash(particleHash[index])*d_cellSize
									+ 0.5f*d_cellSize;
			// this now calls the virtual function that is problem specific
			eulerVel = InputProblem_imposeOpenBoundaryCondition(info, absPos);
		}
		newEulerVel[index] = eulerVel;
	}
}

} // end of cuIO namespace

extern "C"
{

void
setioboundconstants(
	float3	const&	worldOrigin,
	uint3	const&	gridSize,
	float3	const&	cellSize)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuIO::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuIO::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuIO::d_gridSize, &gridSize, sizeof(uint3)));
}

}

void
InputProblem::imposeOpenBoundaryConditionHost(
			float4*			newEulerVel,
	const	particleinfo*	info,
	const	float4*			oldPos,
	const	uint			numParticles,
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

	cuIO::InputProblem_imposeOpenBoundaryConditionDevice<<< numBlocks, numThreads, dummy_shared >>>
		(newEulerVel, oldPos, numParticles, particleHash);

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("imposeOpenBoundaryCondition kernel execution failed");
}

#endif
