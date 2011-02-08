#include <stdio.h>

#include "particledefine.h"

#include "euler.cuh"
#include "euler_kernel.cu"

extern "C"
{
void
euler(	float4*		oldPos,
		float4*		oldVel,
		particleinfo* info,
		float4*		forces,
		float4*		xsph,
		float4*		newPos,
		float4*		newVel,
		uint		numParticles,
		float		dt,
		float		dt2,
		int			step,
		float		t,
		bool		xsphcorr,
		bool		periodicbound)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_INTEGRATE, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	// execute the kernel
	if (step == 1) {
		if (periodicbound) {
			if (xsphcorr)
				eulerXsphDevice<1, true><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt2, dt2, t);
			else
				eulerDevice<1, true><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt2, dt2, t);
		} else {
			if (xsphcorr)
				eulerXsphDevice<1, false><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt2, dt2, t);
			else
				eulerDevice<1, false><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt2, dt2, t);
		}
	} else if (step == 2) {
		if (periodicbound) {
			if (xsphcorr)
				eulerXsphDevice<2, true><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt, dt2, t);
			else
				eulerDevice<2, true><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt, dt2, t);
		} else {
			if (xsphcorr)
				eulerXsphDevice<2, false><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt, dt2, t);
			else
				eulerDevice<2, false><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt, dt2, t);
		}
	} // if (step == 2)

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Kernel execution failed");
}
}
