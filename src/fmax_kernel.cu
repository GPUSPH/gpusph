#ifndef _FLOATMAX_KERNEL_
#define _FLOATMAX_KERNEL_

#include "vector_math.h"

template <unsigned int blockSize>
__global__ void
floatMaxDevice(float *g_idata, float *g_odata, unsigned int n)
{
	extern __shared__ float sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridSize).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	float maximum = max(g_idata[i], g_idata[i+blockSize]);
	i += gridSize;
	while (i < n)
	{
		float temp = fmaxf(g_idata[i], g_idata[i+blockSize]);
		maximum = fmaxf(maximum, temp);
		i += gridSize;
	}
	sdata[tid] = maximum;
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = fmaxf(sdata[tid], sdata[tid +  64]); } __syncthreads(); }

	if (tid < 32)
	{
		if (blockSize >=  64) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 32]); __syncthreads(); }
		if (blockSize >=  32) { sdata[tid] = fmaxf(sdata[tid], sdata[tid + 16]); __syncthreads(); }
		if (blockSize >=  16) { sdata[tid] = fmaxf(sdata[tid], sdata[tid +  8]); __syncthreads(); }
		if (blockSize >=   8) { sdata[tid] = fmaxf(sdata[tid], sdata[tid +  4]); __syncthreads(); }
		if (blockSize >=   4) { sdata[tid] = fmaxf(sdata[tid], sdata[tid +  2]); __syncthreads(); }
		if (blockSize >=   2) { sdata[tid] = fmaxf(sdata[tid], sdata[tid +  1]); __syncthreads(); }
	}

	// write result for this block to global mem
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

#endif
