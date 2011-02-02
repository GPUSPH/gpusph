/*
* Copyright 1993-2007 NVIDIA Corporation.  All rights reserved.
*
* NOTICE TO USER:
*
* This source code is subject to NVIDIA ownership rights under U.S. and
* international Copyright laws.
*
* NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
* CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
* IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
* REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
* MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
* IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
* OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
* OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
* OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
* OR PERFORMANCE OF THIS SOURCE CODE.
*
* U.S. Government End Users.  This source code is a "commercial item" as
* that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
* "commercial computer software" and "commercial computer software
* documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
* and is provided to the U.S. Government only as a commercial end item.
* Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
* 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
* source code with only those rights set forth herein.
*/

/*
	Parallel reduction

	This sample shows how to perform a reduction operation on an array of values
	to produce a single value.

	Reductions are a very common computation in parallel algorithms.  Any time
	an array of values needs to be reduced to a single value using a binary
	associative operator, a reduction can be used.  Example applications include
	statistics computaions such as mean and standard deviation, and image
	processing applications such as finding the total luminance of an
	image.

	This code performs sum reductions, but any associative operator such as
	min() or max() could also be used.

	It assumes the input size is a power of 2.

	COMMAND LINE ARGUMENTS

	"--shmoo":		 Test performance for 1 to 32M elements with each of the 7 different kernels
	"--n=<N>":		 Specify the number of elements to reduce (default 1048576)
	"--threads=<N>":   Specify the number of threads per block (default 128)
	"--kernel=<N>":	Specify which kernel to run (0-6, default 6)
	"--maxblocks=<N>": Specify the maximum number of thread blocks to launch (kernel 6 only, default 64)
	"--cpufinal":	  Read back the per-block results and do final sum of block sums on CPU (default false)
	"--cputhresh=<N>": The threshold of number of blocks sums below which to perform a CPU final reduction (default 1)

*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

//#include <cutil.h>
#include "fmax.cuh"
#include "fmax_kernel.cu"
#include "cuda_call.h"

extern "C" {

void floatMax(float *d_idata, float *d_odata, int size, int threads, int blocks)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);
	int smemSize = threads * sizeof(float);

	switch (threads) {
		case 512:
			floatMaxDevice<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			floatMaxDevice<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			floatMaxDevice<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			floatMaxDevice< 64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			floatMaxDevice< 32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			floatMaxDevice< 16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			floatMaxDevice<  8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			floatMaxDevice<  4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			floatMaxDevice<  2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			floatMaxDevice<  1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		}
}


void getNumBlocksAndThreads(int size, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
	if (size == 1)
		threads = 1;
	else
		threads = (size < maxThreads*2) ? size / 2 : maxThreads;
	blocks = size / (threads * 2);
	blocks = min(maxBlocks, blocks);
}


float getMax(   int  size,
				float* d_idata,
				float* d_odata)
{
	float maximum = 0.0f;

	int numBlocks = 0;
	int numThreads = 0;
	getNumBlocksAndThreads(size, MAX_BLOCKS_FMAX, MAX_THREADS_FMAX, numBlocks, numThreads);
	CUDA_SAFE_CALL(cudaMemset(d_odata, 0, numBlocks*sizeof(float)));

	// execute the kernel for the first time
	floatMax(d_idata, d_odata, size, numThreads, numBlocks);

	// check if kernel execution generated an error
	CUT_CHECK_ERROR("Kernel execution failed");

	// sum partial block sums on GPU
	int s = numBlocks;
	while(s > 1) {
		int threads = 0, blocks = 0;
		getNumBlocksAndThreads(s, MAX_BLOCKS_FMAX, MAX_THREADS_FMAX, blocks, threads);
		floatMax(d_odata, d_odata, s, threads, blocks);
		s = s / (threads*2);
		}

	// copy final sum from device to host
	CUDA_SAFE_CALL( cudaMemcpy( &maximum, d_odata, sizeof(float), cudaMemcpyDeviceToHost) );

	return maximum;
}

}

