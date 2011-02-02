#ifndef _FMAX_CUH_
#define _FMAX_CUH_

#define MAX_BLOCKS_FMAX			64
#define MAX_THREADS_FMAX		128

extern "C"
{

void getNumBlocksAndThreads(int size,
							int maxBlocks,
							int maxThreads,
							int &blocks,
							int &threads);

float getMax(   int  size,
				float* d_idata,
				float* d_odata);
}
#endif