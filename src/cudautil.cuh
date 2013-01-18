#ifndef _CUDAUTIL_CUH_
#define _CUDAUTIL_CUH_

#include "Options.h"
#include "cuda_call.h"

cudaDeviceProp checkCUDA(const Options&);

void allocateArray(void **devPtr, size_t size);

void freeArray(void *devPtr);

void threadSync();

#endif
