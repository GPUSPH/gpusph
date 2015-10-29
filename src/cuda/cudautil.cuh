#ifndef _CUDAUTIL_CUH_
#define _CUDAUTIL_CUH_

#include "GlobalData.h"
#include "Options.h"
#include "cuda_call.h"

cudaDeviceProp checkCUDA(const GlobalData* gdata, uint devnum);

#endif
