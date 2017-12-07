#ifndef _CUDAUTIL_H_
#define _CUDAUTIL_H_

#include "GlobalData.h"
#include "Options.h"
#include "cuda_call.h"

cudaDeviceProp checkCUDA(const GlobalData* gdata, uint devnum);

#endif
