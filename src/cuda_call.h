#ifndef _CUDA_SAFE_CALL_
#define _CUDA_SAFE_CALL_

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CUDA_SAFE_CALL_NOSYNC(err)	__cudaSafeCallNoSync(err, __FILE__, __LINE__)
#define CUDA_SAFE_CALL(err)			__cudaSafeCall(err, __FILE__, __LINE__)
#define CUT_CHECK_ERROR(err)		__cutilGetLastError(err, __FILE__, __LINE__)

inline void __cudaSafeCallNoSync( cudaError err, const char *file, const int line )
{
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cudaSafeCallNoSync() Runtime API error %d : %s.\n",
                file, line, (int)err, cudaGetErrorString( err ));
        exit(-1);
    }
}


inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
	cudaDeviceSynchronize();
    if( cudaSuccess != err) {
		fprintf(stderr, "%s(%i) : cudaSafeCall() Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}


inline void __cutilGetLastError( const char *errorMessage, const char *file, const int line )
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "%s(%i) : cutilCheckMsg() CUTIL CUDA error : %s : (%d) %s.\n",
                file, line, errorMessage, (int)err, cudaGetErrorString( err ) );
        exit(-1);
    }
}

#endif

