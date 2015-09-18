#ifndef _CUDA_SAFE_CALL_
#define _CUDA_SAFE_CALL_

#include <pthread.h>
#include <stdexcept>
#include <sstream>

#include <cuda_runtime.h>

#define CUDA_SAFE_CALL_NOSYNC(err)	__cudaSafeCallNoSync(err, __FILE__, __LINE__, __func__)
#define CUDA_SAFE_CALL(err)			__cudaSafeCall(err, __FILE__, __LINE__, __func__)
#define CUT_CHECK_ERROR(err)		__cutilGetSyncError(err, __FILE__, __LINE__, __func__)
#define KERNEL_CHECK_ERROR			CUT_CHECK_ERROR("kernel execution failed")

inline void __cudaSafeCallNoSync(cudaError err,
	const char *file, const int line, const char *func,
	const char *method="cudaSafeCallNoSync",
	const char *message=NULL)
{
	if (cudaSuccess != err) {
		std::stringstream errmsg;
		errmsg	<< file << "(" << line << ") : in " << func
			<< "() @ thread 0x" << pthread_self() << " : " << method << "()";
		if (message)
			errmsg << " - " << message << " -";
		errmsg << " runtime API error " << err << " : " <<  cudaGetErrorString(err);
		throw std::runtime_error(errmsg.str());
	}
}


inline void __cudaSafeCall(cudaError err,
	const char *file, const int line, const char *func)
{
	if (err == cudaSuccess) err = cudaDeviceSynchronize();
	__cudaSafeCallNoSync(err, file, line, func, "cudaSafeCall");
}

/* We use CUT_CHECK_ERROR after launching kernels in large switches, so we actually
 * want to synchronize on the device before checking for errors. Because of this, we
 * alias CUT_CHECK_ERROR to the`__cutilGetSyncError` below rather than the `__cutilGetLastError`
 * here, which we keep because it might turn useful in other contexts.
 * TODO: check effect of this on multi-GPU */

inline void __cutilGetLastError(const char *errorMessage,
	const char *file, const int line, const char *func)
{
	cudaError_t err = cudaGetLastError();
	__cudaSafeCallNoSync(err, file, line, func, "getLastError", errorMessage);
}


inline void __cutilGetSyncError(const char *errorMessage,
	const char *file, const int line, const char *func)
{
	cudaError_t err = cudaDeviceSynchronize();
	__cudaSafeCallNoSync(err, file, line, func, "getSyncError", errorMessage);
}

#endif

