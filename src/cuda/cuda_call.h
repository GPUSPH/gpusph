/*  Copyright (c) 2011-2018 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
 */
#ifndef _CUDA_SAFE_CALL_
#define _CUDA_SAFE_CALL_

#include <thread>
#include <stdexcept>
#include <sstream>

#include <cuda_runtime.h>

#define CUDA_SAFE_CALL_NOSYNC(err)	__cudaSafeCallNoSync(err, __FILE__, __LINE__, __func__)
#define CUDA_SAFE_CALL(err)			__cudaSafeCall(err, __FILE__, __LINE__, __func__)
#define CUT_CHECK_ERROR(err)		__cutilGetSyncError(err, __FILE__, __LINE__, __func__)

//! Is this a multi-device run?
//! On the host we use the MULTI_DEVICE macro that extracts the information from gdata,
//! which is not available inside the framework, so we use this global boolean
//! that will be initialized on host
extern bool is_multi_device;

/*! We should not sync after every kernel call, in order to improve host/device parallelism,
 *  but we don't have proper locking mechanism for buffers.
 *  This isn't a problem in single GPU, since we only use one command stream, but it might be
 *  in multi-GPU. So by default we sync only if is_multi_device.
 *
 */
#define KERNEL_CHECK_ERROR			if (is_multi_device || g_debug.sync_kernels) CUT_CHECK_ERROR("kernel execution failed")

inline void __cudaSafeCallNoSync(cudaError err,
	const char *file, const int line, const char *func,
	const char *method="cudaSafeCallNoSync",
	const char *message=NULL)
{
	if (cudaSuccess != err) {
		std::stringstream errmsg;
		errmsg	<< file << "(" << line << ") : in " << func
			<< "() @ thread 0x" << std::this_thread::get_id() << " : " << method << "()";
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

