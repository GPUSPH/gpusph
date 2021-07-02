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

#ifndef CUDA_SAFE_CALL_H
#define CUDA_SAFE_CALL_H

#include "safe_call.h"

#include <cuda_runtime_api.h>

inline void cudaSafeCallNoSync(cudaError err,
	const char *file, const int line, const char *func,
	const char *method=NULL,
	const char *message=NULL)
{
	if (!method)
		method = __func__;
	if (cudaSuccess != err) {
		devAPICallFailed(err, cudaGetErrorString(err),
			file, line, func, method, message);
	}
}


inline void cudaSafeCall(cudaError err,
	const char *file, const int line, const char *func)
{
	if (err == cudaSuccess) err = cudaDeviceSynchronize();
	cudaSafeCallNoSync(err, file, line, func, __func__);
}

inline void cudaGetLastError(const char *errorMessage,
	const char *file, const int line, const char *func)
{
	cudaError_t err = cudaGetLastError();
	cudaSafeCallNoSync(err, file, line, func, __func__, errorMessage);
}


inline void cudaGetSyncError(const char *errorMessage,
	const char *file, const int line, const char *func)
{
	cudaError_t err = cudaDeviceSynchronize();
	cudaSafeCallNoSync(err, file, line, func, __func__, errorMessage);
}

#define devAPISafeCall       cudaSafeCall
#define devAPISafeCallNoSync cudaSafeCallNoSync
#define devAPIGetLastError   cudaGetLastError
#define devAPIGetSyncError   cudaGetSyncError

#endif
