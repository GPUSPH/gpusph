/*  Copyright (c) 2021 INGV, EDF, UniCT, JHU

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

#ifndef DEM_PARAMS_H
#define DEM_PARAMS_H

#include <cuda_runtime_api.h>
#include "cache_preference.h"

//! ENABLE_DEM-related texture parameters
struct dem_params
{
private:
#if DISABLE_DEM_TEXTURE
	float *demArray;
	int width;
	int height;
#else
	cudaTextureObject_t demTex;
#endif

public:
	// There is a (thread-local) global dem_params object,
	// whose construction is managed separately.
	// All uses of dem_params as substructures otherwise
	// will simply copy-initialize from the global object.
	dem_params();

	//! Fetch the DEM content at position x, y.
	//! NOTE: x, y are in DEM-relative coordinates
	//! This maps to a simple texture fetch, unless DEM texture usage is disabled,
	//! in which case we do the bilinear interpolation manually
	//! NOTE: CUDA textures are pixel centered, so we need a 0.5 units fixup
	//! on the x and y coordinates. This is done AUTOMATICALLY, and should not be done
	//! by the caller. This way:
	//! 1. the cell centering remains an implementation detail that can be changed
	//!    when supporting other platorms that use a different conventions
	//! 2. in the DISABLE_DEM_TEXTURE case it avoids a + 0.5f (on call) - 0.5f (on fetch)
	__device__ __forceinline__ float
	fetchDem(float x, float y) const
	{
#if DISABLE_DEM_TEXTURE
		const float xb = clamp(x, 0.f, width - 1.f);
		const float yb = clamp(y, 0.f, height - 1.f);
		// find the vertices of the square this point belongs to,
		// and ensure we are within the domain covered by the DEM
		// (outer points will be squashed to the edge values)
		const int    i   = floor(xb);
		const int    j   = floor(yb);
		const int    ip1 = clamp(i + 1, 0, width - 1);
		const int    jp1 = clamp(j + 1, 0, height - 1);
		const float pa  = xb - i;
		const float pb  = yb - j;
		const float ma  = 1 - pa;
		const float mb  = 1 - pb;
		const float z00 = ma*mb*demArray[i   + j   * width];
		const float z10 = pa*mb*demArray[ip1 + j   * width];
		const float z01 = ma*pb*demArray[i   + jp1 * width];
		const float z11 = pa*pb*demArray[ip1 + jp1 * width];

		return z00 + z10 + z01 + z11;
#else
		// CUDA textures are pixel centered, so we need to shift the requsted coordinate by 0.5f
		return tex2D<float>(demTex, x + 0.5f, y + 0.5f);
#endif
	}
};

#endif /* DEM_PARAMS_H */
