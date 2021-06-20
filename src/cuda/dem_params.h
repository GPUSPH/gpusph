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

//! ENABLE_DEM-related texture parameters
struct dem_params
{
private:
	cudaTextureObject_t demTex;

public:
	// There is a (thread-local) global dem_params object,
	// whose construction is managed separately.
	// All uses of dem_params as substructures otherwise
	// will simply copy-initialize from the global object.
	dem_params();

	//! Fetch the DEM content at position x, y.
	//! NOTE: x, y are in DEM-relative coordinates
	__device__ __forceinline__ float
	fetchDem(float x, float y) const
	{
		return tex2D<float>(demTex, x, y);
	}
};

#endif /* DEM_PARAMS_H */
