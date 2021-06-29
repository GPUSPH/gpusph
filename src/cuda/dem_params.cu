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

/*! \file dem_param.cu
 * Implementation of the instantiation and manipulation of the global DEM object
 */

#include <memory> // unique_ptr
#include <cstring> // memset

#include "dem_params.h"
#include "safe_call.h"

using namespace std;

//! The global dem_params object is not an actual dem_params.
//! It has a slightly different interface, because it's what manages the creation
//! (and destruction) of the CUDA array and texture objects associated with the DEM.
struct internal_dem_params
{
#if DISABLE_DEM_TEXTURE
	float *demArray;
	int width;
	int height;
#else
	cudaArray *dDem;
	cudaTextureObject_t demTex;
#endif

	internal_dem_params() :
#if DISABLE_DEM_TEXTURE
		demArray(NULL),
		width(0),
		height(0)
#else
		dDem(NULL),
		demTex()
#endif
	{}

	~internal_dem_params()
	{
#if DISABLE_DEM_TEXTURE
		if (demArray)
			SAFE_CALL(cudaFree(demArray));
#else
		if (!dDem)
			return; // nothing to do if DEM was not allocated
		SAFE_CALL(cudaDestroyTextureObject(demTex));
		SAFE_CALL(cudaFreeArray(dDem));
#endif
	}

	void setDEM(const float *hDem, int width, int height)
	{
		const size_t size = width*height*sizeof(float);
#if DISABLE_DEM_TEXTURE
		// TODO pitched allocation
		SAFE_CALL( cudaMalloc(&demArray, size));
		SAFE_CALL( cudaMemcpy(demArray, hDem, size, cudaMemcpyHostToDevice));
		this->width = width;
		this->height = height;
#else
		// Allocating, reading and copying DEM
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
		SAFE_CALL( cudaMallocArray( &dDem, &channelDesc, width, height ));
		SAFE_CALL( cudaMemcpyToArray( dDem, 0, 0, hDem, size, cudaMemcpyHostToDevice));

		cudaTextureDesc dem_tex_desc;
		memset(&dem_tex_desc, 0, sizeof(dem_tex_desc));

		dem_tex_desc.addressMode[0] = cudaAddressModeClamp;
		dem_tex_desc.addressMode[1] = cudaAddressModeClamp;
		dem_tex_desc.filterMode = cudaFilterModeLinear;
		dem_tex_desc.normalizedCoords = false;
		dem_tex_desc.readMode = cudaReadModeElementType;

		cudaResourceDesc dem_res_desc;
		dem_res_desc.resType = cudaResourceTypeArray;
		dem_res_desc.res.array.array = dDem;

		SAFE_CALL(cudaCreateTextureObject(&demTex, &dem_res_desc, &dem_tex_desc, NULL));
#endif
	}
};

thread_local unique_ptr<internal_dem_params> global_dem_params;

dem_params::dem_params() :
#if DISABLE_DEM_TEXTURE
	demArray(global_dem_params->demArray),
	width(global_dem_params->width),
	height(global_dem_params->height)
#else
	demTex(global_dem_params->demTex)
#endif
{}
