/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

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
#ifndef _TEXTURES_CUH_
#define _TEXTURES_CUH_

#include "particledefine.h"

// textures for particle position, velocity and flags
texture<float4, 1, cudaReadModeElementType> posTex;		// position and mass
texture<float4, 1, cudaReadModeElementType> velTex;		// velocity and density
texture<float4, 1, cudaReadModeElementType> volTex;		// volume
texture<float4, 1, cudaReadModeElementType> boundTex;		// boundary elements
texture<float4, 1, cudaReadModeElementType> gamTex;		// gradient gamma
texture<particleinfo, 1, cudaReadModeElementType> infoTex;	// info
texture<vertexinfo, 1, cudaReadModeElementType> vertTex;	// vertices
texture<float, 1, cudaReadModeElementType> keps_kTex;	// k for k-e model
texture<float, 1, cudaReadModeElementType> keps_eTex;	// e for k-e model
texture<float, 1, cudaReadModeElementType> tviscTex;	// eddy viscosity
texture<float4, 1, cudaReadModeElementType> eulerVelTex;		// eulerian velocity and density

// SPS matrix
// TODO these should probably be coalesced in a float4 + float2 texture
texture<float2, 1, cudaReadModeElementType> tau0Tex;
texture<float2, 1, cudaReadModeElementType> tau1Tex;
texture<float2, 1, cudaReadModeElementType> tau2Tex;

// neib list management
texture<uint, 1, cudaReadModeElementType> cellStartTex;		 // first particle index in cell table
texture<uint, 1, cudaReadModeElementType> cellEndTex;		 // last particle index in cell table
#endif
