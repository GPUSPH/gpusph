/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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
#ifndef _TEXTURES_CUH_
#define _TEXTURES_CUH_

// On devices with compute capability 2.x, we want to distribute cache load
// between L1 cache and the texture cache. On older architectures (no L1 cache)
// we prefer using textures for all read-only arrays. Define PREFER_L1 to 1 or
// 0 accordingly. On 3.x, the L1 cache is only used for register spills, so
// exclude it from PREFER_L1. We keep PREFER_L1 on Maxwell because tests indicate
// that using textures leads to no improvement at best (and possibly some minor
// performance loss)

#if defined(__COMPUTE__)
#if __COMPUTE__ >= 20 && __COMPUTE__/10 != 3
#define PREFER_L1 1
#else
#define PREFER_L1 0
#endif
#endif

#include "particleinfo.h"

//! \name Textures for particle position, velocity, flags etc
/*! We basically need _at least_ a texture for each of the
 * non-ephemeral particle property, plus one for the ephemeral properties
 * which we must read from the neighbors, when leveraging the texture cache:
 * \todo ideally, we should define these programmatically, and even better define our own
 * derived
 *
 * template<flag_t Buffer>
 * struct bufferTexture : texture< BufferTraits<Buffer>::element_type > {};
 *
 * (SPS etc would need a slightly different treatment though)
 * OTOH, NVIDIA hardware is moving towards a shared L1/texture cache, and if this keeps up
 * we might get rid of textures altogether
 * @{
 */
texture<float4, 1, cudaReadModeElementType> posTex;		///< position and mass
texture<float4, 1, cudaReadModeElementType> velTex;		///< velocity and density
texture<float4, 1, cudaReadModeElementType> volTex;		///< volume
texture<float, 1, cudaReadModeElementType> energyTex;	///< internal energy
texture<float4, 1, cudaReadModeElementType> boundTex;	///< boundary elements
texture<float4, 1, cudaReadModeElementType> gamTex;		///< gradient gamma
texture<particleinfo, 1, cudaReadModeElementType> infoTex;	///< info
texture<vertexinfo, 1, cudaReadModeElementType> vertTex;	///< vertices
texture<float, 1, cudaReadModeElementType> keps_kTex;	///< k for k-e model
texture<float, 1, cudaReadModeElementType> keps_eTex;	///< e for k-e model
texture<float, 1, cudaReadModeElementType> tviscTex;	///< eddy viscosity
texture<float, 1, cudaReadModeElementType> effpresTex;	// effective pressure
texture<float4, 1, cudaReadModeElementType> eulerVelTex;	///< eulerian velocity and density

//! SPS matrix
/*! \todo these should probably be coalesced in a float4 + float2 texture
 * @{
 */
texture<float2, 1, cudaReadModeElementType> tau0Tex;
texture<float2, 1, cudaReadModeElementType> tau1Tex;
texture<float2, 1, cudaReadModeElementType> tau2Tex;
/** @} */

//! neib list management
/*! @{ */
texture<uint, 1, cudaReadModeElementType> cellStartTex;	///< first particle index in cell table
texture<uint, 1, cudaReadModeElementType> cellEndTex;	///< last particle index in cell table
/** @} */

/** @} */
#endif
