/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

// textures for particle position and velocity
texture<float4, 1, cudaReadModeElementType> posTex;		// position and mass
texture<float4, 1, cudaReadModeElementType> velTex;		// velocity and density
texture<float2, 1, cudaReadModeElementType> tau0Tex;		// velocity and density
texture<float2, 1, cudaReadModeElementType> tau1Tex;		// velocity and density
texture<float2, 1, cudaReadModeElementType> tau2Tex;		// velocity and density
texture<particleinfo, 1, cudaReadModeElementType> infoTex;	// info
texture<float, 1, cudaReadModeElementType> viscTex;		// viscosity for power law rheology

texture<uint, 1, cudaReadModeElementType> cellStartTex;		 // first particle index in cell table
texture<uint, 1, cudaReadModeElementType> cellEndTex;		 // first particle index in cell table
#endif
