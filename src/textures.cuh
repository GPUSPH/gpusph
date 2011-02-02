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
