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

#include <stdio.h>
#include "cudpp/cudpp.h"

#include "textures.cuh"
#include "forces.cuh"
#include "particledefine.h"

cudaArray*  dDem = NULL;

/* These defines give a shorthand for the kernel with a given correction,
   viscosity, xsph and dt options. They will be used in forces.cu for
   consistency */
#define _FORCES_KERNEL_NAME(visc, xsph, dt) forces_##visc##_##xsph##dt##Device
#define FORCES_KERNEL_NAME(visc, xsph, dt) _FORCES_KERNEL_NAME(visc, xsph, dt)

#include "forces_kernel.cu"

#define NOT_IMPLEMENTED_CHECK(what, arg) \
		default: \
			fprintf(stderr, #what " %s (%u) not implemented\n", what##Name[arg], arg); \
			exit(1)

#define KERNEL_CHECK(kernel, boundarytype, periodic, formulation, visc, dem) \
	case kernel: \
		if (!dtadapt && !xsphcorr) \
				FORCES_KERNEL_NAME(visc,,)<kernel, boundarytype, periodic, dem, formulation><<< numBlocks, numThreads >>>\
						(forces, neibsList, numParticles, slength, influenceradius); \
		else if (!dtadapt && xsphcorr) \
				FORCES_KERNEL_NAME(visc, Xsph,)<kernel, boundarytype, periodic, dem, formulation><<< numBlocks, numThreads >>>\
						(forces, xsph, neibsList, numParticles, slength, influenceradius); \
		else if (dtadapt && !xsphcorr) \
				FORCES_KERNEL_NAME(visc,, Dt)<kernel, boundarytype, periodic, dem, formulation><<< numBlocks, numThreads >>>\
						(forces, neibsList, numParticles, slength, influenceradius, cfl); \
		else if (dtadapt && xsphcorr) \
				FORCES_KERNEL_NAME(visc, Xsph, Dt)<kernel, boundarytype, periodic, dem, formulation><<< numBlocks, numThreads >>>\
						(forces, xsph, neibsList, numParticles, slength, influenceradius, cfl); \
		break

#define KERNEL_SWITCH(formulation, boundarytype, periodic, visc, dem) \
	switch (kerneltype) { \
		KERNEL_CHECK(CUBICSPLINE,	boundarytype, periodic, formulation, visc, dem); \
		KERNEL_CHECK(QUADRATIC,		boundarytype, periodic, formulation, visc, dem); \
		KERNEL_CHECK(WENDLAND,		boundarytype, periodic, formulation, visc, dem); \
		NOT_IMPLEMENTED_CHECK(Kernel, kerneltype); \
	}

#define FORMULATION_CHECK(formulation, boundarytype, periodic, visc, dem) \
	case formulation: \
		KERNEL_SWITCH(formulation, boundarytype, periodic, visc, dem) \
		break

#define FORMULATION_SWITCH(boundarytype, periodic, visc, dem) \
	switch (sph_formulation) { \
		FORMULATION_CHECK(SPH_F1, boundarytype, periodic, visc, dem); \
		FORMULATION_CHECK(SPH_F2, boundarytype, periodic, visc, dem); \
		NOT_IMPLEMENTED_CHECK(SPHFormulation, sph_formulation); \
	}

#define VISC_CHECK(boundarytype, periodic, visc, dem) \
	case visc: \
		FORMULATION_SWITCH(boundarytype, periodic, visc, dem) \
		break

#define VISC_CHECK_STANDARD(boundarytype, periodic, dem) \
		VISC_CHECK(boundarytype, periodic, ARTVISC, dem); \
		VISC_CHECK(boundarytype, periodic, DYNAMICVISC, dem); \
		VISC_CHECK(boundarytype, periodic, KINEMATICVISC, dem);\
		VISC_CHECK(boundarytype, periodic, SPSVISC, dem);


#define VISC_SWITCH(boundarytype, periodic, dem) \
	switch (visctype) { \
		VISC_CHECK_STANDARD(boundarytype, periodic, dem); \
		NOT_IMPLEMENTED_CHECK(Viscosity, visctype); \
	}

//		VISC_CHECK_SPS(boundarytype, periodic, dem); \

#define BOUNDARY_CHECK(boundary, periodic, dem) \
	case boundary: \
		VISC_SWITCH(boundary, periodic, dem) \
		break

#define BOUNDARY_SWITCH(periodic, dem) \
	switch (boundarytype) { \
		BOUNDARY_CHECK(LJ_BOUNDARY, periodic, dem); \
		BOUNDARY_CHECK(MK_BOUNDARY, periodic, dem); \
		NOT_IMPLEMENTED_CHECK(Boundary, boundarytype); \
	}

#define SPS_CHECK(kernel, periodic) \
	case kernel: \
		SPSstressMatrixDevice<kernel, periodic><<< numBlocks, numThreads >>> \
				(tau[0], tau[1], tau[2], neibsList, numParticles, slength, influenceradius); \
		break

#define XSPH_CHECK(kernel, periodic) \
	case kernel: \
		xsphDevice<kernel, periodic><<< numBlocks, numThreads >>> \
				(xsph, neibsList, numParticles, slength, influenceradius); \
	break

#define SHEPARD_CHECK(kernel, periodic) \
	case kernel: \
		shepardDevice<kernel, periodic><<< numBlocks, numThreads >>> \
				 (newVel, neibsList, numParticles, slength, influenceradius); \
	break

#define MLS_CHECK(kernel, periodic) \
	case kernel: \
		MlsDevice<kernel, periodic><<< numBlocks, numThreads >>> \
				(newVel, neibsList, numParticles, slength, influenceradius); \
	break

#define VORT_CHECK(kernel, periodic) \
	case kernel: \
		calcVortDevice<kernel, periodic><<< numBlocks, numThreads >>> \
				 (vort, neibsList, numParticles, slength, influenceradius); \
	break

extern "C"
{

float
forces(	float4*			pos,
		float4*			vel,
		float4*			forces,
		float4*			xsph,
		particleinfo	*info,
		uint*			neibsList,
		uint			numParticles,
		float			slength,
		float			dt,
		bool			dtadapt,
		float			dtadaptfactor,
		bool			xsphcorr,
		KernelType		kerneltype,
		float			influenceradius,
		ViscosityType	visctype,
		float			visccoeff,
		float*			cfl,
		float*			tempfmax,
		uint			numPartsFmax,
		CUDPPHandle		scanplan,
		float2*			tau[],
		bool			periodicbound,
		SPHFormulation	sph_formulation,
		BoundaryType	boundarytype,
		bool			usedem)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_FORCES, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel for computing SPS stress matrix, if needed
	if (visctype == SPSVISC) {
		if (periodicbound) {
			switch (kerneltype) {
				SPS_CHECK(CUBICSPLINE, true);
				SPS_CHECK(QUADRATIC, true);
				SPS_CHECK(WENDLAND, true);
			}
			// check if kernel invocation generated an error
			CUT_CHECK_ERROR("SPS kernel execution failed");
		} else {
			switch (kerneltype) {
				SPS_CHECK(CUBICSPLINE, false);
				SPS_CHECK(QUADRATIC, false);
				SPS_CHECK(WENDLAND, false);
			}
			// check if kernel invocation generated an error
			CUT_CHECK_ERROR("SPS visc kernel execution failed");
		}
		CUDA_SAFE_CALL(cudaBindTexture(0, tau0Tex, tau[0], numParticles*sizeof(float2)));
		CUDA_SAFE_CALL(cudaBindTexture(0, tau1Tex, tau[1], numParticles*sizeof(float2)));
		CUDA_SAFE_CALL(cudaBindTexture(0, tau2Tex, tau[2], numParticles*sizeof(float2)));
	}

	if (usedem) {
		if (periodicbound) {
			BOUNDARY_SWITCH(true, true)
		} else {
			BOUNDARY_SWITCH(false, true)
		}
	} else {
		if (periodicbound) {
			BOUNDARY_SWITCH(true, false)
		} else {
			BOUNDARY_SWITCH(false, false)
		}
	}
	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Forces kernel execution failed");

	if (visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaUnbindTexture(tau0Tex));
		CUDA_SAFE_CALL(cudaUnbindTexture(tau1Tex));
		CUDA_SAFE_CALL(cudaUnbindTexture(tau2Tex));
	}

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	if (dtadapt) {
		cudppScan(scanplan, tempfmax, cfl, numPartsFmax);
		float maxcfl = 0;
		CUDA_SAFE_CALL(cudaMemcpy((void*)(&maxcfl), tempfmax, sizeof(float), cudaMemcpyDeviceToHost));
		
		dt = dtadaptfactor*sqrtf(slength/maxcfl);

		if (visctype != ARTVISC) {
			/* Stability condition from viscosity h²/ν */
			float dt_visc = slength*slength/visccoeff;
			switch (visctype) {
				case KINEMATICVISC:
				case SPSVISC:
				/* ν = visccoeff/4 for kinematic viscosity */
					dt_visc *= 4;
					break;

				case DYNAMICVISC:
				/* ν = visccoeff for dynamic viscosity */
					break;
				}
			dt_visc *= 0.125;
			if (dt_visc < dt)
				dt = dt_visc;
		}
	}
	return dt;
}


void
xsph(	float4*		pos,
		float4*		vel,
		float4*		forces,
		float4*		xsph,
		particleinfo	*info,
		uint*		neibsList,
		uint		numParticles,
		float		slength,
		int			kerneltype,
		float		influenceradius,
		bool		periodicbound)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_FORCES, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));

	// execute the kernel
	if (periodicbound) {
		switch (kerneltype) {
			XSPH_CHECK(CUBICSPLINE, true);
			XSPH_CHECK(QUADRATIC, true);
			XSPH_CHECK(WENDLAND, true);
		}
	} else {
		switch (kerneltype) {
			XSPH_CHECK(CUBICSPLINE, false);
			XSPH_CHECK(QUADRATIC, false);
			XSPH_CHECK(WENDLAND, false);
		}
	}

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Xsph kernel execution failed");
}


void
shepard(float4*		pos,
		float4*		oldVel,
		float4*		newVel,
		particleinfo	*info,
		uint*		neibsList,
		uint		numParticles,
		float		slength,
		int			kerneltype,
		float		influenceradius,
		bool		periodicbound)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_SHEPARD, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	if (periodicbound) {
		switch (kerneltype) {
			SHEPARD_CHECK(CUBICSPLINE, true);
			SHEPARD_CHECK(QUADRATIC, true);
			SHEPARD_CHECK(WENDLAND, true);
		}
	} else {
		switch (kerneltype) {
			SHEPARD_CHECK(CUBICSPLINE, false);
			SHEPARD_CHECK(QUADRATIC, false);
			SHEPARD_CHECK(WENDLAND, false);
		}
	}

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Shepard kernel execution failed");
}


void
mls(float4*		pos,
	float4*		oldVel,
	float4*		newVel,
	particleinfo	*info,
	uint*		neibsList,
	uint		numParticles,
	float		slength,
	int			kerneltype,
	float		influenceradius,
	bool		periodicbound)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_MLS, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	if (periodicbound) {
		switch (kerneltype) {
			MLS_CHECK(CUBICSPLINE, true);
			MLS_CHECK(QUADRATIC, true);
			MLS_CHECK(WENDLAND, true);
		}
	} else {
		switch (kerneltype) {
			MLS_CHECK(CUBICSPLINE, false);
			MLS_CHECK(QUADRATIC, false);
			MLS_CHECK(WENDLAND, false);
		}
	}


	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Mls kernel execution failed");
}

void
vorticity(	float4*		pos,
			float4*		vel,
			float3*		vort,
			particleinfo	*info,
			uint*		neibsList,
			uint		numParticles,
			float		slength,
			int			kerneltype,
			float		influenceradius,
			bool		periodicbound)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_CALCVORT, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	if (periodicbound) {
		switch (kerneltype) {
			VORT_CHECK(CUBICSPLINE, true);
			VORT_CHECK(QUADRATIC, true);
			VORT_CHECK(WENDLAND, true);
		}
	} else {
		switch (kerneltype) {
			VORT_CHECK(CUBICSPLINE, false);
			VORT_CHECK(QUADRATIC, false);
			VORT_CHECK(WENDLAND, false);
		}
	}

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Shepard kernel execution failed");
}


void setDemTexture(float *hDem, int width, int height)
{
	// Allocating, reading and copying DEM
	unsigned int size = width*height*sizeof(float);
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	CUDA_SAFE_CALL( cudaMallocArray( &dDem, &channelDesc, width, height ));
	CUDA_SAFE_CALL( cudaMemcpyToArray( dDem, 0, 0, hDem, size, cudaMemcpyHostToDevice));

	demTex.addressMode[0] = cudaAddressModeClamp;
	demTex.addressMode[1] = cudaAddressModeClamp;
	demTex.filterMode = cudaFilterModeLinear;
	demTex.normalized = false;

	CUDA_SAFE_CALL( cudaBindTextureToArray(demTex, dDem, channelDesc));
}


void releaseDemTexture()
{
	CUDA_SAFE_CALL(cudaFreeArray(dDem));
}

} // extern "C"

#undef KERNEL_CHECK
#undef KERNEL_SWITCH
#undef VISC_CHECK
#undef VISC_SWITCH
#undef XSPH_CHECK
#undef SHEPARD_CHECK
#undef MLS_CHECK
#undef SPS_CHECK
#undef VORT_CHECK

/* These were defined in forces_kernel.cu */
#undef _FORCES_KERNEL_NAME
#undef FORCES_KERNEL_NAME
