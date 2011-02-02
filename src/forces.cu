#include <stdio.h>
#include "textures.cuh"
#include "forces.cuh"
// local_cudpp change:
#include "fmax.cuh"
//#include "local_cudpp.h"
#include "particledefine.h"

cudaArray*  dDem = NULL;

/* These defines give a shorthand for the kernel with a given correction,
   viscosity, xsph and dt options. They will be used in forces.cu for
   consistency */
#define _CUDA_KERNEL_NAME(visc, xsph, dt) forces_##visc##_##xsph##dt##Device
#define CUDA_KERNEL_NAME(visc, xsph, dt) _CUDA_KERNEL_NAME(visc, xsph, dt)

#include "forces_kernel.cu"

#define NOT_IMPLEMENTED_CHECK(what, arg) \
		default: \
			fprintf(stderr, #what " %s (%u) not implemented\n", what##Name[arg], arg); \
			exit(1)

#define KERNEL_CHECK(kernel, boundarytype, periodic, formulation, visc, dem) \
	case kernel: \
		if (!dtadapt && !xsphcorr) \
				CUDA_KERNEL_NAME(visc,,)<kernel, boundarytype, periodic, dem, formulation><<< numBlocks, numThreads >>>\
						(forces, neibsList, numParticles, slength, influenceradius); \
		else if (!dtadapt && xsphcorr) \
				CUDA_KERNEL_NAME(visc, Xsph,)<kernel, boundarytype, periodic, dem, formulation><<< numBlocks, numThreads >>>\
						(forces, xsph, neibsList, numParticles, slength, influenceradius); \
		else if (dtadapt && !xsphcorr) \
				CUDA_KERNEL_NAME(visc,, Dt)<kernel, boundarytype, periodic, dem, formulation><<< numBlocks, numThreads >>>\
						(forces, neibsList, numParticles, slength, influenceradius, cfl); \
		else if (dtadapt && xsphcorr) \
				CUDA_KERNEL_NAME(visc, Xsph, Dt)<kernel, boundarytype, periodic, dem, formulation><<< numBlocks, numThreads >>>\
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

// local_cudpp change: added custom getMax
/* float local_cudpp_getMax(int size, float* d_idata, float* d_tempdata) {
	float res;
	local_cudppMax(d_tempdata, d_idata, size, 0);
	const void * d_origin = ((void*)(d_tempdata  + size-1));
	CUDA_SAFE_CALL(cudaMemcpy((void*)(&res), d_origin, sizeof(float), cudaMemcpyDeviceToHost));
	return res;
} // */

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
		float*			visc,
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
		// local_cudpp change: use custom getMax
		float maxcfl = getMax(numPartsFmax, cfl, tempfmax);
		//float maxcfl = local_cudpp_getMax(numPartsFmax, cfl, tempfmax);

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
#undef _CUDA_KERNEL_NAME
#undef CUDA_KERNEL_NAME
