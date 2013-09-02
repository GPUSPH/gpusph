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
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

#include "textures.cuh"
#include "forces.cuh"

#include "cuda_call.h"

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

#define KERNEL_CHECK(kernel, boundarytype, formulation, visc, dem) \
	case kernel: \
		if (!dtadapt && !xsphcorr) \
				cuforces::FORCES_KERNEL_NAME(visc,,)<kernel, boundarytype, dem, formulation><<< numBlocks, numThreads, dummy_shared >>>\
						(pos, forces, particleHash, cellStart, neibsList, numParticles, slength, influenceradius, rbforces, rbtorques); \
		else if (!dtadapt && xsphcorr) \
				cuforces::FORCES_KERNEL_NAME(visc, Xsph,)<kernel, boundarytype, dem, formulation><<< numBlocks, numThreads, dummy_shared >>>\
						(pos, forces, particleHash, cellStart, xsph, neibsList, numParticles, slength, influenceradius, rbforces, rbtorques); \
		else if (dtadapt && !xsphcorr) \
				cuforces::FORCES_KERNEL_NAME(visc,, Dt)<kernel, boundarytype, dem, formulation><<< numBlocks, numThreads, dummy_shared >>>\
						(pos, forces, particleHash, cellStart, neibsList, numParticles, slength, influenceradius, rbforces, rbtorques, cfl); \
		else if (dtadapt && xsphcorr) \
				cuforces::FORCES_KERNEL_NAME(visc, Xsph, Dt)<kernel, boundarytype, dem, formulation><<< numBlocks, numThreads, dummy_shared >>>\
						(pos, forces, particleHash, cellStart, xsph, neibsList, numParticles, slength, influenceradius, rbforces, rbtorques, cfl); \
		break

#define KERNEL_SWITCH(formulation, boundarytype, visc, dem) \
	switch (kerneltype) { \
		KERNEL_CHECK(CUBICSPLINE,	boundarytype, formulation, visc, dem); \
		KERNEL_CHECK(WENDLAND,		boundarytype, formulation, visc, dem); \
		NOT_IMPLEMENTED_CHECK(Kernel, kerneltype); \
	}

#define FORMULATION_CHECK(formulation, boundarytype, visc, dem) \
	case formulation: \
		KERNEL_SWITCH(formulation, boundarytype, visc, dem) \
		break

#define FORMULATION_SWITCH(boundarytype, visc, dem) \
	switch (sph_formulation) { \
		FORMULATION_CHECK(SPH_F1, boundarytype, visc, dem); \
		FORMULATION_CHECK(SPH_F2, boundarytype, visc, dem); \
		NOT_IMPLEMENTED_CHECK(SPHFormulation, sph_formulation); \
	}

#define VISC_CHECK(boundarytype, visc, dem) \
	case visc: \
		FORMULATION_SWITCH(boundarytype, visc, dem) \
		break

#define VISC_CHECK_STANDARD(boundarytype, dem) \
		VISC_CHECK(boundarytype, ARTVISC, dem); \
		VISC_CHECK(boundarytype, DYNAMICVISC, dem); \
		VISC_CHECK(boundarytype, KINEMATICVISC, dem);\
		VISC_CHECK(boundarytype, SPSVISC, dem);


#define VISC_SWITCH(boundarytype, dem) \
	switch (visctype) { \
		VISC_CHECK_STANDARD(boundarytype, dem); \
		NOT_IMPLEMENTED_CHECK(Viscosity, visctype); \
	}

//		VISC_CHECK_SPS(boundarytype, dem); \

#define BOUNDARY_CHECK(boundary, dem) \
	case boundary: \
		VISC_SWITCH(boundary, dem) \
		break

#define BOUNDARY_SWITCH(dem) \
	switch (boundarytype) { \
		BOUNDARY_CHECK(LJ_BOUNDARY, dem); \
		BOUNDARY_CHECK(MK_BOUNDARY, dem); \
		NOT_IMPLEMENTED_CHECK(Boundary, boundarytype); \
	}

#define SPS_CHECK(kernel) \
	case kernel: \
		cuforces::SPSstressMatrixDevice<kernel><<< numBlocks, numThreads, dummy_shared >>> \
				(pos, tau[0], tau[1], tau[2], particleHash, cellStart, neibsList, numParticles, slength, influenceradius); \
		break

#define SHEPARD_CHECK(kernel) \
	case kernel: \
		cuforces::shepardDevice<kernel><<< numBlocks, numThreads, dummy_shared >>> \
				 (pos, newVel, particleHash, cellStart, neibsList, numParticles, slength, influenceradius); \
	break

#define MLS_CHECK(kernel) \
	case kernel: \
		cuforces::MlsDevice<kernel><<< numBlocks, numThreads, dummy_shared >>> \
				(pos, newVel, particleHash, cellStart, neibsList, numParticles, slength, influenceradius); \
	break

#define VORT_CHECK(kernel) \
	case kernel: \
		cuforces::calcVortDevice<kernel><<< numBlocks, numThreads >>> \
				 (vort, particleHash, cellStart, neibsList, numParticles, slength, influenceradius); \
	break

//Testpoints
#define TEST_CHECK(kernel) \
	case kernel: \
		cuforces::calcTestpointsVelocityDevice<kernel><<< numBlocks, numThreads >>> \
				(newVel, particleHash, cellStart, neibsList, numParticles, slength, influenceradius); \
	break

// Free surface detection
#define SURFACE_CHECK(kernel, savenormals) \
	case kernel: \
		cuforces::calcSurfaceparticleDevice<kernel, savenormals><<< numBlocks, numThreads >>> \
				(normals, newInfo, particleHash, cellStart, neibsList, numParticles, slength, influenceradius); \
	break


extern "C"
{
void
setforcesconstants(const SimParams & simparams, const PhysParams & physparams,
					const uint3 gridSize, const float3 cellSize, const uint numParticles)
{
	// Setting kernels and kernels derivative factors
	float h = simparams.slength;
	float h3 = h*h*h;
	float h4 = h3*h;
	float h5 = h4*h;
	float kernelcoeff = 1.0f/(M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_wcoeff_cubicspline, &kernelcoeff, sizeof(float)));
	kernelcoeff = 15.0f/(16.0f*M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_wcoeff_quadratic, &kernelcoeff, sizeof(float)));
	kernelcoeff = 21.0f/(16.0f*M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_wcoeff_wendland, &kernelcoeff, sizeof(float)));

	kernelcoeff = 3.0f/(4.0f*M_PI*h4);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_fcoeff_cubicspline, &kernelcoeff, sizeof(float)));
	kernelcoeff = 15.0f/(32.0f*M_PI*h4);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_fcoeff_quadratic, &kernelcoeff, sizeof(float)));
	kernelcoeff = 105.0f/(128.0f*M_PI*h5);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_fcoeff_wendland, &kernelcoeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_numfluids, &physparams.numFluids, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rho0, &physparams.rho0, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_bcoeff, &physparams.bcoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gammacoeff, &physparams.gammacoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_sscoeff, &physparams.sscoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_sspowercoeff, &physparams.sspowercoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gravity, &physparams.gravity, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_dcoeff, &physparams.dcoeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_p1coeff, &physparams.p1coeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_p2coeff, &physparams.p2coeff, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_MK_K, &physparams.MK_K, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_MK_d, &physparams.MK_d, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_MK_beta, &physparams.MK_beta, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_r0, &physparams.r0, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_visccoeff, &physparams.visccoeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_epsartvisc, &physparams.epsartvisc, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_ewres, &physparams.ewres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_nsres, &physparams.nsres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_demdx, &physparams.demdx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_demdy, &physparams.demdy, sizeof(float)));
	float demdxdy = physparams.demdx*physparams.demdy;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_demdxdy, &demdxdy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_demzmin, &physparams.demzmin, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_smagfactor, &physparams.smagfactor, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_kspsfactor, &physparams.kspsfactor, sizeof(float)));

	float partsurf = physparams.partsurf;
	if (partsurf == 0.0f)
		partsurf = physparams.r0*physparams.r0;
		// partsurf = (6.0 - M_PI)*physparams.r0*physparams.r0/4;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_partsurf, &partsurf, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_cosconeanglefluid, &physparams.cosconeanglefluid, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_cosconeanglenonfluid, &physparams.cosconeanglenonfluid, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_objectobjectdf, &physparams.objectobjectdf, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_objectboundarydf, &physparams.objectboundarydf, sizeof(float)));

	uint maxneibsnum_time_numparticles = simparams.maxneibsnum*numParticles;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_maxneibsnum_time_numparticles, &maxneibsnum_time_numparticles, sizeof(uint)));

	// Neibs cell to offset table
	char3 cell_to_offset[27];
	for(char z=-1; z<=1; z++) {
		for(char y=-1; y<=1; y++) {
			for(char x=-1; x<=1; x++) {
				int i = (x + 1) + (y + 1)*3 + (z + 1)*9;
				cell_to_offset[i] =  make_char3(x, y, z);
			}
		}
	}
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_cell_to_offset, cell_to_offset, 27*sizeof(char3)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gridSize, &gridSize, sizeof(uint3)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_cellSize, &cellSize, sizeof(float3)));
}


void
getforcesconstants(PhysParams & physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.numFluids, cuforces::d_numfluids, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.rho0, cuforces::d_rho0, MAX_FLUID_TYPES*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.gravity, cuforces::d_gravity, sizeof(float3), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.bcoeff, cuforces::d_bcoeff, MAX_FLUID_TYPES*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.gammacoeff, cuforces::d_gammacoeff, MAX_FLUID_TYPES*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.sscoeff, cuforces::d_sscoeff, MAX_FLUID_TYPES*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.sspowercoeff, cuforces::d_sspowercoeff, MAX_FLUID_TYPES*sizeof(float), 0));

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.dcoeff, cuforces::d_dcoeff, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.p1coeff, cuforces::d_p1coeff, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.p2coeff, cuforces::d_p2coeff, sizeof(float), 0));

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.MK_K, cuforces::d_MK_K, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.MK_d, cuforces::d_MK_d, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.MK_beta, cuforces::d_MK_beta, sizeof(float), 0));

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.r0, cuforces::d_r0, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.visccoeff, cuforces::d_visccoeff, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.epsartvisc, cuforces::d_epsartvisc, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.ewres, cuforces::d_ewres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.nsres, cuforces::d_nsres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.demdx, cuforces::d_demdx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.demdy, cuforces::d_demdy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.demzmin, cuforces::d_demzmin, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.smagfactor, cuforces::d_smagfactor, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams.kspsfactor, cuforces::d_kspsfactor, sizeof(float)));
}


void
setplaneconstants(int numPlanes, float* PlanesDiv, float4* Planes)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_planes, Planes, numPlanes*sizeof(float4)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_plane_div, PlanesDiv, numPlanes*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_numplanes, &numPlanes, sizeof(uint)));
}


void
setgravity(float3 gravity)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gravity, &gravity, sizeof(float3)));
}


void
setforcesrbcg(float3* cg, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rbcg, cg, numbodies*sizeof(float3)));
}


void
setforcesrbstart(uint* rbfirstindex, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rbstartindex, rbfirstindex, numbodies*sizeof(uint)));
}


float
forces(	float4*			pos,
		float4*			vel,
		float4*			forces,
		float4*			rbforces,
		float4*			rbtorques,
		float4*			xsph,
		particleinfo	*info,
		uint*			particleHash,
		uint*			cellStart,
		neibdata*		neibsList,
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
		float*			tempCfl,
		uint			numPartsFmax,
		float2*			tau[],
		SPHFormulation	sph_formulation,
		BoundaryType	boundarytype,
		bool			usedem)
{
	int dummy_shared = 0;
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel for computing SPS stress matrix, if needed
	if (visctype == SPSVISC) {	// thread per particle
		int numThreads = min(BLOCK_SIZE_SPS, numParticles);
		int numBlocks = (int) ceil(numParticles / (float) numThreads);
		#if (__COMPUTE__ == 20)
		dummy_shared = 2560;
		#endif
		switch (kerneltype) {
			SPS_CHECK(CUBICSPLINE);
			SPS_CHECK(QUADRATIC);
			SPS_CHECK(WENDLAND);
		}

		// check if kernel invocation generated an error
		CUT_CHECK_ERROR("SPS kernel execution failed");
		
		CUDA_SAFE_CALL(cudaBindTexture(0, tau0Tex, tau[0], numParticles*sizeof(float2)));
		CUDA_SAFE_CALL(cudaBindTexture(0, tau1Tex, tau[1], numParticles*sizeof(float2)));
		CUDA_SAFE_CALL(cudaBindTexture(0, tau2Tex, tau[2], numParticles*sizeof(float2)));
	}
	
	// thread per particle
	int numThreads = min(BLOCK_SIZE_FORCES, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);		
	#if (__COMPUTE__ == 20)
	if (visctype == SPSVISC)
		dummy_shared = 3328 - dtadapt*BLOCK_SIZE_FORCES*4;
	else
		dummy_shared = 2560 - dtadapt*BLOCK_SIZE_FORCES*4;
	#endif
	if (usedem)
		BOUNDARY_SWITCH(true)
	else
		BOUNDARY_SWITCH(false)

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Forces kernel execution failed");

	if (visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaUnbindTexture(tau0Tex));
		CUDA_SAFE_CALL(cudaUnbindTexture(tau1Tex));
		CUDA_SAFE_CALL(cudaUnbindTexture(tau2Tex));
	}
	
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	if (dtadapt) {
		float maxcfl = cflmax(numPartsFmax, cfl, tempCfl);
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
shepard(float4*		pos,
		float4*		oldVel,
		float4*		newVel,
		particleinfo	*info,
		uint*		particleHash,
		uint*		cellStart,
		neibdata*	neibsList,
		uint		numParticles,
		float		slength,
		int			kerneltype,
		float		influenceradius)
{
	int dummy_shared = 0;
	// thread per particle
	int numThreads = min(BLOCK_SIZE_SHEPARD, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
	
	// execute the kernel
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif
	switch (kerneltype) {
		SHEPARD_CHECK(CUBICSPLINE);
//		SHEPARD_CHECK(QUADRATIC);
		SHEPARD_CHECK(WENDLAND);
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Shepard kernel execution failed");
	
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

}


void
mls(float4*		pos,
	float4*		oldVel,
	float4*		newVel,
	particleinfo	*info,
	uint*		particleHash,
	uint*		cellStart,
	neibdata*	neibsList,
	uint		numParticles,
	float		slength,
	int			kerneltype,
	float		influenceradius)
{
	int dummy_shared = 0;
	// thread per particle
	int numThreads = min(BLOCK_SIZE_MLS, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel		
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif
	switch (kerneltype) {
		MLS_CHECK(CUBICSPLINE);
//		MLS_CHECK(QUADRATIC);
		MLS_CHECK(WENDLAND);
	}
	
	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Mls kernel execution failed");

	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
}

void
vorticity(	float4*		pos,
			float4*		vel,
			float3*		vort,
			particleinfo	*info,
			uint*		particleHash,
			uint*		cellStart,
			neibdata*	neibsList,
			uint		numParticles,
			float		slength,
			int			kerneltype,
			float		influenceradius)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_CALCVORT, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	switch (kerneltype) {
		VORT_CHECK(CUBICSPLINE);
//		VORT_CHECK(QUADRATIC);
		VORT_CHECK(WENDLAND);
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Shepard kernel execution failed");
	
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
}

//Testpoints
void
testpoints( float4*		pos,
			float4*		newVel,
			particleinfo	*info,
			uint*		particleHash,
			uint*		cellStart,
			neibdata*	neibsList,
			uint		numParticles,
			float		slength,
			int			kerneltype,
			float		influenceradius)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_CALCTEST, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, newVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	switch (kerneltype) {
		TEST_CHECK(CUBICSPLINE);
//		TEST_CHECK(QUADRATIC);
		TEST_CHECK(WENDLAND);
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("test kernel execution failed");
	
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
}

// Free surface detection
void
surfaceparticle(	float4*		pos,
					float4*     vel,
					float4*		normals,
					particleinfo	*info,
					particleinfo	*newInfo,
					uint*		particleHash,
					uint*		cellStart,
					neibdata*	neibsList,
					uint		numParticles,
					float		slength,
					int			kerneltype,
					float		influenceradius,
					bool        savenormals)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_CALCTEST, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	if (savenormals){
			switch (kerneltype) {
				SURFACE_CHECK(CUBICSPLINE, true);
//				SURFACE_CHECK(QUADRATIC, true);
				SURFACE_CHECK(WENDLAND, true);
			}
		}
	else {
			switch (kerneltype) {
				SURFACE_CHECK(CUBICSPLINE, false);
//				SURFACE_CHECK(QUADRATIC, false);
				SURFACE_CHECK(WENDLAND, false);
			}
		}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("surface kernel execution failed");
	
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
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


void reduceRbForces(float4*		forces,
					float4*		torques,
					uint*		rbnum,
					uint*		lastindex,
					float3*		totalforce,
					float3*		totaltorque,
					uint		numbodies,
					uint		numBodiesParticles)
{
	thrust::device_ptr<float4> forces_devptr = thrust::device_pointer_cast(forces);
	thrust::device_ptr<float4> torques_devptr = thrust::device_pointer_cast(torques);
	thrust::device_ptr<uint> rbnum_devptr = thrust::device_pointer_cast(rbnum);
	thrust::equal_to<uint> binary_pred;
	thrust::plus<float4> binary_op;

	thrust::inclusive_scan_by_key(rbnum_devptr, rbnum_devptr + numBodiesParticles, 
				forces_devptr, forces_devptr, binary_pred, binary_op);
	thrust::inclusive_scan_by_key(rbnum_devptr, rbnum_devptr + numBodiesParticles, 
				torques_devptr, torques_devptr, binary_pred, binary_op);
	
	for (int i = 0; i < numbodies; i++) {
		float4 temp;
		void * ddata = (void *) (forces + lastindex[i]);
		CUDA_SAFE_CALL(cudaMemcpy((void *) &temp, ddata, sizeof(float4), cudaMemcpyDeviceToHost));
		totalforce[i] = as_float3(temp);
		
		ddata = (void *) (torques + lastindex[i]);
		CUDA_SAFE_CALL(cudaMemcpy((void *) &temp, ddata, sizeof(float4), cudaMemcpyDeviceToHost));
		totaltorque[i] = as_float3(temp);
		}
}


void 
reducefmax(	const int	size, 
			const int	threads, 
			const int	blocks, 
			float		*d_idata, 
			float		*d_odata)
{
	dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, 1, 1);

	// when there is only one warp per block, we need to allocate two warps 
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
	{
		case 512:
			cuforces::fmaxDevice<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 256:
			cuforces::fmaxDevice<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 128:
			cuforces::fmaxDevice<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 64:
			cuforces::fmaxDevice<64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 32:
			cuforces::fmaxDevice<32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case 16:
			cuforces::fmaxDevice<16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  8:
			cuforces::fmaxDevice<8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  4:
			cuforces::fmaxDevice<4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  2:
			cuforces::fmaxDevice<2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
		case  1:
			cuforces::fmaxDevice<1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size); break;
	}
}


uint nextPow2(uint x ) 
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


#define MIN(x,y) ((x < y) ? x : y)
void getNumBlocksAndThreads(const uint	n, 
							const uint	maxBlocks, 
							const uint	maxThreads, 
							uint		&blocks, 
							uint		&threads)
{
	threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
	blocks = (n + (threads * 2 - 1)) / (threads * 2);
	blocks = MIN(maxBlocks, blocks);
}


uint
getNumPartsFmax(const uint n)
{
	return (int) ceil(n / (float) min(BLOCK_SIZE_FORCES, n));
}
	

uint
getFmaxTempStorageSize(const uint n)
{
	uint numBlocks, numThreads;
	getNumBlocksAndThreads(n, MAX_BLOCKS_FMAX, BLOCK_SIZE_FMAX, numBlocks, numThreads);
	return numBlocks*sizeof(float);
}


float
cflmax( const uint	n,
		float*		cfl,
		float*		tempCfl)
{
	uint numBlocks = 0;
	uint numThreads = 0;
	float max = 0.0f;
	
	getNumBlocksAndThreads(n, MAX_BLOCKS_FMAX, BLOCK_SIZE_FMAX, numBlocks, numThreads);
		
	// execute the kernel
	reducefmax(n, numThreads, numBlocks, cfl, tempCfl);

	// check if kernel execution generated an error
	CUT_CHECK_ERROR("fmax kernel execution failed");

	uint s = numBlocks;
	while(s > 1) 
	{
		uint threads = 0, blocks = 0;
		getNumBlocksAndThreads(s, MAX_BLOCKS_FMAX, BLOCK_SIZE_FMAX, blocks, threads);

		reducefmax(s, threads, blocks, tempCfl, tempCfl);
		CUT_CHECK_ERROR("fmax kernel execution failed");

		s = (s + (threads*2-1)) / (threads*2);
	}

	CUDA_SAFE_CALL(cudaMemcpy(&max, tempCfl, sizeof(float), cudaMemcpyDeviceToHost));
	
	return max;
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
#undef TEST_CHECK
#undef SURFACE_CHECK

/* These were defined in forces_kernel.cu */
#undef _FORCES_KERNEL_NAME
#undef FORCES_KERNEL_NAME
