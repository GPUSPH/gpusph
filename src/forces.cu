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

#include <stdio.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/functional.h>

#include "textures.cuh"
#include "forces.cuh"

#include "utils.h"
#include "cuda_call.h"

cudaArray*  dDem = NULL;

/* Auxiliary data for parallel reductions */
size_t	reduce_blocks = 0;
size_t	reduce_blocksize_max = 0;
size_t	reduce_bs2 = 0;
size_t	reduce_shmem_max = 0;
void*	reduce_buffer = NULL;

/* These defines give a shorthand for the kernel with a given correction,
   viscosity, xsph and dt options. They will be used in forces.cu for
   consistency */
#define _FORCES_KERNEL_NAME(visc, xsph, dt) forces_##visc##_##xsph##dt##Device
#define FORCES_KERNEL_NAME(visc, xsph, dt) _FORCES_KERNEL_NAME(visc, xsph, dt)

#define _FORCE_PAIR_NAME(visc, xsph) force_pair_##visc##_##xsph
#define FORCE_PAIR_NAME(visc, xsph) _FORCE_PAIR_NAME(visc, xsph)

#include "forces_kernel.cu"

#define NOT_IMPLEMENTED_CHECK(what, arg) \
		default: \
			fprintf(stderr, #what " %s (%u) not implemented\n", what##Name[arg], arg); \
			exit(1)

#define KERNEL_CHECK(kernel, boundarytype, formulation, visc, dem) \
	case kernel: \
		if (!dtadapt && !xsphcorr) \
				cuforces::FORCES_KERNEL_NAME(visc,,)<kernel, boundarytype, dem, formulation><<< numBlocks, numThreads, dummy_shared >>>\
						(pos, vertPos0, vertPos1, vertPos2, newGGam, forces, keps_dkde, turbvisc, particleHash, cellStart, neibsList, particleRangeEnd, deltap, slength, influenceradius, epsilon, movingBoundaries, rbforces, rbtorques); \
		else if (!dtadapt && xsphcorr) \
				cuforces::FORCES_KERNEL_NAME(visc, Xsph,)<kernel, boundarytype, dem, formulation><<< numBlocks, numThreads, dummy_shared >>>\
						(pos, vertPos0, vertPos1, vertPos2, newGGam, forces, keps_dkde, turbvisc, particleHash, cellStart, xsph, neibsList, particleRangeEnd, deltap, slength, influenceradius, epsilon, movingBoundaries, rbforces, rbtorques); \
		else if (dtadapt && !xsphcorr) \
				cuforces::FORCES_KERNEL_NAME(visc,, Dt)<kernel, boundarytype, dem, formulation><<< numBlocks, numThreads, dummy_shared >>>\
						(pos, vertPos0, vertPos1, vertPos2, newGGam, forces, keps_dkde, turbvisc, particleHash, cellStart, neibsList, particleRangeEnd, deltap, slength, influenceradius, epsilon, movingBoundaries, rbforces, rbtorques, cfl, cflTVisc); \
		else if (dtadapt && xsphcorr) \
				cuforces::FORCES_KERNEL_NAME(visc, Xsph, Dt)<kernel, boundarytype, dem, formulation><<< numBlocks, numThreads, dummy_shared >>>\
						(pos, vertPos0, vertPos1, vertPos2, newGGam, forces, keps_dkde, turbvisc, particleHash, cellStart, xsph, neibsList, particleRangeEnd, deltap, slength, influenceradius, epsilon, movingBoundaries, rbforces, rbtorques, cfl, cflTVisc); \
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
		VISC_CHECK(boundarytype, SPSVISC, dem); \
		VISC_CHECK(boundarytype, KEPSVISC, dem);

#define VISC_SWITCH(boundarytype, dem) \
	switch (visctype) { \
		VISC_CHECK_STANDARD(boundarytype, dem); \
		NOT_IMPLEMENTED_CHECK(Viscosity, visctype); \
	}

#define BOUNDARY_CHECK(boundary, dem) \
	case boundary: \
		VISC_SWITCH(boundary, dem) \
		break

#define BOUNDARY_SWITCH(dem) \
	switch (boundarytype) { \
		BOUNDARY_CHECK(LJ_BOUNDARY, dem); \
		BOUNDARY_CHECK(MK_BOUNDARY, dem); \
		BOUNDARY_CHECK(SA_BOUNDARY, dem); \
		NOT_IMPLEMENTED_CHECK(Boundary, boundarytype); \
	}

#define SPS_CHECK(kernel) \
	case kernel: \
		cuforces::SPSstressMatrixDevice<kernel><<< numBlocks, numThreads, dummy_shared >>> \
				(pos, tau[0], tau[1], tau[2], particleHash, cellStart, neibsList, particleRangeEnd, slength, influenceradius); \
		break

#define SHEPARD_CHECK(kernel) \
	case kernel: \
		cuforces::shepardDevice<kernel><<< numBlocks, numThreads, dummy_shared >>> \
				 (pos, newVel, particleHash, cellStart, neibsList, particleRangeEnd, slength, influenceradius); \
	break

#define MLS_CHECK(kernel) \
	case kernel: \
		cuforces::MlsDevice<kernel><<< numBlocks, numThreads, dummy_shared >>> \
				(pos, newVel, particleHash, cellStart, neibsList, particleRangeEnd, slength, influenceradius); \
	break

#define VORT_CHECK(kernel) \
	case kernel: \
		cuforces::calcVortDevice<kernel><<< numBlocks, numThreads >>> \
				 (pos, vort, particleHash, cellStart, neibsList, particleRangeEnd, slength, influenceradius); \
	break

//Testpoints
#define TEST_CHECK(kernel) \
	case kernel: \
		cuforces::calcTestpointsVelocityDevice<kernel><<< numBlocks, numThreads >>> \
				(pos, newVel, particleHash, cellStart, neibsList, particleRangeEnd, slength, influenceradius); \
	break

// Free surface detection
#define SURFACE_CHECK(kernel, savenormals) \
	case kernel: \
		cuforces::calcSurfaceparticleDevice<kernel, savenormals><<< numBlocks, numThreads >>> \
				(pos, normals, newInfo, particleHash, cellStart, neibsList, particleRangeEnd, slength, influenceradius); \
	break

#define DYNBOUNDARY_CHECK(kernel) \
	case kernel: \
		cuforces::dynamicBoundConditionsDevice<kernel><<< numBlocks, numThreads, dummy_shared >>> \
				 (oldPos, oldVel, oldTKE, oldEps, newGam, particleHash, cellStart, neibsList, particleRangeEnd, deltap, slength, influenceradius, initStep); \
	break

extern "C"
{
void
setforcesconstants(const SimParams *simparams, const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
	idx_t const& allocatedParticles)
{
	// Setting kernels and kernels derivative factors
	float h = simparams->slength;
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
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_numfluids, &physparams->numFluids, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rho0, &physparams->rho0, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_bcoeff, &physparams->bcoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gammacoeff, &physparams->gammacoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_sscoeff, &physparams->sscoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_sspowercoeff, &physparams->sspowercoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gravity, &physparams->gravity, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_dcoeff, &physparams->dcoeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_p1coeff, &physparams->p1coeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_p2coeff, &physparams->p2coeff, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_MK_K, &physparams->MK_K, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_MK_d, &physparams->MK_d, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_MK_beta, &physparams->MK_beta, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_r0, &physparams->r0, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_visccoeff, &physparams->visccoeff, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_epsartvisc, &physparams->epsartvisc, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_ewres, &physparams->ewres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_nsres, &physparams->nsres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_demdx, &physparams->demdx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_demdy, &physparams->demdy, sizeof(float)));
	float demdxdy = physparams->demdx*physparams->demdy;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_demdxdy, &demdxdy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_demzmin, &physparams->demzmin, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_smagfactor, &physparams->smagfactor, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_kspsfactor, &physparams->kspsfactor, sizeof(float)));

	float partsurf = physparams->partsurf;
	if (partsurf == 0.0f)
		partsurf = physparams->r0*physparams->r0;
		// partsurf = (6.0 - M_PI)*physparams->r0*physparams->r0/4;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_partsurf, &partsurf, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_cosconeanglefluid, &physparams->cosconeanglefluid, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_cosconeanglenonfluid, &physparams->cosconeanglenonfluid, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_objectobjectdf, &physparams->objectobjectdf, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_objectboundarydf, &physparams->objectboundarydf, sizeof(float)));

	idx_t neiblist_end = simparams->maxneibsnum*allocatedParticles;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_neiblist_stride, &allocatedParticles, sizeof(idx_t)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_neiblist_end, &neiblist_end, sizeof(idx_t)));

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

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gridSize, &gridSize, sizeof(uint3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_ferrari, &simparams->ferrari, sizeof(float)));
}


void
getforcesconstants(PhysParams *physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->numFluids, cuforces::d_numfluids, sizeof(int)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->rho0, cuforces::d_rho0, MAX_FLUID_TYPES*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->gravity, cuforces::d_gravity, sizeof(float3), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->bcoeff, cuforces::d_bcoeff, MAX_FLUID_TYPES*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->gammacoeff, cuforces::d_gammacoeff, MAX_FLUID_TYPES*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->sscoeff, cuforces::d_sscoeff, MAX_FLUID_TYPES*sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->sspowercoeff, cuforces::d_sspowercoeff, MAX_FLUID_TYPES*sizeof(float), 0));

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->dcoeff, cuforces::d_dcoeff, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->p1coeff, cuforces::d_p1coeff, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->p2coeff, cuforces::d_p2coeff, sizeof(float), 0));

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->MK_K, cuforces::d_MK_K, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->MK_d, cuforces::d_MK_d, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->MK_beta, cuforces::d_MK_beta, sizeof(float), 0));

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->r0, cuforces::d_r0, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->visccoeff, cuforces::d_visccoeff, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->epsartvisc, cuforces::d_epsartvisc, sizeof(float), 0));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->ewres, cuforces::d_ewres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->nsres, cuforces::d_nsres, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->demdx, cuforces::d_demdx, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->demdy, cuforces::d_demdy, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->demzmin, cuforces::d_demzmin, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->smagfactor, cuforces::d_smagfactor, sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->kspsfactor, cuforces::d_kspsfactor, sizeof(float)));
}

void
setplaneconstants(int numPlanes, const float* PlanesDiv, const float4* Planes)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_planes, Planes, numPlanes*sizeof(float4)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_plane_div, PlanesDiv, numPlanes*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_numplanes, &numPlanes, sizeof(uint)));
}

void
setgravity(float3 const& gravity)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_gravity, &gravity, sizeof(float3)));
}


void
setforcesrbcg(const float3* cg, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rbcg, cg, numbodies*sizeof(float3)));
}


void
setforcesrbstart(const uint* rbfirstindex, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuforces::d_rbstartindex, rbfirstindex, numbodies*sizeof(uint)));
}

void
sps(		float2*			tau[],
	const	float4	*pos,
	const	float4	*vel,
const	particleinfo	*info,
	const	hashKey	*particleHash,
	const	uint	*cellStart,
	const	neibdata*neibsList,
			uint	numParticles,
			uint	particleRangeEnd,
			float	slength,
		KernelType	kerneltype,
			float	influenceradius)
{
	int dummy_shared = 0;
	// bind textures to read all particles, not only internal ones
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	uint numThreads = min(BLOCK_SIZE_SPS, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	switch (kerneltype) {
		SPS_CHECK(CUBICSPLINE);
		//SPS_CHECK(QUADRATIC);
		SPS_CHECK(WENDLAND);
		NOT_IMPLEMENTED_CHECK(Kernel, kerneltype);
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("SPS kernel execution failed");

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif

	CUDA_SAFE_CALL(cudaBindTexture(0, tau0Tex, tau[0], numParticles*sizeof(float2)));
	CUDA_SAFE_CALL(cudaBindTexture(0, tau1Tex, tau[1], numParticles*sizeof(float2)));
	CUDA_SAFE_CALL(cudaBindTexture(0, tau2Tex, tau[2], numParticles*sizeof(float2)));
}

float
forces(
	const	float4	*pos,
			float2	*vertPos[],
	const	float4	*vel,
			float4	*forces,
	const	float4	*oldGGam,
			float4	*newGGam,
	const	float4	*boundelem,
			float4	*rbforces,
			float4	*rbtorques,
			float4	*xsph,
	const	particleinfo	*info,
	const	hashKey	*particleHash,
	const	uint	*cellStart,
	const	neibdata*neibsList,
			uint	numParticles,
			uint	particleRangeEnd,
			float	deltap,
			float	slength,
			float	dt,
			bool	dtadapt,
			float	dtadaptfactor,
			bool	xsphcorr,
	KernelType		kerneltype,
			float	influenceradius,
	const	float	epsilon,
	const	bool	movingBoundaries,
	ViscosityType	visctype,
			float	visccoeff,
			float	*turbvisc,
			float	*keps_tke,
			float	*keps_eps,
			float2	*keps_dkde,
			float	*cfl,
			float	*cflTVisc,
			float	*tempCfl,
	SPHFormulation	sph_formulation,
	BoundaryType	boundarytype,
			bool	usedem)
{
	int dummy_shared = 0;
	float2 *vertPos0, *vertPos1, *vertPos2;
	vertPos0 = vertPos1 = vertPos2 = NULL;

	// bind textures to read all particles, not only internal ones
	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaBindTexture(0, gamTex, oldGGam, numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelem, numParticles*sizeof(float4)));
		vertPos0 = vertPos[0];
		vertPos1 = vertPos[1];
		vertPos2 = vertPos[2];
	}

	if (visctype == KEPSVISC) {
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_kTex, keps_tke, numParticles*sizeof(float)));
		CUDA_SAFE_CALL(cudaBindTexture(0, keps_eTex, keps_eps, numParticles*sizeof(float)));
	}

	// thread per particle
	uint numThreads = min(BLOCK_SIZE_FORCES, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);
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

	if (boundarytype == SA_BOUNDARY) {
		CUDA_SAFE_CALL(cudaUnbindTexture(gamTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
	}

	if (visctype == KEPSVISC) {
		CUDA_SAFE_CALL(cudaUnbindTexture(keps_kTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(keps_eTex));
	}

	if (dtadapt) {
		// cfl holds one value per block in the forces kernel call,
		// so it holds numBlocks elements
		float maxcfl = cflmax(numBlocks, cfl, tempCfl);
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
				case KEPSVISC:
					dt_visc = slength*slength/(visccoeff + cflmax(numBlocks, cflTVisc, tempCfl));
					break;
				NOT_IMPLEMENTED_CHECK(Viscosity, visctype);
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
		hashKey*		particleHash,
		uint*		cellStart,
		neibdata*	neibsList,
		uint		numParticles,
		uint		particleRangeEnd,
		float		slength,
		int			kerneltype,
		float		influenceradius)
{
	int dummy_shared = 0;
	// thread per particle
	uint numThreads = min(BLOCK_SIZE_SHEPARD, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	#if (__COMPUTE__ >= 20)
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
	hashKey*		particleHash,
	uint*		cellStart,
	neibdata*	neibsList,
	uint		numParticles,
	uint		particleRangeEnd,
	float		slength,
	int			kerneltype,
	float		influenceradius)
{
	int dummy_shared = 0;
	// thread per particle
	uint numThreads = min(BLOCK_SIZE_MLS, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, oldVel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	#if (__COMPUTE__ >= 20)
	dummy_shared = 2560;
	#endif
	switch (kerneltype) {
		MLS_CHECK(CUBICSPLINE);
//		MLS_CHECK(QUADRATIC);
		MLS_CHECK(WENDLAND);
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Mls kernel execution failed");

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
}

void
vorticity(	float4*		pos,
			float4*		vel,
			float3*		vort,
			particleinfo	*info,
			hashKey*		particleHash,
			uint*		cellStart,
			neibdata*	neibsList,
			uint		numParticles,
			uint		particleRangeEnd,
			float		slength,
			int			kerneltype,
			float		influenceradius)
{
	// thread per particle
	uint numThreads = min(BLOCK_SIZE_CALCVORT, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	switch (kerneltype) {
		VORT_CHECK(CUBICSPLINE);
//		VORT_CHECK(QUADRATIC);
		VORT_CHECK(WENDLAND);
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Vorticity kernel execution failed");

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
}

//Testpoints
void
testpoints( const float4*		pos,
			float4*		newVel,
			particleinfo	*info,
			hashKey*		particleHash,
			uint*		cellStart,
			neibdata*	neibsList,
			uint		numParticles,
			uint		particleRangeEnd,
			float		slength,
			int			kerneltype,
			float		influenceradius)
{
	// thread per particle
	uint numThreads = min(BLOCK_SIZE_CALCTEST, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
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

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
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
					hashKey*		particleHash,
					uint*		cellStart,
					neibdata*	neibsList,
					uint		numParticles,
					uint		particleRangeEnd,
					float		slength,
					int			kerneltype,
					float		influenceradius,
					bool        savenormals)
{
	// thread per particle
	uint numThreads = min(BLOCK_SIZE_CALCTEST, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
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

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
}


void setDemTexture(const float *hDem, int width, int height)
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

	// For the segmented scan, we use rbnum (number of object per object particle) as key (first and second parameters
	// of inclusive_scan_by_key are the begin and the end of the array of keys); forces or torques as input and output
	// the scan is in place); equal_to as data-key operator and plus as scan operator. The sums are in the last position
	// of each segment (thus we retrieve them by using lastindex values).

	thrust::inclusive_scan_by_key(rbnum_devptr, rbnum_devptr + numBodiesParticles,
				forces_devptr, forces_devptr, binary_pred, binary_op);
	thrust::inclusive_scan_by_key(rbnum_devptr, rbnum_devptr + numBodiesParticles,
				torques_devptr, torques_devptr, binary_pred, binary_op);

	for (uint i = 0; i < numbodies; i++) {
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

// returns the number of elements in the (starting) fmax array, assuming n particles.
// this is _exactly_ the number of blocks in the grid launch for the forces kernel over n
// particles, since the forces kernel pre-reduces the cfl values, producing one value
// per block instead of one per particle
uint
getFmaxElements(const uint n)
{
	return div_up(n, min(BLOCK_SIZE_FORCES, n));
}


uint
getFmaxTempElements(const uint n)
{
	uint numBlocks, numThreads;
	getNumBlocksAndThreads(n, MAX_BLOCKS_FMAX, BLOCK_SIZE_FMAX, numBlocks, numThreads);
	return numBlocks;
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

/* Reductions */
void set_reduction_params(void* buffer, size_t blocks,
		size_t blocksize_max, size_t shmem_max)
{
	reduce_blocks = blocks;
	// in the second step of a reduction, a single block is launched, whose size
	// should be the smallest power of two that covers the number of blocks used
	// in the previous reduction run
	reduce_bs2 = 32;
	while (reduce_bs2 < blocks)
		reduce_bs2<<=1;

	reduce_blocksize_max = blocksize_max;
	reduce_shmem_max = shmem_max;
	reduce_buffer = buffer;
}

void unset_reduction_params()
{
	CUDA_SAFE_CALL(cudaFree(reduce_buffer));
	reduce_buffer = NULL;
}

// Compute system energy
void calc_energy(
		float4			*output,
	const	float4		*pos,
	const	float4		*vel,
	const	particleinfo	*pinfo,
	const	hashKey		*particleHash,
		uint			numParticles,
		uint			numFluids)
{
	// shmem needed by a single thread
	size_t shmem_thread = numFluids*sizeof(float4)*2;
	size_t blocksize_max = reduce_shmem_max/shmem_thread;
	if (blocksize_max > reduce_blocksize_max)
		blocksize_max = reduce_blocksize_max;

	size_t blocksize = 32;
	while (blocksize*2 < blocksize_max)
		blocksize<<=1;

	cuforces::calcEnergiesDevice<<<reduce_blocks, blocksize, blocksize*shmem_thread>>>(
			pos, vel, pinfo, particleHash, numParticles, numFluids, (float4*)reduce_buffer);
	CUT_CHECK_ERROR("System energy stage 1 failed");

	cuforces::calcEnergies2Device<<<1, reduce_bs2, reduce_bs2*shmem_thread>>>(
			(float4*)reduce_buffer, reduce_blocks, numFluids);
	CUT_CHECK_ERROR("System energy stage 2 failed");
	CUDA_SAFE_CALL(cudaMemcpy(output, reduce_buffer, numFluids*sizeof(float4), cudaMemcpyDeviceToHost));
}

void
calcPrivate(const	float4*			pos,
			const	float4*			vel,
			const	particleinfo*	info,
					float*			priv,
			const	hashKey*		particleHash,
			const	uint*			cellStart,
					neibdata*		neibsList,
					float			slength,
					float			inflRadius,
					uint			numParticles,
					uint			particleRangeEnd)
{
	uint numThreads = min(BLOCK_SIZE_FORCES, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));

	//execute kernel
	cuforces::calcPrivateDevice<<<numBlocks, numThreads>>>
		(	pos,
			priv,
			particleHash,
			cellStart,
			neibsList,
			slength,
			inflRadius,
			numParticles);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(velTex));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("UpdatePositions kernel execution failed");
}

void
updateBoundValues(	float4*		oldVel,
			float*		oldTKE,
			float*		oldEps,
			vertexinfo*	vertices,
			particleinfo*	info,
			uint		numParticles,
			uint		particleRangeEnd,
			bool		initStep)
{
	uint numThreads = min(BLOCK_SIZE_FORCES, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
	CUDA_SAFE_CALL(cudaBindTexture(0, vertTex, vertices, numParticles*sizeof(vertexinfo)));

	//execute kernel
	cuforces::updateBoundValuesDevice<<<numBlocks, numThreads>>>(oldVel, oldTKE, oldEps, numParticles, initStep);

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(vertTex));

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("UpdateBoundValues kernel execution failed");
}

void
dynamicBoundConditions(	const float4*		oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			float4*			newGam,
			const float4*	boundelement,
			const particleinfo*	info,
			const hashKey*		particleHash,
			const uint*		cellStart,
			const neibdata*	neibsList,
			const uint		numParticles,
			const uint		particleRangeEnd,
			const float		deltap,
			const float		slength,
			const int		kerneltype,
			const float		influenceradius,
			const bool		initStep)
{
	int dummy_shared = 0;

	uint numThreads = min(BLOCK_SIZE_SHEPARD, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaBindTexture(0, posTex, oldPos, numParticles*sizeof(float4)));
	#endif
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	if(initStep)
		CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif
	// execute the kernel
	switch (kerneltype) {
		DYNBOUNDARY_CHECK(CUBICSPLINE);
//		DYNBOUNDARY_CHECK(QUADRATIC);
		DYNBOUNDARY_CHECK(WENDLAND);
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("DynamicBoundConditions kernel execution failed");

	#if (__COMPUTE__ < 20)
	CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
	#endif
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	if(initStep)
		CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

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
#undef KEPS_CHECK
#undef VORT_CHECK
#undef TEST_CHECK
#undef SURFACE_CHECK
#undef INITGRADGAMMA_CHECK
#undef UPDATEGAMMA_CHECK

/* These were defined in forces_kernel.cu */
#undef _FORCES_KERNEL_NAME
#undef FORCES_KERNEL_NAME
