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
#include <stdexcept>

#include "define_buffers.h"
#include "engine_integration.h"
#include "utils.h"
#include "euler_params.h"
#include "density_sum_params.h"

#include "euler_kernel.cu"
#include "density_sum_kernel.cu"

#define BLOCK_SIZE_INTEGRATE	256

template<
	SPHFormulation sph_formulation,
	BoundaryType boundarytype,
	KernelType kerneltype,
	typename ViscSpec,
	flag_t simflags>
class CUDAPredCorrEngine : public AbstractIntegrationEngine
{

void
setconstants(const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
	idx_t const& allocatedParticles, int const& neiblistsize, float const& slength)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_epsxsph, &physparams->epsxsph, sizeof(float)));

	idx_t neiblist_end = neiblistsize*allocatedParticles;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_neiblist_stride, &allocatedParticles, sizeof(idx_t)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuneibs::d_neiblist_end, &neiblist_end, sizeof(idx_t)));

	const float h3 = slength*slength*slength;
	float kernelcoeff = 1.0f/(M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_wcoeff_cubicspline, &kernelcoeff, sizeof(float)));
	kernelcoeff = 15.0f/(16.0f*M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_wcoeff_quadratic, &kernelcoeff, sizeof(float)));
	kernelcoeff = 21.0f/(16.0f*M_PI*h3);
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_wcoeff_wendland, &kernelcoeff, sizeof(float)));
}

void
getconstants(PhysParams *physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->epsxsph, cueuler::d_epsxsph, sizeof(float), 0));
}

void
setrbcg(const int3* cgGridPos, const float3* cgPos, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbcgGridPos, cgGridPos, numbodies*sizeof(int3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbcgPos, cgPos, numbodies*sizeof(float3)));
}

void
setrbtrans(const float3* trans, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbtrans, trans, numbodies*sizeof(float3)));
}

void
setrblinearvel(const float3* linearvel, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rblinearvel, linearvel, numbodies*sizeof(float3)));
}

void
setrbangularvel(const float3* angularvel, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbangularvel, angularvel, numbodies*sizeof(float3)));
}

void
setrbsteprot(const float* rot, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbsteprot, rot, 9*numbodies*sizeof(float)));
}

// TODO FIXME density summation is only currently supported for SA_BOUNDARY, and the code
// is designed for it (no conditional gamma terms etc). It should be redesigned
// to extend support to other formulations as well.
// For the time being we SFINAE its “actual” implementation in this secondary method
template<BoundaryType _boundarytype>
enable_if_t<_boundarytype == SA_BOUNDARY>
density_sum_impl(
		BufferList const& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	epsilon,
		const	float	slength,
		const	float	influenceradius)
{
	// thread per particle
	uint numThreads = BLOCK_SIZE_INTEGRATE;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// the template is on PT_FLUID, but in reality it's for PT_FLUID and PT_VERTEX
	density_sum_params<kerneltype, PT_FLUID, simflags> volumic_params(
		bufread, bufwrite, particleRangeEnd, dt, t, step, slength, influenceradius);

	cudensity_sum::densitySumVolumicDevice<kerneltype, simflags><<< numBlocks, numThreads >>>(volumic_params);

	density_sum_params<kerneltype, PT_BOUNDARY, simflags> boundary_params(
		bufread, bufwrite, particleRangeEnd, dt, t, step, slength, influenceradius);

	cudensity_sum::densitySumBoundaryDevice<kerneltype, simflags><<< numBlocks, numThreads >>>(boundary_params);

	if (simflags & ENABLE_MOVING_BODIES) {
		// VERTEX gamma is always integrated directly
		integrate_gamma_params<PT_VERTEX, kerneltype, simflags> vertex_params(
			bufread, bufwrite,
			particleRangeEnd,
			dt, t, step,
			epsilon, slength, influenceradius);
		cudensity_sum::integrateGammaDevice<<< numBlocks, numThreads >>>(vertex_params);
	} else {
		/* We got them from the buffer lists already, reuse the params structure members.
		 */
		const particleinfo *info = volumic_params.info;
		const float4 *oldgGam = volumic_params.oldgGam;
			  float4 *newgGam = volumic_params.newgGam;
		cueuler::copyTypeDataDevice<PT_VERTEX><<< numBlocks, numThreads >>>(
			info, oldgGam, newgGam, particleRangeEnd);
		cueuler::copyTypeDataDevice<PT_BOUNDARY><<< numBlocks, numThreads >>>(
			info, oldgGam, newgGam, particleRangeEnd);
	}

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}
template<BoundaryType _boundarytype>
enable_if_t<_boundarytype != SA_BOUNDARY>
density_sum_impl(
		BufferList const& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	epsilon,
		const	float	slength,
		const	float	influenceradius)
{
	throw std::runtime_error("density summation is currently only supported with SA_BOUNDARY");
}

void
density_sum(
		BufferList const& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	epsilon,
		const	float	slength,
		const	float	influenceradius)
{
	density_sum_impl<boundarytype>(bufread, bufwrite,
		numParticles, particleRangeEnd,
		dt, step, t, epsilon, slength, influenceradius);
}

// SFINAE implementation of integrate_gamma
template<BoundaryType _boundarytype>
enable_if_t<_boundarytype == SA_BOUNDARY>
integrate_gamma_impl(
		BufferList const& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	epsilon,
		const	float	slength,
		const	float	influenceradius)
{
	// thread per particle
	uint numThreads = BLOCK_SIZE_INTEGRATE;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	integrate_gamma_params<PT_FLUID, kerneltype, simflags> fluid_params(
		bufread, bufwrite,
		particleRangeEnd,
		dt, t, step,
		epsilon, slength, influenceradius);

	// to see why integrateGammaDevice is in the cudensity_sum namespace, see the documentation
	// of the kernel
	cudensity_sum::integrateGammaDevice<<< numBlocks, numThreads >>>(fluid_params);

	if (simflags & ENABLE_MOVING_BODIES) {
		integrate_gamma_params<PT_VERTEX, kerneltype, simflags> vertex_params(fluid_params);
		cudensity_sum::integrateGammaDevice<<< numBlocks, numThreads >>>(vertex_params);
	} else {
		/* We got them from the buffer lists already, reuse the params structure members.
		 */
		const particleinfo *info = fluid_params.info;
		const float4 *oldgGam = fluid_params.oldgGam;
			  float4 *newgGam = fluid_params.newgGam;
		cueuler::copyTypeDataDevice<PT_VERTEX><<< numBlocks, numThreads >>>(
			info, oldgGam, newgGam, particleRangeEnd);
		cueuler::copyTypeDataDevice<PT_BOUNDARY><<< numBlocks, numThreads >>>(
			info, oldgGam, newgGam, particleRangeEnd);
	}

	KERNEL_CHECK_ERROR;
}
template<BoundaryType _boundarytype>
enable_if_t<_boundarytype != SA_BOUNDARY>
integrate_gamma_impl(
		BufferList const& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	epsilon,
		const	float	slength,
		const	float	influenceradius)
{
	throw std::runtime_error("integrate_gamma called without SA_BOUNDARY");
}

void
integrate_gamma(
		BufferList const& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	epsilon,
		const	float	slength,
		const	float	influenceradius)
{
	integrate_gamma_impl<boundarytype>(bufread, bufwrite,
		numParticles, particleRangeEnd,
		dt, step, t, epsilon, slength, influenceradius);
}


void
apply_density_diffusion(
	BufferList const& bufread,
	BufferList& bufwrite,
	const	uint	numParticles,
	const	uint	particleRangeEnd,
	const	float	dt)
{
	// thread per particle
	uint numThreads = BLOCK_SIZE_INTEGRATE;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// This is a trivial integration of the density in position write
	cueuler::updateDensityDevice<<<numBlocks, numThreads>>>(
		bufread.getData<BUFFER_INFO>(), bufread.getData<BUFFER_FORCES>(),
		bufwrite.getData<BUFFER_VEL>(),
		numParticles, particleRangeEnd, dt);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}


void
basicstep(
		BufferList const& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	slength,
		const	float	influenceradius)
{
	// thread per particle
	uint numThreads = BLOCK_SIZE_INTEGRATE;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	if (step == 1)
		cueuler::eulerDevice<<< numBlocks, numThreads >>>(
			euler_params<kerneltype, sph_formulation, boundarytype, ViscSpec, simflags, 1>(
				bufread, bufwrite, numParticles, dt, t));
	else if (step == 2)
		cueuler::eulerDevice<<< numBlocks, numThreads >>>(
			euler_params<kerneltype, sph_formulation, boundarytype, ViscSpec, simflags, 2>(
				bufread, bufwrite, numParticles, dt, t));
	else
		throw std::invalid_argument("unsupported predcorr timestep");

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

/// Disables free surface boundary particles during the repacking process
	void
disableFreeSurfParts(		float4*			pos,
		const	particleinfo*	info,
		const	uint			numParticles,
		const	uint			particleRangeEnd)
{
	uint numThreads = BLOCK_SIZE_INTEGRATE;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	//execute kernel
	cueuler::disableFreeSurfPartsDevice<<<numBlocks, numThreads>>>
		(	pos,
			numParticles);

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}


};

