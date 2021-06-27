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
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius)
{
	// thread per particle
	uint numThreads = BLOCK_SIZE_INTEGRATE;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// Kernel functor types
	using densitySumVolumicDevice = cudensity_sum::densitySumVolumicDevice<sph_formulation, kerneltype, simflags>;
	using densitySumBoundaryDevice = cudensity_sum::densitySumBoundaryDevice<kerneltype, simflags>;

	// We explicitly instantiate the volumic kernel functor,
	// since we'll use some of its members also for the “no moving bodies”
	// gamma correction case below
	densitySumVolumicDevice volumic_kernel(
		bufread, bufwrite, particleRangeEnd, dt, t, step, deltap, slength, influenceradius);
	execute_kernel(volumic_kernel, numBlocks, numThreads);

	// for symmetry
	densitySumBoundaryDevice boundary_kernel(
		bufread, bufwrite, particleRangeEnd, dt, t, step, deltap, slength, influenceradius);

	execute_kernel(boundary_kernel, numBlocks, numThreads);

	if (HAS_MOVING_BODIES(simflags)) {
		// VERTEX gamma is always integrated directly
		using integrate_gamma_params = integrate_gamma_params<PT_VERTEX, kerneltype, simflags>;
		execute_kernel(
			cudensity_sum::integrateGammaDevice<integrate_gamma_params>(
				bufread, bufwrite,
				particleRangeEnd,
				dt, t, step,
				epsilon, slength, influenceradius),
			numBlocks, numThreads);
	} else {
		/* We got them from the buffer lists already, reuse the params structure members.
		 */
		const particleinfo *info = volumic_kernel.info;
		const float4 *oldgGam = volumic_kernel.oldgGam;
			  float4 *newgGam = volumic_kernel.newgGam;
		execute_kernel(
			cueuler::copyTypeDataDevice<PT_VERTEX, float4>(info, oldgGam, newgGam, particleRangeEnd),
			numBlocks, numThreads);
		execute_kernel(
			cueuler::copyTypeDataDevice<PT_BOUNDARY, float4>(info, oldgGam, newgGam, particleRangeEnd),
			numBlocks, numThreads);
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
		const	float	deltap,
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
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius)
{
	density_sum_impl<boundarytype>(bufread, bufwrite,
		numParticles, particleRangeEnd,
		dt, step, t, epsilon, deltap, slength, influenceradius);
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
		const	float	influenceradius,
		const	RunMode	run_mode)
{
	// thread per particle
	uint numThreads = BLOCK_SIZE_INTEGRATE;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	if (run_mode == REPACK) {
		using integrate_gamma_params = integrate_gamma_repack_params<PT_FLUID, kerneltype, simflags>;
		// to see why integrateGammaDevice is in the cudensity_sum namespace, see the documentation
		// of the kernel
		// We explicitly instantiate the kernel functor,
		// since we'll use some of its members also for the copyTypeData kernel calls after the gamma integration
		cudensity_sum::integrateGammaDevice<integrate_gamma_params> fluid_gamma_kernel(
			bufread, bufwrite,
			particleRangeEnd,
			dt, t, step,
			epsilon, slength, influenceradius);
		execute_kernel(fluid_gamma_kernel, numBlocks, numThreads);
		/* We got them from the buffer lists already, reuse the params structure members.
		 */
		const particleinfo *info = fluid_gamma_kernel.info;
		const float4 *oldgGam = fluid_gamma_kernel.oldgGam;
			  float4 *newgGam = fluid_gamma_kernel.newgGam;
		execute_kernel(
			cueuler::copyTypeDataDevice<PT_VERTEX, float4>(info, oldgGam, newgGam, particleRangeEnd),
			numBlocks, numThreads);
		execute_kernel(
			cueuler::copyTypeDataDevice<PT_BOUNDARY, float4>(info, oldgGam, newgGam, particleRangeEnd),
			numBlocks, numThreads);
	} else {
		using integrate_fluid_gamma_params = integrate_gamma_params<PT_FLUID, kerneltype, simflags>;
		// see if() branch
		cudensity_sum::integrateGammaDevice<integrate_fluid_gamma_params> fluid_gamma_kernel(
			bufread, bufwrite,
			particleRangeEnd,
			dt, t, step,
			epsilon, slength, influenceradius);
		execute_kernel(fluid_gamma_kernel, numBlocks, numThreads);

		if (HAS_MOVING_BODIES(simflags)) {
			// integrate gamma, using the same parameters used for the fluid integration
			using integrate_vertex_gamma_params = integrate_gamma_params<PT_VERTEX, kerneltype, simflags>;
			cudensity_sum::integrateGammaDevice<integrate_vertex_gamma_params> vertex_gamma_kernel(fluid_gamma_kernel);
			execute_kernel(vertex_gamma_kernel, numBlocks, numThreads);
		} else {
			/* We got them from the buffer lists already, reuse the params structure members.
			 */
			const particleinfo *info = fluid_gamma_kernel.info;
			const float4 *oldgGam = fluid_gamma_kernel.oldgGam;
			float4 *newgGam = fluid_gamma_kernel.newgGam;
			execute_kernel(
				cueuler::copyTypeDataDevice<PT_VERTEX, float4>(info, oldgGam, newgGam, particleRangeEnd),
				numBlocks, numThreads);
			execute_kernel(
				cueuler::copyTypeDataDevice<PT_BOUNDARY, float4>(info, oldgGam, newgGam, particleRangeEnd),
				numBlocks, numThreads);
		}
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
		const	float	influenceradius,
		const	RunMode	run_mode)
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
		const	float	influenceradius,
		const	RunMode	run_mode)
{
	integrate_gamma_impl<boundarytype>(bufread, bufwrite,
		numParticles, particleRangeEnd,
		dt, step, t, epsilon, slength, influenceradius, run_mode);
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
	execute_kernel(
		cueuler::updateDensityDevice(bufread, bufwrite, particleRangeEnd, dt),
		numBlocks, numThreads);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}


uint
basicstep(
		BufferList const& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	slength,
		const	float	influenceradius,
		const	RunMode	run_mode)
{
	const bool nancheck = g_debug.nans;

	uint nans_found = 0;

	if (nancheck)
		CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_nans_found, &nans_found, sizeof(nans_found)));

	// thread per particle
	uint numThreads = BLOCK_SIZE_INTEGRATE;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// execute the kernel
#define EULER_STEP(step) case step: \
	if (run_mode == REPACK) { \
		euler_repack_params<kerneltype, boundarytype, simflags, step> \
			params(bufread, bufwrite, numParticles, dt, t); \
		cueuler::eulerDevice<<< numBlocks, numThreads >>>(params); \
		if (nancheck) cueuler::nanCheckDevice<<< numBlocks, numThreads>>>(params); \
	} else { \
		euler_params<kerneltype, sph_formulation, boundarytype, ViscSpec, simflags, step> \
			params(bufread, bufwrite, numParticles, dt, t); \
		cueuler::eulerDevice<<< numBlocks, numThreads >>>(params); \
		if (nancheck) cueuler::nanCheckDevice<<< numBlocks, numThreads>>>(params); \
	} \
	break;
	switch (step) {
		EULER_STEP(1);
		EULER_STEP(2);
	default:
		throw std::invalid_argument("unsupported predcorr timestep");
	}
	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	if (nancheck)
		CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&nans_found, cueuler::d_nans_found, sizeof(nans_found)));

	return nans_found;

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

	//execute kernel
	cueuler::disableFreeSurfPartsDevice<<<numBlocks, numThreads>>>
		(	pos, info,
			numParticles);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}


};

