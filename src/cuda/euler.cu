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

#include "euler_kernel.cu"

#define BLOCK_SIZE_INTEGRATE	256

template<
	SPHFormulation sph_formulation,
	BoundaryType boundarytype,
	KernelType kerneltype,
	ViscosityType visctype,
	flag_t simflags>
class CUDAPredCorrEngine : public AbstractIntegrationEngine
{

void
setconstants(const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
	idx_t const& allocatedParticles, int const& maxneibsnum, float const& slength)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_epsxsph, &physparams->epsxsph, sizeof(float)));

	idx_t neiblist_end = maxneibsnum*allocatedParticles;
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_neiblist_stride, &allocatedParticles, sizeof(idx_t)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_neiblist_end, &neiblist_end, sizeof(idx_t)));

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

void
basicstep(
		MultiBufferList::const_iterator bufread,
		MultiBufferList::iterator bufreadUpdate,
		MultiBufferList::iterator bufwrite,
		const	uint	*cellStart,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	float	dt2,
		const	int		step,
		const	float	t,
		const	float	slength,
		const	float	influenceradius)
{
	// thread per particle
	uint numThreads = BLOCK_SIZE_INTEGRATE;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	const float4  *oldPos = bufread->getData<BUFFER_POS>();
	const hashKey *particleHash = bufread->getData<BUFFER_HASH>();
	const float4  *oldVol = bufread->getData<BUFFER_VOLUME>();
	const float4 *oldEulerVel = bufread->getData<BUFFER_EULERVEL>();
	const float *oldTKE = bufread->getData<BUFFER_TKE>();
	const float *oldEps = bufread->getData<BUFFER_EPSILON>();
	const particleinfo *info = bufread->getData<BUFFER_INFO>();
	const neibdata *neibsList = bufread->getData<BUFFER_NEIBSLIST>();
	const float2 * const *vertPos = bufread->getRawPtr<BUFFER_VERTPOS>();

	const float4 *forces = bufread->getData<BUFFER_FORCES>();
	const float2 *contupd = bufread->getData<BUFFER_CONTUPD>();
	const float3 *keps_dkde = bufread->getData<BUFFER_DKDE>();
	const float4 *xsph = bufread->getData<BUFFER_XSPH>();

	// The following two arrays are update in case ENABLE_DENSITY_SUM is set
	// so they are taken from the non-const bufreadUpdate
	float4  *oldVel = bufreadUpdate->getData<BUFFER_VEL>();
	float4 *oldgGam = bufreadUpdate->getData<BUFFER_GRADGAMMA>();

	float4 *newPos = bufwrite->getData<BUFFER_POS>();
	float4 *newVel = bufwrite->getData<BUFFER_VEL>();
	float4 *newVol = bufwrite->getData<BUFFER_VOLUME>();
	float4 *newEulerVel = bufwrite->getData<BUFFER_EULERVEL>();
	float4 *newgGam = bufwrite->getData<BUFFER_GRADGAMMA>();
	float *newTKE = bufwrite->getData<BUFFER_TKE>();
	float *newEps = bufwrite->getData<BUFFER_EPSILON>();
	// boundary elements are updated in-place; only used for rotation in the second step
	float4 *newBoundElement = bufwrite->getData<BUFFER_BOUNDELEMENTS>();

	euler_params<kerneltype, sph_formulation, boundarytype, visctype, simflags> params(
			newPos, newVel, oldPos, particleHash, oldVel, info, forces, numParticles, dt, dt2, t, step,
			xsph,
			oldgGam, newgGam, contupd, newEulerVel, newBoundElement, vertPos, oldEulerVel, slength, influenceradius, neibsList, cellStart,
			newTKE, newEps, oldTKE, oldEps, keps_dkde,
			newVol, oldVol);

	if (step == 1) {
		cueuler::eulerDevice<kerneltype, sph_formulation, boundarytype, visctype, simflags><<< numBlocks, numThreads >>>(params);
	} else if (step == 2) {
		cueuler::eulerDevice<kerneltype, sph_formulation, boundarytype, visctype, simflags><<< numBlocks, numThreads >>>(params);
	} else {
		throw std::invalid_argument("unsupported predcorr timestep");
	}

#undef ARGS

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

};

