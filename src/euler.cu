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

#include "euler.cuh"
#include "euler_kernel.cu"

#include "utils.h"

template<BoundaryType boundarytype, bool xsphcorr>
void
CUDAPredCorrEngine<boundarytype, xsphcorr>::
setconstants(const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_epsxsph, &physparams->epsxsph, sizeof(float)));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_gridSize, &gridSize, sizeof(uint3)));
}

template<BoundaryType boundarytype, bool xsphcorr>
void
CUDAPredCorrEngine<boundarytype, xsphcorr>::
getconstants(PhysParams *physparams)
{
	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&physparams->epsxsph, cueuler::d_epsxsph, sizeof(float), 0));
}

template<BoundaryType boundarytype, bool xsphcorr>
void
CUDAPredCorrEngine<boundarytype, xsphcorr>::
setmbdata(const float4* MbData, uint size)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_mbdata, MbData, size));
}

template<BoundaryType boundarytype, bool xsphcorr>
void
CUDAPredCorrEngine<boundarytype, xsphcorr>::
setrbcg(const float3* cg, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbcg, cg, numbodies*sizeof(float3)));
}

template<BoundaryType boundarytype, bool xsphcorr>
void
CUDAPredCorrEngine<boundarytype, xsphcorr>::
setrbtrans(const float3* trans, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbtrans, trans, numbodies*sizeof(float3)));
}

template<BoundaryType boundarytype, bool xsphcorr>
void
CUDAPredCorrEngine<boundarytype, xsphcorr>::
setrblinearvel(const float3* linearvel, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rblinearvel, linearvel, numbodies*sizeof(float3)));
	//printf("Upload linear vel: %e %e %e\n", linearvel[0].x, linearvel[0].y, linearvel[0].z);
}

template<BoundaryType boundarytype, bool xsphcorr>
void
CUDAPredCorrEngine<boundarytype, xsphcorr>::
setrbangularvel(const float3* angularvel, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbangularvel, angularvel, numbodies*sizeof(float3)));
	//printf("Upload angular vel: %e %e %e\n", angularvel[0].x, angularvel[0].y, angularvel[0].z);
}

template<BoundaryType boundarytype, bool xsphcorr>
void
CUDAPredCorrEngine<boundarytype, xsphcorr>::
setrbsteprot(const float* rot, int numbodies)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cueuler::d_rbsteprot, rot, 9*numbodies*sizeof(float)));
}

template<BoundaryType boundarytype, bool xsphcorr>
void
CUDAPredCorrEngine<boundarytype, xsphcorr>::
basicstep(
	const	float4	*oldPos,
	const	hashKey	*particleHash,
	const	float4	*oldVel,
	const	float4	*oldEulerVel,
	const	float4	*oldgGam,
	const	float	*oldTKE,
	const	float	*oldEps,
	const	particleinfo	*info,
	const	float4	*forces,
	const	float2	*contupd,
	const	float3	*keps_dkde,
	const	float4	*xsph,
			float4	*newPos,
			float4	*newVel,
			float4	*newEulerVel,
			float4	*newgGam,
			float	*newTKE,
			float	*newEps,
	// boundary elements are updated in-place, only used for rotation in the second step
			float4	*newBoundElement,
	const	uint	numParticles,
	const	uint	particleRangeEnd,
	const	float	dt,
	const	float	dt2,
	const	int		step,
	const	float	t)
{
	// thread per particle
	uint numThreads = min(BLOCK_SIZE_INTEGRATE, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

#define ARGS oldPos, particleHash, oldVel, oldEulerVel, oldgGam, oldTKE, oldEps, \
	info, forces, contupd, keps_dkde, xsph, newPos, newVel, newEulerVel, newgGam, newTKE, newEps, newBoundElement, particleRangeEnd, dt, dt2, t

	if (step == 1) {
		cueuler::eulerDevice<1, xsphcorr, boundarytype><<< numBlocks, numThreads >>>(ARGS);
	} else if (step == 2) {
		cueuler::eulerDevice<2, xsphcorr, boundarytype><<< numBlocks, numThreads >>>(ARGS);
	} else {
		throw std::invalid_argument("unsupported predcorr timestep");
	}

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Euler kernel execution failed");
}

// Force the instantiation of all instances
// TODO this is until the engines are turned into header-only classes

#define DECLARE_PREDCORRENGINE(btype) \
	template class CUDAPredCorrEngine<btype, false>; \
	template class CUDAPredCorrEngine<btype, true>;

DECLARE_PREDCORRENGINE(LJ_BOUNDARY)
DECLARE_PREDCORRENGINE(MK_BOUNDARY)
DECLARE_PREDCORRENGINE(SA_BOUNDARY)
DECLARE_PREDCORRENGINE(DYN_BOUNDARY)

