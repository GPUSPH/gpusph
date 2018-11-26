/*  Copyright 2018 Giuseppe Bilotta, Alexis Hérault, Robert A. Dalrymple, Ciro Del Negro

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

/*! \file
 * ViscEngine CUDA kernels
 */

#ifndef _VISC_KERNEL_
#define _VISC_KERNEL_

#include "visc_params.h"
#include "visc_avg.cu"

#include "tensor.cu"

// Include files which we access the namespace of.
// This is normally not needed because all files are included indirectly
// in the problem file via the inclusion of cudasimframework.cu, but if
// we ever need to compile it on its own, having the include here helps a lot
// (also improves autocompletion and real-time error detection in smart editors)
#include "sph_core.cu"
#include "phys_core.cu"
#include "buildneibs_kernel.cu"

// TODO these block sizes should be autotuned
#if (__COMPUTE__ >= 20)
	#define BLOCK_SIZE_SPS			128
	#define MIN_BLOCKS_SPS			6
#else
	#define BLOCK_SIZE_SPS			128
	#define MIN_BLOCKS_SPS			1
#endif

/** \namespace cuvisc
 * \brief Contains all device functions/kernels/variables used for viscosity computation
 *
 */
namespace cuvisc {

using namespace cusph;
using namespace cuphys;
using namespace cuneibs;

/************************************************************************************************************/
/*		   Device functions to compute the laminar viscous contributions                                    */
/************************************************************************************************************/

//! Artificial viscosity
__device__ __forceinline__ float
artvisc(	const float	vel_dot_pos,
			const float	rho,
			const float	neib_rho,
			const float	sspeed,
			const float	neib_sspeed,
			const float	r,
			const float	slength)
{
	return vel_dot_pos*slength*d_artvisccoeff*(sspeed + neib_sspeed)/
			((r*r + d_epsartvisc)*(rho + neib_rho));
}

/************************************************************************************************************/


/************************************************************************************************************/
/*		   Kernels for computing SPS tensor and SPS viscosity												*/
/************************************************************************************************************/

//! A functor that writes out turbvisc for SPS visc
template<bool>
struct write_sps_turbvisc
{
	template<typename FP>
	__device__ __forceinline__
	static void
	with(FP const& params, const uint index, const float turbvisc)
	{ /* do nothing */ }
};

template<>
template<typename FP>
__device__ __forceinline__ void
write_sps_turbvisc<true>::with(FP const& params, const uint index, const float turbvisc)
{ params.turbvisc[index] = turbvisc; }

//! A functor that writes out tau for SPS visc
template<bool>
struct write_sps_tau
{
	template<typename FP>
	__device__ __forceinline__
	static void
	with(FP const& params, const uint index, symtensor3 const& tau)
	{ /* do nothing */ }
};

template<>
template<typename FP>
__device__ __forceinline__ void
write_sps_tau<true>::with(FP const& params, const uint index, symtensor3 const& tau)
{
	storeTau(tau, index, params.tau0, params.tau1, params.tau2);
}

/************************************************************************************************************/


//! Compute SPS matrix
/*!
 Compute the Sub-Particle-Stress (SPS) Tensor matrix for all Particles
 WITHOUT Kernel correction

 Procedure:

 (1) compute velocity gradients

 (2) compute turbulent eddy viscosity (non-dynamic)

 (3) compute turbulent shear stresses

 (4) return SPS tensor matrix (tau) divided by rho^2
*/
template<KernelType kerneltype,
	BoundaryType boundarytype,
	uint simflags>
__global__ void
__launch_bounds__(BLOCK_SIZE_SPS, MIN_BLOCKS_SPS)
SPSstressMatrixDevice(sps_params<kerneltype, boundarytype, simflags> params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	// Compute SPS matrix only for any kind of particles
	// TODO testpoints should also compute SPS, it'd be useful
	// when we will enable SPS saving to disk
	const particleinfo info = tex1Dfetch(infoTex, index);

	// read particle data from sorted arrays
	#if PREFER_L1
	const float4 pos = params.posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	// skip inactive particles
	if (INACTIVE(pos))
		return;

	const float4 vel = tex1Dfetch(velTex, index);

	// Gradients of the the velocity components
	float3 dvx = make_float3(0.0f);
	float3 dvy = make_float3(0.0f);
	float3 dvz = make_float3(0.0f);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

	// Loop over all neighbors to compute their contribution to the velocity gradient
	// TODO: check which particle types should contribute with SA
	for_each_neib2(PT_FLUID, PT_BOUNDARY, index, pos, gridPos, params.cellStart, params.neibsList) {

		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos(
		#if PREFER_L1
			params.posArray[neib_index]
		#else
			tex1Dfetch(posTex, neib_index)
		#endif
			);

		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);
		const float r = length3(relPos);

		// skip inactive particles and particles outside of the kernel support
		if (INACTIVE(relPos) || r >= params.influenceradius)
			continue;

		// Compute relative velocity
		// Now relVel is a float4 and neib density is stored in relVel.w
		const float4 relVel = as_float3(vel) - tex1Dfetch(velTex, neib_index);

		const float neib_rho = physical_density(relVel.w, fluid_num(neib_info));
		const float f = F<kerneltype>(r, params.slength)*relPos.w/neib_rho;	// 1/r ∂Wij/∂r Vj

		// Velocity Gradients
		dvx -= relVel.x*as_float3(relPos)*f;	// dvx = -∑mj/ρj vxij (ri - rj)/r ∂Wij/∂r
		dvy -= relVel.y*as_float3(relPos)*f;	// dvy = -∑mj/ρj vyij (ri - rj)/r ∂Wij/∂r
		dvz -= relVel.z*as_float3(relPos)*f;	// dvz = -∑mj/ρj vzij (ri - rj)/r ∂Wij/∂r
	} // end of loop through neighbors


	// SPS stress matrix elements
	symtensor3 tau;

	// Calculate Sub-Particle Scale viscosity
	// and special turbulent terms
	float SijSij_bytwo = 2.0f*(dvx.x*dvx.x + dvy.y*dvy.y + dvz.z*dvz.z);	// 2*SijSij = 2.0((∂vx/∂x)^2 + (∂vy/∂yx)^2 + (∂vz/∂z)^2)
	float temp = dvx.y + dvy.x;		// 2*SijSij += (∂vx/∂y + ∂vy/∂x)^2
	tau.xy = temp;
	SijSij_bytwo += temp*temp;
	temp = dvx.z + dvz.x;			// 2*SijSij += (∂vx/∂z + ∂vz/∂x)^2
	tau.xz = temp;
	SijSij_bytwo += temp*temp;
	temp = dvy.z + dvz.y;			// 2*SijSij += (∂vy/∂z + ∂vz/∂y)^2
	tau.yz = temp;
	SijSij_bytwo += temp*temp;
	const float S = sqrtf(SijSij_bytwo);
	const float nu_SPS = d_smagfactor*S;		// Dalrymple & Rogers (2006): eq. (12)
	const float divu_SPS = 0.6666666666f*nu_SPS*(dvx.x + dvy.y + dvz.z);
	const float Blinetal_SPS = d_kspsfactor*SijSij_bytwo;

	// Storing the turbulent viscosity for each particle
	write_sps_turbvisc<simflags & SPSK_STORE_TURBVISC>::with(params, index, nu_SPS);

	// Shear Stress matrix = TAU (pronounced taf)
	// Dalrymple & Rogers (2006): eq. (10)
	if (simflags & SPSK_STORE_TAU) {

		const float rho = physical_density(vel.w, fluid_num(info));

		tau.xx = nu_SPS*(dvx.x + dvx.x) - divu_SPS - Blinetal_SPS;	// tau11 = tau_xx/ρ^2
		tau.xx /= rho;
		tau.xy *= nu_SPS/rho;								// tau12 = tau_xy/ρ^2
		tau.xz *= nu_SPS/rho;								// tau13 = tau_xz/ρ^2
		tau.yy = nu_SPS*(dvy.y + dvy.y) - divu_SPS - Blinetal_SPS;	// tau22 = tau_yy/ρ^2
		tau.yy /= rho;
		tau.yz *= nu_SPS/rho;								// tau23 = tau_yz/ρ^2
		tau.zz = nu_SPS*(dvz.z + dvz.z) - divu_SPS - Blinetal_SPS;	// tau33 = tau_zz/ρ^2
		tau.zz /= rho;

		write_sps_tau<simflags & SPSK_STORE_TAU>::with(params, index, tau);
	}
}

}

#endif
