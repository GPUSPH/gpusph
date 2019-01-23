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
#include "device_core.cu"
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
/*		   Device functions to compute the shear rate                                                       */
/************************************************************************************************************/

//! Shear rate computation return
/*! When computing the shear rate, we can return three possible tensors:
 * - the shear rate tensor τ = (∇v + (∇v)ᵀ)/2;
 * - its double D = ∇v + (∇v)ᵀ;
 * - a matrix assembled from the shear tensor diagonal and the doubled one off-diagonal
 *
 * The purpose here is to share the common computational effort, and return the values
 * that are most useful in applications.
 */
enum ShearRateReturnType {
	TENSOR,
	DOUBLED_TENSOR,
	MIXED_TENSOR
};

//! Compute ∇v + (∇v)ᵀ or its doubled version
template<KernelType kerneltype, ShearRateReturnType ret_type>
__device__ __forceinline__
symtensor3 shearRate(
	int index, /* particle index */
	int3 const& gridPos, /* particle grid position */
	float4 const& pos, /* particle position (in cell) */
	float4 const& vel, /* particle velocity */
	neibs_list_params const& params) /* parameters needed to walk the neighbors list */
{
	// Gradients of the the velocity components
	float3 dvx = make_float3(0.0f);
	float3 dvy = make_float3(0.0f);
	float3 dvz = make_float3(0.0f);

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

		const float3 relPos_f = as_float3(relPos)*f;
		// Velocity Gradients
		dvx -= relVel.x*relPos_f;	// dvx = -∑mj/ρj vxij (ri - rj)/r ∂Wij/∂r
		dvy -= relVel.y*relPos_f;	// dvy = -∑mj/ρj vyij (ri - rj)/r ∂Wij/∂r
		dvz -= relVel.z*relPos_f;	// dvz = -∑mj/ρj vzij (ri - rj)/r ∂Wij/∂r
	} // end of loop through neighbors

	symtensor3 ret;
	/* Start by storing the mixed version: non-doubled diagonal elements,
	 * doubled off-diagonal elements
	 */
	ret.xx = dvx.x;
	ret.xy = dvx.y + dvy.x;
	ret.xz = dvx.z + dvz.x;
	ret.yy = dvy.y;
	ret.yz = dvy.z + dvz.y;
	ret.zz = dvz.z;

	/* If the doubled version was requested, double the diagonal elements */
	if (ret_type == DOUBLED_TENSOR) {
		ret.xx *= 2.0f;
		ret.yy *= 2.0f;
		ret.zz *= 2.0f;
	}
	/* If the actual tensor was requested, halve the off-diagonal elements */
	if (ret_type == TENSOR) {
		ret.xy *= 0.5f;
		ret.xz *= 0.5f;
		ret.yz *= 0.5f;
	}

	return ret;
}

//! Compute the square of the double contraction norm of the shear rate tensor D:D/2
/*! Given velocity (vx, vy, vz), we compute the shear rate norm as the square root
 * of D:D/2 = 2τ.τ = 2 ( (∂vx/∂x)² + (∂vy/∂y)² + (∂vz/∂z)² ) +
 *   (∂vx/∂y + ∂vy/∂x)² + (∂vx/∂z + ∂vz/∂x)² + (∂vy/∂z + ∂vz/∂y)²
 * following e.g. Alexandrou et al. (2001) JNNFM doi:10.1016/S0377-0257(01)00127-6.
 * This function returns its squared value (i.e. before taking the square root).
 *
 * The template parameter is used to determine how the tensor is encoded,
 * i.e. if we are being passed τ, D or the mixed form.
 *
 * \todo Other authors seem to use the second invariant defined as (Tr(D)^2 - Tr(D.D))/2,
 * we should explore the differences between the two, and possibly offer the option
 * to choose which norm to use.
 */
template<ShearRateReturnType shRate_type>
__device__ __forceinline__
float shearRateNorm2(symtensor3 const& shRate)
{
	// Start by adding the diagonal entries
	float diag_terms = shRate.xx*shRate.xx + shRate.yy*shRate.yy + shRate.zz*shRate.zz;
	/* If shRate encodes D, we need to halve since we got
	 * 4 ( (∂vx/∂x)² + (∂vy/∂y)² + (∂vz/∂z)² ),
	 * otherwise we need to double since we only got
	 * (∂vx/∂x)² + (∂vy/∂y)² + (∂vz/∂z)².
	 */
	if (shRate_type == DOUBLED_TENSOR)
		diag_terms *= 0.5f;
	else
		diag_terms *= 2.0f;

	float off_terms =  shRate.xy*shRate.xy + shRate.xz*shRate.xz + shRate.yz*shRate.yz;
	/* If shRate encodes τ, we need to multiply by 4 since we got
	 * ( (∂vx/∂y + ∂vy/∂x)² + (∂vx/∂z + ∂vz/∂x)² + (∂vy/∂z + ∂vz/∂y)² )/4
	 */
	if (shRate_type == TENSOR)
		off_terms *= 4.0f;

	return diag_terms + off_terms;
}

/************************************************************************************************************/

/************************************************************************************************************/
/*		Kernels and functions to compute the effective viscosity for generalized Newtonian rheologies		*/
/************************************************************************************************************/

//! Polynomial approximation to (1 - exp(-x))/x, of the given order
/** This can be written in Horner form as
 * (1 - x/2 ( 1 - x/3 ( 1 - x/4 (...))))
 * which we can write iteratively
 */
template<unsigned int order>
__device__ __forceinline__
float
horner_one_minus_exp_minus_over(float x, float inner)
{
	constexpr float div=(-1.0f/(order+1.0f));
	return horner_one_minus_exp_minus_over<order - 1>(x, fmaf(x*inner, div, 1.0f));
}

template<>
__device__ __forceinline__
float
horner_one_minus_exp_minus_over<1>(float x, float inner)
{
	return fmaf(x*inner, -0.5f, 1.0f);
}

template<unsigned int order>
__device__ __forceinline__
float
horner_one_minus_exp_minus_over(float x)
{
	constexpr float div=(-1.0f/(order+1.0f));
	return horner_one_minus_exp_minus_over<order - 1>(x, fmaf(x, div, 1.0f));
}

template<>
__device__ __forceinline__
float
horner_one_minus_exp_minus_over<1>(float x)
{
	return fmaf(x, -0.5f, 1.0f);
}

//! Forms of yield strength contribution
/** We have three forms for the yield strengt contribution:
 * - no contribution (e.g. from power law)
 * - standard contribution (y_s/\dot\gamma), which can become infinite
 * - regularized contribution (Papanastasiou etc)
 */
enum YsContrib
{
	NO_YS, ///< no yield strength
	STD_YS, ///< standard form
	REG_YS ///< regularized
};

//! Statically determine the yield strength contribution for the given rheological model
template<RheologyType rheologytype>
__device__ __forceinline__
constexpr YsContrib
yield_strength_type()
{
	return
		REGULARIZED_RHEOLOGY(rheologytype) ? REG_YS : // yield with regularization
		YIELDING_RHEOLOGY(rheologytype) ? STD_YS : // yield without regularization
			NO_YS; // everything else: should be just Newtonian and power-law
}
template<typename KP>
__device__ __forceinline__
enable_if_t< yield_strength_type<KP::rheologytype>() == NO_YS, float >
viscYieldTerm(int fluid, float shrate, KP const& params)
{
	return 0.0f;
}

//! Standard contribution from the yield strength
/** This has the potential to become infinite at vanishing shear rates.
 */
template<typename KP>
__device__ __forceinline__
enable_if_t< yield_strength_type<KP::rheologytype>() == STD_YS, float>
viscYieldTerm(int fluid, float shrate, KP const& params)
{
	return d_yield_strength[fluid]/shrate;
}

//! Regularized contribution from the yield strength
/** Instead of dividing by the shear rate norm, we multiply by
 * (1 - exp(-m \dot\gamma))/\dot\gamma
 * which tends to m at vanishing shear rates.
 */
template<typename KP>
__device__ __forceinline__
enable_if_t< yield_strength_type<KP::rheologytype>() == REG_YS, float >
viscYieldTerm(int fluid, float shrate, KP const& params)
{
	const float m = d_visc_regularization_param[fluid];
	// we use a Taylor series for shrate < 1/m,
	// the exponential form for shrate >= 1/m
	// TODO allow customization of linearization order,
	// 8 has currently been chosen because it should create
	// a relative error of less than 2.5e-7
	const float mx = m*shrate;
	float reg = 0.0f;
	if (mx < 1)
		reg = m*horner_one_minus_exp_minus_over<8>(mx);
	else
		reg = (1 - expf(-mx))/shrate;

	return d_yield_strength[fluid]*reg;
}

//! Linear dependency on the shear rate
/** For BINGHAM and PAPANASTASIOU */
template<typename KP>
__device__ __forceinline__
enable_if_t<not NONLINEAR_RHEOLOGY(KP::rheologytype), float >
viscShearTerm(int fluid, float shrate, KP const& params)
{
	return d_visccoeff[fluid];
}

//! Power-law dependency on the shear rate
/** For POWER_LAW, HERSCHEL_BULKLEY and ALEXANDROU */
template<typename KP>
__device__ __forceinline__
enable_if_t<POWERLAW_RHEOLOGY(KP::rheologytype), float >
viscShearTerm(int fluid, float shrate, KP const& params)
{
	return d_visccoeff[fluid]*powf(shrate, d_visc_nonlinear_param[fluid] - 1);
}

//! Exponential dependency on the shear rate
/** For DEKEE_TURCOTTE and ZHU */
template<typename KP>
__device__ __forceinline__
enable_if_t<EXPONENTIAL_RHEOLOGY(KP::rheologytype), float >
viscShearTerm(int fluid, float shrate, KP const& params)
{
	return d_visccoeff[fluid]*expf( -d_visc_nonlinear_param[fluid]*shrate );
}

//! Compute the kinematic viscosity from the dynamic one
/** This is only done if the computationa viscosity model is KINEMATIC,
 * or if adaptive timestepping is enabled
 */
template<typename KP>
__device__ __forceinline__
enable_if_t< (KP::ViscSpec::compvisc == KINEMATIC) ||
	(KP::simflags & ENABLE_DTADAPT)>
compute_kinvisc(KP const& params, float effvisc, float rhotilde, int fluid,
	float &kinvisc)
{
	kinvisc = effvisc/physical_density(rhotilde, fluid);
}
template<typename KP>
__device__ __forceinline__
enable_if_t< (KP::ViscSpec::compvisc != KINEMATIC) &&
	not (KP::simflags & ENABLE_DTADAPT)>
compute_kinvisc(KP const& params, float effvisc, float rhotilde, int fluid,
	float &kinvisc)
{ /* do nothing */ }

//! Store the effective viscosity (DYNAMIC computational model)
/*! In this case we store the viscosity as-is, since we computed the dynamic viscosity */
template<typename KP>
__device__ __forceinline__
enable_if_t< KP::ViscSpec::compvisc == DYNAMIC >
store_effective_visc(KP const& params, int index, float effvisc, float /* kinvisc */)
{
	params.effvisc[index] = effvisc;
}

//! Store the effective viscosity (KINEMATIC computational model)
/*! In this case we divide by the physical density */
template<typename KP>
__device__ __forceinline__
enable_if_t< KP::ViscSpec::compvisc == KINEMATIC >
store_effective_visc(KP const& params, int index, float /* effvisc */, float kinvisc)
{
	params.effvisc[index] = kinvisc;
}

//! Reduce the kinematic viscosity, if ENABLE_DTADAPT
template<typename KP>
__device__ __forceinline__
enable_if_t< KP::simflags & ENABLE_DTADAPT >
reduce_kinvisc(KP const& params, float kinvisc)
{
	__shared__ float sm_max[BLOCK_SIZE_SPS];
	sm_max[threadIdx.x] = kinvisc;
	maxBlockReduce(sm_max, params.cfl, 0);
}
//! Do nothing to reduce the kinematic viscosity, if not ENABLE_DTADAPT
template<typename KP>
__device__ __forceinline__
enable_if_t< not (KP::simflags & ENABLE_DTADAPT) >
reduce_kinvisc(KP const& params, float kinvisc)
{ /* do nothing */ }

//! Per-particle effective viscosity computation
/** The individual contributions are factored out in viscShearTerm and viscYieldTerm,
 * the common part (shear rate norm computation, clamping, storage) is in this kernel.
 * Moreover, in the case of adaptive time-stepping, we want the kernel to return
 * the maximum computed kinematic viscosity, so we do a pre-reduction to the CFL array.
 */
template<typename KP,
	KernelType kerneltype = KP::kerneltype,
	BoundaryType boundarytype = KP::boundarytype>
__global__ void
effectiveViscDevice(KP params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	float kinvisc = 0.0f;

	// do { } while (0) around the main body so that we can bail out
	// to the reduction
	do {
		if (index >= params.numParticles)
			break;

		const particleinfo info = tex1Dfetch(infoTex, index);

		// fluid number
		const int fluid = fluid_num(info);

		// read particle data from sorted arrays
#if PREFER_L1
		const float4 pos = params.posArray[index];
#else
		const float4 pos = tex1Dfetch(posTex, index);
#endif

		// skip inactive particles
		if (INACTIVE(pos))
			break;

		const float4 vel = tex1Dfetch(velTex, index);

		// Compute grid position of current particle
		const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

		// shear rate tensor
		symtensor3 tau = shearRate<kerneltype, MIXED_TENSOR>(index, gridPos, pos, vel, params);

		// shear rate norm
		const float SijSij_bytwo = shearRateNorm2<MIXED_TENSOR>(tau);
		const float S = sqrtf(SijSij_bytwo);

		float effvisc = 0.0f;

		// effective viscosity contribution from the shear rate norm
		// (e.g. k \dot\gamma^n for power law and Herschel–Bulkley)
		if (d_visccoeff[fluid] != 0.0f)
			effvisc += viscShearTerm(fluid, S, params);

		// add effective viscosity contribution from the yield strength
		// (e.g. tau_0/\dot\gamma for Bingham and Herschel–Bulkley)
		if (d_yield_strength[fluid] != 0.0f)
			effvisc += viscYieldTerm(fluid, S, params);

		// Clamp to the user-set limiting viscosity
		effvisc = fminf(effvisc, d_limiting_kinvisc*d_rho0[fluid]);

		compute_kinvisc(params, effvisc, vel.w, fluid, kinvisc);
		store_effective_visc(params, index, effvisc, kinvisc);
	} while (0);

	reduce_kinvisc(params, kinvisc);

}


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

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

	symtensor3 tau = shearRate<kerneltype, MIXED_TENSOR>(index, gridPos, pos, vel, params);

	// Calculate Sub-Particle Scale viscosity
	// and special turbulent terms
	const float SijSij_bytwo = shearRateNorm2<MIXED_TENSOR>(tau);
	const float S = sqrtf(SijSij_bytwo);
	const float nu_SPS = d_smagfactor*S;		// Dalrymple & Rogers (2006): eq. (12)
	const float divu_SPS = 0.6666666666f*nu_SPS*(tau.xx + tau.yy + tau.zz);
	const float Blinetal_SPS = d_kspsfactor*SijSij_bytwo;

	// Storing the turbulent viscosity for each particle
	write_sps_turbvisc<simflags & SPSK_STORE_TURBVISC>::with(params, index, nu_SPS);

	// Shear Stress matrix = TAU (pronounced taf)
	// Dalrymple & Rogers (2006): eq. (10)
	if (simflags & SPSK_STORE_TAU) {

		const float rho = physical_density(vel.w, fluid_num(info));

		/* Since tau stores the diagonal components non-doubled, but we need the doubled
		 * ones, we double them here */

		tau.xx = nu_SPS*(tau.xx+tau.xx) - divu_SPS - Blinetal_SPS;	// tau11 = tau_xx/ρ^2
		tau.xx /= rho;
		tau.xy *= nu_SPS/rho;								// tau12 = tau_xy/ρ^2
		tau.xz *= nu_SPS/rho;								// tau13 = tau_xz/ρ^2
		tau.yy = nu_SPS*(tau.yy+tau.yy) - divu_SPS - Blinetal_SPS;	// tau22 = tau_yy/ρ^2
		tau.yy /= rho;
		tau.yz *= nu_SPS/rho;								// tau23 = tau_yz/ρ^2
		tau.zz = nu_SPS*(tau.zz+tau.zz) - divu_SPS - Blinetal_SPS;	// tau33 = tau_zz/ρ^2
		tau.zz /= rho;

		write_sps_tau<simflags & SPSK_STORE_TAU>::with(params, index, tau);
	}
}

}

#endif
