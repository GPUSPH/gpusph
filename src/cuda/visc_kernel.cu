/*  Copyright (c) 2018-2019 INGV, EDF, UniCT, JHU

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

struct common_shear_rate_pdata
{
	const int index;
	const int3 gridPos;
	const float4 pos;
	const float4 vel;
	const particleinfo info;
	const uint fluid;

	template<typename KP>// kernel params
	__device__ __forceinline__
	common_shear_rate_pdata(int _index, KP const& params) :
		index(_index),
		gridPos( calcGridPosFromParticleHash(params.particleHash[index]) ),
		pos(params.fetchPos(index)),
		vel(params.fetchVel(index)),
		info( params.fetchInfo(index) ),
		fluid( fluid_num(info) )
	{}
};

struct sa_shear_rate_pdata
{
	const float gamma;
	const float ref_volume0;

	template<typename KP>// kernel params
	__device__ __forceinline__
	sa_shear_rate_pdata(int index, KP const& params) :
		gamma( params.fetchGradGamma(index).w),
		ref_volume0(params.deltap*params.deltap*params.deltap)
	{}
};

template<BoundaryType _boundarytype>
struct shear_rate_pdata :
	common_shear_rate_pdata,
	COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_shear_rate_pdata)
{
	static constexpr BoundaryType boundarytype = _boundarytype;

	template<typename KP>// kernel params
	__device__ __forceinline__
	shear_rate_pdata(int index, KP const& params) :
		common_shear_rate_pdata(index, params),
		COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_shear_rate_pdata)(index, params)
	{}
};

struct common_shear_rate_ndata
{
	const uint index;
	const float4 relPos;
	const float4 relVel;
	const particleinfo info;
	const float r;
	const uint fluid;
	const float rho;

	template<typename NeibIter, typename P_t, typename KP>
	__device__ __forceinline__
	common_shear_rate_ndata(NeibIter const& neib_iter, P_t const& pdata, KP const& params) :
		index(neib_iter.neib_index()),
		relPos(neib_iter.relPos(params.fetchPos(index))),
		info(params.fetchInfo(index)),
		r(length3(relPos)),
		relVel( make_float3(pdata.vel) - params.fetchVel(index) ),
		fluid(fluid_num(info)),
		rho(physical_density(relVel.w, fluid))
	{}
};

template<BoundaryType boundarytype>
struct shear_rate_ndata :
	common_shear_rate_ndata
	// we whould add the SA_BOUNDARY-specific data here too,
	// but it's neighbor-dependent and we didn't do a splitneibs for this too
	// (and it might not be worth it)
{
	template<typename NeibIter, typename KP>
	__device__ __forceinline__
	shear_rate_ndata(NeibIter const& neib_iter, shear_rate_pdata<boundarytype> const& pdata, KP const& params) :
		common_shear_rate_ndata(neib_iter, pdata, params)
	{}

};

//! Shear rate contribution (multiplier for -relVel)
//! Non-SA case
template<typename P_t, typename N_t, typename KP>
__device__ __forceinline__
enable_if_t<KP::boundarytype != SA_BOUNDARY, float3>
shear_rate_contrib(P_t const& pdata, N_t const& ndata, KP const& params)
{
	// dvx = -∑mj/ρj vxij (ri - rj)/r ∂Wij/∂r
	// dvy = -∑mj/ρj vyij (ri - rj)/r ∂Wij/∂r
	// dvz = -∑mj/ρj vzij (ri - rj)/r ∂Wij/∂r
	const float f = F<KP::kerneltype>(ndata.r, params.slength);	// 1/r ∂Wij/∂r
	const float weight = f*ndata.relPos.w/ndata.rho; // F_ij * V_j
	return make_float3(ndata.relPos)*weight;
}

//! Shear rate contribution (multiplier for -relVel)
//! SA case
template<typename P_t, typename N_t, typename KP>
__device__ __forceinline__
enable_if_t<KP::boundarytype == SA_BOUNDARY, float3>
shear_rate_contrib(P_t const& pdata, N_t const& ndata, KP const& params)
{
	// but multiplication by 1/gamma wil be done in shear_rate_fixup
	// dvx = -1/gamma*∑mj/ρj vxij (ri - rj)/r ∂Wij/∂r + 1/gamma*∑ vxij (ri - rj)/r ∂gamma/∂r
	// dvy = -1/gamma*∑mj/ρj vyij (ri - rj)/r ∂Wij/∂r + 1/gamma*∑ vyij (ri - rj)/r ∂gamma/∂r
	// dvz = -1/gamma*∑mj/ρj vzij (ri - rj)/r ∂Wij/∂r + 1/gamma*∑ vzij (ri - rj)/r ∂gamma/∂r
	const auto nptype = PART_TYPE(ndata.info);

	if (nptype != PT_BOUNDARY) {
		const float f = F<KP::kerneltype>(ndata.r, params.slength);	// 1/r ∂Wij/∂r
		const float weight = f*ndata.relPos.w/ndata.rho; // F_ij * V_j
		return make_float3(ndata.relPos)*weight;
	} else {
		const float4 belem = params.fetchBound(ndata.index);
		const float3 normal_s = as_float3(belem);
		const float3 q = as_float3(ndata.relPos)/params.slength;
		float3 q_vb[3];
		calcVertexRelPos(q_vb, belem,
			params.vertPos0[ndata.index], params.vertPos1[ndata.index], params.vertPos2[ndata.index], params.slength);
		const float ggamAS = gradGamma<KP::kerneltype>(params.slength, q, q_vb, normal_s);

		return -ggamAS*normal_s;
	}
}

//! Fixup right hand-side term with SA boundary elements contribution
//! Non-SA case: nothing to do.
template<typename N_t, typename KP>
__device__ __forceinline__
enable_if_t<KP::boundarytype != SA_BOUNDARY>
sa_boundary_jacobi_build_vector(float &B, N_t const& ndata, KP const& params)
{ /* do nothing */}

//! Fixup right hand-side term with SA boundary elements contribution
//! SA case: nothing to do.
template<typename N_t, typename KP>
__device__ __forceinline__
enable_if_t<KP::boundarytype == SA_BOUNDARY>
sa_boundary_jacobi_build_vector(float &B, N_t const& ndata, KP const& params)
{
	// Definition of delta_rho
	const float delta_rho = cuphys::d_numfluids > 1 ? abs(d_rho0[0]-d_rho0[1]) : d_rho0[0];

	const float4 belem = params.fetchBound(ndata.index);
	const float3 normal_s = as_float3(belem);
	const float3 q = as_float3(ndata.relPos)/params.slength;
	float3 q_vb[3];
	calcVertexRelPos(q_vb, belem,
		params.vertPos0[ndata.index], params.vertPos1[ndata.index], params.vertPos2[ndata.index], params.slength);
	const float ggamAS = gradGamma<KP::kerneltype>(params.slength, q, q_vb, normal_s);
	float r_as(fmax(fabs(dot(as_float3(ndata.relPos), normal_s)), params.deltap));

	// Contribution to the boundary elements to the right hand-side term
	B += delta_rho*dot(d_gravity, normal_s)*ggamAS;
}


//! Post-neib-iteration fixup for dvx, dvy, dvz
//! Non-SA case
template<typename P_t, typename KP>
__device__ __forceinline__
enable_if_t<KP::boundarytype != SA_BOUNDARY>
shear_rate_fixup(float3& dvx, float3& dvy, float3& dvz, P_t const& pdata, KP const& params)
{ /* do nothing */ }

//! Post-neib-iteration fixup for dvx, dvy, dvz
//! SA case
template<typename P_t, typename KP>
__device__ __forceinline__
enable_if_t<KP::boundarytype == SA_BOUNDARY>
shear_rate_fixup(float3& dvx, float3& dvy, float3& dvz, P_t const& pdata, KP const& params)
{
	const float multiplier = 1./pdata.gamma;
	dvx *= multiplier;
	dvy *= multiplier;
	dvz *= multiplier;
}

//! Compute ∇v + (∇v)ᵀ or its doubled version
template<
	typename KP,
	KernelType kerneltype = KP::kerneltype,
	BoundaryType boundarytype = KP::boundarytype,
	typename P_t = shear_rate_pdata<boundarytype>
>
__device__ __forceinline__
void velocity_gradient(P_t const& pdata, KP const& params, float3& dvx, float3& dvy, float3& dvz)
{
	// Loop over all neighbors to compute their contribution to the velocity gradient
	for_every_neib(boundarytype, pdata.index, pdata.pos, pdata.gridPos, params.cellStart, params.neibsList) {

		shear_rate_ndata<boundarytype> ndata(neib_iter, pdata, params);

		// skip inactive particles and particles outside of the kernel support
		if (INACTIVE(ndata.relPos) || ndata.r >= params.influenceradius)
			continue;

		const float3 relVel_multiplier = shear_rate_contrib(pdata, ndata, params);
		// Velocity Gradients
		dvx -= ndata.relVel.x*relVel_multiplier;
		dvy -= ndata.relVel.y*relVel_multiplier;
		dvz -= ndata.relVel.z*relVel_multiplier;
	} // end of loop through neighbors

	shear_rate_fixup(dvx, dvy, dvz, pdata, params);
}


//! Compute ∇v + (∇v)ᵀ or its doubled version
template<
	ShearRateReturnType ret_type,
	typename KP,
	KernelType kerneltype = KP::kerneltype,
	BoundaryType boundarytype = KP::boundarytype,
	typename P_t = shear_rate_pdata<boundarytype>
>
__device__ __forceinline__
symtensor3 shearRate(
	P_t const& pdata,
	KP const& params)
{
	// Gradients of the the velocity components
	float3 dvx = make_float3(0.0f);
	float3 dvy = make_float3(0.0f);
	float3 dvz = make_float3(0.0f);

	velocity_gradient(pdata, params, dvx, dvy, dvz);

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

template<typename P_t, typename KP>
__device__ __forceinline__
enable_if_t< yield_strength_type<KP::rheologytype>() == NO_YS, float >
viscYieldTerm(P_t const& pdata, float shrate, KP const& params)
{
	return 0.0f;
}

//! Standard contribution from the yield strength
/** This has the potential to become infinite at vanishing shear rates.
 */
template<typename P_t, typename KP>
__device__ __forceinline__
enable_if_t< yield_strength_type<KP::rheologytype>() == STD_YS, float>
viscYieldTerm(P_t const& pdata, float shrate, KP const& params)
{
	return d_yield_strength[pdata.fluid]/shrate;
}

//! Regularized contribution from the yield strength
/** Instead of dividing by the shear rate norm, we multiply by
 * (1 - exp(-m \dot\gamma))/\dot\gamma
 * which tends to m at vanishing shear rates.
 */
template<typename P_t, typename KP>
__device__ __forceinline__
enable_if_t< yield_strength_type<KP::rheologytype>() == REG_YS, float >
viscYieldTerm(P_t const& pdata, float shrate, KP const& params)
{
	const float m = d_visc_regularization_param[pdata.fluid];
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

	return d_yield_strength[pdata.fluid]*reg;
}

//! Linear dependency on the shear rate
/** For BINGHAM and PAPANASTASIOU */
template<typename P_t, typename KP>
__device__ __forceinline__
enable_if_t<not NONLINEAR_RHEOLOGY(KP::rheologytype) && KP::rheologytype != GRANULAR, float >
viscShearTerm(P_t const& pdata, float shrate, KP const& params)
{
	return d_visccoeff[pdata.fluid];
}

//! Power-law dependency on the shear rate
/** For POWER_LAW, HERSCHEL_BULKLEY and ALEXANDROU */
template<typename P_t, typename KP>
__device__ __forceinline__
enable_if_t<POWERLAW_RHEOLOGY(KP::rheologytype), float >
viscShearTerm(P_t const& pdata, float shrate, KP const& params)
{
	return d_visccoeff[pdata.fluid]*powf(shrate, d_visc_nonlinear_param[pdata.fluid] - 1);
}

//! Exponential dependency on the shear rate
/** For DEKEE_TURCOTTE and ZHU */
template<typename P_t, typename KP>
__device__ __forceinline__
enable_if_t<EXPONENTIAL_RHEOLOGY(KP::rheologytype), float >
viscShearTerm(P_t const& pdata, float shrate, KP const& params)
{
	return d_visccoeff[pdata.fluid]*expf( -d_visc_nonlinear_param[pdata.fluid]*shrate );
}

//! Granular flow shear rate contribution
/** TODO: this is actually a yield-stress-like contribution for sediment particles,
 * with a non-constant yield stress computed from the effective pressure, but for
 * fluid particles it's the standard Newtonian contribution. In the future we also
 * want to have a full \mu(I) contribution, in which case it'll probably be
 * a good idea to split this into a viscYieldTerm with dynamic yield stress
 */
template<typename P_t, typename KP>
__device__ __forceinline__
enable_if_t<KP::rheologytype == GRANULAR, float >
viscShearTerm(P_t const& pdata, float shrate, KP const& params)
{
	// TODO use a pdata structure
	const float effpres = params.fetchEffPres(pdata.index);

	// for non-fluid particles we do not compute viscosity,
	// we will use the central particle viscosity during interaction
	if (!FLUID(pdata.info))
		return NAN;

	// Newtonian rheology is assumed for the pure fluid component
	if (!SEDIMENT(pdata.info))
		return d_visccoeff[pdata.fluid];

#define SQRT3_TIMES_2 3.46410161514f
	// granular rheology is used for sediment
	const float tau_y = SQRT3_TIMES_2*d_sinpsi[pdata.fluid]/(3.f - d_sinpsi[pdata.fluid])*effpres;
	return tau_y/shrate;
}

//! Clamp the viscosity
/** General case: viscosity cannot be higher than the limiting viscosity */
template<typename KP>
__device__ __forceinline__
enable_if_t< KP::rheologytype != GRANULAR, float >
clamp_visc(KP const& params, float effvisc, int fluid)
{
	return fminf(effvisc, d_limiting_kinvisc*d_rho0[fluid]);
}

//! Clamp the viscosity
/** Granular flow case: the viscosity is also clamped from below,
 * since the sediment viscosity cannot be lower than the interstitial fluid viscosity
 */
template<typename KP>
__device__ __forceinline__
enable_if_t< KP::rheologytype == GRANULAR, float >
clamp_visc(KP const& params, float effvisc, int fluid)
{
	return clamp(effvisc, d_visccoeff[fluid], d_limiting_kinvisc*d_rho0[fluid]);
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

//! Reduce the Jacobi error
__device__ __forceinline__ void
reduce_jacobi_error(float* cfl, float error)
{
	__shared__ float sm_max[BLOCK_SIZE_SPS];
	sm_max[threadIdx.x] = error;
	maxBlockReduce(sm_max, cfl, 0);
}

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

		shear_rate_pdata<boundarytype> pdata(index, params);

		if (KP::rheologytype == GRANULAR && !FLUID(pdata.info))
			break;

		// skip inactive particles
		if (INACTIVE(pdata.pos))
			break;

		// shear rate tensor
		symtensor3 tau = shearRate<MIXED_TENSOR>(pdata, params);

		// shear rate norm
		const float SijSij_bytwo = shearRateNorm2<MIXED_TENSOR>(tau);
		const float S = sqrtf(SijSij_bytwo);

		float effvisc = 0.0f;

		// effective viscosity contribution from the shear rate norm
		// (e.g. k \dot\gamma^n for power law and Herschel–Bulkley)
		if (d_visccoeff[pdata.fluid] != 0.0f)
			effvisc += viscShearTerm(pdata, S, params);

		// add effective viscosity contribution from the yield strength
		// (e.g. tau_0/\dot\gamma for Bingham and Herschel–Bulkley)
		if (d_yield_strength[pdata.fluid] != 0.0f)
			effvisc += viscYieldTerm(pdata, S, params);

		// Clamp to the user-set limiting viscosity
		effvisc = clamp_visc(params, effvisc, pdata.fluid);

		compute_kinvisc(params, effvisc, pdata.vel.w, pdata.fluid, kinvisc);
		store_effective_visc(params, pdata.index, effvisc, kinvisc);
	} while (0);

	reduce_kinvisc(params, kinvisc);

}


/************************************************************************************************************/
/*		   Kernels for computing SPS tensor and SPS viscosity												*/
/************************************************************************************************************/

//! Write out SPS turbulent viscosity, if SPSK_STORE_TURBVISC is enabled
template<typename FP>
__device__ __forceinline__
enable_if_t<not (FP::sps_simflags & SPSK_STORE_TURBVISC)>
write_sps_turbvisc(FP const& params, const uint index, const float turbvisc)
{ /* do nothing */ }

template<typename FP>
__device__ __forceinline__
enable_if_t<(FP::sps_simflags & SPSK_STORE_TURBVISC)>
write_sps_turbvisc(FP const& params, const uint index, const float turbvisc)
{ params.turbvisc[index] = turbvisc; }

//! Write out SPS stress tensor (tau), if SPSK_STORE_TAU is enabled
template<typename FP>
__device__ __forceinline__
enable_if_t<not (FP::sps_simflags & SPSK_STORE_TAU)>
write_sps_tau(FP const& params, const uint index, symtensor3 const& tau)
{ /* do nothing */ }

template<typename FP>
__device__ __forceinline__
enable_if_t<(FP::sps_simflags & SPSK_STORE_TAU)>
write_sps_tau(FP const& params, const uint index, symtensor3 const& tau)
{ params.storeTau(tau, index); }

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
	uint sps_simflags>
__global__ void
__launch_bounds__(BLOCK_SIZE_SPS, MIN_BLOCKS_SPS)
SPSstressMatrixDevice(sps_params<kerneltype, boundarytype, sps_simflags> params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	shear_rate_pdata<boundarytype> pdata(index, params);

	// skip inactive particles
	if (INACTIVE(pdata.pos))
		return;

	symtensor3 tau = shearRate<MIXED_TENSOR>(pdata, params);

	// Calculate Sub-Particle Scale viscosity
	// and special turbulent terms
	const float SijSij_bytwo = shearRateNorm2<MIXED_TENSOR>(tau);
	const float S = sqrtf(SijSij_bytwo);
	const float nu_SPS = d_smagfactor*S;		// Dalrymple & Rogers (2006): eq. (12)
	const float divu_SPS = 0.6666666666f*nu_SPS*(tau.xx + tau.yy + tau.zz);
	const float Blinetal_SPS = d_kspsfactor*SijSij_bytwo;

	// Storing the turbulent viscosity for each particle
	write_sps_turbvisc(params, index, nu_SPS);

	// Shear Stress matrix = TAU (pronounced taf)
	// Dalrymple & Rogers (2006): eq. (10)
	if (sps_simflags & SPSK_STORE_TAU) {

		const float rho = physical_density(pdata.vel.w, pdata.fluid);

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

		write_sps_tau(params, index, tau);
	}
}

__global__ void
__launch_bounds__(BLOCK_SIZE_SPS, MIN_BLOCKS_SPS)
jacobiFSBoundaryConditionsDevice(
	pos_info_wrapper params,
	float * __restrict__ effpres,
	uint numParticles,
	float deltap)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	const float4 pos = params.fetchPos(index);

	const particleinfo info = params.fetchInfo(index);
	const ParticleType cptype = PART_TYPE(info);

	// skip inactive particles
	if (INACTIVE(pos))
		return;

	// Fluid number
	const uint p_fluid_num = fluid_num(info);
	const uint numFluids = cuphys::d_numfluids;

	// Definition of delta_rho
	float delta_rho = d_rho0[0];
	if (numFluids > 1) delta_rho = abs(d_rho0[0] - d_rho0[1]);

	// * for free-surface particles, the Dirichlet condition is enforced
	if (cptype == PT_FLUID && SEDIMENT(info) && (SURFACE(info) || INTERFACE(info))) {
		effpres[index] = deltap*delta_rho*length(d_gravity)/2.;
	} else {
		return;
	}
}

template<KernelType kerneltype,
	BoundaryType boundarytype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SPS, MIN_BLOCKS_SPS)
jacobiWallBoundaryConditionsDevice(jacobi_wall_boundary_params<kerneltype, boundarytype> params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	// the vertex Neumann condition is enforced through the interpolation of
	// free particles effpres values. Thus it has to be re-computed every
	// Jacobi iteration. We need to monitor the backward error of vertex
	// particles to ensure that the solver keeps iterating until the vertex
	// effpres value does not vary anymore.
	float backErr = 0.f;

	// do { } while (0) around the main body so that we can bail out
	// to the reduction
	do {
		if (index >= params.numParticles)
			break;

		const float oldEffPres = params.effpres[index];
		// effpres initilization:
		// * for vertex and boundary, effpres will be calculated from a
		// Shepard interpolation to enforce a Neuman condition.
		// * for free particles, effpres is not modifies here. It is set 
		// to its old value.
		float newEffPres = 0.f;

		// read particle data from sorted arrays
		const float4 pos = params.fetchPos(index);

		const particleinfo info = params.fetchInfo(index);
		const ParticleType cptype = PART_TYPE(info);

		// skip inactive particles
		if (INACTIVE(pos))
			break;

		const float4 vel = params.fetchVel(index);

		// Compute grid position of current particle
		const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

		// Fluid number
		const uint p_fluid_num = fluid_num(info);
		const uint numFluids = cuphys::d_numfluids;

		// Definition of delta_rho
		float delta_rho = d_rho0[0];
		if (numFluids > 1) delta_rho = abs(d_rho0[0] - d_rho0[1]);

		if ((boundarytype != SA_BOUNDARY && cptype == PT_BOUNDARY) ||
		(boundarytype == SA_BOUNDARY && cptype == PT_VERTEX)) {
			float alpha = 0.f; // shepard filter
			// loop over fluid neibs
			for_each_neib(PT_FLUID, index, pos, gridPos, params.cellStart, params.neibsList) {
				const uint neib_index = neib_iter.neib_index();

				// Compute relative position vector and distance
				// Now relPos is a float4 and neib mass is stored in relPos.w
				const float4 relPos = neib_iter.relPos( params.fetchPos(neib_index) );

				const float neib_oldEffPres = params.effpres[neib_index];
				const particleinfo neib_info = params.fetchInfo(neib_index);

				// skip inactive particles
				if (INACTIVE(relPos))
					continue;

				if (SEDIMENT(neib_info)) {
					const float r = length3(relPos);

					// Compute relative velocity
					// Now relVel is a float4 and neib density is stored in relVel.w
					const float4 relVel = as_float3(vel) - params.fetchVel(neib_index);
					const ParticleType nptype = PART_TYPE(neib_info);

					// Fluid numbers
					const uint neib_fluid_num = fluid_num(neib_info);

					// contribution of free particles
					const float w = W<kerneltype>(r, params.slength);	// Wij	
					const float neib_volume = relPos.w/physical_density(relVel.w, fluid_num(neib_info));
					newEffPres += fmax(neib_volume*(neib_oldEffPres + delta_rho*dot(d_gravity, as_float3(relPos)))*w, 0.f);
					alpha += neib_volume*w;
				}
			} // end of loop through neighbors
			if (alpha > 0.f) {
				newEffPres /= alpha;
				// Compute a ref pressure for the current case.
				float refpres = delta_rho*(d_sscoeff[0]/10.)*(d_sscoeff[0]/10.);
				backErr = abs(newEffPres-oldEffPres)/refpres;
			} else {
				newEffPres = 0.f;
			}
			params.effpres[index] = newEffPres;
		}
	} while (0);
	reduce_jacobi_error(params.cfl, backErr);
}

// Jacobi vectors building
template<
	typename KP,
	KernelType kerneltype = KP::kerneltype,
	BoundaryType boundarytype = KP::boundarytype,
	typename P_t = shear_rate_pdata<boundarytype>
>
__global__ void
__launch_bounds__(BLOCK_SIZE_SPS, MIN_BLOCKS_SPS)
jacobiBuildVectorsDevice(KP params,
	float4 *jacobiBuffer)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	shear_rate_pdata<boundarytype> pdata(index, params);

	// Initialize vectors
	float D = 0;
	float Rx = 0;
	float B = 0;

	// read particle data from sorted arrays
	const float4 pos = params.fetchPos(index);
	const particleinfo info = params.fetchInfo(index);
	const ParticleType cptype = PART_TYPE(info);

	// skip inactive particles
	if (INACTIVE(pos))
		return;

	const float4 vel = params.fetchVel(index);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

	// Jacobi vectors are built for free particles of sediment that are not at
	// the interface nor at the free-surface.
	if (cptype == PT_FLUID && SEDIMENT(info) && !INTERFACE(info) && !SURFACE(info))
	{
		/* Loop over neighbours */
		for_every_neib(boundarytype, pdata.index, pdata.pos, pdata.gridPos, params.cellStart, params.neibsList) {

			const shear_rate_ndata<boundarytype> ndata(neib_iter, pdata, params);

			const float neib_oldEffPres = params.fetchEffPres(ndata.index);

			// skip inactive particles
			if (INACTIVE(ndata.relPos) || ndata.r >= params.influenceradius)
				continue;

			const ParticleType nptype = PART_TYPE(ndata.info);

			// contribution of vertex and free particles of sediment
			if ((nptype == PT_FLUID && SEDIMENT(ndata.info)) ||
					(boundarytype != SA_BOUNDARY && nptype == PT_BOUNDARY) ||
					(boundarytype == SA_BOUNDARY && nptype == PT_VERTEX))
			{
				const float f = F<kerneltype>(ndata.r, params.slength);	// 1/r ∂Wij/∂r
				const float neib_volume = ndata.relPos.w/ndata.rho;

				D += neib_volume*f;

				// sediment fluid neibs contribute to the matrix if they are not at the interface nor at the free-surface
				if (nptype == PT_FLUID && !INTERFACE(ndata.info) && !SURFACE(ndata.info)) {
					Rx -= neib_volume*neib_oldEffPres*f;

					// vertex and free particles neibs of sediment that are at the interface
					// or at the free-surface contribute to the right hand-side vector B
				} else {
					B += neib_volume*neib_oldEffPres*f;
				}
				// contribution of boundary elements to the free particle Jacobi vectors
			} else if (boundarytype == SA_BOUNDARY && nptype == PT_BOUNDARY) {
				sa_boundary_jacobi_build_vector(B, ndata, params);
			}
		} // end of loop through neighbors
	}
	jacobiBuffer[index] = make_float4(D, Rx, B, NAN);
}

// Compute effective pressure for PT_FLUID particles from Jacobi vectors.
// Store the residual in cfl.
__global__ void
__launch_bounds__(BLOCK_SIZE_SPS, MIN_BLOCKS_SPS)
jacobiUpdateEffPresDevice(jacobi_update_params params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	float residual = 0.f;

	// do { } while (0) around the main body so that we can bail out
	// to the reduction
	do {
		if (index >= params.numParticles)
			break;

		float newEffPres = 0.f;

		const particleinfo info = params.fetchInfo(index);
		const ParticleType cptype = PART_TYPE(info);

		// Reference pressure
		const float refpres = d_rho0[fluid_num(info)]*d_sqC0[fluid_num(info)]/100;
		// effpres is updated for free particle of sediment that are not at the interface nor
		// nor at the free-surface.
		if (cptype == PT_FLUID && SEDIMENT(info) && !INTERFACE(info) && !SURFACE(info)) {
			const float4 jB = params.jacobiBuffer[index];
			const float D = jB.x;
			const float Rx = jB.y;
			const float B = jB.z;
			newEffPres = (B - Rx)/D;
			// Prevent NaN values.
			if (newEffPres == newEffPres) {
				params.effpres[index] = newEffPres;
			} else {
				params.effpres[index] = 0;
			}
			residual = (D*newEffPres + Rx - B)/refpres;
		}
	} while (0);

	reduce_jacobi_error(params.cfl, residual);

}

}

#endif
