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

/*
 * Device code.
 */

#ifndef _DENSITY_SUM_KERNEL_
#define _DENSITY_SUM_KERNEL_

#include "particledefine.h"
#include "textures.cuh"
#include "multi_gpu_defines.h"

namespace cudensity_sum {

using namespace cusph;
using namespace cuphys;
using namespace cuneibs;
using namespace cueuler;

struct density_sum_particle_output
{
	float4	gGamNp1;
	float	rho;

	__device__ __forceinline__
	density_sum_particle_output() :
		gGamNp1(make_float4(0.0f)),
		rho(0.0f)
	{}
};

struct common_density_sum_particle_data
{
	const	uint	index;
	const	particleinfo	info;
	const	ParticleType	ptype;
	const	float4	force;
	const	int3	gridPos;
	float4	posN;
	float4	posNp1;
	float4	vel;
	const	float4	gGamN;

	__device__ __forceinline__
	common_density_sum_particle_data(const uint _index, common_density_sum_params params) :
		index(_index),
		info(params.info[index]),
		ptype(static_cast<ParticleType>(PART_TYPE(info))),
		force(params.forces[index]),
		gridPos(calcGridPosFromParticleHash(params.particleHash[index])),
		posN(params.oldPos[index]),
		posNp1(params.newPos[index]),
		vel(params.oldVel[index]),
		gGamN(params.oldgGam[index])
	{}
};

struct open_boundary_particle_data
{
	float4	eulerVel;

	__device__ __forceinline__
	open_boundary_particle_data(const uint index, common_density_sum_params params) :
		eulerVel(params.oldEulerVel[index])
	{}
};

/// The actual density_sum_particle_data struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	ParticleType _ntype,
	flag_t _simflags>
struct density_sum_particle_data :
	common_density_sum_particle_data,
	COND_STRUCT(_simflags & ENABLE_INLET_OUTLET,
				open_boundary_particle_data)
{
	static const KernelType kerneltype = _kerneltype;
	static const ParticleType ntype = _ntype;
	static const flag_t simflags = _simflags;

	// shorthand for the type of the density_sum params
	typedef density_sum_params<kerneltype, ntype, simflags> params_t;

	// determine specialization automatically based on info and params
	__device__ __forceinline__
	density_sum_particle_data(const uint _index, params_t const& params) :
		common_density_sum_particle_data(_index, params),
		COND_STRUCT(_simflags & ENABLE_INLET_OUTLET,
					open_boundary_particle_data)(_index, params)
	{}
};

template<KernelType kerneltype>
__device__ __forceinline__
static void
computeDensitySumVolumicTerms(
	const	float4			posN,
			float4			posNp1,
	const	int				index,
	const	float			dt,
	const	float			half_dt,
	const	float			influenceradius,
	const	float			slength,
	const	float4			*oldPos,
	const	float4			*newPos,
	const	float4			*oldVel,
	const	float4			*eulerVel,
	const	float4			*forces,
	const	particleinfo	*pinfo,
	const	hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
	const	int				step,
			float			&sumPmwN,
			float			&sumPmwNp1,
			float			&sumVmwDelta)
{
	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Loop over fluid and vertex neighbors
	for_each_neib2(PT_FLUID, PT_VERTEX, index, posN, gridPos, cellStart, neibsList) {
		const uint neib_index = neib_iter.neib_index();
		const particleinfo neib_info = pinfo[neib_index];

		const float4 posN_neib = oldPos[neib_index];

		if (INACTIVE(posN_neib)) continue;

		/* TODO FIXME splitneibs merge: the MOVING object support here was dropped in the splitneibs branch */

		const float4 posNp1_neib = newPos[neib_index];

		// vector r_{ab} at time N
		const float4 relPosN = neib_iter.relPos(posN_neib);
		// vector r_{ab} at time N+1 = r_{ab}^N + (r_a^{N+1} - r_a^{N}) - (r_b^{N+1} - r_b^N)
		const float4 relPosNp1 = neib_iter.relPos(posNp1_neib) + (posNp1 - posN);

		// -sum_{P\V_{io}} m^n w^n
		if (!IO_BOUNDARY(neib_info)) {
			const float rN = length3(relPosN);
			sumPmwN -= relPosN.w*W<kerneltype>(rN, slength);
		}

		// sum_{P} m^n w^{n+1}
		const float rNp1 = length3(relPosNp1);
		if (rNp1 < influenceradius)
			sumPmwNp1 += relPosN.w*W<kerneltype>(rNp1, slength);

		if (IO_BOUNDARY(neib_info)) {
			// compute - sum_{V^{io}} m^n w(r + delta r)
			const float4 deltaR = dt*(eulerVel[neib_index] - oldVel[neib_index]);
			const float newDist = length3(relPosN + deltaR);
			if (newDist < influenceradius)
				sumVmwDelta -= relPosN.w*W<kerneltype>(newDist, slength);
		}
	}
}

struct common_gamma_sum_terms {
	// collects sum_{S} (gradGam^{n+1} + gradGam^n)/2 . (r^{n+1} - r^{n})
	float gGamDotR;
	// gradGam
	float3 gGam;

	__device__ __forceinline__
	common_gamma_sum_terms() :
		gGamDotR(0.0f),
		gGam(make_float3(0.0f))
	{}
};

/// Gamma summation terms in case of I/O
struct io_gamma_sum_terms {
	// sum_{S^{io}} (gradGam(r + delta r)).delta r
	float sumSgamDelta;

	__device__ __forceinline__
	io_gamma_sum_terms() :
		sumSgamDelta(0.0f)
	{}
};

template<KernelType _kerneltype, flag_t simflags>
struct gamma_sum_terms :
	common_gamma_sum_terms,
	COND_STRUCT(simflags & ENABLE_INLET_OUTLET, io_gamma_sum_terms)
{
	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr bool has_io = simflags & ENABLE_INLET_OUTLET;
};

template<typename GammaTermT, typename OutputT>
using enable_if_IO = typename std::enable_if<GammaTermT::has_io, OutputT>::type;
template<typename GammaTermT, typename OutputT>
using enable_if_not_IO = typename std::enable_if<!GammaTermT::has_io, OutputT>::type;

/* contribution to grad gamma integration from I/O,
 * only if I/O is active
 */
template<typename GammaTermT>
__device__ __forceinline__
enable_if_not_IO<GammaTermT, void>
io_gamma_contrib(GammaTermT& sumGam, ...)
{ /* default case, nothing to do */ };

template<typename GammaTermT>
__device__ __forceinline__
enable_if_IO<GammaTermT, void>
io_gamma_contrib(GammaTermT &sumGam, int neib_index, particleinfo const& neib_info,
	float4 * __restrict__ eulerVel,
	float4 * __restrict__ oldVel,
	float3 const qN,
	float3 const ns,
	float3 * vertexRelPos,
	float dt,
	float slength)
{
		if (IO_BOUNDARY(neib_info)) {
			// sum_{S^{io}} (gradGam(r + delta r)).delta r
			const float3 deltaR = dt*as_float3(eulerVel[neib_index] - oldVel[neib_index]);
			const float3 qDelta = qN + deltaR/slength;
			const float3 gGamDelta = gradGamma<GammaTermT::kerneltype>(slength, qDelta, vertexRelPos, ns)*ns;
			sumGam.sumSgamDelta += dot(deltaR, gGamDelta);
		}
};

// Compute the imposedGamma for densitySumBoundaryDevice, depending on IO conditions
template<typename GammaTermT>
__device__ __forceinline__
enable_if_not_IO<GammaTermT, float>
compute_imposed_gamma(float oldGam, GammaTermT const& sumGam, float sumSgamN)
{
	return oldGam;
}
template<typename GammaTermT>
__device__ __forceinline__
enable_if_IO<GammaTermT, float>
compute_imposed_gamma(float oldGam, GammaTermT const& sumGam, float sumSgamN)
{
	float imposed = oldGam + (sumGam.sumSgamDelta + sumSgamN)/2.0f;
	// clipping of the imposed gamma
	if (imposed > 1.0f)
		imposed = 1.0f;
	else if (imposed < 0.1f)
		imposed = 0.1f;

	return imposed;
}

// TODO use more structs to collect params
template<KernelType kerneltype, flag_t simflags>
__device__ __forceinline__
static void
computeDensitySumBoundaryTerms(
	const	float4			posN,
			float4			posNp1,
	const	int				index,
	const	float			dt,
	const	float			half_dt,
	const	float			influenceradius,
	const	float			slength,
	const	float4			*oldPos,
	const	float4			*newPos,
	const	float4			*oldVel,
	const	float4			*eulerVel,
	const	particleinfo	*pinfo,
	const	float4			*boundElement,
	const	float2			*vPos0,
	const	float2			*vPos1,
	const	float2			*vPos2,
	const	hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
	const	int				step,
	gamma_sum_terms<kerneltype, simflags> &sumGam)
{
	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Loop over BOUNDARY neighbors
	for_each_neib(PT_BOUNDARY, index, posN, gridPos, cellStart, neibsList) {
		const uint neib_index = neib_iter.neib_index();
		const particleinfo neib_info = pinfo[neib_index];

		const float4 posN_neib = oldPos[neib_index];

		if (INACTIVE(posN_neib)) continue;

		const float4 posNp1_neib = newPos[neib_index];

		// vector r_{ab} at time N
		const float4 qN = neib_iter.relPos(posN_neib)/slength;
		// vector r_{ab} at time N+1 = r_{ab}^N + (r_a^{N+1} - r_a^{N}) - (r_b^{N+1} - r_b^N)
		const float4 qNp1 = (neib_iter.relPos(posN) + posNp1 - posNp1_neib)/slength;

		// normal of segment
		const float3 ns = as_float3(boundElement[neib_index]);

		// TODO vertexRelPos does not account for movement of the object atm
		float3 vertexRelPos[3];
		calcVertexRelPos(vertexRelPos, ns, vPos0[neib_index], vPos1[neib_index], vPos2[neib_index], slength);

		// sum_S 1/2*(gradGam^n + gradGam^{n+1})*relVel
		const float3 gGamN   = gradGamma<kerneltype>(slength, as_float3(qN),   vertexRelPos, ns)*ns;
		const float3 gGamNp1 = gradGamma<kerneltype>(slength, as_float3(qNp1), vertexRelPos, ns)*ns;
		sumGam.gGamDotR += 0.5f*dot(gGamN + gGamNp1, as_float3(qNp1 - qN));
		sumGam.gGam += gGamNp1;

		io_gamma_contrib(sumGam, neib_index, neib_info,
			eulerVel, oldVel, make_float3(qN), ns, vertexRelPos, dt, slength);
	}
	sumGam.gGamDotR *= slength;
}

/// Computes the density based on an integral formulation of the continuity equation
/*! Updates the density of fluid particles
 *
 *	\param[in] oldPos : previous particle's position
 *	\param[in] hashKey : particle's hash
 *	\param[in] oldVel : previous particle's velocity
 *	\param[in] oldEulerVel : previous eulerian velocities for ??? <- TODO
 *	\param[in] oldGam : previous values of gradient of gamma
 *	\param[in] okdTKE : previous values of k, for k-e model
 *	\param[in] oldEps : previous values of e, for k-e model
 *	\param[in] particleInfo : particle's information
 *	\param[in] forces : derivative of particle's velocity and density
 *	\param[in] dgamdt : time derivative of gamma
 *	\param[in] keps_dkde : derivative of ??? <- TODO
 *	\param[in] xsph : SPH mean of velocities used for xsph correction
 *	\param[out] newPos : updated particle's position
 *	\param[out] newVel : updated particle's  velocity
 *	\param[out] newEulerVel : updated eulerian velocities for ??? <- TODO
 *	\param[out] newgGam : updated values of gradient of gamma
 *	\param[out] newTKE : updated values of k, for k-e model
 *	\param[out] newEps : updated values of e, for k-e model
 *	\param[in,out] newBoundElement : ??? <- TODO
 *	\param[in] numParticles : total number of particles
 *	\param[in] full_dt  : time step (dt)
 *	\param[in] half_dt : half of time step (dt/2)
 *	\param[in] t : simualation time
 *
 *	\tparam step : integration step (1, 2)
 *	\tparam boundarytype : type of boundary
 *	\tparam kerneltype : type of kernel
 *	\tparam simflags : simulation flags
 */
//TODO templatize vars like other kernels
template<KernelType kerneltype,
	flag_t simflags>
__global__ void
densitySumVolumicDevice(
	// parameters are the same for fluid and vertex
	density_sum_params<kerneltype, PT_FLUID, simflags> params)
{
	const int index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	// only perform density integration for fluid particles
	if (index >= params.numParticles || !FLUID(params.info[index]))
		return;

	// We use dt/2 on the first step, the actual dt on the second step
	const float dt = (params.step == 1) ? params.half_dt : params.full_dt;

	density_sum_particle_data<kerneltype, PT_FLUID, simflags> pdata(index, params);

	density_sum_particle_output pout;

	// continuity equation based on particle positions
	// - sum_{P\V^{io}} m^n w^n
	float sumPmwN = 0.0f;
	// sum_{P} m^n w^{n+1}
	float sumPmwNp1 = 0.0f;
	// - sum_{V^{io}} m^n w(r + delta r)
	float sumVmwDelta = 0.0f;
	// compute new terms based on r^{n+1} and \delta r
	computeDensitySumVolumicTerms<kerneltype>(
		pdata.posN,
		pdata.posNp1,
		index,
		dt,
		params.half_dt,
		params.influenceradius,
		params.slength,
		params.oldPos,
		params.newPos,
		params.oldVel,
		params.oldEulerVel,
		params.forces,
		params.info,
		params.particleHash,
		params.cellStart,
		params.neibsList,
		params.numParticles,
		params.step,
		sumPmwN,
		sumPmwNp1,
		sumVmwDelta);

	params.forces[index].w = sumPmwNp1 + sumPmwN + sumVmwDelta;
}

/// Computes the density based on an integral formulation of the continuity equation
/*! Updates the density of fluid particles
 *
 *	\param[in] oldPos : previous particle's position
 *	\param[in] hashKey : particle's hash
 *	\param[in] oldVel : previous particle's velocity
 *	\param[in] oldEulerVel : previous eulerian velocities for ??? <- TODO
 *	\param[in] oldGam : previous values of gradient of gamma
 *	\param[in] okdTKE : previous values of k, for k-e model
 *	\param[in] oldEps : previous values of e, for k-e model
 *	\param[in] particleInfo : particle's information
 *	\param[in] forces : derivative of particle's velocity and density
 *	\param[in] dgamdt : time derivative of gamma
 *	\param[in] keps_dkde : derivative of ??? <- TODO
 *	\param[in] xsph : SPH mean of velocities used for xsph correction
 *	\param[out] newPos : updated particle's position
 *	\param[out] newVel : updated particle's  velocity
 *	\param[out] newEulerVel : updated eulerian velocities for ??? <- TODO
 *	\param[out] newgGam : updated values of gradient of gamma
 *	\param[out] newTKE : updated values of k, for k-e model
 *	\param[out] newEps : updated values of e, for k-e model
 *	\param[in,out] newBoundElement : ??? <- TODO
 *	\param[in] numParticles : total number of particles
 *	\param[in] full_dt  : time step (dt)
 *	\param[in] half_dt : half of time step (dt/2)
 *	\param[in] t : simualation time
 *
 *	\tparam step : integration step (1, 2)
 *	\tparam boundarytype : type of boundary
 *	\tparam kerneltype : type of kernel
 *	\tparam simflags : simulation flags
 */
//TODO templatize vars like other kernels
template<KernelType kerneltype,
	flag_t simflags>
__global__ void
densitySumBoundaryDevice(
	density_sum_params<kerneltype, PT_BOUNDARY, simflags> params)
{
	const int index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	// only perform density integration for fluid particles
	if (index >= params.numParticles || !FLUID(params.info[index]))
		return;

	// We use dt/2 on the first step, the actual dt on the second step
	const float dt = (params.step == 1) ? params.half_dt : params.full_dt;

	density_sum_particle_data<kerneltype, PT_BOUNDARY, simflags> pdata(index, params);

	density_sum_particle_output pout;

	// continuity equation based on particle positions
	// sum_{S^{io}} (gradGam^n).delta r
	/* GB TODO FIXME this is spurious when not using IO, and definitely needs
	 * some thought about IO too, particularly when using density summation.
	 * As a provisional splitneibs-merge fix, set it to zero, we'll re-evaluate
	 * it when reintroduing open boundaries.
	 */
#if 0
	const float sumSgamN = dt*params.dgamdt[index];
#else
	const float sumSgamN = 0;
#endif

	gamma_sum_terms<kerneltype, simflags> sumGam;

	computeDensitySumBoundaryTerms(
		pdata.posN,
		pdata.posNp1,
		index,
		dt,
		params.half_dt,
		params.influenceradius,
		params.slength,
		params.oldPos,
		params.newPos,
		params.oldVel,
		params.oldEulerVel,
		params.info,
		params.newBoundElement,
		params.vertPos0,
		params.vertPos1,
		params.vertPos2,
		params.particleHash,
		params.cellStart,
		params.neibsList,
		params.numParticles,
		params.step,
		sumGam);

	pout.gGamNp1.x = sumGam.gGam.x;
	pout.gGamNp1.y = sumGam.gGam.y;
	pout.gGamNp1.z = sumGam.gGam.z;

	// gamma terms
	// AM-TODO what about this term to remove 1/2 dgamdt?
	//const float4 gGamN = pdata.newgGam;// - (step-1.0)*make_float4(0.0f, 0.0f, 0.0f, gGamDotR/2.0f);
	pout.gGamNp1.w = pdata.gGamN.w + sumGam.gGamDotR;

	// now compute a new gamma based on the eulerian velocity of the boundary
	float imposedGam = compute_imposed_gamma(pdata.gGamN.w, sumGam, sumSgamN);

	// generate new density based on previously computed values
	pout.rho = (imposedGam*pdata.vel.w + params.forces[index].w)/pout.gGamNp1.w;

	// clipping of new gamma
	// this needs to happen after the density update because otherwise density jumps can occur
	if (pout.gGamNp1.w > 1.0f || length3(pout.gGamNp1)*params.slength < 1e-10f)
		pout.gGamNp1.w = 1.0f;
	else if (pout.gGamNp1.w < 0.1f)
		pout.gGamNp1.w = 0.1f;

	// output of updated variables:
	// density
	params.newVel[index].w = pout.rho;
	// gamma
	params.newgGam[index] = pout.gGamNp1;
}

/// Integrate gamma
/** Gamma is always integrated using a “density sum” approach,
 * from the difference of the particle distribution at step n
 * and at step n+1 (hence why the kernel is here in
 * the density sum namespace)
*/
template<KernelType kerneltype, flag_t simflags>
__global__ void
integrateGammaDevice(
	const	float4	* __restrict__ gGamN, ///< previous gamma and its gradient
			float4	* __restrict__ gGamNp1, ///< [out] new gamma and its gradient
	const	float4	* __restrict__ posN, ///< positions at step n
	const	float4	* __restrict__ posNp1, ///< positions at step n+1
	const	float4	* __restrict__ velN, ///< velocities at step n
	const	float4	* __restrict__ velNp1, ///< velocities at step n+1
	const	hashKey	* __restrict__ particleHash, ///< particle hash
	const	particleinfo * __restrict__ info, ///< particle info
	const	float4	* __restrict__ boundElementN, ///< boundary elements at step n
	const	float4	* __restrict__ boundElementNp1, ///< boundary elements at step n+1
	const	float2	* __restrict__ vPos0,
	const	float2	* __restrict__ vPos1,
	const	float2	* __restrict__ vPos2,
	const	neibdata *__restrict__ neibsList,
	const	uint	* __restrict__ cellStart,
	const	uint	particleRangeEnd, ///< max number of particles
	const	float	full_dt, ///< time step (dt)
	const	float	half_dt, ///< half of time step (dt/2)
	const	float	t, ///< simulation time
	const	uint	step, ///< integrator step
	const	float	slength,
	const	float	influenceradius)
{
	const int index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	// only perform density integration for fluid particles
	if (index >= particleRangeEnd || !FLUID(info[index]))
		return;

	// We use dt/2 on the first step, the actual dt on the second step
	const float dt = (step == 1) ? half_dt : full_dt;

	gamma_sum_terms<kerneltype, simflags> sumGam;

	computeDensitySumBoundaryTerms(
		posN[index], posNp1[index], index,
		dt, half_dt, influenceradius, slength,
		posN, posNp1, velN, NULL /* TODO oldEulerVel, only for I/O */,
		info,
		boundElementNp1, vPos0, vPos1, vPos2,
		particleHash, cellStart, neibsList, particleRangeEnd,
		step, sumGam);


	gGamNp1[index] = make_float4(
		sumGam.gGam, gGamN[index].w + sumGam.gGamDotR);
}

} // end of namespace cudensity_sum
#endif
