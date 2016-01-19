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
	float4	newgGam;
	float4	gGamNp1;
	float	rho;

	__device__ __forceinline__
	density_sum_particle_output() :
		newgGam(make_float4(0.0f)),
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
	const	float4	velc;
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
		velc(vel + (params.step - 1.0f)*force*params.half_dt),
		gGamN(params.newgGam[index])
	{}
};

struct open_boundary_particle_data
{
	float4	eulerVel;

	__device__ __forceinline__
	open_boundary_particle_data(const uint index, sa_boundary_density_sum_params params) :
		eulerVel(params.oldEulerVel[index])
	{}
};

/// The actual density_sum_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	BoundaryType _boundarytype,
	ViscosityType _visctype,
	flag_t _simflags>
struct density_sum_particle_data :
	common_density_sum_particle_data,
	COND_STRUCT(_simflags & ENABLE_INLET_OUTLET,
				open_boundary_particle_data)
{
	static const KernelType kerneltype = _kerneltype;
	static const SPHFormulation sph_formulation = _sph_formulation;
	static const BoundaryType boundarytype = _boundarytype;
	static const ViscosityType visctype = _visctype;
	static const flag_t simflags = _simflags;

	// shorthand for the type of the density_sum params
	typedef density_sum_params<kerneltype, sph_formulation, boundarytype, visctype, simflags> params_t;

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
computeDensitySumTerms(
	const	float4			posN,
			float4			posNp1,
	const	float4			velN,
	const	int				index,
	const	float			dt,
	const	float			half_dt,
	const	float			influenceradius,
	const	float			slength,
	const	float4			*oldPos,
	const	float4			*oldVel,
	const	float4			*eulerVel,
	const	float4			*forces,
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
			float			&sumPmwN,
			float			&sumPmwNp1,
			float			&sumVmwDelta,
			float			&sumSgamDelta,
			float			&gGamDotR,
			float3			&gGam)
{
	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// posNp1Obj cotains the position of a with the inverse movement of the object
	float3 posNp1Obj;
	// savedObjId identifies for which object id the posNp1Obj vector is computed
	// because it is very likely that we only have one type of object per fluid particle
	uint savedObjId = UINT_MAX;

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(posN, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);
		const particleinfo neib_info = pinfo[neib_index];

		const float4 posN_neib = oldPos[neib_index];

		if (INACTIVE(posN_neib)) continue;

		const float4 velN_neib = oldVel[neib_index] + (FLUID(neib_info) ? (step - 1)*forces[neib_index]*half_dt : make_float4(0.0f));
		// fluid particles are moved every time-step according the the velocity
		// vertex parts and boundary elements are moved only in the first integration step according to the velocity
		// in the second step they are moved according to the solid body movement
		float4 posNp1_neib = posN_neib;
		if (MOVING(neib_info) && step == 2) { // this implies VERTEX(neib_info) || BOUNDARY(neib_info)
			// now the following trick is employed for moving objects, instead of moving the segment and all vertices
			// the fluid is moved virtually in opposite direction. this requires only one position to be recomputed
			// and not all of them. additionally, the normal stays the same.
			const uint i = object(neib_info)-1;
			// if savedObjId is equal to i that means that we have already computed the virtual position of the fluid
			// with respect to the opposite movement of the object, so we can reuse that information, if not we need
			// to compute it
			if (i != savedObjId) {
				// first move the fluid particle in opposite direction of the body translation
				float4 virtPos = posNp1 - make_float4(d_rbtrans[i]);
				// compute position with respect to center of gravity
				const float3 virtPosCG = d_worldOrigin + as_float3(virtPos) + calcGridPosFromParticleHash(particleHash[index])*d_cellSize + 0.5f*d_cellSize - d_rbcgPos[i];
				// apply inverse rotation matrix to position
				applycounterrot(&d_rbsteprot[9*i], virtPosCG, virtPos);
				// now store the virtual position
				posNp1Obj = as_float3(virtPos);
				// and the id for which this virtual position was computed
				savedObjId = i;
			}
			// set the Np1 position of a to the virtual position that is saved
			posNp1 = make_float4(posNp1Obj);
		}
		else if (FLUID(neib_info) || (MOVING(neib_info) && step==1)) {
			posNp1_neib += dt*velN_neib;
		}
		// vector r_{ab} at time N
		const float4 relPosN = pos_corr - posN_neib;
		// vector r_{ab} at time N+1 = r_{ab}^N + (r_a^{N+1} - r_a^{N}) - (r_b^{N+1} - r_b^N)
		const float4 relPosNp1 = make_float4(pos_corr) + posNp1 - posN - posNp1_neib;

		if (FLUID(neib_info) || VERTEX(neib_info)) {

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
		else if (BOUNDARY(neib_info)) {

			// normal of segment
			const float3 ns = as_float3(boundElement[neib_index]); // TODO this could be the new normal already

			// vectors r_{v_i,s}
			uint j = 0;
			// Get index j for which n_s is minimal
			if (fabs(ns.x) > fabs(ns.y))
				j = 1;
			if ((1-j)*fabs(ns.x) + j*fabs(ns.y) > fabs(ns.z))
				j = 2;
			// compute the first coordinate which is a 2-D rotated version of the normal
			const float3 coord1 = normalize(make_float3(
				// switch over j to give: 0 -> (0, z, -y); 1 -> (-z, 0, x); 2 -> (y, -x, 0)
				-((j==1)*ns.z) +  (j == 2)*ns.y ,  // -z if j == 1, y if j == 2
				  (j==0)*ns.z  - ((j == 2)*ns.x),  // z if j == 0, -x if j == 2
				-((j==0)*ns.y) +  (j == 1)*ns.x ));// -y if j == 0, x if j == 1
			// the second coordinate is the cross product between the normal and the first coordinate
			const float3 coord2 = cross(ns, coord1);
			// relative positions of vertices with respect to the segment
			const float3 vertexRelPos[3] = { -(vPos0[neib_index].x*coord1 + vPos0[neib_index].y*coord2), // e.g. v0 = r_{v0} - r_s
											 -(vPos1[neib_index].x*coord1 + vPos1[neib_index].y*coord2),
											 -(vPos2[neib_index].x*coord1 + vPos2[neib_index].y*coord2) };

			// sum_S 1/2*(gradGam^n + gradGam^{n+1})*relVel
			const float3 gGamN   = gradGamma<kerneltype>(slength, as_float3(relPosN),   vertexRelPos, ns)*ns;
			const float3 gGamNp1 = gradGamma<kerneltype>(slength, as_float3(relPosNp1), vertexRelPos, ns)*ns;
			gGamDotR += 0.5f*dot(gGamN + gGamNp1, as_float3(relPosNp1 - relPosN));
			gGam += gGamNp1;

			if (IO_BOUNDARY(neib_info)) {
				// sum_{S^{io}} (gradGam(r + delta r)).delta r
				const float3 deltaR = dt*as_float3(eulerVel[neib_index] - oldVel[neib_index]);
				const float3 relPosDelta = as_float3(relPosN) + deltaR;
				const float3 gGamDelta = gradGamma<kerneltype>(slength, relPosDelta, vertexRelPos, ns)*ns;
				sumSgamDelta += dot(deltaR, gGamDelta);
			}
		}
	}
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
template<KernelType kerneltype, SPHFormulation sph_formulation, BoundaryType boundarytype, ViscosityType visctype, flag_t simflags>
__global__ void
densitySumDevice(
	density_sum_params<kerneltype, sph_formulation, boundarytype, visctype, simflags> params)
{
	const int index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	// only perform density integration for fluid particles
	if (index >= params.numParticles || type(params.info[index]) != PT_FLUID)
		return;

	// We use dt/2 on the first step, the actual dt on the second step
	const float dt = (params.step == 1) ? params.half_dt : params.full_dt;

	density_sum_particle_data<kerneltype, sph_formulation, boundarytype, visctype, simflags> pdata(index, params);

	density_sum_particle_output pout;

	// continuity equation based on particle positions
	// - sum_{P\V^{io}} m^n w^n
	float sumPmwN = 0.0f;
	// sum_{S^{io}} (gradGam^n).delta r
	const float sumSgamN = dt*params.dgamdt[index];
	// sum_{P} m^n w^{n+1}
	float sumPmwNp1 = 0.0f;
	// - sum_{V^{io}} m^n w(r + delta r)
	float sumVmwDelta = 0.0f;
	// sum_{S^{io}} (gradGam(r + delta r)).delta r
	float sumSgamDelta = 0.0f;
	// sum_{S} (gradGam^{n+1} + gradGam^n)/2 . (r^{n+1} - r^{n})
	float gGamDotR = 0.0f;
	// compute new terms based on r^{n+1} and \delta r
	computeDensitySumTerms<kerneltype>(
		pdata.posN,
		pdata.posNp1,
		pdata.velc,
		index,
		dt,
		params.half_dt,
		params.influenceradius,
		params.slength,
		params.oldPos,
		params.oldVel,
		params.oldEulerVel,
		params.forces,
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
		sumPmwN,
		sumPmwNp1,
		sumVmwDelta,
		sumSgamDelta,
		gGamDotR,
		as_float3(pout.gGamNp1));

	// gamma terms
	// AM-TODO what about this term to remove 1/2 dgamdt?
	//const float4 gGamN = pdata.newgGam;// - (step-1.0)*make_float4(0.0f, 0.0f, 0.0f, gGamDotR/2.0f);
	pout.gGamNp1.w = pdata.gGamN.w + gGamDotR;

	// now compute a new gamma based on the eulerian velocity of the boundary
	float imposedGam = pdata.gGamN.w + (sumSgamDelta + sumSgamN)/2.0f;
	// clipping of the imposed gamma
	if (imposedGam > 1.0f)
		imposedGam = 1.0f;
	else if (imposedGam < 0.1f)
		imposedGam = 0.1f;

	// generate new density based on previously computed values
	pout.rho = (imposedGam*pdata.vel.w + sumPmwNp1 + sumPmwN + sumVmwDelta)/pout.gGamNp1.w;

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
	// we need to save the gamma at time N in the first integrator step
	if (params.step==1)
		params.oldgGam[index] = pdata.gGamN;
}

} // end of namespace cudensity_sum
#endif
