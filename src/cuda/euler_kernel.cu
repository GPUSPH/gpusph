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

#ifndef _EULER_KERNEL_
#define _EULER_KERNEL_

#include "particledefine.h"
#include "textures.cuh"
#include "multi_gpu_defines.h"

namespace cueuler {

__constant__ float	d_epsxsph;
__constant__ float3	d_maxlimit;
__constant__ float3	d_minlimit;

__constant__ int3	d_rbcgGridPos[MAX_BODIES]; //< cell of the center of gravity
__constant__ float3	d_rbcgPos[MAX_BODIES]; //< in-cell coordinate of the center of gravity
__constant__ float3	d_rbtrans[MAX_BODIES];
__constant__ float3	d_rblinearvel[MAX_BODIES];
__constant__ float3	d_rbangularvel[MAX_BODIES];
__constant__ float	d_rbsteprot[9*MAX_BODIES];

__constant__ idx_t	d_neiblist_end; // maxneibsnum * number of allocated particles
__constant__ idx_t	d_neiblist_stride; // stride between neighbors of the same particle

#include "cellgrid.cuh"

#include "sph_core_utils.cuh"

/// Apply rotation to a given vector
/*! Apply the rotation given by the matrix rot to the vector relPos.
 *  The change in the relPos vector due to the rotation is computed
 *  and added to the pos vector.
 *
 *	\param[in] rot : rotation matrix
 *	\param[in] relPos: position with respect to center of gravity
 *	\param[in] pos: position with respect to the local cell center
 *
 *	\return local postion rotated according to rot
 */
__device__ __forceinline__ void
applyrot(const float* rot, const float3 & relPos, float4 & pos)
{
	// Applying rotation
	pos.x += (rot[0] - 1.0f)*relPos.x + rot[1]*relPos.y + rot[2]*relPos.z;
	pos.y += rot[3]*relPos.x + (rot[4] - 1.0f)*relPos.y + rot[5]*relPos.z;
	pos.z += rot[6]*relPos.x + rot[7]*relPos.y + (rot[8] - 1.0f)*relPos.z;
}

/// Apply counter rotation to a given vector
/*! Apply the inverse rotation given by the matrix rot to the vector relPos.
 *  The change in the relPos vector due to the rotation is computed
 *  and added to the pos vector.
 *
 *	\param[in] rot : rotation matrix
 *	\param[in] relPos: position with respect to center of gravity
 *	\param[in] pos: position with respect to the local cell center
 *
 *	\return local postion rotated according to rot^{-1}
 */
__device__ __forceinline__ void
applycounterrot(const float* rot, const float3 & relPos, float4 & pos)
{
	// Applying counter rotation (using R^{-1} = R^T)
	pos.x += (rot[0] - 1.0f)*relPos.x + rot[3]*relPos.y + rot[6]*relPos.z;
	pos.y += rot[1]*relPos.x + (rot[4] - 1.0f)*relPos.y + rot[7]*relPos.z;
	pos.z += rot[2]*relPos.x + rot[5]*relPos.y + (rot[8] - 1.0f)*relPos.z;
}

__device__ __forceinline__ void
applyrot2(float* rot, float3 & pos, const float3 & cg)
{
	float3 relpos = pos - cg;
	float3 new_relpos;

	// Applying rotation
	new_relpos.x = rot[0]*relpos.x + rot[1]*relpos.y + rot[2]*relpos.z;
	new_relpos.y = rot[3]*relpos.x + rot[4]*relpos.y + rot[5]*relpos.z;
	new_relpos.z = rot[6]*relpos.x + rot[7]*relpos.y + rot[8]*relpos.z;

	pos.x = new_relpos.x + cg.x;
	pos.y = new_relpos.y + cg.y;
	pos.z = new_relpos.z + cg.z;
}

template<bool densitySum>
struct sa_integrate_continuity_equation
{
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
				const float3 gGamN   = gradGamma<kerneltype>(as_float3(relPosN),   vertexRelPos, ns, slength)*ns;
				const float3 gGamNp1 = gradGamma<kerneltype>(as_float3(relPosNp1), vertexRelPos, ns, slength)*ns;
				gGamDotR += 0.5f*dot(gGamN + gGamNp1, as_float3(relPosNp1 - relPosN));
				gGam += gGamNp1;

				if (IO_BOUNDARY(neib_info)) {
					// sum_{S^{io}} (gradGam(r + delta r)).delta r
					const float3 deltaR = dt*as_float3(eulerVel[neib_index] - oldVel[neib_index]);
					const float3 relPosDelta = as_float3(relPosN) + deltaR;
					const float3 gGamDelta = gradGamma<kerneltype>(relPosDelta, vertexRelPos, ns, slength)*ns;
					sumSgamDelta += dot(deltaR, gGamDelta);
				}
			}
		}
	}

	template<KernelType kerneltype>
	__device__ __forceinline__
	static void
	with(
		const	float	halfDensity,
				float	&newDensity,
		const	uint	index,
		const	float4	newPPos,
		const	float4	newPVelc,
				float4	&oldgGam,
				float4	&newgGam,
		const	float4	*oldPos,
				float4	*oldVel,
		const	float4	*oldEulerVel,
		const	float4	*forces,
		const	particleinfo	*pinfo,
		const	float4	*newBoundElement,
		const	float2	*vPos0,
		const	float2	*vPos1,
		const	float2	*vPos2,
		const	hashKey	*particleHash,
		const	uint	*cellStart,
		const	neibdata	*neibsList,
		const	uint	numParticles,
		const	float2	contupd,
		const	float	densityForce,
		const	float	dt,
		const	float	full_dt,
		const	float	half_dt,
		const	float	slength,
		const	float	influenceradius,
		const	int		step)
	{}
};

template<>
template<KernelType kerneltype>
__device__ __forceinline__ void
sa_integrate_continuity_equation<true>::with(
	const	float	halfDensity,
			float	&newDensity,
	const	uint	index,
	const	float4	newPPos,
	const	float4	newPVelc,
			float4	&oldgGam,
			float4	&newgGam,
	const	float4	*oldPos,
			float4	*oldVel,
	const	float4	*oldEulerVel,
	const	float4	*forces,
	const	particleinfo	*pinfo,
	const	float4	*newBoundElement,
	const	float2	*vPos0,
	const	float2	*vPos1,
	const	float2	*vPos2,
	const	hashKey	*particleHash,
	const	uint	*cellStart,
	const	neibdata	*neibsList,
	const	uint	numParticles,
	const	float2	contupd,
	const	float	densityForce,
	const	float	dt,
	const	float	full_dt,
	const	float	half_dt,
	const	float	slength,
	const	float	influenceradius,
	const	int		step)
{
	const float oldDensity = newDensity;
	// Rhie and Chow filter
	if (step == 1) {
		// Update rho^n with the filter to obtain a new rho^n
		newDensity += 2.0f*oldDensity*full_dt*full_dt*densityForce;
		// update array with filtered rho value
		oldVel[index].w = newDensity;
	}
	// continuity equation based on particle positions
	// - sum_{P\V^{io}} m^n w^n
	const float sumPmwN = contupd.x;
	// sum_{S^{io}} (gradGam^n).delta r
	const float sumSgamN = dt*contupd.y;
	// sum_{P} m^n w^{n+1}
	float sumPmwNp1 = 0.0f;
	// - sum_{V^{io}} m^n w(r + delta r)
	float sumVmwDelta = 0.0f;
	// sum_{S^{io}} (gradGam(r + delta r)).delta r
	float sumSgamDelta = 0.0f;
	// sum_{S} (gradGam^{n+1} + gradGam^n)/2 . (r^{n+1} - r^{n})
	float gGamDotR = 0.0f;
	float3 gGam= make_float3(0.0f);
	// compute new terms based on r^{n+1} and \delta r
	computeDensitySumTerms<kerneltype>(
		oldPos[index],
		newPPos,
		newPVelc,
		index,
		dt,
		half_dt,
		influenceradius,
		slength,
		oldPos,
		oldVel,
		oldEulerVel,
		forces,
		pinfo,
		newBoundElement,
		vPos0,
		vPos1,
		vPos2,
		particleHash,
		cellStart,
		neibsList,
		numParticles,
		step,
		sumPmwNp1,
		sumVmwDelta,
		sumSgamDelta,
		gGamDotR,
		gGam);

	// gamma terms
	const float4 gGamN = newgGam;// - (step-1.0)*make_float4(0.0f, 0.0f, 0.0f, gGamDotR/2.0f);
	float gamNp1 = gGamN.w + gGamDotR;

	// now compute a new gamma based on the eulerian velocity of the boundary
	float imposedGam = gGamN.w + (sumSgamDelta + sumSgamN)/2.0f;
	// clipping of the imposed gamma
	if (imposedGam > 1.0f)
		imposedGam = 1.0f;
	else if (imposedGam < 0.1f)
		imposedGam = 0.1f;

	// generate new density based on previously computed values
	newDensity = (imposedGam*newDensity + sumPmwNp1 + sumPmwN + sumVmwDelta)/gamNp1;

	// clipping of new gamma
	// this needs to happen after the density update because otherwise density jumps can occur
	if (gamNp1 > 1.0f || length(gGam)*slength < 1e-10f)
		gamNp1 = 1.0f;
	else if (gamNp1 < 0.1f)
		gamNp1 = 0.1f;
	// we need to save the gamma at time N in the first integrator step
	if (step==1)
		oldgGam = gGamN;

	// update new gamma with time integrated version
	newgGam.x = gGam.x;
	newgGam.y = gGam.y;
	newgGam.z = gGam.z;
	newgGam.w = gamNp1;
}

template<>
template<KernelType kerneltype>
__device__ __forceinline__ void
sa_integrate_continuity_equation<false>::with(
	const	float	halfDensity,
			float	&newDensity,
	const	uint	index,
	const	float4	newPPos,
	const	float4	newPVelc,
			float4	&oldgGam,
			float4	&newgGam,
	const	float4	*oldPos,
			float4	*oldVel,
	const	float4	*oldEulerVel,
	const	float4	*forces,
	const	particleinfo	*pinfo,
	const	float4	*newBoundElement,
	const	float2	*vPos0,
	const	float2	*vPos1,
	const	float2	*vPos2,
	const	hashKey	*particleHash,
	const	uint	*cellStart,
	const	neibdata	*neibsList,
	const	uint	numParticles,
	const	float2	contupd,
	const	float	densityForce,
	const	float	dt,
	const	float	full_dt,
	const	float	half_dt,
	const	float	slength,
	const	float	influenceradius,
	const	int		step)
{
	const float oldDensity = newDensity;
	newDensity = 0.0f;
	// Updating particle density
	// For step 1:
	//	  vel = vel(n+1/2) = vel(n) + f(n)*dt/2
	// For step 2:
	//	  vel = vel(n+1) = vel(n) + f(n+1/2)*dt
	// Improved continuity equation
	if (step == 1) {
		// gamma at time n+1 (gGam.w is gamma at time n)
		const float gamN = newgGam.w;
		const float gamNp1 = gamN + dt*contupd.y;
		newDensity = gamN/gamNp1*(oldDensity + dt*densityForce) + dt*contupd.x;
	}
	else {
		const float gamNp1o2 = newgGam.w;
		const float rhoNp1o2 = halfDensity;
		// gamma at time n (gGam.w is gamma at time n+1/2)
		const float gamN   = oldgGam.w;
		// gamma at time n+1
		const float gamNp1 = gamN + dt*contupd.y;
		newDensity = 1.0f/gamNp1*(gamN*oldDensity + dt*oldDensity/rhoNp1o2*gamNp1o2*densityForce) + dt*contupd.x;
	}
	// Classical continuity equation
	//newDensity += dt*densityForce + dt*contupd.x - dt*oldDensity/newgGam.w*contupd.y;
}

#include "euler_kernel.def"

}
#endif
