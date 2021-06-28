/*  Copyright (c) 2013-2019 INGV, EDF, UniCT, JHU

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

/*
 * Device code.
 */

#ifndef _POST_PROCESS_KERNEL_
#define _POST_PROCESS_KERNEL_

#include "particledefine.h"
#include "vector_math.h"
#include "multi_gpu_defines.h"
#include "GlobalData.h"

#include "neibs_list_params.h"

#if __COMPUTE__ < 20
#define printf(...) /* eliminate printf from 1.x */
#endif

namespace cupostprocess {

using namespace cugeom;
using namespace cusph;
using namespace cuphys;
using namespace cuneibs;
using namespace cubounds;

/************************************************************************************************************/
/*					   Auxiliary kernels used for post processing										    */
/************************************************************************************************************/

//! Computes the vorticity field
template<KernelType kerneltype, BoundaryType boundarytype>
__global__ void
calcVortDevice(neibs_interaction_params<boundarytype> params, float3* __restrict__ vorticity)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// computing vorticity only for active fluid particles
	const particleinfo info = params.fetchInfo(index);
	const float4 pos = params.fetchPos(index);

	if (NOT_FLUID(info) || INACTIVE(pos)) {
		vorticity[index] = make_float3(NAN);
		return;
	}

	const float4 vel = params.fetchVel(index);

	float3 vort = make_float3(0.0f);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

	// First loop over all FLUID neighbors
	for_each_neib(PT_FLUID, index, pos, gridPos, params.cellStart, params.neibsList) {
		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos( params.fetchPos(neib_index) );

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length3(relPos);

		// Compute relative velocity
		// Now relVel is a float4 and neib density is stored in relVel.w
		const float4 relVel = as_float3(vel) - params.fetchVel(neib_index);
		const particleinfo neib_info = params.fetchInfo(neib_index);

		// Compute vorticity
		if (r < params.influenceradius) {
			const float f = F<kerneltype>(r, params.slength)*relPos.w/physical_density(relVel.w,fluid_num(neib_info));	// ∂Wij/∂r*Vj
			// vxij = vxi - vxj and same for vyij and vzij
			vort.x += f*(relVel.y*relPos.z - relVel.z*relPos.y);		// vort.x = ∑(vyij(zi - zj) - vzij*(yi - yj))*∂Wij/∂r*Vj
			vort.y += f*(relVel.z*relPos.x - relVel.x*relPos.z);		// vort.y = ∑(vzij(xi - xj) - vxij*(zi - zj))*∂Wij/∂r*Vj
			vort.z += f*(relVel.x*relPos.y - relVel.y*relPos.x);		// vort.x = ∑(vxij(yi - yj) - vyij*(xi - xj))*∂Wij/∂r*Vj
		}
	} // end of loop trough neighbors

	vorticity[index] = vort;
}

template<BoundaryType boundarytype, typename ViscSpec
	, bool has_keps = ViscSpec::turbmodel == KEPSILON
	, typename neibs_params = neibs_interaction_params<boundarytype>
	, typename keps_tex_cond = typename COND_STRUCT(has_keps, keps_tex_params)
	, typename keps_cond = typename COND_STRUCT(has_keps, keps_params<true>)
	>
struct testpoints_params :
	neibs_params,
	keps_tex_cond,
	keps_cond
{
	float4 * __restrict__ newVel;

	testpoints_params(BufferList const& bufread, BufferList& bufwrite,
		uint particleRangeEnd, float slength, float influenceRadius)
	:
		neibs_params(bufread, particleRangeEnd, slength, influenceRadius),
		// calcTestpointsDevice does an in-place update, so the keps textures
		// are taken from the write buffers
		keps_tex_cond(bufwrite),
		keps_cond(bufwrite),
		newVel(bufwrite.getData<BUFFER_VEL>())
	{}
};

struct testpoints_keps_pdata
{
	float tke_avg;
	float eps_avg;

	__device__ __forceinline__
	testpoints_keps_pdata() : tke_avg(0.0f), eps_avg(0.0f) {}

	__device__ __forceinline__
	void add_keps_contrib(keps_tex_params const& params, const uint index, const float w)
	{
		tke_avg += w*params.fetchTke(index);
		eps_avg += w*params.fetchEpsilon(index);
	}

	__device__ __forceinline__
	void normalize(float alpha)
	{
		tke_avg /= alpha;
		eps_avg /= alpha;
	}

	__device__ __forceinline__
	void reset()
	{
		tke_avg = eps_avg = 0.0f;
	}

	__device__ __forceinline__
	void store_keps(keps_params<true> const& params, const uint index)
	{
		params.keps_tke[index] = tke_avg;
		params.keps_eps[index] = eps_avg;
	}
};

struct testpoints_nokeps_pdata
{
	using dummy_params = empty<keps_params<true>>;

	__device__ __forceinline__
	void add_keps_contrib(dummy_params const& params, const uint index, const float w) {};

	__device__ __forceinline__
	void normalize(float alpha) {}

	__device__ __forceinline__
	void reset() {}

	__device__ __forceinline__
	void store_keps(dummy_params const&, const uint index) {}
};

template<typename ViscSpec
	, bool has_keps = ViscSpec::turbmodel == KEPSILON
	, typename keps_pdata = typename conditional<has_keps, testpoints_keps_pdata, testpoints_nokeps_pdata>::type
	>
struct testpoints_pdata : keps_pdata
{
	float4	vel_avg;
	float	alpha; // shepard normalization factor

	__device__ __forceinline__
	testpoints_pdata() : keps_pdata(), vel_avg(make_float4(0.0f)), alpha(0.0f) {}

	__device__ __forceinline__
	void normalize() {
		if (alpha > 1e-5f) {
			vel_avg /= alpha;
			keps_pdata::normalize(alpha);
		} else {
			vel_avg = make_float4(0.0f);
			keps_pdata::reset();
		}
	}
};


//! Compute the values of velocity, density, k and epsilon at test points
template<KernelType kerneltype,
	BoundaryType boundarytype,
	typename ViscSpec>
__global__ void
calcTestpointsDevice(testpoints_params<boundarytype, ViscSpec> params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	const particleinfo info = params.fetchInfo(index);
	if(!TESTPOINT(info))
		return;

	const float4 pos = params.fetchPos(index);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

	testpoints_pdata<ViscSpec> pdata;

	// First loop over FLUID and VERTEX neighbors (VERTEX only in SA case)
	for_each_neib2(PT_FLUID, (boundarytype == SA_BOUNDARY ? PT_VERTEX : PT_NONE),
			index, pos, gridPos, params.cellStart, params.neibsList) {

		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos( params.fetchPos(neib_index) );

		const float r = length3(relPos);

		const particleinfo neib_info = params.fetchInfo(neib_index);

		if (r < params.influenceradius) {
			const float4 neib_vel = params.fetchVel(neib_index);
			const float w = W<kerneltype>(r, params.slength)*relPos.w/physical_density(neib_vel.w,fluid_num(neib_info));	// Wij*mj
			// Velocity
			pdata.vel_avg.x += w*neib_vel.x;
			pdata.vel_avg.y += w*neib_vel.y;
			pdata.vel_avg.z += w*neib_vel.z;
			//Pressure
			pdata.vel_avg.w += w*P(neib_vel.w, fluid_num(neib_info));

			// TKE contribution (if present)
			pdata.add_keps_contrib(params, neib_index, w);

			//Shepard filter
			pdata.alpha += w;
		}
	}

	pdata.normalize();

	params.newVel[index] = pdata.vel_avg;

	pdata.store_keps(params, index);
}


//! Identifies particles which form the free-surface
template<KernelType kerneltype, BoundaryType boundarytype, flag_t simflags, bool savenormals>
__global__ void
calcSurfaceparticleDevice(
	neibs_interaction_params<boundarytype> params,
	float4*	__restrict__ normals,
	particleinfo* __restrict__ newInfo)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	particleinfo info = params.fetchInfo(index);

	const float4 pos = params.fetchPos(index);

	float4 normal = make_float4(0.0f);

	if (NOT_FLUID(info) || INACTIVE(pos)) {
		if (savenormals)
			normals[index] = make_float4(NAN);
		return;
	}

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

	CLEAR_FLAG(info, FG_SURFACE);

	// self contribution to normalization: W(0)*vol
	normal.w = W<kerneltype>(0.0f, params.slength)*pos.w/physical_density(params.fetchVel(index).w,fluid_num(info));

	// First loop over all neighbors
	for_every_neib(boundarytype, index, pos, gridPos, params.cellStart, params.neibsList) {

		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos( params.fetchPos(neib_index) );

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length3(relPos);

		// read neighbor data from sorted arrays
		const particleinfo neib_info = params.fetchInfo(neib_index);

		// neighbor volume
		const float neib_vol = relPos.w/physical_density(params.fetchVel(neib_index).w, fluid_num(neib_info));

		if (r < params.influenceradius) {
			const float f = F<kerneltype>(r, params.slength)*neib_vol; // 1/r ∂Wij/∂r Vj
			normal.x -= f * relPos.x;
			normal.y -= f * relPos.y;
			normal.z -= f * relPos.z;
			normal.w += W<kerneltype>(r, params.slength)*neib_vol;	// Wij*Vj ;

		}
	}

	// Checking the planes
	if (simflags & ENABLE_PLANES)
		for (uint i = 0; i < d_numplanes; ++i) {
			const float r = PlaneDistance(gridPos, as_float3(pos), d_plane[i]);
			if (r < params.influenceradius) {
				// since our current normal is still unnormalized, the plane normal
				// contribution must be scaled up to match the length of the current normal
				as_float3(normal) += d_plane[i].normal*length3(normal);
			}
		}

	const float normal_length = length3(normal);

	int nc = 0;

	// Second loop over all neighbors
	for_every_neib(boundarytype, index, pos, gridPos, params.cellStart, params.neibsList) {

		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos( params.fetchPos(neib_index) );

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length3(relPos);

		float cosconeangle;

		const particleinfo neib_info = params.fetchInfo(neib_index);

		if (r < params.influenceradius) {
			float criteria = -dot3(normal, relPos);
			if (FLUID(neib_info))
				cosconeangle = d_cosconeanglefluid;
			else
				cosconeangle = d_cosconeanglenonfluid;

			if (criteria > r*normal_length*cosconeangle)
				nc++;
		}

	}

	if (!nc)
		SET_FLAG(info, FG_SURFACE);

	newInfo[index] = info;

	if (savenormals) {
		normal.x /= normal_length;
		normal.y /= normal_length;
		normal.z /= normal_length;
		normals[index] = normal;
		}

}

//! Identifies particles at the interface of two fluids and at the free-surface
template<KernelType kerneltype, BoundaryType boundarytype, flag_t simflags, bool savenormals>
__global__
enable_if_t<boundarytype != SA_BOUNDARY>
calcInterfaceparticleDevice(
	neibs_interaction_params<boundarytype> params,
	float4*	__restrict__ normals,
	particleinfo* __restrict__ newInfo,
	// deltap is only actually used by the SA_BOUNDARY specialization below
	// TODO FIXME reorganize this so this is not needed
	const float deltap)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	particleinfo info = params.fetchInfo(index);

	const float4 pos = params.fetchPos(index);

	float4 normal_fs = make_float4(0.0f); // free-surface
	float4 normal_if = make_float4(0.0f); // interface

	if (NOT_FLUID(info) || INACTIVE(pos)) {
		// NOTE: inactive particles will keep their last surface flag status
		newInfo[index] = info;
		if (savenormals)
			normals[index] = make_float4(NAN);
		return;
	}

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

	CLEAR_FLAG(info, FG_SURFACE);
	CLEAR_FLAG(info, FG_INTERFACE);

	// Particle physical density and volume
	const float p_rho = physical_density(params.fetchVel(index).w,fluid_num(info));
	const float p_volume = pos.w/p_rho;

	// self contribution to normalization: W(0)*vol
	normal_fs.w = W<kerneltype>(0.0f, params.slength)*p_volume;
	normal_if.w = W<kerneltype>(0.0f, params.slength)*p_volume;

	// First loop over all neighbors
	for_every_neib(boundarytype, index, pos, gridPos, params.cellStart, params.neibsList) {

		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos( params.fetchPos(neib_index) );

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const particleinfo n_info = params.fetchInfo(neib_index);
		const float r = length3(relPos);

		// neighbor physical density and volume
		const float n_rho = physical_density(params.fetchVel(neib_index).w, fluid_num(n_info));
		const float n_volume = relPos.w/n_rho;

		if (r < params.influenceradius) {
			const float f = F<kerneltype>(r, params.slength); // 1/r ∂Wij/∂r
			normal_fs.x -= f * relPos.x;
			normal_fs.y -= f * relPos.y;
			normal_fs.z -= f * relPos.z;
			normal_fs.w += W<kerneltype>(r, params.slength)*n_volume;	// Wij*Vj ;
			if ((fluid_num(info) == fluid_num(n_info) || NOT_FLUID(n_info))) {
				normal_if.x -= f * relPos.x;
				normal_if.y -= f * relPos.y;
				normal_if.z -= f * relPos.z;
				normal_if.w += W<kerneltype>(r, params.slength)*n_volume;	// Wij*Vj ;
			}
		}
	}

	normal_fs.x *= p_volume;
	normal_fs.y *= p_volume;
	normal_fs.z *= p_volume;

	normal_if.x *= p_volume;
	normal_if.y *= p_volume;
	normal_if.z *= p_volume;

	const float normal_fs_length = length3(normal_fs);
	const float normal_if_length = length3(normal_if);

	int nc_fs = 0;
	int nc_if = 0;

	// Second loop over all neighbors
	for_every_neib(boundarytype, index, pos, gridPos, params.cellStart, params.neibsList) {

		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos( params.fetchPos(neib_index) );

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length3(relPos);

		float cosconeangle;

		const particleinfo n_info = params.fetchInfo(neib_index);

		if (r < params.influenceradius) {
			float criteria_fs = -dot3(normal_fs, relPos);
			if (FLUID(n_info))
				cosconeangle = d_cosconeanglefluid;
			else
				cosconeangle = d_cosconeanglenonfluid;


			if (criteria_fs > r*normal_fs_length*cosconeangle)
				nc_fs++;

			if ((fluid_num(info) == fluid_num(n_info) || NOT_FLUID(n_info))) {
				float criteria_if = -dot3(normal_if, relPos);
				if (FLUID(n_info))
					cosconeangle = d_cosconeanglefluid;
				else
					cosconeangle = d_cosconeanglenonfluid;

				//cosconeangle = d_cosconeanglefluid;

				if (criteria_if > r*normal_if_length*cosconeangle)
					nc_if++;
			}
		}
	}

	if (!nc_fs)
		SET_FLAG(info, FG_SURFACE);

	if (!nc_if && nc_fs)
		SET_FLAG(info, FG_INTERFACE);

	newInfo[index] = info;

	if (savenormals) {
		normal_fs.x /= normal_fs_length;
		normal_fs.y /= normal_fs_length;
		normal_fs.z /= normal_fs_length;

		normal_if.x /= normal_if_length;
		normal_if.y /= normal_if_length;
		normal_if.z /= normal_if_length;

		if (!nc_if && nc_fs) {
			normals[index] = normal_if;
		} else {
			normals[index] = normal_fs;
		}
	}
}

//! Identifies particles at the interface of two fluids and at the free-surface
// SA_BOUNDARY specialization
template<KernelType kerneltype, BoundaryType boundarytype, flag_t simflags, bool savenormals>
__global__
enable_if_t<boundarytype == SA_BOUNDARY>
calcInterfaceparticleDevice(
	neibs_interaction_params<boundarytype> params,
	float4*	__restrict__ normals,
	particleinfo* __restrict__ newInfo,
	const float deltap)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	particleinfo info = params.fetchInfo(index);

	const float4 pos = params.fetchPos(index);

	float4 normal_fs = make_float4(0.0f); // free-surface
	float4 normal_if = make_float4(0.0f); // interface

	if (NOT_FLUID(info) || INACTIVE(pos)) {
		// NOTE: inactive particles will keep their last surface flag status
		newInfo[index] = info;
		if (savenormals)
			normals[index] = make_float4(NAN);
		return;
	}

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

	CLEAR_FLAG(info, FG_SURFACE);
	CLEAR_FLAG(info, FG_INTERFACE);

	// Particle physical density and volume
	const float p_rho = physical_density(params.fetchVel(index).w,fluid_num(info));
	const float p_volume = pos.w/p_rho;

	// self contribution to normalization: W(0)*vol
	normal_fs.w = W<kerneltype>(0.0f, params.slength)*p_volume;
	normal_if.w = W<kerneltype>(0.0f, params.slength)*p_volume;

	// First loop over all neighbors
	for_every_neib(boundarytype, index, pos, gridPos, params.cellStart, params.neibsList) {

		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos( params.fetchPos(neib_index) );

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const particleinfo n_info = params.fetchInfo(neib_index);
		const ParticleType nptype = PART_TYPE(n_info);
		const float r = length3(relPos);

		// neighbor physical density and volume
		const float n_rho = physical_density(params.fetchVel(neib_index).w, fluid_num(n_info));

		if (nptype == PT_FLUID || nptype == PT_VERTEX) {
			const float n_volume = relPos.w/n_rho;
			const float n_volume0 = relPos.w/d_rho0[fluid_num(n_info)];
			float n_theta = n_volume/n_volume0;

			if (r < params.influenceradius) {
				const float f = F<kerneltype>(r, params.slength); // 1/r ∂Wij/∂r
				normal_fs.x -= f * n_theta * relPos.x;
				normal_fs.y -= f * n_theta * relPos.y;
				normal_fs.z -= f * n_theta * relPos.z;
				normal_fs.w += W<kerneltype>(r, params.slength)*n_volume;	// Wij*Vj ;
				if ((fluid_num(info) == fluid_num(n_info) || NOT_FLUID(n_info))) {
					normal_if.x -= f * n_theta * relPos.x;
					normal_if.y -= f * n_theta * relPos.y;
					normal_if.z -= f * n_theta * relPos.z;
					normal_if.w += W<kerneltype>(r, params.slength)*n_volume;	// Wij*Vj ;
				}
			}
		} else if (nptype == PT_BOUNDARY && r < params.influenceradius) {
			const float4 belem = params.fetchBound(neib_index);
			const float3 normal_s = as_float3(belem);
			const float3 q = as_float3(relPos)/params.slength;
			float3 q_vb[3];
			calcVertexRelPos(q_vb, belem,
					params.vertPos0[neib_index], params.vertPos1[neib_index], params.vertPos2[neib_index], params.slength);
			const float ggamAS = gradGamma<kerneltype>(params.slength, q, q_vb, normal_s);
			/* Actual volume should be calculated from the actual (interpolated) density
			 * and the mass. But boundary elements do not have mass. To get their actual 
			 * interpolated volume, we use the fact that:
			 *
			 * 	 n_ref_volume/n_ref_volume0 = n_rho0/n_rho
			 * 	 ==> n_ref_volume = n_ref_volume0*n_rho0/n_rho 
			 * 
			 * with n_ref_volume0 = deltap^3. Note that we are computing the reference volume
			 * here which is defined by:
			 *
			 * 	n_ref_volume = n_volume/n_theta
			 */
			const float n_ref_volume0 = deltap*deltap*deltap;
			const float n_ref_volume = n_ref_volume0*d_rho0[fluid_num(n_info)]/n_rho;


			// Free-surface
			normal_fs.x += ggamAS / n_ref_volume * relPos.x * normal_s.x;
			normal_fs.y += ggamAS / n_ref_volume * relPos.y * normal_s.y;
			normal_fs.z += ggamAS / n_ref_volume * relPos.z * normal_s.z;
			// Interface
			normal_if.x += ggamAS / n_ref_volume * relPos.x * normal_s.x;
			normal_if.y += ggamAS / n_ref_volume * relPos.y * normal_s.y;
			normal_if.z += ggamAS / n_ref_volume * relPos.z * normal_s.z;
		}
	}

	const float gamma = params.fetchGradGamma(index).w;
	normal_fs.x *= p_volume/gamma;
	normal_fs.y *= p_volume/gamma;
	normal_fs.z *= p_volume/gamma;
	normal_fs.w /= gamma;

	normal_if.x *= p_volume/gamma;
	normal_if.y *= p_volume/gamma;
	normal_if.z *= p_volume/gamma;
	normal_if.w /= gamma;

	const float normal_fs_length = length3(normal_fs);
	const float normal_if_length = length3(normal_if);

	int nc_fs = 0;
	int nc_if = 0;

	// Second loop over all neighbors
	for_every_neib(boundarytype, index, pos, gridPos, params.cellStart, params.neibsList) {

		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos( params.fetchPos(neib_index) );

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length3(relPos);

		float cosconeangle;

		const particleinfo n_info = params.fetchInfo(neib_index);

		if (r < params.influenceradius) {
			float criteria_fs = -dot3(normal_fs, relPos);
			cosconeangle = d_cosconeanglefluid;

			if (criteria_fs > r*normal_fs_length*cosconeangle)
				nc_fs++;

			if ((fluid_num(info) == fluid_num(n_info) || NOT_FLUID(n_info))) {
				float criteria_if = -dot3(normal_if, relPos);
				cosconeangle = d_cosconeanglefluid;

				if (criteria_if > r*normal_if_length*cosconeangle)
					nc_if++;
			}
		}
	}

	if (!nc_fs)
		SET_FLAG(info, FG_SURFACE);

	if (!nc_if && nc_fs)
		SET_FLAG(info, FG_INTERFACE);

	newInfo[index] = info;

	if (savenormals) {
		normal_fs.x /= normal_fs_length;
		normal_fs.y /= normal_fs_length;
		normal_fs.z /= normal_fs_length;

		normal_if.x /= normal_if_length;
		normal_if.y /= normal_if_length;
		normal_if.z /= normal_if_length;

		if (!nc_if && nc_fs) {
			normals[index] = normal_if;
		} else {
			normals[index] = normal_fs;
		}
	}
}


// TODO documentation
__global__ void
fluxComputationDevice
			(	const	particleinfo	*pinfo,
				const	float4			*eulerVel,
				const	float4			*boundElement,
						float			*d_IOflux,
				const	uint			numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if(index < numParticles) {
		const particleinfo info = pinfo[index];
		if (IO_BOUNDARY(info) && BOUNDARY(info)) {
			const float4 normal = boundElement[index];
			atomicAdd(&d_IOflux[object(info)], normal.w*dot3(eulerVel[index],normal));
		}
	}
}

/************************************************************************************************************/
/*					   Parallel reduction kernels															*/
/************************************************************************************************************/

extern __shared__ float4 shmem4[];

//! Computes the energy of all particles
extern "C" __global__
void calcEnergiesDevice(
	const		float4	*pPos,
	const		float4	*pVel,
	const	particleinfo	*pInfo,
	const		hashKey	*particleHash,
		uint	numParticles,
		uint	numFluids,
		float4	*output
		)
{
	// shared memory for this kernel should be sized to
	// blockDim.x*numFluids*sizeof(float4)*2

	uint gid = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	uint stride = INTMUL(gridDim.x,blockDim.x);
	// .x kinetic, .y potential, .z internal
	float4 energy[MAX_FLUID_TYPES], E_k[MAX_FLUID_TYPES];

#pragma unroll
	for (uint i = 0; i < MAX_FLUID_TYPES; ++i)
		energy[i] = E_k[i] = make_float4(0.0f);

	while (gid < numParticles) {
		const float4 pos = pPos[gid];
		const float4 vel = pVel[gid];
		const int3 gridPos = calcGridPosFromParticleHash( particleHash[gid] );
		particleinfo pinfo = pInfo[gid];
		if (FLUID(pinfo)) {
			uint fnum = fluid_num(pinfo);
			float v2 = kahan_sqlength(as_float3(vel));
			// TODO improve precision by splitting the float part from the grid part
			float gh = kahan_dot(d_gravity, as_float3(pos) + (make_float3(gridPos) + 0.5f)*d_cellSize);
			kahan_add(energy[fnum].x, pos.w*v2/2, E_k[fnum].x);
			kahan_add(energy[fnum].y, -pos.w*gh, E_k[fnum].y);
			// internal elastic energy
			float gamma = d_gammacoeff[fnum];
			float gm1 = d_gammacoeff[fnum]-1;
			float rho0 = d_rho0[fnum];
			float elen = __powf(vel.w/rho0, gm1)/gm1 + rho0/vel.w - gamma/gm1;
			float ssp = soundSpeed(vel.w, fnum);
			elen *= ssp*ssp/gamma;
			kahan_add(energy[fnum].z, pos.w*elen, E_k[fnum].z);
		}
		gid += stride;
	}

	uint lid = threadIdx.x;
	for (uint offset = blockDim.x/2; offset; offset >>= 1) {
		stride = offset*numFluids; // stride between fields in shmem4 memory
		if (lid >= offset && lid < 2*offset) {
			for (uint i = 0; i < numFluids; ++i) {
				uint idx = lid + offset*i;
				shmem4[idx] = energy[i];
				idx += stride;
				shmem4[idx] = E_k[i];
			}
		}
		__syncthreads();
		if (lid < offset) {
			for (uint i = 0; i < numFluids; ++i) {
				uint idx = lid + offset*(i+1);
				float4 other = shmem4[idx];
				idx += stride;
				float4 oth_k = shmem4[idx];
				kahan_add(energy[i].x, oth_k.x, E_k[i].x);
				kahan_add(energy[i].x, other.x, E_k[i].x);
				kahan_add(energy[i].y, oth_k.y, E_k[i].y);
				kahan_add(energy[i].y, other.y, E_k[i].y);
				kahan_add(energy[i].z, oth_k.z, E_k[i].z);
				kahan_add(energy[i].z, other.z, E_k[i].z);
			}
		}
	}

	if (lid == 0) {
		for (uint i = 0; i < numFluids; ++i) {
			output[blockIdx.x + INTMUL(gridDim.x,i)] = energy[i];
			output[blockIdx.x + INTMUL(gridDim.x,numFluids+i)] = E_k[i];
		}
	}
}

//! Sum the previously computed energy up (across threads)
extern "C" __global__
void calcEnergies2Device(
		float4* buffer,
		uint	prev_blocks,
		uint	numFluids)
{
	// shared memory for this kernel should be sized to
	// blockDim.x*numFluids*sizeof(float4)*2

	uint gid = threadIdx.x;
	float4 energy[MAX_FLUID_TYPES];
	float4 E_k[MAX_FLUID_TYPES];
	for (uint i = 0; i < numFluids; ++i) {
		if (gid < prev_blocks) {
			energy[i] = buffer[gid + prev_blocks*i];
			E_k[i] = buffer[gid + prev_blocks*(numFluids+i)];
		} else {
			energy[i] = E_k[i] = make_float4(0.0f);
		}
	}

	uint stride;
	for (uint offset = blockDim.x/2; offset; offset >>= 1) {
		stride = offset*numFluids; // stride between fields in shmem4 memory
		if (gid >= offset && gid < 2*offset) {
			for (uint i = 0; i < numFluids; ++i) {
				uint idx = gid + offset*i;
				shmem4[idx] = energy[i];
				idx += stride;
				shmem4[idx] = E_k[i];
			}
		}
		__syncthreads();
		if (gid < offset) {
			for (uint i = 0; i < numFluids; ++i) {
				uint idx = gid + offset*(i+1);
				float4 other = shmem4[idx];
				idx += stride;
				float4 oth_k = shmem4[idx];
				kahan_add(energy[i].x, oth_k.x, E_k[i].x);
				kahan_add(energy[i].x, other.x, E_k[i].x);
				kahan_add(energy[i].y, oth_k.y, E_k[i].y);
				kahan_add(energy[i].y, other.y, E_k[i].y);
				kahan_add(energy[i].z, oth_k.z, E_k[i].z);
				kahan_add(energy[i].z, other.z, E_k[i].z);
			}
		}
	}

	if (gid == 0) {
		for (uint i = 0; i < numFluids; ++i)
			buffer[i] = energy[i] + E_k[i];
	}
}

} //namespace cupostprocess

#endif
