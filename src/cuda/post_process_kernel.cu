/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A.
 	Dalrymple, Eugenio Rustico, Ciro Del Negro

	Conservatoire National des Arts et Metiers, Paris, France

	Istituto Nazionale di Geofisica e Vulcanologia,
    Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

#ifndef _POST_PROCESS_KERNEL_
#define _POST_PROCESS_KERNEL_

#include "particledefine.h"
#include "textures.cuh"
#include "vector_math.h"
#include "multi_gpu_defines.h"
#include "GlobalData.h"

#if __COMPUTE__ < 20
#define printf(...) /* eliminate printf from 1.x */
#endif

namespace cupostprocess {

using namespace cubounds;

// Core SPH functions
#include "sph_core_utils.cuh"

/** \name Device constants
 *  @{ */
__constant__ idx_t	d_neiblist_end;			///< maximum number of neighbors * number of allocated particles
__constant__ idx_t	d_neiblist_stride;		///< stride between neighbors of the same particle

// Free surface detection
__constant__ float	d_cosconeanglefluid;
__constant__ float	d_cosconeanglenonfluid;


/************************************************************************************************************/
/*					   Auxiliary kernels used for post processing										    */
/************************************************************************************************************/

//! Computes the vorticity field
template<KernelType kerneltype>
__global__ void
calcVortDevice(	const	float4*		posArray,
						float3*		vorticity,
				const	hashKey*		particleHash,
				const	uint*		cellStart,
				const	neibdata*	neibsList,
				const	uint		numParticles,
				const	float		slength,
				const	float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// computing vorticity only for fluid particles
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (NOT_FLUID(info))
		return;

	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif
	const float4 vel = tex1Dfetch(velTex, index);

	float3 vort = make_float3(0.0f);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// First loop over all neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
		const float4 relPos = pos_corr - posArray[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		// Compute relative velocity
		// Now relVel is a float4 and neib density is stored in relVel.w
		const float4 relVel = as_float3(vel) - tex1Dfetch(velTex, neib_index);
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// Compute vorticity
		if (r < influenceradius && FLUID(neib_info)) {
			const float f = F<kerneltype>(r, slength)*relPos.w/relVel.w;	// ∂Wij/∂r*Vj
			// vxij = vxi - vxj and same for vyij and vzij
			vort.x += f*(relVel.y*relPos.z - relVel.z*relPos.y);		// vort.x = ∑(vyij(zi - zj) - vzij*(yi - yj))*∂Wij/∂r*Vj
			vort.y += f*(relVel.z*relPos.x - relVel.x*relPos.z);		// vort.y = ∑(vzij(xi - xj) - vxij*(zi - zj))*∂Wij/∂r*Vj
			vort.z += f*(relVel.x*relPos.y - relVel.y*relPos.x);		// vort.x = ∑(vxij(yi - yj) - vyij*(xi - xj))*∂Wij/∂r*Vj
		}
	} // end of loop trough neighbors

	vorticity[index] = vort;
}


//! Compute the values of velocity, density, k and epsilon at test points
template<KernelType kerneltype>
__global__ void
calcTestpointsVelocityDevice(	const float4*	oldPos,
								float4*			newVel,
								float*			newTke,
								float*			newEpsilon,
								const hashKey*	particleHash,
								const uint*		cellStart,
								const neibdata*	neibsList,
								const uint		numParticles,
								const float		slength,
								const float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	const particleinfo info = tex1Dfetch(infoTex, index);
	if(!TESTPOINT(info))
		return;

	#if (__COMPUTE__ >= 20)
	const float4 pos = oldPos[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	// this is the velocity (x,y,z) and pressure (w)
	float4 velavg = make_float4(0.0f);
	// this is for k/epsilon
	float tkeavg = 0.0f;
	float epsavg = 0.0f;
	// this is the shepard filter sum(w_b w_{ab})
	float alpha = 0.0f;

	// Compute grid position of current particle
	int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// First loop over all neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if (__COMPUTE__ >= 20)
		const float4 relPos = pos_corr - oldPos[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && (FLUID(neib_info) || VERTEX(neib_info))) {
			const float4 neib_vel = tex1Dfetch(velTex, neib_index);
			const float w = W<kerneltype>(r, slength)*relPos.w/neib_vel.w;	// Wij*mj
			//Velocity
			velavg.x += w*neib_vel.x;
			velavg.y += w*neib_vel.y;
			velavg.z += w*neib_vel.z;
			//Pressure
			velavg.w += w*P(neib_vel.w, fluid_num(neib_info));
			// Turbulent kinetic energy
			if(newTke){
				const float neib_tke = tex1Dfetch(keps_kTex, neib_index);
				tkeavg += w*neib_tke;
			}
			if(newEpsilon){
				const float neib_eps = tex1Dfetch(keps_eTex, neib_index);
				epsavg += w*neib_eps;
			}
			//Shepard filter
			alpha += w;
		}
	}

	// Renormalization by the Shepard filter
	if(alpha>1e-5f) {
		velavg /= alpha;
		if(newTke)
			tkeavg /= alpha;
		if(newEpsilon)
			epsavg /= alpha;
	}
	else {
		velavg = make_float4(0.0f);
		if(newTke)
			tkeavg = 0.0f;
		if(newEpsilon)
			epsavg = 0.0f;
	}

	newVel[index] = velavg;
	if(newTke)
		newTke[index] = tkeavg;
	if(newEpsilon)
		newEpsilon[index] = epsavg;
}


//! Identifies particles which form the free-surface
template<KernelType kerneltype, flag_t simflags, bool savenormals>
__global__ void
calcSurfaceparticleDevice(	const	float4*			posArray,
									float4*			normals,
									particleinfo*	newInfo,
							const	hashKey*		particleHash,
							const	uint*			cellStart,
							const	neibdata*		neibsList,
							const	uint			numParticles,
							const	float			slength,
							const	float			influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	particleinfo info = tex1Dfetch(infoTex, index);

	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif
	float4 normal = make_float4(0.0f);

	if (NOT_FLUID(info) || INACTIVE(pos)) {
		// NOTE: inactive particles will keep their last surface flag status
		newInfo[index] = info;
		return;
	}

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	CLEAR_FLAG(info, FG_SURFACE);

	// self contribution to normalization: W(0)*vol
	normal.w = W<kerneltype>(0.0f, slength)*pos.w/tex1Dfetch(velTex, index).w;

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// First loop over all neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
		const float4 relPos = pos_corr - posArray[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		// neighbor volume
		const float neib_vol = relPos.w/tex1Dfetch(velTex, neib_index).w;

		if (r < influenceradius) {
			const float f = F<kerneltype>(r, slength)*neib_vol; // 1/r ∂Wij/∂r Vj
			normal.x -= f * relPos.x;
			normal.y -= f * relPos.y;
			normal.z -= f * relPos.z;
			normal.w += W<kerneltype>(r, slength)*neib_vol;	// Wij*Vj ;

		}
	}

	// Checking the planes
	if (simflags & ENABLE_PLANES)
		for (uint i = 0; i < d_numplanes; ++i) {
			const float r = PlaneDistance(gridPos, as_float3(pos), d_plane[i]);
			if (r < influenceradius) {
				// since our current normal is still unnormalized, the plane normal
				// contribution must be scaled up to match the length of the current normal
				as_float3(normal) += d_plane[i].normal*length3(normal);
			}
		}

	const float normal_length = length3(normal);

	// Second loop over all neighbors

	// Resetting persistent variables across getNeibData
	neib_cellnum = 0;
	neib_cell_base_index = 0;

	// loop over all the neighbors (Second loop)
	int nc = 0;
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
		const float4 relPos = pos_corr - posArray[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length3(relPos);

		float cosconeangle;

		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius) {
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

//! Compute a private variable
/*!
 This function computes an arbitrary passive array. It can be used for debugging purposes or passive scalars
*/
__global__ void
calcPrivateDevice(	const	float4*		pos_array,
							float*		priv,
					const	hashKey*	particleHash,
					const	uint*		cellStart,
					const	neibdata*	neibsList,
					const	float		slength,
					const	float		inflRadius,
							uint		numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if(index < numParticles) {
		#if( __COMPUTE__ >= 20)
		float4 pos = pos_array[index];
		#else
		float4 pos = tex1Dfetch(posTex, index);
		#endif
		const particleinfo info = tex1Dfetch(infoTex, index);
		float4 vel = tex1Dfetch(velTex, index);

		// Compute grid position of current particle
		const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

		// Persistent variables across getNeibData calls
		char neib_cellnum = 0;
		uint neib_cell_base_index = 0;
		float3 pos_corr;

		priv[index] = 0;

		// Loop over all the neighbors
		for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
			neibdata neib_data = neibsList[i + index];

			if (neib_data == 0xffff) break;

			const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
						neib_cellnum, neib_cell_base_index);

			// Compute relative position vector and distance

			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);
			#if( __COMPUTE__ >= 20)
			const float4 relPos = pos_corr - pos_array[neib_index];
			#else
			const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
			#endif
			float r = length(as_float3(relPos));
			if (r < inflRadius)
				priv[index] += 1;
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
