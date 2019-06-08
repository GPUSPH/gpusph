/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

using namespace cusph;
using namespace cuphys;
using namespace cuneibs;

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

#include "euler_kernel.def"

// Trivial kernel to update the density of fluid particles
__global__ void
updateDensityDevice(
	const	particleinfo * __restrict__ pinfo,
	const	float4 * __restrict__ forces,
			float4 * __restrict__ vel,
			uint numParticles,
			uint particleRangeEnd,
			float dt)
{
	const int index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	if (index >= particleRangeEnd)
		return;

	particleinfo info = pinfo[index];
	if (!FLUID(info))
		return;

	float rho = vel[index].w;
	float delta = forces[index].w*dt;

	vel[index].w = rho + delta;
}

// Trivial kernel to copy the value of a buffer for particles of a given type
template<ParticleType cptype, typename DataType>
__global__ void
copyTypeDataDevice(
	const	particleinfo * __restrict__ pinfo,
	const	DataType * __restrict__ oldVal,
			DataType * __restrict__ newVal,
			uint particleRangeEnd)
{
	const int index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	if (index >= particleRangeEnd)
		return;

	particleinfo info = pinfo[index];
	if (PART_TYPE(info) != cptype)
		return;

	newVal[index] = oldVal[index];
}

/*!
 This kernel is only used for repacking in combination with the free surface particle identification.
 As soon as repacking is finished the free surface particles are removed by this kernel.
*/
	__global__ void
disableFreeSurfPartsDevice(
		float4*		oldPos,
		const	uint		numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index < numParticles) {
		const particleinfo info = tex1Dfetch(infoTex, index);

		if (SURFACE(info) && NOT_FLUID(info)) {
			float4 pos = oldPos[index];
			if (ACTIVE(pos)) {
				disable_particle(pos);
				oldPos[index] = pos;
			}
		}
	}
}

}
#endif
