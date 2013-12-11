/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
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

#ifndef _EULER_KERNEL_
#define _EULER_KERNEL_

#include "particledefine.h"
#include "textures.cuh"

namespace cueuler {
__constant__ float	d_epsxsph;
__constant__ float3	d_maxlimit;
__constant__ float3	d_minlimit;
__constant__ float3	d_dispvect;
__constant__ float3	d_dispOffset;
__constant__ float4	d_mbdata[MAXMOVINGBOUND];

__constant__ float3	d_rbcg[MAXBODIES];
__constant__ float3	d_rbtrans[MAXBODIES];
__constant__ float	d_rbsteprot[9*MAXBODIES];

/*
 * Device code.
 */

__device__ __forceinline__ void
applyrot(float* rot, float4 & pos, const float3 & cg)
{
	float3 relpos = as_float3(pos) - cg;
	float3 new_relpos;

	// Applying rotation
	new_relpos.x = rot[0]*relpos.x + rot[1]*relpos.y + rot[2]*relpos.z;
	new_relpos.y = rot[3]*relpos.x + rot[4]*relpos.y + rot[5]*relpos.z;
	new_relpos.z = rot[6]*relpos.x + rot[7]*relpos.y + rot[8]*relpos.z;

	pos.x = new_relpos.x + cg.x;
	pos.y = new_relpos.y + cg.y;
	pos.z = new_relpos.z + cg.z;
}


#undef XSPH_KERNEL
#define EULER_KERNEL_NAME eulerDevice
#include "euler_kernel.def"
#undef EULER_KERNEL_NAME

#define XSPH_KERNEL 1
#define EULER_KERNEL_NAME eulerXsphDevice
#include "euler_kernel.def"
#undef XPSH_KERNEL
#undef EULER_KERNEL_NAME
}
#endif
