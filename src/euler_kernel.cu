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

__constant__ float	d_epsxsph;
__constant__ float3	d_maxlimit;
__constant__ float3	d_minlimit;
__constant__ float3 d_dispvect3;
__constant__ float4	d_mbdata[MAXMOVINGBOUND];

__constant__ float3 d_rb_cg2[MAXBODIES];
__constant__ float3 d_rb_trans[MAXBODIES];
__constant__ float	d_rb_steprot[MAXBODIES];


/*
 * Device code.
 */

#undef XSPH_KERNEL
#define EULER_KERNEL_NAME eulerDevice
#include "euler_kernel.def"
#undef EULER_KERNEL_NAME

#define XSPH_KERNEL 1
#define EULER_KERNEL_NAME eulerXsphDevice
#include "euler_kernel.def"
#undef XPSH_KERNEL
#undef EULER_KERNEL_NAME

#endif
