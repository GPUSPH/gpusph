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
