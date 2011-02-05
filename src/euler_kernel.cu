/*
 * Device code.
 */

#ifndef _EULER_KERNEL_
#define _EULER_KERNEL_

#include "particledefine.h"
#include "textures.cuh"

__constant__ float	d_epsxsph;
__constant__ float3	d_mborigin;
__constant__ float3	d_mbv;
__constant__ float	d_mbdisp;
__constant__ float2 d_mbsincostheta;
__constant__ float3	d_maxlimit;
__constant__ float3	d_minlimit;
__constant__ float3 d_dispvect3;

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
