/*
 * Device code.
 */

#ifndef _EULER_KERNEL_
#define _EULER_KERNEL_

#include "particledefine.h"
#include "textures.cuh"

__constant__ float	d_epsxsph;
__constant__ float	d_mbomega;
__constant__ float	d_mbphase;
__constant__ float	d_mbksintheta;
__constant__ float	d_mbamplitude;
__constant__ float3	d_mborigin;
__constant__ float3	d_mbv;
__constant__ float3	d_mbtstart;   //.x piston, .y flap, .z gate
__constant__ float3	d_mbtend;
__constant__ float3	d_maxlimit;
__constant__ float3	d_minlimit;
__constant__ float	d_stroke;   //BDR
__constant__ float	d_paddle_h_SWL;   //BDR
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
