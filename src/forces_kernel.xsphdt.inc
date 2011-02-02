/* All Xsph/Dt variations. Include this after defining CORR_TYPE and
   CORR_<name> */

#undef XSPH_KERNEL
#undef DT_KERNEL

/* NO Xsph, NO Dt */
#include "forces_kernel.visc.inc"

/* Xsph, NO Dt */
#define XSPH_KERNEL Xsph
#include "forces_kernel.visc.inc"

/* Xsph, Dt */
#define DT_KERNEL Dt
#include "forces_kernel.visc.inc"

/* NO Xsph, Dt */
#undef XSPH_KERNEL
#include "forces_kernel.visc.inc"

#undef DT_KERNEL
