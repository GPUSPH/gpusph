/* All viscosity variations. Include this after defining everything else */

#undef VISC_TYPE
#undef VISC_ARTVISC
#undef VISC_DYNAMIC
#undef VISC_KINEMATIC
#undef VISC_SPS

#define VISC_TYPE ARTVISC
#define VISC_ARTVISC
#include "forces_kernel.def"
#undef VISC_TYPE
#undef VISC_ARTVISC

#define VISC_TYPE DYNAMICVISC
#define VISC_DYNAMIC
#include "forces_kernel.def"
#undef VISC_TYPE
#undef VISC_DYNAMIC

#define VISC_TYPE KINEMATICVISC
#define VISC_KINEMATIC
#include "forces_kernel.def"
#undef VISC_TYPE
#undef VISC_KINEMATIC

#define VISC_TYPE SPSVISC
#define VISC_SPS
#include "forces_kernel.def"
#undef VISC_TYPE
#undef VISC_SPS

