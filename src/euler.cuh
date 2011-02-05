#ifndef _EULER_CUH_
#define _EULER_CUH_

#define BLOCK_SIZE_INTEGRATE	192

extern "C"
{
void
euler(	float4*		oldPos,
		float4*		oldVel,	      //particleinfo* info,
		particleinfo* info,
		float4*		forces,
		float4*		xsph,
		float4*		newPos,
		float4*		newVel,
		uint		numParticles,
		float		dt,
		float		dt2,
		int			step,
		float		t,
		bool		xsphcorr,
		bool		periodicbound);
}
#endif
