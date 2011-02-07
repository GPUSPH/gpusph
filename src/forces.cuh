#ifndef _FORCES_CUH_
#define _FORCES_CUH_

#include "cudpp/cudpp.h"

#define BLOCK_SIZE_CALCVORT		128
#define BLOCK_SIZE_SHEPARD		128
#define BLOCK_SIZE_MLS			128
#define BLOCK_SIZE_XSPH			128


extern "C"
{

float
forces(	float4*			pos,
		float4*			vel,
		float4*			forces,
		float4*			xsph,
		particleinfo*	info,
		uint*			neibsList,
		uint			numParticles,
		float			slength,
		float			dt,
		bool			dtadapt,
		float			dtadaptfactor,
		bool			xsphcorr,
		KernelType		kerneltype,
		float			influenceradius,
		ViscosityType	visctype,
		float			visccoeff,
		float*			cfl,
		float*			tempfmax,
		uint			numPartsFmax,
		CUDPPHandle		scanplan,
		float*			visc,
		float2*			tau[],
		bool			periodicbound,
		SPHFormulation		sph_formulation,
		BoundaryType	boundarytype,
		bool			usedem);

void
xsph(	float4*		pos,
		float4*		vel,
		float4*		forces,
		float4*		xsph,
		particleinfo*	info,
		uint*		neibsList,
		uint		numParticles,
		float		slength,
		int			kerneltype,
		float		influenceradius,
		bool		periodicbound);

void
shepard(float4*		pos,
		float4*		oldVel,
		float4*		newVel,
		particleinfo*	info,
		uint*		neibsList,
		uint		numParticles,
		float		slength,
		int			kerneltype,
		float		influenceradius,
		bool		periodicbound);

void
mls(float4*		pos,
	float4*		oldVel,
	float4*		newVel,
	particleinfo*	info,
	uint*		neibsList,
	uint		numParticles,
	float		slength,
	int			kerneltype,
	float		influenceradius,
	bool		periodicbound);

void
vorticity(	float4*		pos,
			float4*		vel,
			float3*		vort,
			particleinfo*	info,
			uint*		neibsList,
			uint		numParticles,
			float		slength,
			int			kerneltype,
			float		influenceradius,
			bool		periodicbound);

void
setDemTexture(float *hDem, int width, int height);

void
releaseDemTexture();
}
#endif
