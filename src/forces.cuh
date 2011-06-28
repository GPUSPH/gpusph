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

#ifndef _FORCES_CUH_
#define _FORCES_CUH_

#include "cudpp/cudpp.h"

#define BLOCK_SIZE_CALCVORT		128
#define BLOCK_SIZE_CALCNODES	128
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
		float2*			tau[],
		bool			periodicbound,
		SPHFormulation	sph_formulation,
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

//Testpoints
void
testpoints(	float4*		pos,
			float4*		newVel,
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
