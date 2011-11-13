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

#if (__COMPUTE__ >= 20)
	#define BLOCK_SIZE_CALCVORT		128
	#define MIN_BLOCKS_CALCVORT		6
	#define BLOCK_SIZE_CALCTEST		128
	#define MIN_BLOCKS_CALCTEST		6
	#define BLOCK_SIZE_SHEPARD		128
	#define MIN_BLOCKS_SHEPARD		6
	#define BLOCK_SIZE_MLS			128
	#define MIN_BLOCKS_MLS			6
	#define BLOCK_SIZE_SPS			128
	#define MIN_BLOCKS_SPS			6
	#define BLOCK_SIZE_FMAX			256
	#define MAX_BLOCKS_FMAX			64
#else
	#define MIN_BLOCKS_FORCES		6
	#define BLOCK_SIZE_CALCVORT		128
	#define MIN_BLOCKS_CALCVORT		1
	#define BLOCK_SIZE_CALCTEST		128
	#define MIN_BLOCKS_CALCTEST		1
	#define BLOCK_SIZE_SHEPARD		224
	#define MIN_BLOCKS_SHEPARD		1
	#define BLOCK_SIZE_MLS			128
	#define MIN_BLOCKS_MLS			1
	#define BLOCK_SIZE_SPS			128
	#define MIN_BLOCKS_SPS			1
	#define MAX_THREADS_FMAX		256
	#define MAX_BLOCKS_FMAX			64
#endif


extern "C"
{

float
forces(	float4*			pos,
		float4*			vel,
		float4*			forces,
		float4*			rbforces,
		float4*			rbtorques,
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
		float*			tempCfl,
		uint			numPartsFmax,
		float2*			tau[],
		bool			periodicbound,
		SPHFormulation	sph_formulation,
		BoundaryType	boundarytype,
		bool			usedem,
		float*			cfl2);


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

// Free surface detection
void
surfaceparticle(	float4*		pos,
			float4*		vel,
		    float4*     normals,
			particleinfo*	info,
			particleinfo*  newInfo,
			uint*		neibsList,
			uint		numParticles,
			float		slength,
			int			kerneltype,
			float		influenceradius,
			bool		periodicbound,
			bool        savenormals);

void
setDemTexture(float *hDem, int width, int height);

void
releaseDemTexture();

void
reduceRbForces(	float4*		forces,
				float4*		torques,
				uint*		rbnum,
				uint*		lastindex,
				float3*		totalforce,
				float3*		totaltorque,
				uint		numbodies,
				uint		numBodiesParticles);

uint
getFmaxTempStorageSize(const uint n);

float
cflmax( const uint	n,
		float*		cfl,
		float*		tempCfl);
}
#endif
