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

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"

/* Important notes on block sizes:
	- all kernels accessing the neighbor list MUST HAVE A BLOCK
	MULTIPLE OF NEIBINDEX_INTERLEAVE
	- a parallel reduction for adaptive dt is done inside forces, block
	size for forces MUST BE A POWER OF 2
 */
#if (__COMPUTE__ >= 20)
	#define BLOCK_SIZE_FORCES		128
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
	#define BLOCK_SIZE_FORCES		64
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
	#define BLOCK_SIZE_FMAX			256
	#define MAX_BLOCKS_FMAX			64
#endif


extern "C"
{
void
setforcesconstants(const SimParams *simaprams, const PhysParams *physparams);

void
getforcesconstants(PhysParams *physparams);

void
setplaneconstants(int numPlanes, const float* PlanesDiv, const float4* Planes);

void
setoutletforces(const PhysParams *phys);

void
setgravity(float3 const& gravity);

void
setforcesrbcg(const float3* cg, int numbodies);

void
setforcesrbstart(const uint* rbfirstindex, int numbodies);

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
		uint			particleRangeEnd,
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
		float2*			tau[],
		bool			periodicbound,
		SPHFormulation	sph_formulation,
		BoundaryType	boundarytype,
		bool			usedem);

void
sps(	float4*			pos,
		float4*			vel,
		particleinfo*	info,
		uint*			neibsList,
		uint			numParticles,
		uint			particleRangeEnd,
		float			slength,
		KernelType		kerneltype,
		float			influenceradius,
		ViscosityType	visctype,
		float2*			tau[],
		bool			periodicbound );

void
shepard(float4*		pos,
		float4*		oldVel,
		float4*		newVel,
		particleinfo*	info,
		uint*		neibsList,
		uint		numParticles,
		uint		particleRangeEnd,
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
	uint		particleRangeEnd,
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
setDemTexture(const float *hDem, int width, int height);

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
getNumPartsFmax(const uint n);

uint
getFmaxTempStorageSize(const uint n);

float
cflmax( const uint	n,
		float*		cfl,
		float*		tempCfl);

/* Reductions */
void set_reduction_params(void* buffer, size_t blocks,
		size_t blocksize_max, size_t shmem_max);
void unset_reduction_params();

// Compute system energy
void calc_energy(
		float4*			output,
		float4	const*	pos,
		float4	const*	vel,
	particleinfo const*	pinfo,
		uint			numParticles,
		uint			numFluids);
}
#endif
