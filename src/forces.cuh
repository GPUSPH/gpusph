/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

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
setforcesconstants(const SimParams *simaprams, const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
	idx_t const& allocatedParticles);

void
getforcesconstants(PhysParams *physparams);

void
setplaneconstants(int numPlanes, const float* PlanesDiv, const float4* Planes);

void
setgravity(float3 const& gravity);

void
setforcesrbcg(const float3* cg, int numbodies);

void
setforcesrbstart(const uint* rbfirstindex, int numbodies);

float
forces(
	const	float4	*pos,
	const	float4	*vel,
			float4	*forces,
	const	float4	*gradgam,
	const	float4	*boundelem,
			float4	*rbforces,
			float4	*rbtorques,
			float4	*xsph,
	const	particleinfo	*info,
	const	hashKey	*particleHash,
	const	uint	*cellStart,
	const	neibdata*neibsList,
			uint	numParticles,
			uint	particleRangeEnd,
			float	deltap,
			float	slength,
			float	dt,
			bool	dtadapt,
			float	dtadaptfactor,
			bool	xsphcorr,
	KernelType		kerneltype,
			float	influenceradius,
	ViscosityType	visctype,
			float	visccoeff,
			float	*turbvisc,
			float	*keps_tke,
			float	*keps_eps,
			float2	*keps_dkde,
			float	*cfl,
			float	*cflTVisc,
			float	*tempCfl,
	SPHFormulation	sph_formulation,
	BoundaryType	boundarytype,
			bool	usedem);

void
sps(		float2*			tau[],
	const	float4	*pos,
	const	float4	*vel,
const	particleinfo	*info,
	const	hashKey	*particleHash,
	const	uint	*cellStart,
	const	neibdata*neibsList,
			uint	numParticles,
			uint	particleRangeEnd,
			float	slength,
		KernelType	kerneltype,
			float	influenceradius);

void
mean_strain_rate(
			float	*strainrate,
	const	float4	*pos,
	const	float4	*vel,
const	particleinfo	*info,
	const	hashKey	*particleHash,
	const	uint	*cellStart,
	const	neibdata*neibsList,
	const	float4	*gradgam,
	const	float4	*boundelem,
			uint	numParticles,
			uint	particleRangeEnd,
			float	slength,
		KernelType	kerneltype,
			float	influenceradius);

void
shepard(float4*		pos,
		float4*		oldVel,
		float4*		newVel,
		particleinfo	*info,
		hashKey*		particleHash,
		uint*		cellStart,
		neibdata*	neibsList,
		uint		numParticles,
		uint		particleRangeEnd,
		float		slength,
		int			kerneltype,
		float		influenceradius);

void
mls(float4*		pos,
	float4*		oldVel,
	float4*		newVel,
	particleinfo	*info,
	hashKey*		particleHash,
	uint*		cellStart,
	neibdata*	neibsList,
	uint		numParticles,
	uint		particleRangeEnd,
	float		slength,
	int			kerneltype,
	float		influenceradius);


void
vorticity(	float4*		pos,
			float4*		vel,
			float3*		vort,
			particleinfo*	info,
			hashKey*		particleHash,
			uint*		cellStart,
			neibdata*	neibsList,
			uint		numParticles,
			uint		particleRangeEnd,
			float		slength,
			int			kerneltype,
			float		influenceradius);

//Testpoints
void
testpoints(	const float4*		pos,
			float4*		newVel,
			particleinfo*	info,
			hashKey*		particleHash,
			uint*		cellStart,
			neibdata*	neibsList,
			uint		numParticles,
			uint		particleRangeEnd,
			float		slength,
			int			kerneltype,
			float		influenceradius);

// Free surface detection
void
surfaceparticle(	float4*		pos,
			float4*		vel,
		    float4*     normals,
			particleinfo*	info,
			particleinfo*  newInfo,
			hashKey*		particleHash,
			uint*		cellStart,
			neibdata*	neibsList,
			uint		numParticles,
			uint		particleRangeEnd,
			float		slength,
			int			kerneltype,
			float		influenceradius,
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
getFmaxElements(const uint n);

uint
getFmaxTempElements(const uint n);

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
			float4			*output,
	const	float4			*pos,
	const	float4			*vel,
	const	particleinfo	*pinfo,
	const	hashKey			*particleHash,
			uint			numParticles,
			uint			numFluids);

// Computes initial values of the gamma gradient
void
initGradGamma(	float4*		oldPos,
		float4*		newPos,
		float4*		virtualVel,
		particleinfo*	info,
		float4*		boundElement,
		float4*		gradGamma,
		const hashKey*	particleHash,
		const uint*	cellStart,
		neibdata*	neibsList,
		uint		numParticles,
		uint		particleRangeEnd,
		float		deltap,
		float		slength,
		float		inflRadius,
		int			kerneltype);

// Computes current value of the gamma gradient and update gamma value
// according to the evolution equation { dGamma/dt = gradGamma * relVel }
void
updateGamma(			float4*			oldPos,
				const	float4*			newPos,
						float4*			virtualVel,
						particleinfo*	info,
						float4*			boundElement,
						float4*			oldGam,
						float4*			newGam,
						float2*			vertPos[],
				const	hashKey*		particleHash,
				const	uint*			cellStart,
						neibdata*		neibsList,
						uint			numParticles,
						uint			particleRangeEnd,
						float			slength,
						float			inflRadius,
						float			virtDt,
						bool			predcor,
						int				kerneltype);

//Moves particles back to their initial positions during initialization of gamma
void
updatePositions(	float4*		oldPos,
			float4*		newPos,
			float4*		virtualVel,
			particleinfo*	info,
			float		virtDt,
			uint		numParticles,
			uint		particleRangeEnd);

// calculate a private scalar for debugging or a passive value
void
calcPrivate(const	float4*			pos,
			const	float4*			vel,
			const	particleinfo*	info,
					float*			priv,
			const	hashKey*		particleHash,
			const	uint*			cellStart,
					neibdata*		neibsList,
					float			slength,
					float			inflRadius,
					uint			numParticles,
					uint			particleRangeEnd);

// Recomputes values at the boundary elements (currently only density) as an average
// over three vertices of this element
void
updateBoundValues(	float4*		oldVel,
			float*		oldTKE,
			float*		oldEps,
			vertexinfo*	vertices,
			particleinfo*	info,
			uint		numParticles,
			uint		particleRangeEnd,
			bool		initStep);

// Recomputes values at the vertex particles, following procedure similar to Shepard filter.
// Only fluid particles are taken into summation
// oldVel array is used to read density of fluid particles and to write density of vertex particles.
// There is no need to use two velocity arrays (read and write) and swap them after.
void
dynamicBoundConditions(	const float4*		oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			const particleinfo*	info,
			const hashKey*		particleHash,
			const uint*		cellStart,
			const neibdata*	neibsList,
			const uint		numParticles,
			const uint		particleRangeEnd,
			const float		deltap,
			const float		slength,
			const int		kerneltype,
			const float		influenceradius);

}

#endif
