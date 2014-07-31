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

void
forces_bind_textures(	const	float4	*pos,
						const	float4	*vel,
						const	float4	*oldGGam,
						const	float4	*boundelem,
						const	particleinfo	*info,
						uint	numParticles,
						ViscosityType	visctype,
						float	*keps_tke,
						float	*keps_eps,
						BoundaryType	boundarytype);

void
forces_unbind_textures(	ViscosityType	visctype,
						BoundaryType	boundarytype);

float
forces_dtreduce(	float	slength,
				float	dtadaptfactor,
		ViscosityType	visctype,
				float	visccoeff,
				float	*cfl,
				float	*cflTVisc,
				float	*tempCfl,
				uint	numBlocks);

uint
forces(
	const	float4	*pos,
	const	float2	* const vertPos[],
	const	float4	*vel,
			float4	*forces,
	const	float4	*oldGGam,
			float4	*newGGam,
	const	float4	*boundelem,
			float4	*rbforces,
			float4	*rbtorques,
			float4	*xsph,
	const	particleinfo	*info,
	const	hashKey	*particleHash,
	const	uint	*cellStart,
	const	neibdata*neibsList,
			uint	numParticles,
			uint	fromParticle,
			uint	toParticle,
			float	deltap,
			float	slength,
			bool	dtadapt,
			float	dtadaptfactor,
			bool	xsphcorr,
	KernelType		kerneltype,
			float	influenceradius,
	const	float	epsilon,
	const	bool	movingBoundaries,
	ViscosityType	visctype,
			float	visccoeff,
			float	*turbvisc,
			float	*keps_tke,
			float	*keps_eps,
			float3	*keps_dkde,
			float	*cfl,
			float	*cflTVisc,
			float	*tempCfl,
			uint	cflOffset,
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
testpoints(	const float4*	pos,
			float4*			newVel,
			float*			newTke,
			float*			newEpsilon,
			particleinfo*	info,
			hashKey*		particleHash,
			uint*			cellStart,
			neibdata*		neibsList,
			uint			numParticles,
			uint			particleRangeEnd,
			float			slength,
			int				kerneltype,
			float			influenceradius);

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

// Computes the boundary conditions on segments using the information from the fluid (on solid walls used for Neumann boundary conditions).
void
saSegmentBoundaryConditions(
			float4*			oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			float4*			oldEulerVel,
			vertexinfo*		vertices,
	const	float2	* const vertPos[],
	const	float4*			boundelement,
	const	particleinfo*	info,
	const	hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	int				kerneltype,
	const	float			influenceradius,
	const	bool			initStep,
	const	bool			inoutBoundaries);

// There is no need to use two velocity arrays (read and write) and swap them after.
// Computes the boundary conditions on vertex particles using the values from the segments associated to it. Also creates particles for inflow boundary conditions.
// Data is only read from fluid and segments and written only on vertices.
void
saVertexBoundaryConditions(
			float4*			oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			float4*			newGam,
			float4*			oldEulerVel,
			float4*			forces,
	const	float4*			boundelement,
	const	vertexinfo*		vertices,
			particleinfo*	info,
			hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
			uint*			newNumParticles,
	const	uint			particleRangeEnd,
	const	float			dt,
	const	int				step,
	const	float			deltap,
	const	float			slength,
	const	int				kerneltype,
	const	float			influenceradius,
	const	uint&			numActiveParticles,
	const	bool			initStep);

}

#endif
