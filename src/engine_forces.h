/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _FORCESENGINE_H
#define _FORCESENGINE_H

/* Abstract ForcesEngine base class; it simply defines the interface
 * of the ForcesEngine
 * TODO FIXME in this transition phase it just mirros the exact same
 * set of methods that were exposed in forces, with the same
 * signatures, but the design probably needs to be improved. */

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"

class AbstractForcesEngine
{
public:
	virtual void
	setconstants(const SimParams *simparams, const PhysParams *physparams,
		float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
		idx_t const& allocatedParticles) = 0;

	virtual void
	getconstants(PhysParams *physparams) = 0;

	virtual void
	setplanes(int numPlanes, const float *planesDiv, const float4 *planes) = 0;

	virtual void
	setgravity(float3 const& gravity) = 0;

	// Rigit Body methods

	virtual void
	setrbcg(const float3* cg, int numbodies) = 0;

	virtual void
	setrbstart(const int* rbfirstindex, int numbodies) = 0;

	virtual void
	reduceRbForces(	float4	*forces,
					float4	*torques,
					uint	*rbnum,
					uint	*lastindex,
					float3	*totalforce,
					float3	*totaltorque,
					uint	numbodies,
					uint	numBodiesParticles) = 0;


	// TODO texture bind/unbind and dtreduce for forces should be handled in a different way:
	// they are sets of operations to be done before/after the forces kernel, which are
	// called separately because the kernel might be split into inner/outer calls, and
	// the pre-/post- calls have to do before the first and after the last
	virtual void
	bind_textures(
		const	float4	*pos,
		const	float4	*vel,
		const	float4	*eulerVel,
		const	float4	*oldGGam,
		const	float4	*boundelem,
		const	particleinfo	*info,
		const	float	*keps_tke,
		const	float	*keps_eps,
		uint	numParticles) = 0;

	virtual void
	unbind_textures() = 0;

	virtual void
	setDEM(const float *hDem, int width, int height) = 0;

	virtual void
	unsetDEM() = 0;

	// Striping support: round a number of particles down to the largest multiple
	// of the block size that is not greater than it
	virtual uint
	round_particles(uint numparts) = 0;

	// basic forces step. returns the number of blocks launched
	// (which is the number of blocks to launch dtreduce on
	virtual uint
	basicstep(
		const	float4	*pos,
		const	float2	* const vertPos[],
		const	float4	*vel,
				float4	*forces,
				float2	*contupd,
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
				float	dtadaptfactor,
				float	influenceradius,
		const	float	epsilon,
				uint	*IOwaterdepth,
				float	visccoeff,
				float	*turbvisc,
				float	*keps_tke,
				float	*keps_eps,
				float3	*keps_dkde,
				float	*cfl,
				float	*cflTVisc,
				float	*tempCfl,
				uint	cflOffset) = 0;

	// Reduction methods

	virtual uint
	getFmaxElements(const uint n) = 0;

	virtual uint
	getFmaxTempElements(const uint n) = 0;

	virtual float
	dtreduce(	float	slength,
				float	dtadaptfactor,
				float	visccoeff,
				float	*cfl,
				float	*cflTVisc,
				float	*tempCfl,
				uint	numBlocks) = 0;

};

/// TODO AbstractPostProcessEngine and AbstractBoundaryConditionsEngine
/// are presently just horrible hacks to speed up the transition to header-only
/// engine definitions
//
class AbstractPostProcessEngine
{
public:

virtual void
vorticity(const	float4*		pos,
		const	float4*		vel,
			float3*		vort,
		const	particleinfo	*info,
		const	hashKey*		particleHash,
		const	uint*		cellStart,
		const	neibdata*	neibsList,
			uint		numParticles,
			uint		particleRangeEnd,
			float		slength,
			float		influenceradius) = 0;

//Testpoints
virtual void
testpoints(	const float4*	pos,
			float4*			newVel,
			float*			newTke,
			float*			newEpsilon,
			const particleinfo*	info,
			const hashKey*		particleHash,
			const uint*			cellStart,
			const neibdata*		neibsList,
			uint			numParticles,
			uint			particleRangeEnd,
			float			slength,
			float			influenceradius) = 0;

// Free surface detection
virtual void
surfaceparticle(const	float4*		pos,
				const	float4*     vel,
					float4*		normals,
				const	particleinfo	*info,
					particleinfo	*newInfo,
				const	hashKey*		particleHash,
				const	uint*		cellStart,
				const	neibdata*	neibsList,
					uint		numParticles,
					uint		particleRangeEnd,
					float		slength,
					float		influenceradius,
					bool		savenormals) = 0;

// calculate a private scalar for debugging or a passive value
virtual void
calcPrivate(const	float4*			pos,
			const	float4*			vel,
			const	particleinfo*	info,
					float*			priv,
			const	hashKey*		particleHash,
			const	uint*			cellStart,
			const	neibdata*		neibsList,
					float			slength,
					float			inflRadius,
					uint			numParticles,
					uint			particleRangeEnd) = 0;
};

class AbstractBoundaryConditionsEngine
{
public:

// Computes the boundary conditions on segments using the information from the fluid (on solid walls used for Neumann boundary conditions).
virtual void
saSegmentBoundaryConditions(
			float4*			oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			float4*			oldEulerVel,
			float4*			oldGGam,
			vertexinfo*		vertices,
	const	uint*			vertIDToIndex,
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
	const	float			influenceradius,
	const	bool			initStep) = 0;

// There is no need to use two velocity arrays (read and write) and swap them after.
// Computes the boundary conditions on vertex particles using the values from the segments associated to it. Also creates particles for inflow boundary conditions.
// Data is only read from fluid and segments and written only on vertices.
virtual void
saVertexBoundaryConditions(
			float4*			oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			float4*			oldGGam,
			float4*			oldEulerVel,
			float4*			forces,
			float2*			contupd,
	const	float4*			boundelement,
			vertexinfo*		vertices,
	const	uint*			vertIDToIndex,
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
	const	float			influenceradius,
	const	uint&			newIDsOffset,
	const	bool			initStep) = 0;

// disables particles that went through boundaries when open boundaries are used
virtual void
disableOutgoingParts(		float4*			pos,
							vertexinfo*		vertices,
					const	particleinfo*	info,
					const	uint			numParticles,
					const	uint			particleRangeEnd) = 0;

// downloads the per device waterdepth from the GPU
virtual void
downloadIOwaterdepth(
			uint*	h_IOwaterdepth,
	const	uint*	d_IOwaterdepth,
	const	uint	numObjects) = 0;

// upload the global waterdepth to the GPU
virtual void
uploadIOwaterdepth(
	const	uint*	h_IOwaterdepth,
			uint*	d_IOwaterdepth,
	const	uint	numObjects) = 0;

// identifies vertices at the corners of open boundaries
virtual void
saIdentifyCornerVertices(
	const	float4*			oldPos,
	const	float4*			boundelement,
			particleinfo*	info,
	const	hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			eps) = 0;

// finds the closest vertex particles for segments which have no vertices themselves that are of
// the same object type and are no corner particles
virtual void
saFindClosestVertex(
	const	float4*			oldPos,
			particleinfo*	info,
			vertexinfo*		vertices,
	const	uint*			vertIDToIndex,
	const	hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
	const	uint			particleRangeEnd) = 0;

};
#endif
