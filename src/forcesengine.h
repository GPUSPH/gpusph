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

	virtual void
	setrbcg(const float3* cg, int numbodies) = 0;

	virtual void
	setrbstart(const int* rbfirstindex, int numbodies) = 0;

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

	virtual float
	dtreduce(	float	slength,
				float	dtadaptfactor,
				float	visccoeff,
				float	*cfl,
				float	*cflTVisc,
				float	*tempCfl,
				uint	numBlocks) = 0;

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
};
#endif
