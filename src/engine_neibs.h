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

#ifndef _NEIBSENGINE_H
#define _NEIBSENGINE_H

/* Abstract NeibsEngine base class; it simply defines the interface
 * of the NeibsEngine
 * TODO FIXME in this transition phase it just mirros the exact same
 * set of methods that were exposed in buildneibs, with the same
 * signatures, but the design probably needs to be improved. */

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "timing.h"
#include "buffer.h"

class AbstractNeibsEngine
{
public:
	virtual void
	setconstants(const SimParams *simparams, const PhysParams *physparams,
		float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
		idx_t const& allocatedParticles) = 0;

	virtual void
	getconstants(SimParams *simparams, PhysParams *physparams) = 0;

	virtual void
	resetinfo() = 0;

	virtual void
	getinfo(TimingInfo &timingInfo) = 0;

	virtual void
	calcHash(float4*	pos,
		 hashKey*	particleHash,
		 uint*		particleIndex,
		 const particleinfo* particleInfo,
		 uint*		compactDeviceMap,
		 const uint		numParticles) = 0;

	virtual void
	fixHash(hashKey*	particleHash,
			uint*		particleIndex,
			const particleinfo* particleInfo,
			uint*		compactDeviceMap,
			const uint		numParticles) = 0;

	virtual void
	reorderDataAndFindCellStart(
		uint*				cellStart,			// output: cell start index
		uint*				cellEnd,			// output: cell end index
		uint*				segmentStart,		// output: segment start

		const hashKey*		particleHash,		// input: sorted grid hashes
		const uint*			particleIndex,		// input: sorted particle indices

		MultiBufferList::iterator sorted_buffers,		// output: sorted buffers
		MultiBufferList::const_iterator unsorted_buffers, // input: buffers to sort

		const uint			numParticles,		// input: number of particles in input buffers
		uint*				newNumParticles)	// output: number of active particles found
	= 0;

	virtual void
	updateVertIDToIndex(const particleinfo*	particleInfo,	// input: particle's information
						uint*			vertIDToIndex,	// output: vertIDToIndex array
						const uint		numParticles)	// input: total number of particles
	= 0;

	virtual void
	buildNeibsList(	neibdata*			neibsList,
					const float4*		pos,
					const particleinfo*	info,
					vertexinfo*			vertices,
					const float4		*boundelem,
					float2*				vertPos[],
					const uint*			vertIDToIndex,
					const hashKey*		particleHash,
					const uint*			cellStart,
					const uint*			cellEnd,
					const uint			numParticles,
					const uint			particleRangeEnd,
					const uint			gridCells,
					const float			sqinfluenceradius,
					const float			boundNlSqInflRad) = 0;

	virtual void
	sort(	hashKey	*particleHash,
			uint	*particleIndex,
			uint	numParticles) = 0;
};
#endif
