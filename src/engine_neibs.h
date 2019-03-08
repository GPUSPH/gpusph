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

/*! \file
 * Contains the abstract interface for the NeibsEngine
 */

#ifndef _NEIBSENGINE_H
#define _NEIBSENGINE_H

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "timing.h"
#include "buffer.h"

/*! Abstract class that defines the interface for the NeibsEngine.
 * The NeibsEngine is the part of the framework that takes care of building
 * the neighbors list (and associated tasks, such as computing the
 * particle hash and handling periodicity).
 */
class AbstractNeibsEngine
{
public:
	virtual ~AbstractNeibsEngine() {}

	/// Set the device constants
	virtual void
	setconstants(const SimParams *simparams, const PhysParams *physparams,
		float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
		idx_t const& allocatedParticles) = 0;

	/// Get the device constants
	virtual void
	getconstants(SimParams *simparams, PhysParams *physparams) = 0;

	/// Reset iteration information (timing, max neibs, etc)
	virtual void
	resetinfo() = 0;

	/// Get the current timing information
	virtual void
	getinfo(TimingInfo &timingInfo) = 0;

	/// Compute the particle hash, and disable particles that have
	/// flown out of the domain.
	virtual void
	calcHash(const BufferList& bufread,
			BufferList& bufwrite,
			const uint	numParticles) = 0;

	/// Update the particle hash computed on host.
	/// This is done to mark cells and particles that belong to other devices
	virtual void
	fixHash(const BufferList& bufread,
			BufferList& bufwrite,
			const uint	numParticles) = 0;

	/// Sort the data to match the new particle order
	virtual void
	reorderDataAndFindCellStart(
			uint*		segmentStart,
			BufferList& sorted_buffers,
			const BufferList& unsorted_buffers,
			const uint		numParticles,
			uint*			newNumParticles) = 0;

	/// Sort the particles by hash and particleinfo
	virtual void
	sort(	const BufferList& bufread,
			BufferList& bufwrite,
			uint	numParticles) = 0;

	/// Build the neighbors list
	virtual void
	buildNeibsList( const BufferList&	bufread,
						  BufferList&	bufwrite,
					const uint			numParticles,
					const uint			particleRangeEnd,
					const uint			gridCells,
					const float			sqinfluenceradius,
					const float			boundNlSqInflRad) = 0;
};
#endif
