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

#ifndef _INTEGRATIONENGINE_H
#define _INTEGRATIONENGINE_H

/* Abstract IntegrationEngine base class; it simply defines the interface
 * of the IntegrationEngine
 * TODO FIXME in this transition phase it just mirros the exact same
 * set of methods that were exposed in euler, with the same
 * signatures, but the design probably needs to be improved. */

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "buffer.h"

class AbstractIntegrationEngine
{
public:
	virtual void
	setconstants(const PhysParams *physparams, float3 const& worldOrigin,
		uint3 const& gridSize, float3 const& cellSize, idx_t const& allocatedParticles,
		int const& maxneibsnum, float const& slength) = 0;

	virtual void
	getconstants(PhysParams *physparams) = 0;

	virtual void
	setrbcg(const int3* cgGridPos, const float3* cgPos, int numbodies) = 0;

	virtual void
	setrbtrans(const float3* trans, int numbodies) = 0;

	virtual void
	setrbsteprot(const float* rot, int numbodies) = 0;

	virtual void
	setrblinearvel(const float3* linearvel, int numbodies) = 0;

	virtual void
	setrbangularvel(const float3* angularvel, int numbodies) = 0;

	/// Single integration 
	// TODO will probably need to be made more generic for other
	// integration schemes
	virtual void
	basicstep(
		MultiBufferList::const_iterator bufread,	// this is the read only arrays
		MultiBufferList::iterator bufreadUpdate,	// the read array but it will be written to in certain cases (densitySum)
		MultiBufferList::iterator bufwrite,
		const	uint	*cellStart,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	float	dt2,
		const	int		step,
		const	float	t,
		const	float	slength,
		const	float	influenceRadius)
	= 0;

};
#endif
