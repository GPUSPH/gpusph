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

#ifndef _FILTERENGINE_H
#define _FILTERENGINE_H

/* Abstract FilterEngine base class; it simply defines the interface
 * of the FilterEngine.
 * FilterEngines are run periodically (every N iterations) to filter
 * values (typically, smoothing density: MLS, shepard)
 */

#include "particledefine.h"

// TODO as usual, the API needs to be redesigned properly
class AbstractFilterEngine
{
	uint m_frequency; // frequency of the pre-processing (iterations)
public:

	AbstractFilterEngine(uint _frequency) : m_frequency(_frequency)
	{}

	void set_frequency(uint _frequency)
	{ m_frequency = _frequency; }

	inline uint frequency() const
	{ return m_frequency; }

	virtual void setconstants() = 0 ; // TODO
	virtual void getconstants() = 0 ; // TODO

	virtual void
	process(
		const	float4	*pos,
		const	float4	*oldVel,
				float4	*newVel,
		const	particleinfo	*info,
		const	hashKey	*particleHash,
		const	uint	*cellStart,
		const	neibdata*neibsList,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius) = 0;
};
#endif
