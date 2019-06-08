/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

/*! \file
 * Contains the abstract interface for filter engines.
 */

#include "buffer.h"
#include "particledefine.h"

/* Abstract class that defines the interface for each FilterEngine
 * FilterEngines are run periodically (every N iterations) to filter
 * values (typically, smoothing density: MLS, shepard).
 */
class AbstractFilterEngine
{
	uint m_frequency; // frequency of the pre-processing (iterations)
public:

	//! Constructor, sets the frequency
	AbstractFilterEngine(uint _frequency) : m_frequency(_frequency)
	{}

	// destructor must be virtual
	virtual ~AbstractFilterEngine() {}

	//! Override the frequency
	void set_frequency(uint _frequency)
	{ m_frequency = _frequency; }

	//! Get the frequency
	inline uint frequency() const
	{ return m_frequency; }

	//! Getter/setter for device constants
	//! @{
	virtual void setconstants() = 0 ; // TODO
	virtual void getconstants() = 0 ; // TODO
	//! @}


	//! Run the filter
	/*! \note the surrounding code currently assumes that only the velocity
	 * and/or density will be filtered, and will not swap other buffers.
	 * If the need arise, we can extend the engine API to include a method
	 * where the read/written/updated buffers can be specified.
	 */
	virtual void
	process(
		const	BufferList& bufread,
				BufferList& bufwrite,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius) = 0;
};
#endif
