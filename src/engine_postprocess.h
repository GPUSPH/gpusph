/*  Copyright 2015 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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
 * Contains the abstract interface for the PostProcessEngines
 */

#ifndef _POSTPROCENGINE_H
#define _POSTPROCENGINE_H

#include "particledefine.h"
#include "buffer.h"
#include "command_flags.h"
#include "Writer.h"

// define the GlobalData struct so that we can pass pointers to the functions
struct GlobalData;

/*! Abstract class that defines the interface for the PostProcessEngine.
 * PostProcessEngines are run before writes to produce additional information
 * which is not typically needed during a simulation (e.g. vorticity, testpoint
 * values, surface detection etc).
 *
 * NOTE: surface detection might be needed for other purposes in the future!
 */
class AbstractPostProcessEngine
{
protected:
	/// Hold the options enabled for this specific engine
	flag_t m_options;
public:

	/// Construct an engine with the given options
	AbstractPostProcessEngine(flag_t options=NO_FLAGS) :
		m_options(options)
	{}

	// destructor must be virtual
	virtual ~AbstractPostProcessEngine() {}

	/// Get the options set when the engine was created
	flag_t const& get_options() const
	{ return m_options; }

	/// Set device constants
	virtual void
	setconstants(
		const	SimParams	*simparams,
		const	PhysParams	*physparams,
		idx_t	const&		allocatedParticles) const = 0;

	/// Get device constants
	virtual void getconstants() = 0 ; // TODO

	//< Main processing routine
	virtual void
	process(
		const BufferList&	bufread,
		BufferList&		bufwrite,
				uint					numParticles,
				uint					particleRangeEnd,
				uint					deviceIndex,
		const	GlobalData	* const		gdata) = 0;

	//< Returns a list of buffers updated (in the bufwrite list)
	virtual flag_t get_written_buffers() const = 0;
	//< Returns a list of buffers that were updated in-place
	//< in the bufread list
	virtual flag_t get_updated_buffers() const = 0;

	//< Allocation of memory on host
	virtual void
	hostAllocate(const GlobalData * const gdata) = 0;

	//< Main processing routine on host
	virtual void
	hostProcess(const GlobalData * const gdata) = 0;

	//< Main processing routine on host
	virtual void
	write(WriterMap writers, double t) = 0;
};
#endif
