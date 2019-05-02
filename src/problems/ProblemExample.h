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

#ifndef _PROBLEMEXAMPLE_H
#define	_PROBLEMEXAMPLE_H

#define PROBLEM_API 1
#include "Problem.h"

class ProblemExample: public Problem {
	public:
		ProblemExample(GlobalData *);

		// Define this override if you want to be able to compute
		// specific per-particle values that are not defined
		// otherwise
		void calcPrivate(flag_t options,
			BufferList const& bufread,
			BufferList & bufwrite,
			uint numParticles,
			uint particleRangeEnd,
			uint deviceIndex,
			const GlobalData * const gdata) override;

		// Define this override to provide a meaningful name for
		// the BUFFER_PRIVATE data
		std::string get_private_name(flag_t buffer) const override;
};
#endif	/* _PROBLEMEXAMPLE_H */

