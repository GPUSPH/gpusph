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

#ifndef _COMPLETESAEXAMPLE_H
#define	_COMPLETESAEXAMPLE_H

#define PROBLEM_API 1
#include "Problem.h"

// Set to 1 (or true) for velocity driven inlet, 0 (or false) for pressure driven
#define	VELOCITY_DRIVEN			1

// Water level simulated by the pressure inlet
#define INLET_WATER_LEVEL	0.9

// Velocity (m/s) and fading-in time (s) for velocity driven inlet
// Set fading time to 0 to impose immediately INLET_VELOCITY
#define INLET_VELOCITY			4.0
#define INLET_VELOCITY_FADE		1.0

class CompleteSaExample: public Problem {
	private:
	public:
		CompleteSaExample(GlobalData *);
		//virtual ~CompleteSaExample(void);

		uint max_parts(uint);

		void
		imposeBoundaryConditionHost(
			BufferList&		bufwrite,
			BufferList const&	bufread,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numOpenBoundaries,
			const	uint			particleRangeEnd);

		// override standard split
		void fillDeviceMap();
};
#endif	/* _COMPLETESAEXAMPLE_H */

