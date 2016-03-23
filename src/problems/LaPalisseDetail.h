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

#ifndef _LAPALISSEDETAIL_H
#define	_LAPALISSEDETAIL_H

#include "XProblem.h"

// Water level simulated by the pressure inlet
#define INLET_WATER_LEVEL	0.85575f
// Water level after pre-processing fill
#define INITIAL_WATER_LEVEL	0.75f
// Time [s] over which the water should rise from INITIAL to
// INLET_WATER_LEVEL
#define RISE_TIME	6.0f

#define VELOCITY_DRIVEN true
#define PRESSURE_DRIVEN false

class LaPalisseDetail: public XProblem {
	private:
		double			w, l, h;
		double			H;				// water level (used to set D constant)
	public:
		LaPalisseDetail(GlobalData *);
		//virtual ~LaPalisseDetail(void);

		uint max_parts(uint);
		void initializeParticles(BufferList &buffers, const uint numParticles);
		void init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash);

		void
		imposeBoundaryConditionHost(
			MultiBufferList::iterator		bufwrite,
			MultiBufferList::const_iterator	bufread,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numOpenBoundaries,
			const	uint			particleRangeEnd);

		// override standard split
		void fillDeviceMap();
};
#endif	/* _LAPALISSEDETAIL_H */

