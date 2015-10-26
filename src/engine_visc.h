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

#ifndef _VISCENGINE_H
#define _VISCENGINE_H

/* Abstract ViscEngine base class; it simply defines the interface
 * of the ViscEngine.
 * ViscEngines handle the pre-computation of viscosity before the forces.
 * (e.g. SPS, temperature- or rheology-dependent viscosity, etc)
 */

#include "particledefine.h"

// TODO as usual, the API needs to be redesigned properly
class AbstractViscEngine
{
public:
	virtual ~AbstractViscEngine() {}

	virtual void setconstants() = 0 ; // TODO
	virtual void getconstants() = 0 ; // TODO

	virtual void
	process(		float2	*tau[],
					float	*turbvisc,
			const	float4	*pos,
			const	float4	*vel,
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
