/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

#ifndef _EULER_CUH_
#define _EULER_CUH_

#include "particledefine.h"

#define BLOCK_SIZE_INTEGRATE	256

extern "C"
{
void
seteulerconstants(const PhysParams & physparams);

void
geteulerconstants(PhysParams & physparams);

void
setmbdata(float4* MbData, uint size);

void
seteulerrbcg(float3* cg, int numbodies);

void
seteulerrbtrans(float3* trans, int numbodies);

void
seteulerrbsteprot(float* rot, int numbodies);

void
euler(	float4*		oldPos,
		float4*		oldVel,
		particleinfo* info,
		float4*		forces,
		float4*		xsph,
		float4*		newPos,
		float4*		newVel,
		uint		numParticles,
		float		dt,
		float		dt2,
		int			step,
		float		t,
		bool		xsphcorr,
		bool		periodicbound);
}
#endif
