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

#ifndef _EULER_CUH_
#define _EULER_CUH_

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"

#define BLOCK_SIZE_INTEGRATE	256

extern "C"
{
void
seteulerconstants(const PhysParams *physparams,
	float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize);

void
geteulerconstants(PhysParams *physparams);

void
setmbdata(const float4* MbData, uint size);

void
seteulerrbcg(const float3* cg, int numbodies);

void
seteulerrbtrans(const float3* trans, int numbodies);

void
seteulerrbsteprot(const float* rot, int numbodies);

void
euler(	const float4*		oldPos,
		const hashKey*		particleHash,
		const float4*		oldVel,
		const float*		oldTKE,
		const float*		oldEps,
		const particleinfo* info,
		const float4*		forces,
		float2*				keps_dkde,
		const float4*		xsph,
		float4*				newPos,
		float4*				newVel,
		float*				newTKE,
		float*				newEps,
		const uint			numParticles,
		const uint			particleRangeEnd,
		const float			dt,
		const float			dt2,
		const int			step,
		const float			t,
		const bool			xsphcorr);
}
#endif
