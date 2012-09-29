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

#include <stdio.h>

#include "particledefine.h"

#include "euler.cuh"
#include "euler_kernel.cu"

extern "C"
{
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
		bool		periodicbound)
{
	// thread per particle
	int numThreads = min(BLOCK_SIZE_INTEGRATE, numParticles);
	int numBlocks = (int) ceil(numParticles / (float) numThreads);

	// execute the kernel
	if (step == 1) {
		if (xsphcorr) {
			if (periodicbound)
				cueuler::eulerXsphDevice<1, true><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt2, dt2, t);
			else
				cueuler::eulerXsphDevice<1, false><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt2, dt2, t);
		} else {
			if (periodicbound)
				cueuler::eulerDevice<1, true><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces,
									newPos, newVel,
									numParticles, dt2, dt2, t);
			else
				cueuler::eulerDevice<1, false><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces,
									newPos, newVel,
									numParticles, dt2, dt2, t);
		}
	} else if (step == 2) {
		if (xsphcorr) {
			if (periodicbound)
				cueuler::eulerXsphDevice<2, true><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt, dt2, t);
			else
				cueuler::eulerXsphDevice<2, false><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces, xsph,
									newPos, newVel,
									numParticles, dt, dt2, t);
		} else {
			if (periodicbound)
				cueuler::eulerDevice<2, true><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces,
									newPos, newVel,
									numParticles, dt, dt2, t);
			else
				cueuler::eulerDevice<2, false><<< numBlocks, numThreads >>>(oldPos, oldVel, info,
									forces,
									newPos, newVel,
									numParticles, dt, dt2, t);
		}
	} // if (step == 2)

	// check if kernel invocation generated an error
	CUT_CHECK_ERROR("Euler kernel execution failed");
}
}
