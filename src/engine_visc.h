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

/*! \file
 * Contains the abstract interface for the ViscEngine
 */

#ifndef _VISCENGINE_H
#define _VISCENGINE_H

#include "particledefine.h"
#include "buffer.h"

/*! Abstract class that defines the interface for the ViscEngine
 * ViscEngines handle the pre-computation of viscosity before the forces.
 * (e.g. SPS, temperature- or rheology-dependent viscosity, etc)
 */
class AbstractViscEngine
{
public:
	virtual ~AbstractViscEngine() {}

	/// Set device constants
	virtual void setconstants() = 0 ; // TODO
	/// Get device constants
	virtual void getconstants() = 0 ; // TODO

	/// Run the viscosity computation step
	/*! This method runs the necessary computations needed for
	 * viscous computation, such as the stress tensor correction for SPS,
	 * or the viscosity itself in case of per-particle viscosity.
	 * (\see{BUFFER_EFFVISC}).
	 * In the non-Newtonian case, it also returns the maximum kinematic
	 * viscosity.
	 */
	virtual float
	calc_visc(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius) = 0;

	virtual void
	enforce_jacobi_fs_boundary_conditions(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius) = 0;

	virtual float
	enforce_jacobi_wall_boundary_conditions(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius) = 0;

	virtual void
	build_jacobi_vectors(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius) = 0;

	virtual float
	update_jacobi_effpres(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius) = 0;
};
#endif
