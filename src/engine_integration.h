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
 * Contains the abstract interface for the IntegrationEngine
 */

#ifndef _INTEGRATIONENGINE_H
#define _INTEGRATIONENGINE_H

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "buffer.h"

/*! Abstract class that defines the interface for the IntegrationEngine
 */
class AbstractIntegrationEngine
{
public:
	virtual ~AbstractIntegrationEngine() {}

	/// Set the device constants
	virtual void
	setconstants(const PhysParams *physparams, float3 const& worldOrigin,
		uint3 const& gridSize, float3 const& cellSize, idx_t const& allocatedParticles,
		int const& neiblistsize, float const& slength) = 0;

	/// Get the device constants
	virtual void
	getconstants(PhysParams *physparams) = 0;

	/// \defgroup IntegrationEngineRB Rigid Body methods for the IntegrationEngine
	/// @{

	/// Set the center of mass of all rigid bodies
	virtual void
	setrbcg(const int3* cgGridPos, const float3* cgPos, int numbodies) = 0;

	/// Set the translation of all rigid bodies
	virtual void
	setrbtrans(const float3* trans, int numbodies) = 0;

	/// Set the rotation of all rigid bodies
	virtual void
	setrbsteprot(const float* rot, int numbodies) = 0;

	/// Set the linear velocity of all rigid bodies
	virtual void
	setrblinearvel(const float3* linearvel, int numbodies) = 0;

	/// Set the angular velocity of all rigid bodies
	virtual void
	setrbangularvel(const float3* angularvel, int numbodies) = 0;
	/// @}

	/// Integral formulation of continuity equation
	/*! Integrates density and gamma */
	virtual void
	density_sum(
		const BufferList& bufread,	// this is the read only arrays
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	epsilon,
		const	float	slength,
		const	float	influenceRadius) = 0;

	/// Gamma integration from particle distribution
	virtual void
	integrate_gamma(
		const BufferList& bufread,	// particle system at state n
		BufferList& bufreadUpdate,	// particle system at state n+1
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	epsilon,
		const	float	slength,
		const	float	influenceRadius,
		const	RunMode	run_mode) = 0;

	/// Apply density diffusion
	virtual void
	apply_density_diffusion(
		const BufferList& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt) = 0;

	/// Single integration step
	/// TODO will probably need to be made more generic for other
	/// integration schemes
	virtual void
	basicstep(
		const BufferList& bufread,	// this is the read only arrays
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	dt,
		const	int		step,
		const	float	t,
		const	float	slength,
		const	float	influenceRadius,
		const	RunMode	run_mode) = 0;

	// Disable free surface boundary particles after the repacking is achieved
	virtual void
		disableFreeSurfParts(		float4*			pos,
				const	particleinfo*	info,
				const	uint			numParticles,
				const	uint			particleRangeEnd) = 0;

};
#endif
