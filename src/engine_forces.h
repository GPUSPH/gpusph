/*  Copyright (c) 2014-2019 INGV, EDF, UniCT, JHU

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

#ifndef _FORCESENGINE_H
#define _FORCESENGINE_H

/*! \file
 * Contains the abstract interface for the ForcesEngine
 */

#include "particledefine.h"
#include "planes.h"
#include "physparams.h"
#include "simparams.h"
#include "buffer.h"

/*! Abstract class that defines the interface for the ForcesEngine.
 */
class AbstractForcesEngine
{
public:
	virtual ~AbstractForcesEngine() {};

	/// Set the device constants
	virtual void
	setconstants(const SimParams *simparams, const PhysParams *physparams,
		float3 const& worldOrigin, uint3 const& gridSize, float3 const& cellSize,
		idx_t const& allocatedParticles) = 0;

	/// Get the device constants
	virtual void
	getconstants(PhysParams *physparams) = 0;

	/// Upload planes to the device
	virtual void
	setplanes(PlaneList const& planes) = 0;

	/// Update the value of gravity
	virtual void
	setgravity(float3 const& gravity) = 0;

	/// \defgroup ForcesEngineRB Rigid Body methods for the ForcesEngine
	/// @{

	/// Set the center of mass of all rigid bodies
	virtual void
	setrbcg(const int3* cgGridPos, const float3* cgPos, int numbodies) = 0;

	/// Initialize the rigid body variables
	virtual void
	setrbstart(const int* rbfirstindex, int numbodies) = 0;

	/// Initialize the fea variables
	virtual void
	setfeastart(const int2* feanodefirstindex, const int2 *feapartsfirstindex,
		int numfeabodies) = 0;
	virtual void
	setfeanatcoords(const float4* feanatcoords, const uint4* nodes, int numfeaparts) = 0;

	/// Compute total force and torque acting on each rigid body
	virtual void
	reduceRbForces(	BufferList& bufwrite,
					uint	*lastindex,
					float3	*totalforce,
					float3	*totaltorque,
					uint	numbodies,
					uint	numBodiesParticles) = 0;
	/// @}


	// TODO texture bind/unbind and dtreduce for forces should be handled in a different way:
	// they are sets of operations to be done before/after the forces kernel, which are
	// called separately because the kernel might be split into inner/outer calls, and
	// the pre-/post- calls have to do before the first and after the last

	/// Bind textures needed in the forces kernel execution
	virtual void
	bind_textures(const BufferList& bufread, uint numParticles,
		RunMode run_mode) = 0;

	/// Unbind the textures after the forces kernel execution
	virtual void
	unbind_textures(RunMode run_mode) = 0;

	/// Set the DEM
	/// TODO set/unsetDEM should be moved to the BC engine,
	/// and the latter should be called by the destructor
	virtual void
	setDEM(const float *hDem, int width, int height) = 0;

	/// Free DEM-related resources
	virtual void
	unsetDEM() = 0;

	/// Striping support: round a number of particles down to the largest multiple
	/// of the block size that is not greater than it
	virtual uint
	round_particles(uint numparts) = 0;

	/// Compute particle density
	virtual void
	compute_density(const BufferList& bufread,
		BufferList& bufwrite,
		uint numParticles,
		float slength,
		float influenceradius) = 0;

	/// Compute density diffusion term
	virtual void
	compute_density_diffusion(
		const BufferList& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceRadius,
		const	float	dt) = 0;

	/// Basic forces step.
	/// \return the number of blocks launched (which is the number of blocks to
	/// launch dtreduce on)
	virtual uint
	basicstep(
		const BufferList& bufread,
		BufferList& bufwrite,
				uint	numParticles,
				uint	fromParticle,
				uint	toParticle,
				float	deltap,
				float	slength,
				float	dtadaptfactor,
				float	influenceradius,
		const	float	epsilon,
				uint	*IOwaterdepth,
				uint	cflOffset,
		const	RunMode	run_mode,
		const	int		step,
		const	float	dt,
		const	bool	compute_object_forces) = 0;

	// Reduction methods

	/// Get number of elements needed for fmax
	virtual uint
	getFmaxElements(const uint n) = 0;

	/// Get the number of elements needed by intermediate reduction steps
	virtual uint
	getFmaxTempElements(const uint n) = 0;

	/// Find the minimum allowed time-step
	virtual float
	dtreduce(	float	slength,
				float	dtadaptfactor,
				float	sspeed_cfl,
				float	max_kinematic,
				BufferList const& bufread,
				BufferList& bufwrite,
				uint	numBlocks,
				uint	numParticles) = 0;

};
#endif
