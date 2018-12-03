/*  Copyright 2018 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/*! \file command_type.h
 * Commands that GPUSPH can issue to workers via doCommand() calls
 */

//! Next step for workers.
/*! The commands are grouped by category, depending on whether they reflect
 * actual parts of the integrator from those that have a purely “administrative”
 * scope (buffer management etc).
 */
enum CommandType {
	/** \name Administrative commands
	 *
	 * These commands refer to “behind-the-scene” administration of the
	 * buffers and manager/worker relationship.
	 * @{
	 */


	/* Basic worker functionality */

	/// Dummy cycle (do nothing)
	IDLE,
	/// Quit the simulation cycle
	QUIT,


	/* Buffer state management */

	/// Set the state of the given buffers to the given string
	SET_BUFFER_STATE,
	/// Add the given state string to the state of the given buffers
	ADD_BUFFER_STATE,
	/// Set the validity of the given buffers
	SET_BUFFER_VALIDITY,

	/// Swap double-buffered buffers
	SWAP_BUFFERS,


	/* Host-device data exchange */

	/// Dump (device) particle data arrays into shared host arrays
	DUMP,
	/// Dump (device) cellStart and cellEnd into shared host arrays
	DUMP_CELLS,
	/// Dump device segments to shared host arrays, and update number of internal particles
	UPDATE_SEGMENTS,


	/* Multi-GPU data exchange */

	/// Crop particle buffers, dropping all external particles
	CROP,
	/// Append a copy of the external cells to the end of self device arrays
	APPEND_EXTERNAL,
	///	Update the read-only copy of the external cells
	UPDATE_EXTERNAL,


	/* Open boundary (I/O) data exchange */

	/// Download the number of particles on device (in case of inlets/outlets)
	DOWNLOAD_NEWNUMPARTS,
	/// Upload the number of particles to the device
	UPLOAD_NEWNUMPARTS,
	/// Download (partial) computed water depth from device to host
	DOWNLOAD_IOWATERDEPTH,
	/// Upload (total)computed water depth from host to device
	UPLOAD_IOWATERDEPTH,


	/* Physical data exchange and update */

	/// Upload new value of gravity, after problem callback
	UPLOAD_GRAVITY,
	/// Upload planes to devices
	UPLOAD_PLANES,


	/* Moving body data exchange and update */

	/// Compute total force acting on a moving body
	REDUCE_BODIES_FORCES,
	/// Upload centers of gravity of moving bodies for the integration engine
	/// TODO FIXME there shouldn't be a need for separate EULER_ and FORCES_ version
	/// of this, the moving body data should be put in its own namespace
	EULER_UPLOAD_OBJECTS_CG,
	/// Upload centers of gravity of moving bodies for forces computation
	/// TODO FIXME there shouldn't be a need for separate EULER_ and FORCES_ version
	/// of this, the moving body data should be put in its own namespace
	FORCES_UPLOAD_OBJECTS_CG,
	/// Upload translation vector and rotation matrices for moving bodies
	UPLOAD_OBJECTS_MATRICES,
	/// Upload linear and angular velocity of moving bodies
	UPLOAD_OBJECTS_VELOCITIES,

	/** @} */

	/** \name Neighbors list management commands
	 *
	 * These are the commands that define the neighbors list construction
	 *
	 * @{
	 */

	/// Compute particle hashes
	CALCHASH,
	/// Sort particles by hash
	SORT,
	/// Reorder particle data according to the latest SORT, and find the start of each cell
	REORDER,
	/// Build the neighbors list
	BUILDNEIBS,

	/** @} */

	/** \name Pre-force commands
	 *
	 * These includes things such as the imposition of boundary conditions, or
	 * the computation of additional formulation-specific information
	 * (e.g. non-homogeneous viscosity, open boundary conditions etc)
	 *
	 * @{
	 */

	/// Run smoothing filters (e.g. Shepard, MLS)
	FILTER,

	/* SA_BOUNDARY boundary conditions kernels */

	/// SA_BOUNDARY only: compute segment boundary conditions and identify fluid particles
	/// that leave open boundaries
	SA_CALC_SEGMENT_BOUNDARY_CONDITIONS,
	/// SA_BOUNDARY only: compute vertex boundary conditions, including mass update
	/// and generation of new fluid particles at open boundaries.
	/// During initialization, also compute a preliminary ∇γ direction vector
	SA_CALC_VERTEX_BOUNDARY_CONDITIONS,
	/// Compute the normal of a vertex in the initialization step
	SA_COMPUTE_VERTEX_NORMAL,
	/// Initialize gamma for dynamic gamma computation
	SA_INIT_GAMMA,

	/* Open boundary conditions kernels (currently SA-only) */

	/// Count vertices that belong to the same IO and the same segment as an IO vertex
	INIT_IO_MASS_VERTEX_COUNT,
	/// Modifiy initial mass of open boundaries
	INIT_IO_MASS,
	/// Impose problem-specific velocity/pressure on open boundaries
	/// (should update the WRITE buffer in-place)
	IMPOSE_OPEN_BOUNDARY_CONDITION,

	/// SA_BOUNDARY only: identify vertices at corner of open boundaries.
	/// Corner vertices do not generate new particles,
	IDENTIFY_CORNER_VERTICES,
	/// SA_BOUNDARY only: disable particles that went through an open boundary
	DISABLE_OUTGOING_PARTS,

	/// SPH_GRENIER only: compute density
	COMPUTE_DENSITY,

	/// Compute per-particle viscosity and SPS stress matrix
	CALC_VISC,

	/** @} */

	/** \name Forces computation commands
	 *
	 * This is a single kernel, but is split in three forms to allow for asynchronous
	 * computation.
	 * \todo All computational kernels should offer the option to be handled this way,
	 * possibly in an automatic way, there shouldn't be a need for three command types.
	 */

	/// Compute forces, blocking; this runs the whole forces sequence (texture bind, kernele execution, texture
	/// unbinding, dt reduction) and only proceeds on completion
	FORCES_SYNC,
	/// Compute forces, asynchronously: bind textures, launch kernel and return without waiting for kernel completion
	FORCES_ENQUEUE,
	/// Wait for completion of the forces kernel unbind texture, reduce dt
	FORCES_COMPLETE,

	/** @} */

	/** \name Integration and finalization commands
	 *
	 * These are commands pertaining to the update of the particle state after
	 * forces computation, taking into account both direct integration and
	 * additional post-processing e.g. as needed by density diffusion.
	 */

	/// Integration (runs the Euler kernel)
	EULER,
	/// Integration of SABOUNDARY's gamma
	INTEGRATE_GAMMA,
	/// Integration of the density using an integral formulation
	DENSITY_SUM,
	/// Compute the density diffusion term in the case of density sum:
	CALC_DENSITY_DIFFUSION,
	/// Apply density diffusion term in the case of density sum:
	APPLY_DENSITY_DIFFUSION,

	/** @} */

	/** \name Additional computational kernels
	 *
	 * This should include any computational kernel that does not (or at least should not)
	 * impact the simulation results, and currently is limited to the post-processing
	 * kernel invokation
	 *
	 * @{
	 */

	/// Run post-processing filters (e.g. vorticity, testpoints)
	POSTPROCESS,

	/** @} */

};


