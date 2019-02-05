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

/*! \file
 * Actual definition of the commands that GPUSPH can issue to workers
 */

/* This file must be inclued with a definition for the DEFINE_COMMAND macro */

#ifndef DEFINE_COMMAND
#error "Cannot define commands"
#endif

/* Macro to define a command with no buffer usage */
#define DEFINE_COMMAND_NOBUF(_command) DEFINE_COMMAND(_command, NO_BUFFER_USAGE, BUFFER_NONE, BUFFER_NONE, BUFFER_NONE)

/* Macro to define a command with dynamic buffer usage */
#define DEFINE_COMMAND_DYN(_command) DEFINE_COMMAND(_command, DYNAMIC_BUFFER_USAGE, BUFFER_NONE, BUFFER_NONE, BUFFER_NONE)

/* Macro to define a command with static buffer usage */
#define DEFINE_COMMAND_BUF(_command, _reads, _updates, _writes) DEFINE_COMMAND(_command, STATIC_BUFFER_USAGE, _reads, _updates, _writes)

/** \name Administrative commands
 *
 * These commands refer to “behind-the-scene” administration of the
 * buffers and manager/worker relationship.
 * @{
 */

/* Basic worker functionality */

/// Dummy cycle (do nothing)
DEFINE_COMMAND_NOBUF(IDLE)
/// Quit the simulation cycle
DEFINE_COMMAND_NOBUF(QUIT)


/* Buffer state management */

/// Initialize a new (invalid) ParticleSystem state
DEFINE_COMMAND_NOBUF(INIT_STATE)
/// Change the name of a ParticleSystem state
DEFINE_COMMAND_NOBUF(RENAME_STATE)
/// Release a ParticleSystem state
DEFINE_COMMAND_NOBUF(RELEASE_STATE)

/// Remove buffers from a the given ParticleSystem state
/*! Buffers are returned to the pool if they are not shared
 * with other states
 */
DEFINE_COMMAND_DYN(REMOVE_STATE_BUFFERS)
/// Swap buffers between two states, marking the new destination buffer as invalid
/*! Arguments: “source” state, “destination” state, buffer(s)
 * \note order of states is important, because the buffers moved to the
 * “destination” state are marked invalid
 */
DEFINE_COMMAND_DYN(SWAP_STATE_BUFFERS)
/// Move buffers from one state to the other, invalidating them
/*! Arguments: “source” state, “destination” state, buffer(s)
 * This is essentially a combination of REMOVE_STATE_BUFFERS from the source state,
 * followed by an ADD_STATE_BUFFERS to the second state.
 */
DEFINE_COMMAND_DYN(MOVE_STATE_BUFFERS)

/// Share buffers between states
/*! Arguments: “source” state, “destination” state, buffer(s)
 */
DEFINE_COMMAND_DYN(SHARE_BUFFERS)

/* Host-device data exchange */

/// Dump (device) particle data arrays into shared host arrays
DEFINE_COMMAND_DYN(DUMP)
/// Dump (device) cellStart and cellEnd into shared host arrays
DEFINE_COMMAND_NOBUF(DUMP_CELLS)
/// Dump device segments to shared host arrays, and update number of internal particles
DEFINE_COMMAND_NOBUF(UPDATE_SEGMENTS)


/* Multi-GPU data exchange */

/// Crop particle buffers, dropping all external particles
DEFINE_COMMAND_NOBUF(CROP)
/// Append a copy of the external cells to the end of self device arrays
DEFINE_COMMAND_DYN(APPEND_EXTERNAL)
///	Update the read-only copy of the external cells
DEFINE_COMMAND_DYN(UPDATE_EXTERNAL)


/* Open boundary (I/O) data exchange */

/// Download the number of particles on device (in case of inlets/outlets)
DEFINE_COMMAND_NOBUF(DOWNLOAD_NEWNUMPARTS)
/// Upload the number of particles to the device
DEFINE_COMMAND_NOBUF(UPLOAD_NEWNUMPARTS)
/// Download (partial) computed water depth from device to host
DEFINE_COMMAND_NOBUF(DOWNLOAD_IOWATERDEPTH)
/// Upload (total)computed water depth from host to device
DEFINE_COMMAND_NOBUF(UPLOAD_IOWATERDEPTH)


/* Physical data exchange and update */

/// Upload new value of gravity, after problem callback
DEFINE_COMMAND_NOBUF(UPLOAD_GRAVITY)
/// Upload planes to devices
DEFINE_COMMAND_NOBUF(UPLOAD_PLANES)


/* Moving body data exchange and update */

/// Compute total force acting on a moving body
DEFINE_COMMAND_NOBUF(REDUCE_BODIES_FORCES)
/// Upload centers of gravity of moving bodies for the integration engine
/// TODO FIXME there shouldn't be a need for separate EULER_ and FORCES_ version
/// of this, the moving body data should be put in its own namespace
DEFINE_COMMAND_NOBUF(EULER_UPLOAD_OBJECTS_CG)
/// Upload centers of gravity of moving bodies for forces computation
/// TODO FIXME there shouldn't be a need for separate EULER_ and FORCES_ version
/// of this, the moving body data should be put in its own namespace
DEFINE_COMMAND_NOBUF(FORCES_UPLOAD_OBJECTS_CG)
/// Upload translation vector and rotation matrices for moving bodies
DEFINE_COMMAND_NOBUF(UPLOAD_OBJECTS_MATRICES)
/// Upload linear and angular velocity of moving bodies
DEFINE_COMMAND_NOBUF(UPLOAD_OBJECTS_VELOCITIES)

/** @} */

/** \name Neighbors list management commands
 *
 * These are the commands that define the neighbors list construction
 *
 * @{
 */

/// Compute particle hashes
DEFINE_COMMAND_BUF(CALCHASH,
	BUFFER_INFO, // reads
	BUFFER_POS | BUFFER_HASH | BUFFER_PARTINDEX, // updates in-place
	BUFFER_NONE) // writes
/// Sort particles by hash
DEFINE_COMMAND_BUF(SORT,
	BUFFER_INFO | BUFFER_HASH, // reads
	BUFFER_PARTINDEX, // updates in-place
	BUFFER_NONE) // writes
/// Reorder particle data according to the latest SORT, and find the start of each cell
/*! \todo check buffer usage, there might be a discrepancy between IMPORT_BUFFERS and
 * the policy's get_multi_buffered()
 * \todo we should probably define an additional buffer trait declaring if that buffer
 * is a long-term particle property (i.e. follows the particle, evolves with differential
 * equations etc ) or ephemeral (i.e. computed specifically for their immediate
 * usage, e.g. per-particle viscosity)
 */
DEFINE_COMMAND_BUF(REORDER, IMPORT_BUFFERS, BUFFER_NONE, IMPORT_BUFFERS)
/// Build the neighbors list
DEFINE_COMMAND_BUF(BUILDNEIBS,
	BUFFER_POS | BUFFER_INFO | BUFFER_VERTICES | BUFFER_BOUNDELEMENTS | BUFFER_HASH,
	BUFFER_NONE,
	BUFFER_NEIBSLIST | BUFFER_VERTPOS)

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
DEFINE_COMMAND_BUF(FILTER,
	BUFFER_POS | BUFFER_VEL | BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST,
	BUFFER_NONE,
	BUFFER_VEL)

/* SA_BOUNDARY boundary conditions kernels */

/// SA_BOUNDARY only: compute segment boundary conditions and identify fluid particles
/// that leave open boundaries
/// TODO FIXME this also calls findOutgoingSegment on step 2 if open boundaries are enabled
DEFINE_COMMAND_BUF(SA_CALC_SEGMENT_BOUNDARY_CONDITIONS,
	BUFFER_POS | BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST | BUFFER_VERTPOS | BUFFER_BOUNDELEMENTS | BUFFER_VERTICES,
	BUFFER_VEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_EULERVEL | BUFFER_GRADGAMMA,
	BUFFER_NONE)
/// SA_BOUNDARY only: compute vertex boundary conditions, including mass update
/// and generation of new fluid particles at open boundaries.
/// During initialization, also compute a preliminary ∇γ direction vector
/// TODO FIXME this generates new particles if open boundaries are enabled,
/// and to achieve this it touches arrays that should be read-only.
/// Find a way to describe this.
DEFINE_COMMAND_BUF(SA_CALC_VERTEX_BOUNDARY_CONDITIONS,
	BUFFER_POS | BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST | BUFFER_VERTPOS | BUFFER_BOUNDELEMENTS | BUFFER_VERTICES | BUFFER_GRADGAMMA,
	BUFFER_FORCES | BUFFER_POS | BUFFER_GRADGAMMA | BUFFER_VERTICES | BUFFER_NEXTID | BUFFER_VEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_EULERVEL,
	BUFFER_NONE)
/// Compute the normal of a vertex in the initialization step
DEFINE_COMMAND_BUF(SA_COMPUTE_VERTEX_NORMAL,
	BUFFER_VERTICES | BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST,
	BUFFER_BOUNDELEMENTS,
	BUFFER_NONE)
/// Initialize gamma for dynamic gamma computation
DEFINE_COMMAND_BUF(SA_INIT_GAMMA,
	BUFFER_POS | BUFFER_BOUNDELEMENTS | BUFFER_INFO | BUFFER_HASH | BUFFER_HASH | BUFFER_NEIBSLIST | BUFFER_VERTPOS,
	BUFFER_GRADGAMMA,
	BUFFER_NONE)

/* Open boundary conditions kernels (currently SA-only) */

/// Count vertices that belong to the same IO and the same segment as an IO vertex
DEFINE_COMMAND_BUF(INIT_IO_MASS_VERTEX_COUNT,
	BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST | BUFFER_VERTICES,
	BUFFER_FORCES,
	BUFFER_NONE)
/// Modifiy initial mass of open boundaries
DEFINE_COMMAND_BUF(INIT_IO_MASS,
	BUFFER_POS | BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST | BUFFER_VERTICES | BUFFER_FORCES,
	BUFFER_NONE,
	BUFFER_POS)
/// Impose problem-specific velocity/pressure on open boundaries
/// (should update the WRITE buffer in-place)
DEFINE_COMMAND_BUF(IMPOSE_OPEN_BOUNDARY_CONDITION,
	BUFFER_POS | BUFFER_INFO | BUFFER_HASH,
	BUFFER_VEL | BUFFER_EULERVEL | BUFFER_TKE | BUFFER_EPSILON,
	BUFFER_NONE)

/// SA_BOUNDARY only: identify vertices at corner of open boundaries.
/// (Corner vertices do not generate new particles)
DEFINE_COMMAND_BUF(IDENTIFY_CORNER_VERTICES,
	BUFFER_POS | BUFFER_BOUNDELEMENTS | BUFFER_HASH | BUFFER_VERTICES | BUFFER_NEIBSLIST,
	BUFFER_INFO,
	BUFFER_NONE)
/// SA_BOUNDARY only: disable particles that went through an open boundary
DEFINE_COMMAND_BUF(DISABLE_OUTGOING_PARTS,
	BUFFER_INFO,
	BUFFER_POS | BUFFER_VERTICES,
	BUFFER_NONE)

/// SPH_GRENIER only: compute density
DEFINE_COMMAND_BUF(COMPUTE_DENSITY,
	BUFFER_POS | BUFFER_VOLUME | BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST,
	BUFFER_VEL | BUFFER_SIGMA,
	BUFFER_NONE)

/// Compute per-particle viscosity and SPS stress matrix
DEFINE_COMMAND_BUF(CALC_VISC,
	BUFFER_POS | BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST | BUFFER_VEL,
	BUFFER_NONE,
	BUFFER_EFFVISC)

/** @} */

/** \name Forces computation commands
 *
 * This is a single kernel, but is split in three forms to allow for asynchronous
 * computation.
 * \todo All computational kernels should offer the option to be handled this way,
 * possibly in an automatic way, there shouldn't be a need for three command types.
 */

#define FORCES_INPUT_BUFFERS \
	(BUFFER_POS | BUFFER_VEL | BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST | \
	 BUFFER_VERTPOS | BUFFER_GRADGAMMA | BUFFER_BOUNDELEMENTS | BUFFER_TURBVISC | \
	 BUFFER_VOLUME | BUFFER_SIGMA)

#define FORCES_UPDATE_BUFFERS \
		(BUFFER_FORCES | BUFFER_XSPH | BUFFER_TAU | BUFFER_DKDE | BUFFER_CFL | \
		 BUFFER_CFL_GAMMA | BUFFER_CFL_KEPS | BUFFER_CFL_TEMP | \
		 BUFFER_INTERNAL_ENERGY_UPD)

/// Compute forces, blocking; this runs the whole forces sequence (texture bind, kernele execution, texture
/// unbinding, dt reduction) and only proceeds on completion
DEFINE_COMMAND_BUF(FORCES_SYNC,
	FORCES_INPUT_BUFFERS,
	FORCES_UPDATE_BUFFERS,
	BUFFER_NONE)
/// Compute forces, asynchronously: bind textures, launch kernel and return without waiting for kernel completion
DEFINE_COMMAND_BUF(FORCES_ENQUEUE,
	FORCES_INPUT_BUFFERS,
	FORCES_UPDATE_BUFFERS,
	BUFFER_NONE)
/// Wait for completion of the forces kernel unbind texture, reduce dt
DEFINE_COMMAND_BUF(FORCES_COMPLETE,
	FORCES_INPUT_BUFFERS,
	FORCES_UPDATE_BUFFERS,
	BUFFER_NONE)

/** @} */

/** \name Integration and finalization commands
 *
 * These are commands pertaining to the update of the particle state after
 * forces computation, taking into account both direct integration and
 * additional post-processing e.g. as needed by density diffusion.
 */

#define EULER_INPUT_BUFFERS \
		(BUFFER_POS | BUFFER_HASH | BUFFER_VOLUME | BUFFER_VEL | BUFFER_INTERNAL_ENERGY | BUFFER_EULERVEL | \
		 BUFFER_BOUNDELEMENTS /* only if has moving */ | \
		 BUFFER_TKE | BUFFER_EPSILON | BUFFER_INFO | BUFFER_NEIBSLIST /* why? */ | BUFFER_VERTPOS | \
		 BUFFER_FORCES | BUFFER_INTERNAL_ENERGY_UPD | BUFFER_DKDE | BUFFER_XSPH)

#define EULER_OUTPUT_BUFFERS \
		(BUFFER_POS | BUFFER_VEL | BUFFER_VOLUME | BUFFER_INTERNAL_ENERGY | BUFFER_EULERVEL | BUFFER_TKE | \
		 BUFFER_BOUNDELEMENTS /* only if has moving */ | \
		 BUFFER_EPSILON | BUFFER_TURBVISC)

/// Integration (runs the Euler kernel)
DEFINE_COMMAND_BUF(EULER,
	EULER_INPUT_BUFFERS,
	BUFFER_NONE,
	EULER_OUTPUT_BUFFERS)

#define GAMMMA_AND_DENSITY_SUM_INPUT_BUFFERS \
	(BUFFER_POS /* old and new! */ | \
	 BUFFER_VEL /* old and new! */ | \
	 BUFFER_EULERVEL /* old and new! */ | \
	 BUFFER_BOUNDELEMENTS /* old and, if moving bodies, new */ | \
	 BUFFER_VERTPOS | BUFFER_HASH | BUFFER_INFO | BUFFER_VEL | BUFFER_GRADGAMMA)

/// Integration of SABOUNDARY's gamma
DEFINE_COMMAND_BUF(INTEGRATE_GAMMA,
	GAMMMA_AND_DENSITY_SUM_INPUT_BUFFERS | BUFFER_NEIBSLIST,
	BUFFER_GRADGAMMA,
	BUFFER_NONE)

/// Integration of the density using an integral formulation
DEFINE_COMMAND_BUF(DENSITY_SUM,
	GAMMMA_AND_DENSITY_SUM_INPUT_BUFFERS,
	BUFFER_VEL | BUFFER_GRADGAMMA,
	BUFFER_NONE)

/// Compute the density diffusion term in the case of density sum:
DEFINE_COMMAND_BUF(CALC_DENSITY_DIFFUSION,
	BUFFER_BOUNDELEMENTS | BUFFER_POS | BUFFER_VEL | BUFFER_INFO | BUFFER_HASH | BUFFER_NEIBSLIST | BUFFER_GRADGAMMA | BUFFER_VERTPOS,
	BUFFER_FORCES,
	BUFFER_NONE)
/// Apply density diffusion term in the case of density sum:
DEFINE_COMMAND_BUF(APPLY_DENSITY_DIFFUSION,
	BUFFER_INFO | BUFFER_FORCES,
	BUFFER_VEL,
	BUFFER_NONE)

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
/*! Special case of dynamic buffer: the buffer specification is directed
 * by each specific post-processing engine
 */
DEFINE_COMMAND_DYN(POSTPROCESS)

/** @} */

DEFINE_COMMAND_NOBUF(NUM_COMMANDS)
