/*  Copyright 2019 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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
 * Actual definition of the commands that GPUSPH can issue to itself
 */

/* This file must be inclued with a definition for the DEFINE_COMMAND macro */

#ifndef DEFINE_COMMAND
#error "Cannot define commands"
#endif

/* Macro to define a command with no buffer usage */
#define DEFINE_COMMAND_NOBUF(_command) DEFINE_COMMAND(_command, false, NO_BUFFER_USAGE)

/** \name Host commands
 *
 * These are commands that map to methods within GPUSPH itself
 * @{
 */

/// Update array particle offsets
/*! Computes the offsets within each particle system array
 * assigned to each device
 */
DEFINE_COMMAND_NOBUF(UPDATE_ARRAY_INDICES)

/// Run problem callbacks
/*! E.g. set variable gravity
 */
DEFINE_COMMAND_NOBUF(RUN_CALLBACKS)

/// End initialization sequence, ready to enter main cycle
DEFINE_COMMAND_NOBUF(END_OF_INIT)

/// End repacking sequence, ready to enter main cycle
DEFINE_COMMAND_NOBUF(END_OF_REPACKING)

/// Begin an integrator time-step
DEFINE_COMMAND_NOBUF(TIME_STEP_PRELUDE)

/// End an integrator time-step
DEFINE_COMMAND_NOBUF(TIME_STEP_EPILOGUE)

/// Complete the computation of the total force acting on moving bodies
/*! Additional reduction steps needed in the multi-GPU and multi-host cases
 */
DEFINE_COMMAND_NOBUF(REDUCE_BODIES_FORCES_HOST)

/// Problem-specific body forces callback
DEFINE_COMMAND_NOBUF(BODY_FORCES_CALLBACK)

/// Move bodies
/*! Determine new position and velocities for moving bodies
 * (eiter as prescribed by the problem, or by interaction with
 * the Chrono library).
 */
DEFINE_COMMAND_NOBUF(MOVE_BODIES)

/// Find maximum water depth
/*! Find the maximum across all devices of the water depth for each open boundary
 */
DEFINE_COMMAND_NOBUF(FIND_MAX_IOWATERDEPTH)

#if 0 // TODO
/// Compute the total kinetic energy for repacking
DEFINE_COMMAND_NOBUF(COMPUTE_KINETIC_ENERGY)
#endif

/// Check maximum number of neighbors and estimate number of interactions
DEFINE_COMMAND_NOBUF(CHECK_NEIBSNUM)

/// Hnadle pending hotwrites
DEFINE_COMMAND_NOBUF(HANDLE_HOTWRITE)

/// Determine if particles were created in this iteration
DEFINE_COMMAND_NOBUF(CHECK_NEWNUMPARTS)

/// Dump the particle system state for debugging
DEFINE_COMMAND_NOBUF(DEBUG_DUMP)

/// Stop criterion of the Jacobi solver used to compute effective pressure (granular rheology)
/*! Reduce Jacobi backward error and residual, and determine
whether the solver should stop based one simulation paramters
jacobi_maxiter, jacobi_backerr and jacobi_residual.
 */
DEFINE_COMMAND_BUF(JACOBI_STOP_CRITERION, true)
/// Reset the Jacobi solver stop criterion before entering the solver loop.
DEFINE_COMMAND_BUF(JACOBI_RESET_STOP_CRITERION, true)

/// Not an actual command ;-)
DEFINE_COMMAND_NOBUF(NUM_COMMANDS)

/** @} */
