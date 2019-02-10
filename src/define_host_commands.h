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
#define DEFINE_COMMAND_NOBUF(_command) DEFINE_COMMAND(_command, false, NO_BUFFER_USAGE, BUFFER_NONE, BUFFER_NONE, BUFFER_NONE)

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

/// Check maximum number of neighbors and estimate number of interactions
DEFINE_COMMAND_NOBUF(CHECK_NEIBSNUM)

/// Determine if particles were created in this iteration
DEFINE_COMMAND_NOBUF(CHECK_NEWNUMPARTS)

/// Not an actual command ;-)
DEFINE_COMMAND_NOBUF(NUM_COMMANDS)

/** @} */
