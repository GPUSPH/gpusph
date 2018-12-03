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
 * Commands that GPUSPH can issue to workers via doCommand() calls
 */

#include "common_types.h"
#include "define_buffers.h"

//! Next step for workers.
/*! The commands are grouped by category, depending on whether they reflect
 * actual parts of the integrator from those that have a purely “administrative”
 * scope (buffer management etc).
 */

enum CommandType {
#define DEFINE_COMMAND(code, ...) code,
#include "define_commands.h"
#undef DEFINE_COMMAND
};

//! Specification of buffer usage by commands
enum CommandBufferUsage
{
	NO_BUFFER_USAGE, ///< command does not touch any buffer
	STATIC_BUFFER_USAGE, ///< command works on a fixed set of buffers
	DYNAMIC_BUFFER_USAGE ///< command needs a parameter specifying the buffers to operate on
};

template<CommandType T>
struct CommandTraits
{
	static constexpr CommandType command = T;
	static constexpr CommandBufferUsage buffer_usage = NO_BUFFER_USAGE;
	static constexpr flag_t reads = BUFFER_NONE; ///< list of buffers read by this command
	static constexpr flag_t updates = BUFFER_NONE; ///< list of buffers updated in-place by this command
	static constexpr flag_t writes = BUFFER_NONE; ///< list of buffers written by this command

	static const char name[]; ///< printable name of the buffer
};


/* Generic macro for the definition of a command traits structure */
#define DEFINE_COMMAND(_command, _usage, _reads, _updates, _writes) \
template<> \
struct CommandTraits<_command> \
{ \
	static constexpr CommandType command = _command; \
	static constexpr CommandBufferUsage buffer_usage = _usage; \
	static constexpr flag_t reads = _reads; \
	static constexpr flag_t updates = _updates; \
	static constexpr flag_t writes = _writes; \
 \
	static const char name[]; \
};

#include "define_commands.h"

#undef DEFINE_COMMAND

