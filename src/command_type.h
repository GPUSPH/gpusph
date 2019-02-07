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

#ifndef COMMAND_TYPE_H
#define COMMAND_TYPE_H

#include <string>
#include <vector>

#include "command_flags.h"
#include "common_types.h"
#include "define_buffers.h"

//! Next step for workers.
/*! The commands are grouped by category, depending on whether they reflect
 * actual parts of the integrator from those that have a purely “administrative”
 * scope (buffer management etc).
 */
enum CommandName
{
#define DEFINE_COMMAND(code, ...) code,
#include "define_commands.h"
#undef DEFINE_COMMAND
};

//! Array of command names
/*! Maps a CommandName to its string representation.
 * The actual array is defined in src/command_type.cc
 */
extern const char* command_name[];

//! Map CommandName to its C string representation
/*! With proper fencing for undefined commands
 */
inline const char * getCommandName(CommandName cmd)
{
	if (cmd < NUM_COMMANDS)
		return command_name[cmd];
	return "<undefined command>";
}

/*
 * Structures needed to specify command arguments
 */

//! A struct specifying buffers within a state
struct StateBuffers
{
	std::string state;
	flag_t buffers;

	StateBuffers(std::string const& state_, flag_t buffers_) :
		state(state_),
		buffers(buffers_)
	{}
	StateBuffers(std::string const& state_) :
		state(state_),
		buffers(BUFFER_NONE)
	{}
	StateBuffers(flag_t buffers_) :
		state(),
		buffers(buffers_)
	{}
};

//! A command buffer usage specification
/*! This is used to specify which buffers, from which states,
 * the command will read, update or write
 */
typedef std::vector<StateBuffers> CommandBufferArgument;

//! Specification of buffer usage by commands
enum CommandBufferUsage
{
	NO_BUFFER_USAGE, ///< command does not touch any buffer
	STATIC_BUFFER_USAGE, ///< command works on a fixed set of buffers
	DYNAMIC_BUFFER_USAGE ///< command needs a parameter specifying the buffers to operate on
};

//! A full command structure
/*! The distinction between updates and writes specification
 * is that in the updates case the buffer(s) will also be read,
 * and must therefore already be present in the corresponding states,
 * whereas writes are considered ignoring previous content, and
 * can therefore be missing/invalid in the state.
 *
 * If the command applies to a single state, src should be set.
 */
struct CommandStruct
{
	CommandName command; ///< the command
	std::string src; ///< source state (if applicable)
	std::string dst; ///< destination state (if applicable)
	flag_t flags; ///< command flag (e.g. integration step, shared flags, etc)
	CommandBufferArgument reads;
	CommandBufferArgument updates;
	CommandBufferArgument writes;

	CommandStruct(CommandName cmd) :
		command(cmd),
		src(),
		dst(),
		flags(NO_FLAGS),
		reads(),
		updates(),
		writes()
	{}

	// setters
	CommandStruct& set_src(std::string const& src_)
	{ src = src_; return *this; }
	CommandStruct& set_dst(std::string const& dst_)
	{ dst = dst_; return *this; }
	CommandStruct& set_flags(flag_t f)
	{ flags |= f; return *this; }
	CommandStruct& clear_flags(flag_t f)
	{ flags &= ~f; return *this; }
	CommandStruct& reading(StateBuffers const& buf)
	{ reads.push_back(buf); return *this; }
	CommandStruct& updating(StateBuffers const& buf)
	{ updates.push_back(buf); return *this; }
	CommandStruct& writing(StateBuffers const& buf)
	{ writes.push_back(buf); return *this; }
};

inline const char * getCommandName(CommandStruct const& cmd)
{ return getCommandName(cmd.command); }

/*
 * Command traits
 */

// TODO only_internal should be a command trait
template<CommandName T>
struct CommandTraits
{
	static constexpr CommandName command = T;
	static constexpr CommandBufferUsage buffer_usage = NO_BUFFER_USAGE;
};


/* Generic macro for the definition of a command traits structure */
/* TODO reads, updates and writes specifications will be moved to the integrator */
#define DEFINE_COMMAND(_command, _usage, _reads, _updates, _writes) \
template<> \
struct CommandTraits<_command> \
{ \
	static constexpr CommandName command = _command; \
	static constexpr CommandBufferUsage buffer_usage = _usage; \
};

#include "define_commands.h"

#undef DEFINE_COMMAND

#endif
