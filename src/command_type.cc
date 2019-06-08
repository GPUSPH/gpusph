/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

/*! \file
 * This file is only used to hold the actual strings
 * defining the command printable names.
 */

#include "command_type.h"
#include "ParticleSystem.h"

const char* command_name[] = {
#define DEFINE_COMMAND(_command, ...) #_command,
#include "define_commands.h"
#undef DEFINE_COMMAND
};

const BufferList extractExistingBufferList(
	ParticleSystem const& ps,
	std::string const& state, const flag_t buffers)
{
	if (buffers == BUFFER_NONE)
		throw std::invalid_argument("no buffers specified for subsetting state " + state);
	return ps.state_subset_existing(state, buffers);
}

const BufferList extractExistingBufferList(
	ParticleSystem const& ps,
	StateBuffers const& sb)
{
	return extractExistingBufferList(ps, sb.state, sb.buffers);
}

const BufferList extractExistingBufferList(
	ParticleSystem const& ps,
	CommandBufferArgument const& arg)
{
	BufferList ret;
	for (auto const& sb : arg) {
		ret |= extractExistingBufferList(ps, sb);
	}
	return ret;
}

BufferList extractGeneralBufferList(
	ParticleSystem& ps,
	CommandBufferArgument const& arg)
{
	BufferList ret;
	for (auto const& sb : arg) {
		if (sb.buffers == BUFFER_NONE)
			throw std::invalid_argument("no buffers specified for subsetting state " + sb.state);
		ret |= ps.state_subset(sb.state, sb.buffers);
	}
	return ret;
}

BufferList extractGeneralBufferList(
	ParticleSystem& ps,
	CommandBufferArgument const& arg,
	BufferList const& model)
{
	const flag_t missing_spec = model.get_keys();
	BufferList ret;
	for (auto const& sb : arg) {
		flag_t buffers = sb.buffers;
		if (sb.buffers == BUFFER_NONE)
			buffers = missing_spec;
		ret |= ps.state_subset(sb.state, buffers);
	}
	return ret;
}

float undefined_dt(GlobalData const*)
{ return NAN; }
