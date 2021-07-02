/*  Copyright (c) 2011-2018 INGV, EDF, UniCT, JHU

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
 * Debug flags implementation
 */

#include "buffer.h" // for DEBUG_BUFFER_ACCESS
#include "debugflags.h"

#include <sstream>
#include <iostream>

#include <stdexcept>

using namespace std;

DebugFlags g_debug;

void check_buffer_debug_options(DebugFlags const& ret)
{
#if !DEBUG_BUFFER_ACCESS
	if (ret.inspect_buffer_access || ret.clobber_invalid_buffers)
		std::cerr << "WARNING: buffer access inspection requested, "
			"but support not compiled in" << std::endl;
#endif
}

void parse_debug_flags(string const& str)
{
	DebugFlags ret = DebugFlags();

	istringstream in(str);
	string flag;

	while (getline(in, flag, ',')) {
#include "parse-debugflags.h"
		throw invalid_argument("unknown debug flag '" + flag + "'");
	}

	check_buffer_debug_options(ret);

	g_debug = ret;
}

