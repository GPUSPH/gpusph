/*  Copyright 2016 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#include "debugflags.h"

#include <sstream>

#include <stdexcept>

using namespace std;

DebugFlags parse_debug_flags(string const& str)
{
	DebugFlags ret = DebugFlags();

	istringstream in(str);
	string flag;

	while (getline(in, flag, ',')) {
		if (flag == "print_step")
			ret.print_step = 1;
		else if (flag == "neibs")
			ret.neibs = 1;
		else if (flag == "forces")
			ret.forces = 1;
		else if (flag == "inspect_preforce")
			ret.inspect_preforce = 1;
		else
			throw invalid_argument("unknown debug flag '" + flag + "'");
	}

	return ret;
}

