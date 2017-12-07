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

#ifndef DEBUG_FLAGS_H
#define DEBUG_FLAGS_H

#include <string>

/// Bitfield of things to debug
struct DebugFlags {
	/// print each step as it is being executed
	unsigned print_step : 1;
	/// debug the neighbors list on host
	unsigned neibs : 1;
	/// debug forces on host
	unsigned forces : 1;
	/// inspect pre-force particle status on
	unsigned inspect_preforce : 1;
};

/// Get a DebugFlag from a comma-separated list
DebugFlags parse_debug_flags(std::string const& str);

#endif
