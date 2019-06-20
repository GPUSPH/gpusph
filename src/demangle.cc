/*  Copyright (c) 2019 INGV, EDF, UniCT, JHU

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
 * C++ name demangling function implementation
 */

#include <cstdlib>

#ifdef _MSC_VER
#else
#include <cxxabi.h>
#endif

#include "demangle.h"

using namespace std;

string demangle(const char *mangled_name)
{
#ifdef _MSC_VER
	// MSVC actually returns a readable name, but it's in the form
	// `class SomeClass` or `enum SomeEnum`. For our applications,
	// we can clean this up by skipping to after the first space
	string ret = mangled_name;
	auto space = ret.find(' ');
	if (space != string::npos) {
		ret.erase(0, space+1);
	}
#else
	char *tname = abi::__cxa_demangle(mangled_name, NULL, 0, NULL);
	string ret = tname;
	free(tname);
#endif
	return ret;
}
