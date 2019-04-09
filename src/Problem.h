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
 * Problem API version wrangler.
 *
 * \note
 * This file should be included after defining PROBLEM_API to the expected
 * version. The correct API will then be included and exposed as the Problem
 * base class, that the user-specific simulation can derive from.
 */

#ifndef PROBLEM_H
#define PROBLEM_H

/* Individual API specifications are in this form */
template<int version>
class ProblemAPI;

#ifndef PROBLEM_API
#define PROBLEM_API 0
#warning "Please define PROBLEM_API before inclusion of Problem.h"
#endif

//! use STR(SOMEMACRO) to turn the content of SOMEMACRO into a string
#define _STR(x) #x
#define STR(x) _STR(x)

#define __PROBLEM_API_INCLUDE(version) STR(ProblemAPI_##version.h)
#define _PROBLEM_API_INCLUDE(version) __PROBLEM_API_INCLUDE(version)
#define PROBLEM_API_INCLUDE _PROBLEM_API_INCLUDE(PROBLEM_API)

// All problem interfaces end up hooking up on ProblemCore
#include "ProblemCore.h"

#include PROBLEM_API_INCLUDE

typedef ProblemAPI<PROBLEM_API> Problem;

#if PROBLEM_API > 1

#error "Unsupported PROBLEM_API"

#endif

#endif
