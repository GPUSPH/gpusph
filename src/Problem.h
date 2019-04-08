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

#ifndef PROBLEM_H
#define PROBLEM_H

#ifndef PROBLEM_API
#define PROBLEM_API 0
#warning "Please define PROBLEM_API before inclusion of Problem.h"
#endif

// All problem interfaces end up hooking up on ProblemCore
#include "ProblemCore.h"

#if PROBLEM_API == 0

// Problem API version 0 is the “first” Problem API, which exposed what is
// currently the ProblemCore directly
// It should not normally be used directly, as it is an intrinsically unstable
// interface
typedef ProblemCore Problem;

#elif PROBLEM_API == 1

// Problem API version 1 was known as the XProblem interface before GPUSPHv5
// TODO migration still to be done

#else

#error "Unsupported PROBLEM_API"

#endif

#endif
