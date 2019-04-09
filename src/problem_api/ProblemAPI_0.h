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

#ifndef PROBLEM_API_H
#define PROBLEM_API_H

// Problem API version 0 is the “first” Problem API, which exposed what is
// currently the ProblemCore directly
// It should not normally be used directly, as it is an intrinsically unstable
// interface

#define PROBLEM_API 0
#include "Problem.h"

template<>
class ProblemAPI<0> : public ProblemCore
{
	using ProblemCore::ProblemCore; // inherit the constructor
};

#endif


