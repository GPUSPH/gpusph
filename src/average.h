/*  Copyright (c) 2018-2019 INGV, EDF, UniCT, JHU

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
 * Averaging operators
 */

#ifndef AVERAGE_H
#define AVERAGE_H

#include "option_range.h"

#ifdef __CUDACC__
#define _AVG_FUNC_SPEC __host__ __device__ __forceinline__
#else
#define _AVG_FUNC_SPEC inline
#endif

//! Averaging operator names
/**@defpsubsection{viscosityAveraging, VISCOUS_AVERAGING}
 * @inpsection{viscous_options}
 * @default{harmonic}
 * @values{arithmetic, harmonic, geometric}
 * TLT_VISCOUS_AVERAGING
 */
enum AverageOperator
{
	ARITHMETIC, //! (a+b)/2
	HARMONIC, //! 2 ab/(a+b)
	GEOMETRIC //! sqrt(ab)
};

//! Name of the viscous model
#ifndef GPUSPH_MAIN
extern
#endif
const char* AverageOperatorName[GEOMETRIC+1]
#ifdef GPUSPH_MAIN
= {
	"Arithmetic",
	"Harmonic",
	"Geometric",
}
#endif
;

DEFINE_OPTION_RANGE(AverageOperator, AverageOperatorName, ARITHMETIC, GEOMETRIC);

/* Actual operators */

template<AverageOperator>
_AVG_FUNC_SPEC float
average(float a, float b);

template<>
_AVG_FUNC_SPEC float
average<ARITHMETIC>(float a, float b)
{
	return (a+b)*0.5f;
}

template<>
_AVG_FUNC_SPEC float
average<HARMONIC>(float a, float b)
{
	return 2.0f*a*b/(a+b);
}

template<>
_AVG_FUNC_SPEC float
average<GEOMETRIC>(float a, float b)
{
	return sqrtf(a*b);
}

#endif
