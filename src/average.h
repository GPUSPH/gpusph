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
