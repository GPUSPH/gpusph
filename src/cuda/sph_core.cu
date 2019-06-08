/*  Copyright (c) 2014-2018 INGV, EDF, UniCT, JHU

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
#ifndef _SPH_CORE_
#define _SPH_CORE_

/* This header file contains core functions of SPH such as the weighting functions,
 * its derivative, as well as the EOS (including inverse) and the methods to compute
 * Riemann invariants
 */

////////////////////
// V A R I A B L E S
////////////////////

namespace cusph {
//< Kernel normalization constants
__constant__ float	d_wcoeff_cubicspline;			//< coeff = 1/(Pi h^3)
__constant__ float	d_wcoeff_quadratic;				//< coeff = 15/(16 Pi h^3)
__constant__ float	d_wcoeff_wendland;				//< coeff = 21/(16 Pi h^3)

/*! Gaussian kernel is in the form (exp(-r/h)^2 - S)/K, so we have two constants:
  d_wsub_gaussian = S which is exp(-R^2), and the normalization constant
  d_wcoeff_gaussian = 1/K
 */
__constant__ float	d_wsub_gaussian;
__constant__ float	d_wcoeff_gaussian;

//< Kernel derivative normalization constants
__constant__ float	d_fcoeff_cubicspline;			//< coeff = 3/(4Pi h^4)
__constant__ float	d_fcoeff_quadratic;				//< coeff = 15/(32Pi h^4)
__constant__ float	d_fcoeff_wendland;				//< coeff = 105/(128Pi h^5)
__constant__ float	d_fcoeff_gaussian;				//< coeff = wcoeff * 2/h^2

#include "gamma.cuh"

////////////////////
// F U N C T I O N S
////////////////////

/********************************************* SPH kernels **************************************************/
// Return kernel value at distance r, for a given smoothing length
template<KernelType kerneltype>
__device__ __forceinline__ float
W(const float r, const float slength);


// Cubic Spline kernel
template<>
__device__ __forceinline__ float
W<CUBICSPLINE>(const float r, const float slength)
{
	float val = 0.0f;
	const float R = r/slength;

	if (R < 1)
		val = 1.0f - 1.5f*R*R + 0.75f*R*R*R;			// val = 1 - 3/2 R^2 + 3/4 R^3
	else
		val = 0.25f*(2.0f - R)*(2.0f - R)*(2.0f - R);	// val = 1/4 (2 - R)^3

	val *= d_wcoeff_cubicspline;						// coeff = 1/(Pi h^3)

	return val;
}


// Qudratic kernel
template<>
__device__ __forceinline__ float
W<QUADRATIC>(const float r, const float slength)
{
	float val = 0.0f;
	const float R = r/slength;

	val = 0.25f*R*R - R + 1.0f;		// val = 1/4 R^2 -  R + 1
	val *= d_wcoeff_quadratic;		// coeff = 15/(16 Pi h^3)

	return val;
}


// Wendland kernel
template<>
__device__ __forceinline__ float
W<WENDLAND>(const float r, const float slength)
{
	const float R = r/slength;

	float val = 1.0f - 0.5f*R;
	val *= val;
	val *= val;						// val = (1 - R/2)^4
	val *= 1.0f + 2.0f*R;			// val = (2R + 1)(1 - R/2)^4*
	val *= d_wcoeff_wendland;		// coeff = 21/(16 Pi h^3)
	return val;
}


// Gaussia kernel
// W(r, h) = (exp(-(r/h)^2) - exp(-(δ/h)^2))*const
// with δ cut-off radius (i.e. influence radius) (typically, 3h),
// and const normalization constant
template<>
__device__ __forceinline__ float
W<GAUSSIAN>(float r, float slength)
{
	const float R = r/slength;

	float val = expf(-R*R);
	val -= d_wsub_gaussian;
	val *= d_wcoeff_gaussian;
	return val;
}


// Return 1/r dW/dr at distance r, for a given smoothing length
template<KernelType kerneltype>
__device__ __forceinline__ float
F(const float r, const float slength);


template<>
__device__ __forceinline__ float
F<CUBICSPLINE>(const float r, const float slength)
{
	float val = 0.0f;
	const float R = r/slength;

	if (R < 1.0f)
		val = (-4.0f + 3.0f*R)/slength;		// val = (-4 + 3R)/h
	else
		val = -(-2.0f + R)*(-2.0f + R)/r;	// val = -(-2 + R)^2/r
	val *= d_fcoeff_cubicspline;			// coeff = 3/(4Pi h^4)

	return val;
}


template<>
__device__ __forceinline__ float
F<QUADRATIC>(const float r, const float slength)
{
	const float R = r/slength;

	float val = (-2.0f + R)/r;		// val = (-2 + R)/r
	val *= d_fcoeff_quadratic;		// coeff = 15/(32Pi h^4)

	return val;
}


template<>
__device__ __forceinline__ float
F<WENDLAND>(const float r, const float slength)
{
	const float qm2 = r/slength - 2.0f;	// val = (-2 + R)^3
	float val = qm2*qm2*qm2*d_fcoeff_wendland;
	return val;
}


template<>
__device__ __forceinline__ float
F<GAUSSIAN>(const float r, const float slength)
{
	const float R = r/slength;
	float val = -expf(-R*R)*d_fcoeff_gaussian;
	return val;
}

/************************************************************************************************************/

}

#endif
