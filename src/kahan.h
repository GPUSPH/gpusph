/*  Copyright (c) 2012-2019 INGV, EDF, UniCT, JHU

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
 * Kahan summation and related functions
 */

#ifndef _KAHAN_H_
#define _KAHAN_H_

#include "vector_math.h"
#define __spec __host__ __device__

//! Add n floating-point values stored in q
//! using Kahan summation
float
__spec
kahan_sum(const float *q, uint n)
{
	float sum, corr, corr_next, new_sum;
	sum = q[0];
	corr = 0.0f;

	for (uint i=1; i < n; ++i) {
		corr_next = q[i] - corr;
		new_sum = sum + corr_next;
		corr = (new_sum - sum) - corr_next;
		sum = new_sum;
	}

	return sum;
}

//! Kahan sum of three floating-point values
float
__spec
kahan_sum(const float f1, const float f2, const float f3)
{
	float sum, corr, corr_next, new_sum;
	sum = f1;
	corr = 0.0f;

	corr_next = f2 - corr;
	new_sum = sum + corr_next;
	corr = (new_sum - sum) - corr_next;
	sum = new_sum;

	corr_next = f3 - corr;
	new_sum = sum + corr_next;
	corr = (new_sum - sum) - corr_next;
	sum = new_sum;

	return sum;
}

//! Kahan sum of four floating-point values
float
__spec
kahan_sum(const float f1, const float f2, const float f3, const float f4)
{
	float sum, corr, corr_next, new_sum;
	sum = f1;
	corr = 0.0f;

	corr_next = f2 - corr;
	new_sum = sum + corr_next;
	corr = (new_sum - sum) - corr_next;
	sum = new_sum;

	corr_next = f3 - corr;
	new_sum = sum + corr_next;
	corr = (new_sum - sum) - corr_next;
	sum = new_sum;

	corr_next = f4 - corr;
	new_sum = sum + corr_next;
	corr = (new_sum - sum) - corr_next;
	sum = new_sum;

	return sum;
}

//! Kahan sum of five floating-point values
float
__spec
kahan_sum(const float f1, const float f2, const float f3, const float f4, const float f5)
{
	float sum, corr, corr_next, new_sum;
	sum = f1;
	corr = 0.0f;

	corr_next = f2 - corr;
	new_sum = sum + corr_next;
	corr = (new_sum - sum) - corr_next;
	sum = new_sum;

	corr_next = f3 - corr;
	new_sum = sum + corr_next;
	corr = (new_sum - sum) - corr_next;
	sum = new_sum;

	corr_next = f4 - corr;
	new_sum = sum + corr_next;
	corr = (new_sum - sum) - corr_next;
	sum = new_sum;

	corr_next = f5 - corr;
	new_sum = sum + corr_next;
	corr = (new_sum - sum) - corr_next;
	sum = new_sum;

	return sum;
}

//! Check if any component is non-zero
//! @{
inline bool
__spec
operator !(const float3& v) {
	return !(v.x || v.y || v.z);
}

inline bool
__spec
operator !(const float4& v) {
	return !(v.x || v.y || v.z || v.w);
}
//! @}

//! Increment val by add using Kahan summation
//! and storing the remainder in kahan
template<typename T>
inline void
__spec
kahan_add(T &val, const T& add, T &kahan) {
	if (!add)
		return;
	T sub = add - kahan;
	T new_val = val + sub;
	kahan = new_val - val;
	kahan -= sub;
	val = new_val;
}


/* 2D Kahan is just the standard ops */
inline float
__spec
kahan_dot(const float2 &f1, const float2 &f2)
{
	return f1.x*f2.x + f1.y*f2.y;
}

//! Cross-product using Kahan summation
//! @{
inline float
__spec
kahan_dot(const float3 &f1, const float3 &f2)
{
	return kahan_sum(f1.x*f2.x, f1.y*f2.y, f1.z*f2.z);
}

inline float
__spec
kahan_dot(const float4 &f1, const float4 &f2)
{
	return kahan_sum(f1.x*f2.x, f1.y*f2.y, f1.z*f2.z, f1.w*f2.w);
}

// T1, T2 should be either float3 or float4
template<typename T1, typename T2>
inline float
__spec
kahan_dot3(const T1 &f1, const T2 &f2)
{
	return kahan_sum(f1.x*f2.x, f1.y*f2.y, f1.z*f2.z);
}
//! @}

//! Squared vector length using Kahan summation
//! @{
inline float
__spec
kahan_sqlength(const float2 &f1)
{
	return kahan_dot(f1, f1);
}

inline float
__spec
kahan_sqlength(const float3 &f1)
{
	return kahan_dot(f1, f1);
}

inline float
__spec
kahan_sqlength(const float4 &f1)
{
	return kahan_dot(f1, f1);
}

inline float
__spec
kahan_sqlength3(const float4 &f1)
{
	return kahan_dot3(f1, f1);
}
//! @}

//! Vector length using Kahan summation
//! @{
inline float
__spec
kahan_length(const float2 &f1)
{
	return sqrt(kahan_sqlength(f1));
}

inline float
__spec
kahan_length(const float3 &f1)
{
	return sqrt(kahan_sqlength(f1));
}

inline float
__spec
kahan_length(const float4 &f1)
{
	return sqrt(kahan_sqlength(f1));
}
//! @}

// Kahan-Babushka-Neumaier compensated summation
// Remember to add the kahan reminder after adding all terms!
inline float
__spec
kbn_add(float base, float add, float &kahan)
{
	if (add == 0) return base;
	float new_val = base + add;
	/* (A - new_val) + B where A is absmax(val, add) and B is the other */
	const bool add_larger = (fabs(base) < fabs(add));
	const float A = add_larger ? add : base;
	const float B = add_larger ? base : add;
	kahan += (A - new_val) + B;
	return new_val;
}

inline float
__spec
kbn_sum(float const* els, int n)
{
	float total = 0;
	float rem = 0;
	for (int i = 0; i < n; ++i)
		total = kbn_add(total, els[i], rem);
	total += rem;
	return total;
}

inline float
__spec
kbn_sum(float c1, float c2, float c3)
{
	float total = 0;
	float rem = 0;
	total = kbn_add(total, c1, rem);
	total = kbn_add(total, c2, rem);
	total = kbn_add(total, c3, rem);
	total += rem;
	return total;
}

inline float
__spec
kbn_sum(float c1, float c2, float c3, float c4)
{
	float total = 0;
	float rem = 0;
	total = kbn_add(total, c1, rem);
	total = kbn_add(total, c2, rem);
	total = kbn_add(total, c3, rem);
	total = kbn_add(total, c4, rem);
	total += rem;
	return total;
}


//! Cross-product using KBN summation
//! @{
inline float
__spec
kbn_dot(const float3 &f1, const float3 &f2)
{
	return kbn_sum(f1.x*f2.x, f1.y*f2.y, f1.z*f2.z);
}

inline float
__spec
kbn_dot(const float4 &f1, const float4 &f2)
{
	return kbn_sum(f1.x*f2.x, f1.y*f2.y, f1.z*f2.z, f1.w*f2.w);
}
//! @}

#undef __spec

#endif

