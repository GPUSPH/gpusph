/* Kahan summation and related functions */

#ifndef _KAHAN_H_
#define _KAHAN_H_

#include <math_functions.h>
#define __spec __host__ __device__

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

/* special cases for 3 and 4 components */

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

#undef __spec

#endif

