/* Math on CUDA vector types, from cutil_math.h */
#ifndef _VECTOR_MATH_H_
#define _VECTOR_MATH_H_

#include <stdint.h>
#include <math.h>
#include <cmath>

typedef unsigned int uint;
typedef unsigned short ushort;

//float2

struct float2 {
	float x;
	float y;
};

inline float2 make_float2(float x, float y) {
	float2 ret;
	ret.x=x; ret.y=y;
	return ret;
}

//float3

struct float3 {
	float x;
	float y;
	float z;
};

inline float3 make_float3(float x, float y, float z) {
	float3 ret;
	ret.x=x; ret.y=y; ret.z=z;
	return ret;
}

//float4

struct float4 {
	float x;
	float y;
	float z;
	float w;
};

inline float4 make_float4(float x, float y, float z, float w) {
	float4 ret;
	ret.x=x; ret.y=y; ret.z=z; ret.w=w;
	return ret;
}

inline float4 make_float4(float x) {
	float4 v;
	v.x = x; v.y = x; v.z = x; v.w = x;
	return v;
}

//double2

struct double2 {
	double x;
	double y;
};

inline double2 make_double2(double x, double y) {
	double2 ret;
	ret.x=x; ret.y=y;
	return ret;
}

//double3

struct double3 {
	double x;
	double y;
	double z;
};

inline double3 make_double3(double x, double y, double z) {
	double3 ret;
	ret.x=x; ret.y=y; ret.z=z;
	return ret;
}

//double4

struct double4 {
	double x;
	double y;
	double z;
	double w;
};

inline double4 make_double4(double x, double y, double z, double w) {
	double4 ret;
	ret.x=x; ret.y=y; ret.z=z; ret.w=w;
	return ret;
}

inline double4 make_double4(double x) {
	double4 v;
	v.x = x; v.y = x; v.z = x; v.w = x;
	return v;
}

//int2

struct int2 {
	int x;
	int y;
};

inline int2 make_int2(int x, int y) {
	int2 ret;
	ret.x=x; ret.y=y;
	return ret;
}

//int3

struct int3 {
	int x;
	int y;
	int z;
};

inline int3 make_int3(int x, int y, int z) {
	int3 ret;
	ret.x=x; ret.y=y; ret.z=z;
	return ret;
}

//int4

struct int4 {
	int x;
	int y;
	int z;
	int w;
};

//uint3

struct uint3 {
	uint x;
	uint y;
	uint z;
};

inline uint3 make_uint3(uint x, uint y, uint z) {
	uint3 ret;
	ret.x=x; ret.y=y; ret.z=z;
	return ret;
}

//ushort4

struct ushort4 {
	unsigned short x;
	unsigned short y;
	unsigned short z;
	unsigned short w;
};

struct short4 {
	short x;
	short y;
	short z;
	short w;
};

//char3

struct char3 {
	char x;
	char y;
	char z;
};

//uint4

struct uint4 {
	unsigned int x;
	unsigned int y;
	unsigned int z;
	unsigned int w;
};

inline float fminf(float a, float b)
{
	return a < b ? a : b;
}

inline float fmaxf(float a, float b)
{
	return a > b ? a : b;
}

inline int max(int a, int b)
{
	return a > b ? a : b;
}

inline int min(int a, int b)
{
	return a < b ? a : b;
}

inline float rsqrtf(float x)
{
	x = sqrtf(x);
	return (x == 0.0f ? 0.0f : 1.0f / sqrtf(x));
}

#endif
