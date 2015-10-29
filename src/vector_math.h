/* Math on CUDA vector types, inspired by cutil_math.h */

// NOTES: to ensure no downconversions are introduced, this header should
// compile cleanly with -Wconversion enabled. To achieve this, use the following
// care when adding functions:
// * float functions should be used on float POD types (e.g.: fabsf instead of
//   fabs when the argument is a float);
// * double functions should be used double POD types (obviously);
// * explictly cast int/uint to float when doing mixed int/float operations,
//   since int-to-float conversion can actually cause data loss (for values
//   larger than 2^24) and thus -Wconversion warns about them.

#ifndef VECTOR_MATH_H
#define VECTOR_MATH_H

#include "cuda_runtime.h"

////////////////////////////////////////////////////////////////////////////////
typedef unsigned int uint;
typedef unsigned short ushort;

#ifndef __CUDACC__
#include <cmath>

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
	return 1.0f / sqrtf(x);
}

inline double rsqrt(double x)
{
	return 1.0 / sqrt(x);
}

#endif

// float functions
////////////////////////////////////////////////////////////////////////////////

// lerp
static __device__ __forceinline__ __host__ float lerp(const float &a, const float &b, const float &t)
{
	return a + t*(b-a);
}

// clamp
static __device__ __forceinline__ __host__ float clamp(const float &f, const float &a, const float &b)
{
	return fmaxf(a, fminf(f, b));
}

// int2 functions
////////////////////////////////////////////////////////////////////////////////

// negate
static __forceinline__ __host__ __device__ int2 operator-(const int2 &a)
{
	return make_int2(-a.x, -a.y);
}

// addition
static __forceinline__ __host__ __device__ int2 operator+(const int2 &a, const int2 &b)
{
	return make_int2(a.x + b.x, a.y + b.y);
}

static __forceinline__ __host__ __device__ void operator+=(int2 &a, const int2 &b)
{
	a.x += b.x; a.y += b.y;
}

// subtract
static __forceinline__ __host__ __device__ int2 operator-(const int2 &a, const int2 &b)
{
	return make_int2(a.x - b.x, a.y - b.y);
}

static __forceinline__ __host__ __device__ void operator-=(int2 &a, const int2 &b)
{
	a.x -= b.x; a.y -= b.y;
}

// multiply
static __forceinline__ __host__ __device__ int2 operator*(const int2 &a, const int2 &b)
{
	return make_int2(a.x * b.x, a.y * b.y);
}

static __forceinline__ __host__ __device__ int2 operator*(const int2 &a, const int &s)
{
	return make_int2(a.x * s, a.y * s);
}

static __forceinline__ __host__ __device__ int2 operator*(const int &s, const int2 &a)
{
	return make_int2(a.x * s, a.y * s);
}

static __forceinline__ __host__ __device__ void operator*=(int2 &a, const int &s)
{
	a.x *= s; a.y *= s;
}

// show an int3 as an int2
static __forceinline__ __host__ __device__ int2& as_int2(const int3 &v)
{
	return *(int2*)&v;
}

// float2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
static __forceinline__ __host__ __device__ float2 make_float2(const float &s)
{
	return make_float2(s, s);
}

static __forceinline__ __host__ __device__ float2 make_float2(const int2 &a)
{
	return make_float2(float(a.x), float(a.y));
}

// show an float3 as an float2
static __forceinline__ __host__ __device__ float2& as_float2(const float3 &v)
{
	return *(float2*)&v;
}


// negate
static __forceinline__ __host__ __device__ float2 operator-(const float2 &a)
{
	return make_float2(-a.x, -a.y);
}

// addition
static __forceinline__ __host__ __device__ float2 operator+(const float2 &a, const float2 &b)
{
	return make_float2(a.x + b.x, a.y + b.y);
}

static __forceinline__ __host__ __device__ void operator+=(float2 &a, const float2 &b)
{
	a.x += b.x; a.y += b.y;
}

// subtract
static __forceinline__ __host__ __device__ float2 operator-(const float2 &a, const float2 &b)
{
	return make_float2(a.x - b.x, a.y - b.y);
}

static __forceinline__ __host__ __device__ void operator-=(float2 &a, const float2 &b)
{
	a.x -= b.x; a.y -= b.y;
}

// multiply
static __forceinline__ __host__ __device__ float2 operator*(const float2 &a, const float2 &b)
{
	return make_float2(a.x * b.x, a.y * b.y);
}

static __forceinline__ __host__ __device__ float2 operator*(const float2 &a, const float &s)
{
	return make_float2(a.x * s, a.y * s);
}

static __forceinline__ __host__ __device__ float2 operator*(const float &s, const float2 &a)
{
	return make_float2(a.x * s, a.y * s);
}

static __forceinline__ __host__ __device__ void operator*=(float2 &a, const float &s)
{
	a.x *= s; a.y *= s;
}

// divide
static __forceinline__ __host__ __device__ float2 operator/(const float2 &a, const float2 &b)
{
	return make_float2(a.x / b.x, a.y / b.y);
}

static __forceinline__ __host__ __device__ float2 operator/(const float2 &a, const float &s)
{
	float inv = 1.0f / s;
	return a * inv;
}

static __forceinline__ __host__ __device__ float2 operator/(const float &s, const float2 &a)
{
	float inv = 1.0f / s;
	return a * inv;
}

static __forceinline__ __host__ __device__ void operator/=(float2 &a, const float &s)
{
	float inv = 1.0f / s;
	a *= inv;
}

// lerp
static __device__ __forceinline__ __host__ float2 lerp(const float2 &a, const float2 &b, const float &t)
{
	return a + t*(b-a);
}

// clamp
static __device__ __forceinline__ __host__ float2 clamp(const float2 &v, const float &a, const float &b)
{
	return make_float2(clamp(v.x, a, b), clamp(v.y, a, b));
}

static __device__ __forceinline__ __host__ float2 clamp(const float2 &v, const float2 &a, const float2 &b)
{
	return make_float2(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y));
}

// dot product
static __forceinline__ __host__ __device__ float dot(const float2 &a, const float2 &b)
{
	return a.x * b.x + a.y * b.y;
}

// squared length
static __forceinline__ __host__ __device__ float sqlength(const float2 &v)
{
	return dot(v, v);
}

// length
static __forceinline__ __host__ __device__ float length(const float2 &v)
{
	return sqrtf(sqlength(v));
}

// normalize
static __forceinline__ __host__ __device__ float2 normalize(const float2 &v)
{
	float invLen = rsqrtf(sqlength(v));
	return v * invLen;
}

// floor
static __forceinline__ __host__ __device__ float2 floor(const float2 &v)
{
	return make_float2(floorf(v.x), floorf(v.y));
}

// reflect
static __forceinline__ __host__ __device__ float2 reflect(const float2 &i, const float2 &n)
{
	return i - 2.0f * n * dot(n,i);
}

// absolute value
static __forceinline__ __host__ __device__ float2 fabs(const float2 &v)
{
	return make_float2(fabsf(v.x), fabsf(v.y));
}

// length of the vector, computed more robustly when the components of v
// are significantly larger than unity
static __forceinline__ __host__ __device__ float hypot(const float2 &v)
{
	float p, r;
	if (fabs(v.x) > fabs(v.y)) {
		p = fabs(v.x);
		r = v.y/v.x;
	} else {
		p = fabs(v.y);
		r = p > 0 ? v.x/v.y : 0; // avoid a division by 0
	}
	return p*sqrt(1+r*r);
}

// double2 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
static __forceinline__ __host__ __device__ double2 make_double2(const double3 &s)
{
	return make_double2(s.x, s.y); // strip last component
}

// add
static __forceinline__ __host__ __device__ double2 operator+(const double2 &a, const double &b)
{
	return make_double2(a.x + b, a.y + b);
}

// subtract
static __forceinline__ __host__ __device__ double2 operator-(const double2 &a, const double &b)
{
	return make_double2(a.x - b, a.y - b);
}

// dot product
static __forceinline__ __host__ __device__ double dot(const double2 &a, const double2 &b)
{
	return a.x*b.x + a.y*b.y;
}

// squared length
static __forceinline__ __host__ __device__ double sqlength(const double2 &v)
{
	return dot(v, v);
}

// length
static __forceinline__ __host__ __device__ double length(const double2 &v)
{
	return sqrt(sqlength(v));
}

static __forceinline__ __host__ __device__ double2 operator*(const double2 &b, const double a)
{
	return make_double2(a*b.x, a*b.y);
}

static __forceinline__ __host__ __device__ double2 normalize(const double2 &v)
{
	double invLen = rsqrt(sqlength(v));
	return v * invLen;
}

// length of the vector, computed more robustly when the components of v
// are significantly larger than unity
static __forceinline__ __host__ __device__ double hypot(const double2 &v)
{
	double p, r;
	if (fabs(v.x) > fabs(v.y)) {
		p = fabs(v.x);
		r = v.y/v.x;
	} else {
		p = fabs(v.y);
		r = p > 0 ? v.x/v.y : 0; // avoid a division by 0
	}
	return p*sqrt(1+r*r);
}

// float3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
static __forceinline__ __host__ __device__ float3 make_float3(const float &s)
{
	return make_float3(s, s, s);
}

static __forceinline__ __host__ __device__ float3 make_float3(const float2 &a)
{
	return make_float3(a.x, a.y, 0.0f);
}

static __forceinline__ __host__ __device__ float3 make_float3(const float2 &a, const float &s)
{
	return make_float3(a.x, a.y, s);
}

static __forceinline__ __host__ __device__ float3 make_float3(const float4 &a)
{
	return make_float3(a.x, a.y, a.z);  // discards w
}

static __forceinline__ __host__ __device__ float3 make_float3(const int3 &a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}

static __forceinline__ __host__  float3 make_float3(const float *a)
{
	return make_float3(a[0], a[1], a[2]);
}

static __forceinline__ __host__  float3 make_float3(const double *a)
{
	return make_float3(float(a[0]), float(a[1]), float(a[2]));
}

static __forceinline__ __host__  __device__  float3 make_float3(const double3 &a)
{
	return make_float3(float(a.x), float(a.y), float(a.z));
}


// negate
static __forceinline__ __host__ __device__ float3 operator-(const float3 &a)
{
	return make_float3(-a.x, -a.y, -a.z);
}

// min
static __forceinline__ __host__ __device__ float3 fminf(const float3 &a, const float3 &b)
{
	return make_float3(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z));
}

// max
static __forceinline__ __host__ __device__ float3 fmaxf(const float3 &a, const float3 &b)
{
	return make_float3(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z));
}

// addition
static __forceinline__ __host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
	return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __host__ __device__ float3 operator+(float3 a, float b)
{
	return make_float3(a.x + b, a.y + b, a.z + b);
}

static __forceinline__ __host__ __device__ void operator+=(float3 &a, const float3 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
static __forceinline__ __host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
	return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __host__ __device__ float3 operator-(const float3 &a, const float &b)
{
	return make_float3(a.x - b, a.y - b, a.z - b);
}

static __forceinline__ __host__ __device__ void operator-=(float3 &a, const float3 &b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
static __forceinline__ __host__ __device__ float3 operator*(const float3 &a, const float3 &b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __host__ __device__ float3 operator*(const int3 &a, const float3 &b)
{
	return make_float3(float(a.x) * b.x, float(a.y) * b.y, float(a.z) * b.z);
}

static __forceinline__ __host__ __device__ float3 operator*(const float3 &a, const int3 &b)
{
	return make_float3(a.x * float(b.x), a.y * float(b.y), a.z * float(b.z));
}

static __forceinline__ __host__ __device__ float3 operator*(const uint3 &a, const float3 &b)
{
	return make_float3(float(a.x) * b.x, float(a.y) * b.y, float(a.z) * b.z);
}

static __forceinline__ __host__ __device__ float3 operator*(const float3 &a, const uint3 &b)
{
	return make_float3(a.x * float(b.x), a.y * float(b.y), a.z * float(b.z));
}

static __forceinline__ __host__ __device__ float3 operator*(const float3 &a, const float &s)
{
	return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __host__ __device__ float3 operator*(const float &s, const float3 &a)
{
	return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __host__ __device__ void operator*=(float3 &a, const float &s)
{
	a.x *= s; a.y *= s; a.z *= s;
}

// divide
static __forceinline__ __host__ __device__ float3 operator/(const float3 &a, const float3 &b)
{
	return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}
static __forceinline__ __host__ __device__ float3 operator/(const float3 &a, const float &s)
{
	float inv = 1.0f / s;
	return a * inv;
}
static __forceinline__ __host__ __device__ float3 operator/(const float &s, const float3 &a)
{
	float inv = 1.0f / s;
	return a * inv;
}
static __forceinline__ __host__ __device__ void operator/=(float3 &a, const float &s)
{
	float inv = 1.0f / s;
	a *= inv;
}

// lerp
static __device__ __forceinline__ __host__ float3 lerp(const float3 &a, const float3 &b, const float &t)
{
	return a + t*(b-a);
}

// clamp
static __device__ __forceinline__ __host__ float3 clamp(const float3 &v, const float &a, const float &b)
{
	return make_float3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

static __device__ __forceinline__ __host__ float3 clamp(const float3 &v, const float3 &a, const float3 &b)
{
	return make_float3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// dot product
static __forceinline__ __host__ __device__ float dot(const float3 &a, const float3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
static __forceinline__ __host__ __device__ float3 cross(const float3 &a, const float3 &b)
{
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// squared length
static __forceinline__ __host__ __device__ float sqlength(const float3 &v)
{
	return dot(v, v);
}

// length
static __forceinline__ __host__ __device__ float length(const float3 &v)
{
	return sqrtf(sqlength(v));
}

// normalize
static __forceinline__ __host__ __device__ float3 normalize(const float3 &v)
{
	float invLen = rsqrtf(sqlength(v));
	return v * invLen;
}

// floor
static __forceinline__ __host__ __device__ float3 floor(const float3 &v)
{
	return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

// reflect
static __forceinline__ __host__ __device__ float3 reflect(const float3 &i, const float3 &n)
{
	return i - 2.0f * n * dot(n,i);
}

// absolute value
static __forceinline__ __host__ __device__ float3 fabs(const float3 &v)
{
	return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

static __forceinline__ __host__ __device__ float3 rotate(const float3 &v, const float3 &ort, const float &angle)
{
	float vnorm = length(ort);
	float ct = cosf(angle);
	float st = sinf(angle);
	float x = ort.x, y = ort.y, z = ort.z;

	float a11, a12, a13, a21, a22, a23, a31, a32, a33;

	a11 = x*x + (y*y + z*z)*ct;
	a11 /= vnorm*vnorm;

	a22 = y*y + (x*x + z*z)*ct;
	a22 /= vnorm*vnorm;

	a33 = z*z + (x*x + y*y)*ct;
	a33 /= vnorm*vnorm;

	a12 = x*y*(1-ct)-z*vnorm*st;
	a12 /= vnorm*vnorm;

	a21 = x*y*(1-ct)+z*vnorm*st;
	a21 /= vnorm*vnorm;

	a13 = x*z*(1-ct)+y*vnorm*st;
	a13 /= vnorm*vnorm;

	a31 = x*z*(1-ct)-y*vnorm*st;
	a31 /= vnorm*vnorm;

	a23 = y*z*(1-ct)-x*vnorm*st;
	a23 /= vnorm*vnorm;

	a32 = y*z*(1-ct)+x*vnorm*st;
	a32 /= vnorm*vnorm;

	return make_float3(
			a11*v.x+a12*v.y+a13*v.z,
			a21*v.x+a22*v.y+a23*v.z,
			a31*v.x+a32*v.y+a33*v.z);
}

// length of the vector, computed more robustly when the components of v
// are significantly larger than unity
static __forceinline__ __host__ __device__ float hypot(const float3 &v)
{
	float p;
	p = fmax(fmax(fabs(v.x), fabs(v.y)), fabs(v.z));
	if (!p)
		return 0;

	float3 w=v/p;
	return p*length(w);
}

// double3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
static __forceinline__ __host__ __device__ double3 make_double3(const double &s)
{
	return make_double3(s, s, s);
}

static __forceinline__ __host__ __device__ double3 make_double3(const double2 &a)
{
	return make_double3(a.x, a.y, 0.0);
}

static __forceinline__ __host__ __device__ double3 make_double3(const double2 &a, const double &s)
{
	return make_double3(a.x, a.y, s);
}

static __forceinline__ __host__ __device__ double3 make_double3(const double4 &a)
{
	return make_double3(a.x, a.y, a.z);  // discards w
}

static __forceinline__ __host__ __device__ double3 make_double3(const int3 &a)
{
	return make_double3(double(a.x), double(a.y), double(a.z));
}

static __forceinline__ __host__ __device__ double3 make_double3(const float3 &a)
{
	return make_double3(double(a.x), double(a.y), double(a.z));
}

static __forceinline__ __host__  double3 make_double3(const float *a)
{
	return make_double3(double(a[0]), double(a[1]), double(a[2]));
}

static __forceinline__ __host__  double3 make_double3(const double *a)
{
	return make_double3(a[0], a[1], a[2]);
}

// negate
static __forceinline__ __host__ __device__ double3 operator-(const double3 &a)
{
	return make_double3(-a.x, -a.y, -a.z);
}


// addition
static __forceinline__ __host__ __device__ double3 operator+(const double3 &a, const double3 &b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __host__ __device__ double3 operator+(double3 a, double b)
{
	return make_double3(a.x + b, a.y + b, a.z + b);
}

static __forceinline__ __host__ __device__ void operator+=(double3 &a, const double3 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}
static __inline__ __host__ __device__ void operator+=(double3 &a, const double &b)
{
	a.x += b; a.y += b; a.z += b;
}

// subtract
static __forceinline__ __host__ __device__ double3 operator-(const double3 &a, const double3 &b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __host__ __device__ double3 operator-(const double3 &a, const double &b)
{
	return make_double3(a.x - b, a.y - b, a.z - b);
}

static __forceinline__ __host__ __device__ void operator-=(double3 &a, const double3 &b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}
static __inline__ __host__ __device__ void operator-=(double3 &a, const double &b)
{
	a.x -= b; a.y -= b; a.z -= b;
}

// multiply
static __forceinline__ __host__ __device__ double3 operator*(const double3 &a, const double3 &b)
{
	return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __host__ __device__ double3 operator*(const int3 &a, const double3 &b)
{
	return make_double3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __host__ __device__ double3 operator*(const double3 &a, const double &s)
{
	return make_double3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __host__ __device__ double3 operator*(const double &s, const double3 &a)
{
	return make_double3(a.x * s, a.y * s, a.z * s);
}
static __forceinline__ __host__ __device__ void operator*=(double3 &a, const double &s)
{
	a.x *= s; a.y *= s; a.z *= s;
}

// divide
static __forceinline__ __host__ __device__ double3 operator/(const double3 &a, const double3 &b)
{
	return make_double3(a.x / b.x, a.y / b.y, a.z / b.z);
}

static __forceinline__ __host__ __device__ double3 operator/(const double3 &a, const double &s)
{
	double inv = 1.0 / s;
	return a * inv;
}

static __forceinline__ __host__ __device__ double3 operator/(const double &s, const double3 &a)
{
	double inv = 1.0 / s;
	return a * inv;
}

static __forceinline__ __host__ __device__ void operator/=(double3 &a, const double &s)
{
	double inv = 1.0 / s;
	a *= inv;
}


// dot product
static __forceinline__ __host__ __device__ double dot(const double3 &a, const double3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// cross product
static __forceinline__ __host__ __device__ double3 cross(const double3 &a, const double3 &b)
{
	return make_double3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

// squared length
static __forceinline__ __host__ __device__ double sqlength(const double3 &v)
{
	return dot(v, v);
}

// length
static __forceinline__ __host__ __device__ double length(const double3 &v)
{
	return sqrt(sqlength(v));
}

// normalize
static __forceinline__ __host__ __device__ double3 normalize(const double3 &v)
{
	double invLen = rsqrt(sqlength(v));
	return v * invLen;
}

// floor
static __forceinline__ __host__ __device__ double3 floor(const double3 &v)
{
	return make_double3(floor(v.x), floor(v.y), floor(v.z));
}

// absolute value
static __forceinline__ __host__ __device__ double3 fabs(const double3 &v)
{
	return make_double3(fabs(v.x), fabs(v.y), fabs(v.z));
}

// copysign
static __forceinline__ __host__ __device__ double3 copysign(const double3 &v, const double3 &ref)
{
	return make_double3(
		copysign(v.x, ref.x),
		copysign(v.y, ref.y),
		copysign(v.z, ref.z));
}

// length of the vector, computed more robustly when the components of v
// are significantly larger than unity
static __forceinline__ __host__ __device__ double hypot(const double3 &v)
{
	double p;
	p = fmax(fmax(fabs(v.x), fabs(v.y)), fabs(v.z));
	if (!p)
		return 0;

	double3 w=v/p;
	return p*length(w);
}


// double4 functions
////////////////////////////////////////////////////////////////////////////////
static __forceinline__ __host__ __device__ double4 make_double4(double a)
{
	return make_double4(a, a, a, a);
}

static __forceinline__ __host__ __device__ double4 make_double4(const float4& v)
{
	return make_double4(v.x, v.y, v.z, v.w);
}

static __forceinline__ __host__ __device__ double4 make_double4(const double3& v, double a)
{
	return make_double4(v.x, v.y, v.z, a);
}

// sum
static __forceinline__ __host__ __device__ double4 operator+(const double4 &a, const double4 &b)
{
	return make_double4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

static __forceinline__ __host__ __device__ void operator+=(double4 &a, const double4 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// subtract
static __forceinline__ __host__ __device__ double4 operator-(const double4 &a, const double4 &b)
{
	return make_double4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

// multiply
static __forceinline__ __host__ __device__ double4 operator*(const double4 &a, const double4 &b)
{
	return make_double4(a.x * b.x, a.y * b.y, a.z * b.z,  a.w * b.w);
}

static __forceinline__ __host__ __device__ double4 operator*(const double4 &a, const double &s)
{
	return make_double4(a.x * s, a.y * s, a.z * s, a.w * s);
}

static __forceinline__ __host__ __device__ double4 operator*(const double &s, const double4 &a)
{
	return make_double4(a.x * s, a.y * s, a.z * s, a.w * s);
}

// divide
static __forceinline__ __host__ __device__ double4 operator/(const double4 &a, const double &s)
{
	float inv = 1.0f / s;
	return a * inv;
}

// dot product
static __forceinline__ __host__ __device__ double dot(const double4 &a, const double4 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// dot product for double4 but act as if they were double3s
static __forceinline__ __host__ __device__ double dot3(const double4 &a, const double4 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// show a double4 as a double3
static __forceinline__ __host__ __device__ double3& as_double3(const double4 &v)
{
	return *(double3*)&v;
}

// squared length
static __forceinline__ __host__ __device__ double sqlength(const double4 &v)
{
	return dot(v, v);
}

// length for double4 but act as if they are double3s
static __forceinline__ __host__ __device__ float sqlength3(const double4 &v)
{
	return dot3(v, v);
}


// length
static __forceinline__ __host__ __device__ double length(const double4 &v)
{
	return sqrtf(sqlength(v));
}

// length for double4 but act as if they are double3s
static __forceinline__ __host__ __device__ float length3(const double4 &v)
{
	return sqrtf(sqlength3(v));
}

// length of the vector, computed more robustly when the components of v
// are significantly larger than unity
static __forceinline__ __host__ __device__ double hypot(const double4 &v)
{
	double p;
	p = fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w)));
	if (!p)
		return 0;

	double4 w=v/p;
	return p*length(w);
}


// float4 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
static __forceinline__ __host__ __device__ float4 make_float4(const float &s)
{
	return make_float4(s, s, s, s);
}

static __forceinline__ __host__ __device__ float4 make_float4(const float3 &a)
{
	return make_float4(a.x, a.y, a.z, 0.0f);
}

static __forceinline__ __host__ __device__ float4 make_float4(const float3 &a, const float &w)
{
	return make_float4(a.x, a.y, a.z, w);
}
static __forceinline__ __host__ __device__ float4 make_float4(const float2 &a, const float2 &b)
{
	return make_float4(a.x, a.y, b.x, b.y);
}

static __forceinline__ __host__ __device__ float4 make_float4(const int4 &a)
{
	return make_float4(float(a.x), float(a.y), float(a.z), float(a.w));
}

// show a float4 as a float3
static __forceinline__ __host__ __device__ float3& as_float3(const float4 &v)
{
	return *(float3*)&v;
}

// negate
static __forceinline__ __host__ __device__ float4 operator-(const float4 &a)
{
	return make_float4(-a.x, -a.y, -a.z, -a.w);
}

// min
static __forceinline__ __host__ __device__ float4 fminf(const float4 &a, const float4 &b)
{
	return make_float4(fminf(a.x,b.x), fminf(a.y,b.y), fminf(a.z,b.z), fminf(a.w,b.w));
}

// max
static __forceinline__ __host__ __device__ float4 fmaxf(const float4 &a, const float4 &b)
{
	return make_float4(fmaxf(a.x,b.x), fmaxf(a.y,b.y), fmaxf(a.z,b.z), fmaxf(a.w,b.w));
}

// addition
static __forceinline__ __host__ __device__ float4 operator+(const float4 &a, const float4 &b)
{
	return make_float4(a.x + b.x, a.y + b.y, a.z + b.z,  a.w + b.w);
}

static __forceinline__ __host__ __device__ void operator+=(float4 &a, const float4 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
}

// subtract
static __forceinline__ __host__ __device__ float4 operator-(const float4 &a, const float4 &b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  a.w - b.w);
}

// subtract
static __forceinline__ __host__ __device__ float4 operator-(const float3 &a, const float4 &b)
{
	return make_float4(a.x - b.x, a.y - b.y, a.z - b.z,  b.w);
}

static __forceinline__ __host__ __device__ void operator-=(float4 &a, const float4 &b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z; a.w -= b.w;
}

// multiply
static __forceinline__ __host__ __device__ float4 operator*(const float4 &a, const float &s)
{
	return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

static __forceinline__ __host__ __device__ float4 operator*(const float &s, const float4 &a)
{
	return make_float4(a.x * s, a.y * s, a.z * s, a.w * s);
}

static __forceinline__ __host__ __device__ void operator*=(float4 &a, const float &s)
{
	a.x *= s; a.y *= s; a.z *= s; a.w *= s;
}

// divide
static __forceinline__ __host__ __device__ float4 operator/(const float4 &a, const float4 &b)
{
	return make_float4(a.x / b.x, a.y / b.y, a.z / b.z, a.w / b.w);
}

static __forceinline__ __host__ __device__ float4 operator/(const float4 &a, const float &s)
{
	float inv = 1.0f / s;
	return a * inv;
}

static __forceinline__ __host__ __device__ float4 operator/(const float &s, const float4 &a)
{
	float inv = 1.0f / s;
	return a * inv;
}

static __forceinline__ __host__ __device__ void operator/=(float4 &a, float s)
{
	float inv = 1.0f / s;
	a *= inv;
}

// lerp
static __device__ __forceinline__ __host__ float4 lerp(const float4 &a, const float4 &b, const float &t)
{
	return a + t*(b-a);
}

// clamp
static __device__ __forceinline__ __host__ float4 clamp(const float4 &v, const float &a, const float &b)
{
	return make_float4(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b), clamp(v.w, a, b));
}

static __device__ __forceinline__ __host__ float4 clamp(const float4 &v, const float4 &a, const float4 &b)
{
	return make_float4(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z), clamp(v.w, a.w, b.w));
}

// dot product
static __forceinline__ __host__ __device__ float dot(const float4 &a, const float4 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// dot product for float4 but act as if they are float3s
static __forceinline__ __host__ __device__ float dot3(const float4 &a, const float4 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

// as above, but one is a float3
static __forceinline__ __host__ __device__ float dot3(const float3 &a, const float4 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
static __forceinline__ __host__ __device__ float dot3(const float4 &a, const float3 &b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}



// squared length
static __forceinline__ __host__ __device__ float sqlength(const float4 &v)
{
	return dot(v, v);
}

// squared length for float4 but act as if they are float3s
static __forceinline__ __host__ __device__ float sqlength3(const float4 &v)
{
	return dot3(v, v);
}

// length
static __forceinline__ __host__ __device__ float length(const float4 &v)
{
	return sqrtf(sqlength(v));
}

// length for float4 but act as if they are float3s
static __forceinline__ __host__ __device__ float length3(const float4 &v)
{
	return sqrtf(sqlength3(v));
}

// cross product
static __forceinline__ __host__ __device__ float4 cross3(const float4 &a, const float4 &b)
{
	return make_float4(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x, 0.0f);
}

// normalize
static __forceinline__ __host__ __device__ float4 normalize(const float4 &v)
{
	float invLen = rsqrtf(sqlength(v));
	return v * invLen;
}

// normalize for float4 but act as if they are float3s
static __forceinline__ __host__ __device__ float4 normalize3(const float4 &v)
{
	float invLen = rsqrtf(sqlength3(v));
	return make_float4(v.x*invLen, v.y*invLen, v.z*invLen, v.w);
}

// floor
static __forceinline__ __host__ __device__ float4 floor(const float4 &v)
{
	return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

// absolute value
static __forceinline__ __host__ __device__ float4 fabs(const float4 &v)
{
	return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}

// length of the vector, computed more robustly when the components of v
// are significantly larger than unity
static __forceinline__ __host__ __device__ float hypot(const float4 &v)
{
	float p;
	p = fmax(fmax(fabs(v.x), fabs(v.y)), fmax(fabs(v.z), fabs(v.w)));
	if (!p)
		return 0;

	float4 w=v/p;
	return p*length(w);
}

// length of the vector, computed more robustly when the components of v
// are significantly larger than unity
static __forceinline__ __host__ __device__ float hypot3(const float4 &v)
{
	float p;
	p = fmax(fmax(fabs(v.x), fabs(v.y)), fabs(v.z));
	if (!p)
		return 0;

	float3 w = make_float3(v.x/p, v.y/p, v.z/p);
	return p*length(w);
}


// char3 functions
////////////////////////////////////////////////////////////////////////////////
// multiply
static __forceinline__ __host__ __device__ float3 operator*(const char3 &a, const float3 &b)
{
	return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

// int3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
static __forceinline__ __host__ __device__ int3 make_int3(const int &s)
{
	return make_int3(s, s, s);
}

static __forceinline__ __host__ __device__ int3 make_int3(const int3 &s)
{
	return make_int3(s.x, s.y, s.z);
}

static __forceinline__ __host__ __device__ int3 make_int3(const float3 &a)
{
	return make_int3(int(a.x), int(a.y), int(a.z));
}

static __forceinline__ __host__ __device__ int3 make_int3(const double3 &a)
{
	return make_int3(int(a.x), int(a.y), int(a.z));
}

// negate
static __forceinline__ __host__ __device__ int3 operator-(const int3 &a)
{
	return make_int3(-a.x, -a.y, -a.z);
}

// min
static __forceinline__ __host__ __device__ int3 min(const int3 &a, const int3 &b)
{
	return make_int3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

// max
static __forceinline__ __host__ __device__ int3 max(const int3 &a, const int3 &b)
{
	return make_int3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// addition
static __forceinline__ __host__ __device__ int3 operator+(const int3 &a, const int3 &b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __host__ __device__ int3 operator+(const int3 &a, const char3 &b)
{
	return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __host__ __device__ void operator+=(int3 &a, const int3 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
static __forceinline__ __host__ __device__ int3 operator-(const int3 &a, const int3 &b)
{
	return make_int3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __host__ __device__ void operator-=(int3 &a, const int3 &b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
static __forceinline__ __host__ __device__ int3 operator*(const int3 &a, const int3 &b)
{
	return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __host__ __device__ int3 operator*(const int3 &a, const int &s)
{
	return make_int3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __host__ __device__ int3 operator*(const int &s, const int3 &a)
{
	return make_int3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __host__ __device__ void operator*=(int3 &a, const int &s)
{
	a.x *= s; a.y *= s; a.z *= s;
}

// divide
static __forceinline__ __host__ __device__ int3 operator/(const int3 &a, const int3 &b)
{
	return make_int3(a.x / b.x, a.y / b.y, a.z / b.z);
}

static __forceinline__ __host__ __device__ int3 operator/(const int3 &a, const int &s)
{
	return make_int3(a.x / s, a.y / s, a.z / s);
}

static __forceinline__ __host__ __device__ int3 operator/(const int &s, const int3 &a)
{
	return make_int3(a.x / s, a.y / s, a.z / s);
}
static __forceinline__ __host__ __device__ void operator/=(int3 &a, const int &s)
{
	a.x /= s; a.y /= s; a.z /= s;
}

// clamp
static __device__ __forceinline__ __host__ int clamp(const int &f, const int &a, const int &b)
{
	return max(a, min(f, b));
}

static __device__ __forceinline__ __host__ int3 clamp(const int3 &v, const int &a, const int &b)
{
	return make_int3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

static __device__ __forceinline__ __host__ int3 clamp(const int3 &v, const int3 &a, const int3 &b)
{
	return make_int3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

// uint3 functions
////////////////////////////////////////////////////////////////////////////////

// additional constructors
static __forceinline__ __host__ __device__ uint3 make_uint3(const uint &s)
{
	return make_uint3(s, s, s);
}

static __forceinline__ __host__ __device__ uint3 make_uint3(const float3 &a)
{
	return make_uint3(uint(a.x), uint(a.y), uint(a.z));
}

// min
static __forceinline__ __host__ __device__ uint3 min(const uint3 &a, const uint3 &b)
{
	return make_uint3(min(a.x,b.x), min(a.y,b.y), min(a.z,b.z));
}

// max
static __forceinline__ __host__ __device__ uint3 max(const uint3 &a, const uint3 &b)
{
	return make_uint3(max(a.x,b.x), max(a.y,b.y), max(a.z,b.z));
}

// addition
static __forceinline__ __host__ __device__ uint3 operator+(const uint3 &a, const uint3 &b)
{
	return make_uint3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __host__ __device__ void operator+=(uint3 &a, const uint3 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

static __forceinline__ __host__ __device__ void operator+=(uint3 &a, const int3 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

static __forceinline__ __host__ __device__ void operator+=(int3 &a, const uint3 &b)
{
	a.x += b.x; a.y += b.y; a.z += b.z;
}

// subtract
static __forceinline__ __host__ __device__ uint3 operator-(const uint3 &a, const uint3 &b)
{
	return make_uint3(a.x - b.x, a.y - b.y, a.z - b.z);
}

static __forceinline__ __host__ __device__ void operator-=(uint3 &a, const uint3 &b)
{
	a.x -= b.x; a.y -= b.y; a.z -= b.z;
}

// multiply
static __forceinline__ __host__ __device__ uint3 operator*(const uint3 &a, const uint3 &b)
{
	return make_uint3(a.x * b.x, a.y * b.y, a.z * b.z);
}

static __forceinline__ __host__ __device__ uint3 operator*(const uint3 &a, const uint &s)
{
	return make_uint3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __host__ __device__ uint3 operator*(const uint &s, const uint3 &a)
{
	return make_uint3(a.x * s, a.y * s, a.z * s);
}
static __forceinline__ __host__ __device__ void operator*=(uint3 &a, const uint &s)
{
	a.x *= s; a.y *= s; a.z *= s;
}

// divide
static __forceinline__ __host__ __device__ uint3 operator/(const uint3 &a, const uint3 &b)
{
	return make_uint3(a.x / b.x, a.y / b.y, a.z / b.z);
}

static __forceinline__ __host__ __device__ uint3 operator/(const uint3 &a, const uint &s)
{
	return make_uint3(a.x / s, a.y / s, a.z / s);
}

static __forceinline__ __host__ __device__ uint3 operator/(const uint &s, const uint3 &a)
{
	return make_uint3(a.x / s, a.y / s, a.z / s);
}

static __forceinline__ __host__ __device__ void operator/=(uint3 &a, const uint &s)
{
	a.x /= s; a.y /= s; a.z /= s;
}

// clamp
static __device__ __forceinline__ __host__ uint clamp(const uint &f, const uint &a, const uint &b)
{
	return max(a, min(f, b));
}

static __device__ __forceinline__ __host__ uint3 clamp(const uint3 &v, const uint &a, const uint &b)
{
	return make_uint3(clamp(v.x, a, b), clamp(v.y, a, b), clamp(v.z, a, b));
}

static __device__ __forceinline__ __host__ uint3 clamp(const uint3 &v, const uint3 &a, const uint3 &b)
{
	return make_uint3(clamp(v.x, a.x, b.x), clamp(v.y, a.y, b.y), clamp(v.z, a.z, b.z));
}

#endif
