/*  Copyright (c) 2013-2019 INGV, EDF, UniCT, JHU

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
#ifndef TENSOR_IMPL
#define TENSOR_IMPL

#include "tensor.h"
#include "vector_math.h"
#include "kahan.h"

#define __spec __device__ __forceinline__

__spec
void
clear(symtensor3& T)
{
	T.xx = T.xy = T.xz =
		T.yy = T.yz = T.zz = 0.0f;
}

__spec
void
clear(symtensor4& T)
{
	T.xx = T.xy = T.xz = T.xw =
		T.yy = T.yz = T.yw =
		T.zz = T.zw = T.ww = 0.0f;
}

__spec
void
set_identity(symtensor3& T)
{
	T.xx = T.yy = T.zz = 1.0f;
	T.xy = T.xz = T.yz = 0.0f;
}

__spec
symtensor3
identity3(void)
{
	symtensor3 T;

	T.xx = T.yy = T.zz = 1.0f;
	T.xy = T.xz = T.yz = 0.0f;

	return T;
}

__spec
void
set_identity(symtensor4& T)
{
	T.xx = T.yy = T.zz = T.ww = 1;
	T.xy = T.xz = T.xw = T.yz = T.yw = T.zw = 0;
}

// determinant of a 3x3 symmetric tensor
template<Dimensionality dimensions = R3, int dims = space_dimensions_for(dimensions)>
__spec
float
det(symtensor3 const& T)
{
	if (dims == 2)
	{
		return(T.xx*T.yy - T.xy*T.xy);
	}

	float ret = 0;
	ret += T.xx*(T.yy*T.zz - T.yz*T.yz);
	ret -= T.xy*(T.xy*T.zz - T.xz*T.yz);
	ret += T.xz*(T.xy*T.yz - T.xz*T.yy);
	return ret;
}

template<Dimensionality dimensions = R3, int dims = space_dimensions_for(dimensions)>
__spec
float
kbn_det(symtensor3 const& T)
{
	if (dims == 2)
	{
		return(T.xx*T.yy - T.xy*T.xy);
	}

	float ret = 0, kahan = 0;
	ret = kbn_add(ret,  T.xx*(T.yy*T.zz - T.yz*T.yz), kahan);
	ret = kbn_add(ret, -T.xy*(T.xy*T.zz - T.xz*T.yz), kahan);
	ret = kbn_add(ret,  T.xz*(T.xy*T.yz - T.xz*T.yy), kahan);
	return ret + kahan;
}

// determinant of a 4x4 symmetric tensor
__spec
float
det(symtensor4 const& T)
{
	float ret = 0;

	// first minor: ww * (xyz × xyz)
	float M = 0;
	M += T.xx*(T.yy*T.zz - T.yz*T.yz);
	M -= T.xy*(T.xy*T.zz - T.xz*T.yz);
	M += T.xz*(T.xy*T.yz - T.xz*T.yy);
	ret += M*T.ww;

	// second minor: -zw * (xyz × xyw)
	M = 0;
	M += T.xx*(T.yy*T.zw - T.yz*T.yw);
	M -= T.xy*(T.xy*T.zw - T.xz*T.yw);
	M += T.xw*(T.xy*T.yz - T.xz*T.yy);
	ret -= M*T.zw;

	// third minor: yw * (xyz × xzw)
	M = 0;
	M += T.xx*(T.yz*T.zw - T.zz*T.yw);
	M -= T.xz*(T.xy*T.zw - T.xz*T.yw);
	M += T.xw*(T.xy*T.zz - T.xz*T.yz);
	ret += M*T.yw;

	// last minor: xw * (xyz × yzw)
	M = 0;
	M += T.xy*(T.yz*T.zw - T.zz*T.yw);
	M -= T.xz*(T.yy*T.zw - T.yz*T.yw);
	M += T.xw*(T.yy*T.zz - T.yz*T.yz);
	ret -= M*T.xw;

	return ret;
}

// L-infinity norm of a symmetric 4x4 tensor
__spec
float
norm_inf(symtensor4 const& T)
{
	float m = fmaxf(T.xx, T.xy);
	m = fmaxf(m, T.xz);
	m = fmaxf(m, T.xw);
	m = fmaxf(m, T.yy);
	m = fmaxf(m, T.yz);
	m = fmaxf(m, T.yw);
	m = fmaxf(m, T.zz);
	m = fmaxf(m, T.zw);
	m = fmaxf(m, T.ww);
	return m;
}

// compute inverse of tensor when the determinant has been computed already
template<Dimensionality dimensions = R3, int dims = space_dimensions_for(dimensions)>
__spec
symtensor3
inverse(symtensor3 const& T, const float D)
{
	symtensor3 R;

	if (dims == 2)
	{
		R.xx = T.yy/D;
		R.xy = - T.xy/D;
		R.yy = T.xx/D;
		R.xz = 0.0f;
		R.yz = 0.0f;
		R.zz = 1.0f;
		return R;
	}

	R.xx = (T.yy*T.zz - T.yz*T.yz)/D;
	R.xy = (T.xz*T.yz - T.xy*T.zz)/D;
	R.xz = (T.xy*T.yz - T.xz*T.yy)/D;
	R.yy = (T.xx*T.zz - T.xz*T.xz)/D;
	R.yz = (T.xz*T.xy - T.xx*T.yz)/D;
	R.zz = (T.xx*T.yy - T.xy*T.xy)/D;

	return R;
}

// compute inverse of a tensor when the determinant has not been computed already
template<Dimensionality dimensions = R3>
__spec
symtensor3
inverse(symtensor3 const& T)
{
	return inverse<dimensions>(T, det<dimensions>(T));
}

__spec
symtensor3
operator -(symtensor3 const& T1, symtensor3 const& T2)
{
	symtensor3 R;
	R.xx = T1.xx - T2.xx;
	R.xy = T1.xy - T2.xy;
	R.xz = T1.xz - T2.xz;
	R.yy = T1.yy - T2.yy;
	R.yz = T1.yz - T2.yz;
	R.zz = T1.zz - T2.zz;
	return R;
}

__spec
symtensor3
operator +(symtensor3 const& T1, symtensor3 const& T2)
{
	symtensor3 R;
	R.xx = T1.xx + T2.xx;
	R.xy = T1.xy + T2.xy;
	R.xz = T1.xz + T2.xz;
	R.yy = T1.yy + T2.yy;
	R.yz = T1.yz + T2.yz;
	R.zz = T1.zz + T2.zz;
	return R;
}

__spec
symtensor3 &
operator -=(symtensor3 &T1, symtensor3 const& T2)
{
	T1.xx -= T2.xx;
	T1.xy -= T2.xy;
	T1.xz -= T2.xz;
	T1.yy -= T2.yy;
	T1.yz -= T2.yz;
	T1.zz -= T2.zz;
	return T1;
}

__spec
symtensor3 &
operator +=(symtensor3 &T1, symtensor3 const& T2)
{
	T1.xx += T2.xx;
	T1.xy += T2.xy;
	T1.xz += T2.xz;
	T1.yy += T2.yy;
	T1.yz += T2.yz;
	T1.zz += T2.zz;
	return T1;
}

__spec
symtensor3
operator *(symtensor3 const& T1, float f)
{
	symtensor3 R;
	R.xx = T1.xx*f;
	R.xy = T1.xy*f;
	R.xz = T1.xz*f;
	R.yy = T1.yy*f;
	R.yz = T1.yz*f;
	R.zz = T1.zz*f;
	return R;
}

__spec
symtensor3
operator /(symtensor3 const& T1, float f)
{
	symtensor3 R;
	R.xx = T1.xx/f;
	R.xy = T1.xy/f;
	R.xz = T1.xz/f;
	R.yy = T1.yy/f;
	R.yz = T1.yz/f;
	R.zz = T1.zz/f;
	return R;
}


__spec
symtensor3 &
operator /=(symtensor3 &T1, float f)
{
	T1.xx /= f;
	T1.xy /= f;
	T1.xz /= f;
	T1.yy /= f;
	T1.yz /= f;
	T1.zz /= f;
	return T1;
}

// T.v for 3x3 symmetric tensors
template<typename V> // V should be float3 or float4
__spec
float3
dot(symtensor3 const& T, V const& v)
{
	return make_float3(
			T.xx*v.x + T.xy*v.y + T.xz*v.z,
			T.xy*v.x + T.yy*v.y + T.yz*v.z,
			T.xz*v.x + T.yz*v.y + T.zz*v.z);

}

// T.v for 3x3 symmetric tensors, computed using KBN summation
template<typename V>
__spec
float3
kbn_dot(symtensor3 const& T, V const& v)
{
	return make_float3(
			kbn_sum(T.xx*v.x, T.xy*v.y, T.xz*v.z),
			kbn_sum(T.xy*v.x, T.yy*v.y, T.yz*v.z),
			kbn_sum(T.xz*v.x, T.yz*v.y, T.zz*v.z));

}

// v.T.v
template<typename V> // V should be float3 or float4
__spec
float
ddot(symtensor3 const& T, V const& v)
{
	return T.xx*v.x*v.x + T.yy*v.y*v.y + T.zz*v.z*v.z +
		2*(	T.xy*v.x*v.y +
			T.yz*v.y*v.z +
			T.xz*v.x*v.z);
}

// v.T.v computed with KBN
template<typename V> // V should be float3 or float4
__spec
float
kbn_ddot(symtensor3 const& T, V const& v)
{
	return kbn_sum(T.xx*v.x*v.x, T.yy*v.y*v.y, T.zz*v.z*v.z) +
		2*kbn_sum(	T.xy*v.x*v.y,
				T.yz*v.y*v.z,
				T.xz*v.x*v.z);
}


// T.v for 4x4 symmetric tensots
__spec
float4
dot(symtensor4 const& T, float4 const& v)
{
	return make_float4(
			T.xx*v.x + T.xy*v.y + T.xz*v.z + T.xw*v.w,
			T.xy*v.x + T.yy*v.y + T.yz*v.z + T.yw*v.w,
			T.xz*v.x + T.yz*v.y + T.zz*v.z + T.zw*v.w,
			T.xw*v.x + T.yw*v.y + T.zw*v.z + T.ww*v.w);

}

__spec
symtensor3
point_product(symtensor3 const& T1, symtensor3 const& T2)
{
	symtensor3 R;
	R.xx = T1.xx * T2.xx;
	R.xy = T1.xy * T2.xy;
	R.xz = T1.xz * T2.xz;
	R.yy = T1.yy * T2.yy;
	R.yz = T1.yz * T2.yz;
	R.zz = T1.zz * T2.zz;
	return R;
}

__spec
symtensor3
point_sqrt(symtensor3 const& T)
{
	symtensor3 R;
	R.xx = sqrt(abs(T.xx));
	R.xy = sqrt(abs(T.xy));
	R.xz = sqrt(abs(T.xz));
	R.yy = sqrt(abs(T.yy));
	R.yz = sqrt(abs(T.yz));
	R.zz = sqrt(abs(T.zz));
	return R;
}

// v.T.w
__spec
float
dot(float4 const& v, symtensor4 const& T, float4 const& w)
{
	return dot(v, dot(T,w));
}

// v.T.v
__spec
float
ddot(symtensor4 const& T, float4 const& v)
{
	return T.xx*v.x*v.x + T.yy*v.y*v.y + T.zz*v.z*v.z + T.ww*v.w*v.w +
		2*(
			(T.xy*v.y + T.xw*v.w)*v.x +
			(T.yz*v.z + T.yw*v.w)*v.y +
			(T.xz*v.x + T.zw*v.w)*v.z);
}

// First row of the adjugate of a given matrix
__spec
float4
adjugate_row1(symtensor4 const& T)
{
	return make_float4(
		T.yy*T.zz*T.ww + T.yz*T.zw*T.yw + T.yw*T.yz*T.zw - T.yy*T.zw*T.zw - T.yz*T.yz*T.ww - T.yw*T.zz*T.yw,
		T.xy*T.zw*T.zw + T.yz*T.xz*T.ww + T.yw*T.zz*T.xw - T.xy*T.zz*T.ww - T.yz*T.zw*T.xw - T.yw*T.xz*T.zw,
		T.xy*T.yz*T.ww + T.yy*T.zw*T.xw + T.yw*T.xz*T.yw - T.xy*T.zw*T.yw - T.yy*T.xz*T.ww - T.yw*T.yz*T.xw,
		T.xy*T.zz*T.yw + T.yy*T.xz*T.zw + T.yz*T.yz*T.xw - T.xy*T.yz*T.zw - T.yy*T.zz*T.xw - T.yz*T.xz*T.yw);
}

#undef __spec

#endif
