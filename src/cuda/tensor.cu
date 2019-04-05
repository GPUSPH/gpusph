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
	T.xx = T.yy = T.zz = 1;
	T.xy = T.xz = T.yz = 0;
}

__spec
void
set_identity(symtensor4& T)
{
	T.xx = T.yy = T.zz = T.ww = 1;
	T.xy = T.xz = T.xw = T.yz = T.yw = T.zw = 0;
}

// determinant of a 3x3 symmetric tensor
__spec
float
det(symtensor3 const& T)
{
	float ret = 0;
	ret += T.xx*(T.yy*T.zz - T.yz*T.yz);
	ret -= T.xy*(T.xy*T.zz - T.xz*T.yz);
	ret += T.xz*(T.xy*T.yz - T.xz*T.yy);
	return ret;
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

__spec
symtensor3
inverse(symtensor3 const& T)
{
	symtensor3 R;
	float D(det(T));
	R.xx = (T.yy*T.zz - T.yz*T.yz)/D;
	R.xy = (T.xz*T.yz - T.xy*T.zz)/D;
	R.xz = (T.xy*T.yz - T.xz*T.yy)/D;
	R.yy = (T.xx*T.zz - T.xz*T.xz)/D;
	R.yz = (T.xz*T.xy - T.xx*T.yz)/D;
	R.zz = (T.xx*T.yy - T.xy*T.xy)/D;

	return R;
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

__spec
float3
dot(symtensor3 const& T, float3 const& v)
{
	return make_float3(
			T.xx*v.x + T.xy*v.y + T.xz*v.z,
			T.xy*v.y + T.yy*v.y + T.yz*v.z,
			T.xz*v.x + T.yz*v.y + T.zz*v.z);

}

__spec
float3
dot(symtensor3 const& T, float4 const& v)
{
	return make_float3(
			T.xx*v.x + T.xy*v.y + T.xz*v.z,
			T.xy*v.y + T.yy*v.y + T.yz*v.z,
			T.xz*v.x + T.yz*v.y + T.zz*v.z);

}

// T.v
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

/**** Methods for loading/storing tensors from textures and array ****/

#include "textures.cuh"

//! Fetch tau tensor from texture
/*!
 an auxiliary function that fetches the tau tensor
 for particle i from the textures where it's stored
*/
__device__
symtensor3 fetchTau(uint i)
{
	symtensor3 tau;
	float2 temp = tex1Dfetch(tau0Tex, i);
	tau.xx = temp.x;
	tau.xy = temp.y;
	temp = tex1Dfetch(tau1Tex, i);
	tau.xz = temp.x;
	tau.yy = temp.y;
	temp = tex1Dfetch(tau2Tex, i);
	tau.yz = temp.x;
	tau.zz = temp.y;
	return tau;
}

//! Fetch tau tensor from split arrays
/*!
 an auxiliary function that fetches the tau tensor
 for particle i from the arrays where it's stored
*/
__device__
symtensor3 fetchTau(uint i,
	const float2 *__restrict__ tau0,
	const float2 *__restrict__ tau1,
	const float2 *__restrict__ tau2)
{
	symtensor3 tau;
	float2 temp = tau0[i];
	tau.xx = temp.x;
	tau.xy = temp.y;
	temp = tau1[i];
	tau.xz = temp.x;
	tau.yy = temp.y;
	temp = tau2[i];
	tau.yz = temp.x;
	tau.zz = temp.y;
	return tau;
}

//! Store tau tensor to split arrays
__device__
void storeTau(symtensor3 const& tau, uint i,
	float2 *__restrict__ tau0,
	float2 *__restrict__ tau1,
	float2 *__restrict__ tau2)
{
	tau0[i] = make_float2(tau.xx, tau.xy);
	tau1[i] = make_float2(tau.xz, tau.yy);
	tau2[i] = make_float2(tau.yz, tau.zz);
}



#endif
