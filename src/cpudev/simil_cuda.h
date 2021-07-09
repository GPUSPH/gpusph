/*  Copyright (c) 2021 INGV, EDF, UniCT, JHU

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

/*
 * \file Definitions normally found in the cuda_runtime[_api].h headers,
 * emulated for CPU
 */

#ifndef SIMIL_CUDA_H
#define SIMIL_CUDA_H

#include <cstdint>
#include <cstddef>
#include <algorithm> // max

#include "atomic_type.h"

// No memory space specification
#define __device__
#define __constant__
#define __host__
// forceinline
#define __forceinline__ inline __attribute__((always_inline))

#define __align__(num) __attribute__((aligned(num)))

// Data types

#define DEF_STRUCT(base, num, type, ...) \
	struct base##num { type __VA_ARGS__ ; }
#define DEF_ALIGN_STRUCT(base, num, type, ...) \
	struct __align__(num*sizeof(type)) base##num { type __VA_ARGS__ ; }

#define DEF_STRUCTS(base, type) \
	DEF_STRUCT(base, 1, type, x); \
	static __forceinline__ base##1 make_##base##1(type x) \
		{ base##1 ret ; ret.x = x ; return ret; } \
	DEF_ALIGN_STRUCT(base, 2, type, x, y); \
	static __forceinline__ base##2 make_##base##2(type x, type y) \
		{ base##2 ret ; ret.x = x ; ret.y = y ; return ret; } \
	DEF_STRUCT(base, 3, type, x, y, z); \
	static __forceinline__ base##3 make_##base##3(type x, type y, type z) \
		{ base##3 ret ; ret.x = x ; ret.y = y ; ret.z = z ; return ret; } \
	DEF_ALIGN_STRUCT(base, 4, type, x, y, z, w); \
	static __forceinline__ base##4 make_##base##4(type x, type y, type z, type w) \
		{ base##4 ret ; ret.x = x ; ret.y = y ; ret.z = z ; ret.w = w ; return ret; } \

DEF_STRUCTS(char, signed char)
DEF_STRUCTS(uchar, unsigned char)
DEF_STRUCTS(short, short)
DEF_STRUCTS(ushort, unsigned short)
DEF_STRUCTS(int, int32_t)
DEF_STRUCTS(uint, uint32_t)
DEF_STRUCTS(long, int64_t)
DEF_STRUCTS(ulong, uint64_t)

DEF_STRUCTS(float, float)
DEF_STRUCTS(double, double)

// CUDA device functionss

template<typename T>
T atomicCAS(ATOMIC_TYPE(T)* address, T compare, T val)
{
#ifdef _OPENMP
	return std::atomic_compare_exchange_strong(address, &compare, val);
#else
	// no need to actually be atomic
	const T old = *address;
	*address = old == compare ? val : old;
	return old;
#endif
}

template<typename T>
T atomicAdd(ATOMIC_TYPE(T)* address, T val)
{
#ifdef _OPENMP
	return address->fetch_add(val);
#else
	// no need to actually be atomic
	const T old = *address;
	*address = old + val;
	return old;
#endif
}

template<typename T>
T atomicMax(ATOMIC_TYPE(T)* address, T val)
{
#ifdef _OPENMP
	// C++11 doesn't have atomic max, so we simulate it
	// TODO FIXME verify
	T old = address->load();
	while (old < val && !address->compare_exchange_weak(old, val))
	{}

	return old;
#else
	// no need to actually be atomic
	const T old = *address;
	*address = std::max(old, val);
	return old;
#endif
}


#define __powf powf

#endif

