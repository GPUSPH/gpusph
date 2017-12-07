/*  Copyright 2015 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

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

/* Vector type traits, to allow generic programming for (specific) vector types only */

#ifndef _VECTOR_TRAITS_H
#define _VECTOR_TRAITS_H

#include "common_types.h"

/// The vector_traits structure aggregates information about
/// vector types (e.g. float4, int3), such as the type and number
/// of components.
/// It can be used in conjuction with something like enable_if
/// to e.g. limit a template function to only apply to vector types
template<typename V>
struct vector_traits
{
	typedef V component_type;
	enum { components = 0 };
};

#define TRAITS(T, N) \
	template<> \
	struct vector_traits<T##N> \
	{ \
		typedef T component_type; \
		enum { components = N }; \
	}

#define DEFINE_TRAITS(T) \
	TRAITS(T, 1); \
	TRAITS(T, 2); \
	TRAITS(T, 3); \
	TRAITS(T, 4)

DEFINE_TRAITS(char);
DEFINE_TRAITS(uchar);
DEFINE_TRAITS(short);
DEFINE_TRAITS(ushort);
DEFINE_TRAITS(int);
DEFINE_TRAITS(uint);
DEFINE_TRAITS(long);
DEFINE_TRAITS(ulong);

DEFINE_TRAITS(float);
DEFINE_TRAITS(double);

#endif

