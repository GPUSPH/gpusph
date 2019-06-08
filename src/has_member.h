/*  Copyright (c) 2018 INGV, EDF, UniCT, JHU

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
 *
 * \brief Member detector macros.
 *
 * From time to time we need to specialize our function templates
 * based on presence or absence of a particular member
 * (for example, fetch the velocity array from velArray if present,
 * or using the texture otherwise).
 *
 * We can achieve this using SFINAE to define a structure template
 * that maps to std::false_type or std::true_type depending on whether the
 * member is present or not.
 *
 * To simplify creation of new detectors, we wrap the creation of the
 * structures in a macro.
 */

#ifndef HAS_MEMBER_H
#define HAS_MEMBER_H

#include "cpp11_missing.h"

#define DECLARE_MEMBER_DETECTOR(member, detector) \
/* Default: false */ \
template<typename T, typename = std::false_type > struct _##detector : std::false_type {}; \
/* Specialization that due to SFINAE will only be triggered when \
 * FP::member is a valid expression */ \
template<typename T> struct _##detector<T, bool_constant<!sizeof(decltype(T::member))>> : std::true_type {}; \
/* Constexpr function that returns true if the given type has the given member, \
 * false otherwise. Both a no-argument version, and one that derives the type \
 * from the argument, are provided. \
 */ \
template<typename T, bool ret = _##detector<T>::value> \
__host__ __device__ __forceinline__ \
constexpr bool detector() { return ret; } \
template<typename T, bool ret = _##detector<T>::value> \
__host__ __device__ __forceinline__ \
constexpr bool detector(T const&) { return ret; }

/*! Example usage:

   DECLARE_MEMBER_DETECTOR(velArray, has_velArray)

   template<FP> enable_if_t<has_velArray<FP>(), float4> func() { ... }
   template<FP> enable_if_t<!has_velArray<FP>(), float4> func() { ... }

 */


#endif
