/*  Copyright (c) 2011-2018 INGV, EDF, UniCT, JHU

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
 * Types, structures, operators and functions that are useful on host
 * and device, but missing in earlier versions of C++.
 * This used to be about stuff missing before C++11, but it has grown
 * to include stuff that is missing in C++11, defined in C++14/C++17,
 * and that can be defined in C++11 too.
 */

#ifndef _CPP11_MISSING_H
#define _CPP11_MISSING_H

#include <type_traits>

#if __cplusplus < 201103L

/*! decltype is only avaialable on C++11, but both gcc and clang support
 * __decltype even on older standards, so use that (unless decltype was
 * already defined, that happens in certain versions of Clang on the Mac
 * OS X)
 */
#ifndef decltype
#define decltype __decltype
#endif

/*! conditional is trivial to implement pre-C++11.
 * It is used to choose either of two types depending on
 * whether a boolean condition is satisfied or not.
 */
template<bool B, typename T, typename F>
struct conditional { typedef T type; };

template<typename T, typename F>
struct conditional<false, T, F> { typedef F type; };

/*! enable_if is also trivial to implement pre-C++11.
 * It is used as std::enable_if<some_condition, some_type>::type
 * to restrict a function template to the cases where some_condition
 * (generally assembled from the function template parameters) is satisfied.
 */
template<bool B, typename T=void>
struct enable_if {};

template<typename T>
struct enable_if<true, T>
{ typedef T type; };

#else

using std::conditional;
using std::enable_if;

#endif

#if __cplusplus < 201402L

template<bool B, typename T=void>
using enable_if_t = typename enable_if<B, T>::type;

#else
using std::enable_if_t;

#endif

#if __cplusplus >= 201703L
using std::void_t;
#else
template<typename ...>
using void_t = void;
#endif

#if __cplusplus >= 201703L
using std::bool_constant;
#else
template<bool B>
using bool_constant = std::integral_constant<bool, B>;
#endif

#if __cplusplus >= 201703L || _MSC_VER > 1400
using std::as_const;
#else
template <class T>
constexpr typename std::add_const<T>::type& as_const(T& t) noexcept
{
	return t;
}
template <class T>
void as_const(const T&&) = delete;
#endif



#endif
