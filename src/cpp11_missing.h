/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/* Types, structures, operators and functions that are useful on host,
 * but missing in versions of C++ earlier than C++11
 */

#ifndef _CPP11_MISSING_H
#define _CPP11_MISSING_H

#if __cplusplus < 201103L

/* decltype is only avaialable on C++11, but both gcc and clang support
 * __decltype even on older standards, so use it
 */
#define decltype __decltype

/* conditional is trivial to implement pre-C++11.
 * It is used to choose either of two types depending on
 * whether a boolean condition is satisfied or not.
 */
template<bool B, typename T, typename F>
struct conditional { typedef T type; };

template<typename T, typename F>
struct conditional<false, T, F> { typedef F type; };

/* enable_if is also trivial to implement pre-C++11.
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

#include <type_traits>
using std::conditional;
using std::enable_if;

#endif

#endif
