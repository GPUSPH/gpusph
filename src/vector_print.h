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

/* Definition of the << operators for vector types */

#ifndef _VECTOR_PRINT_H
#define _VECTOR_PRINT_H

#include <ostream>
#include <iterator> // ostream_iterator
#include <algorithm> // copy

#include "vector_traits.h"
#include "cpp11_missing.h" // enable_if

/*! overload of ostream operator << for vector types.
 *
 * Note the use of enable_if to only enable this override
 * when V is a vector type (i.e. vector_traits<V>::components is larger than 0),
 * and the use of ostream_iterator to output all components without
 * knowing how many there are.
 */
template<typename V>
typename enable_if<
	(vector_traits<V>::components > 0),
	std::ostream
>::type&
operator<<
(std::ostream& out, V const& val)
{
	typedef vector_traits<V> traits;
	typedef typename traits::component_type T;
	const int N = traits::components;

	// We want to print all components, separated by a specific separator
	// (comma-space aka ", "), regardless of how many there are and how they
	// are named, so we use iterators, traversing our vector type V as if it
	// was an array of N elements of type T. Of course in our case the iterators
	// are just pointers.
	// So, for example, if V is float4, T will be float,
	// begin will be a pointer to the first component of val (.x), and
	// end will be a pointer to the last component (.w).
	const T* begin = (T*)&val;
	const T* end = begin + N - 1;

	// output the opening parenthesis
	out << "(";
	// output all components _except for the last_, appending
	// the separator after each one. Note that end is exclusive, which is what
	// we want, because we don't want to append the separator to that. Also we
	// can skip all this if N == 1
	if (N > 1)
		std::copy(begin, end, std::ostream_iterator<T>(out, ", "));
	// append the last component and the closing parenthesis
	out << *end << ")";
	return out;
}

#endif

