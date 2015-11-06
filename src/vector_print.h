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

/* Output streaming operator for vector types
 * TODO use std::basic_ostream<CharT, Traits> in place of std::ostream
 * for genericity
 */

#ifndef _VECTOR_PRINT_H
#define _VECTOR_PRINT_H

#include <ostream>
#include <iomanip>

#include "vector_traits.h"
#include "cpp11_missing.h" // enable_if

/*! A vector type is streamed out in the form:
 * opening component0 separator component1 separator ... closing
 * where opening, separator and closing may be any string, and by default
 * they are "(", ", " and ")". We want to be able to override any of them,
 * which will be done via manipulators (see below). To simplify the code,
 * we define a struct that holds the three strings.
 */

struct vector_fmt_options {
	const std::string opening;
	const std::string separator;
	const std::string closing;

	// index of the vector format options in the pword hash:
	// used to retrieve/set the current vector format options
	// of a stream
	static const int idx;
};

// Sending a vector_fmt_options to an ostream sets the vector format options
inline
std::ostream&
operator<<(std::ostream& out, vector_fmt_options const& fmt)
{
	// for safe memory management we store a copy of the given options,
	// and delete the preceding one, if any
	delete static_cast<vector_fmt_options*>(out.pword(vector_fmt_options::idx));
	out.pword(vector_fmt_options::idx) = new vector_fmt_options(fmt);
	return out;
}

/// The default options
extern const vector_fmt_options reset_vector_fmt;

/// I/O manipulator to set opening, separator and closing vector format options.
///     cout << set_vector_fmt("{", ", ", "}") << somefloat4 << reset_vector_fmt << endl;
/// will output `somefloat4` in Mathematica form
inline vector_fmt_options set_vector_fmt(const char *open, const char *sep, const char *close)
{
	vector_fmt_options fmt = { open, sep, close };
	return fmt;
}

/// I/O manipulator to set the separator only and clear the opening and closing
inline vector_fmt_options set_vector_fmt(const char *sep)
{
	vector_fmt_options fmt = { "", sep, "" };
	return fmt;
}

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

	const vector_fmt_options *stream_fmt =
		static_cast<const vector_fmt_options *>(out.pword(vector_fmt_options::idx));
	const vector_fmt_options *fmt = stream_fmt ? stream_fmt : &reset_vector_fmt;

	// If the user set the width, (e.g. cout << setw(10) << somefloat4 << endl)
	// we want to apply it to each component, and just those (i.e. not the
	// opening/closing/separator strings). Since stream output operators reset
	// the width, we store it and apply it to the parts we care about.
	// TODO for efficiency, it would be better to manipulate the streambuf
	// directly
	const std::streamsize width = out.width();

	out.width(0);

	// output the opening parenthesis
	out << fmt->opening;

	// We want to print all components, separated by a specific separator,
	// regardless of how many there are and how they are named, so we traverse
	// our vector type V as if it was an array of N elements of type T.
	// So we get a pointer of type T to the first component of val
	// and iterate over all components except for the last
	const T* const ptr = (T*)&val;

#pragma unroll
	for(int i=0 ; i < N - 1; ++i)
		out << std::setw(width) << ptr[i] << fmt->separator;

	// append the last component and the closing parenthesis
	out << std::setw(width) << ptr[N-1] << fmt->closing;

}

#endif

