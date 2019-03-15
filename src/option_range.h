/*  Copyright (c) 2018-2019 INGV, EDF, UniCT, JHU

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
 * Option range structure definition
 */

#ifndef _OPTION_RANGE_H
#define	_OPTION_RANGE_H

#include <stdexcept>
#include <istream>

/* For the string parsing */
#include <algorithm>
#include <cstring>
#include <stdexcept>
#ifdef _MSC_VER
#define DEMANGLE(type_id_name) strdup(type_id_name)
#else
#include <cxxabi.h>
#define DEMANGLE(type_id_name) abi::__cxa_demangle(type_id_name, NULL, 0, NULL)
#endif

#include "cpp11_missing.h"

//! 'Traits'-like structure to determine the first and last valid value for each option
/** This can be used to programmatically iterate over each possible value
 * of a given option. It also links the type to the array of names, which can be
 * leveraged when parsing e.g. user input.
 */

template<typename Option>
struct option_range
{
	static constexpr bool defined = false;
};

#define DEFINE_OPTION_RANGE(_option, _names,_min, _max) \
template<> \
struct option_range<_option> \
{ \
	static constexpr _option min = _min; \
	static constexpr _option max = _max; \
	static constexpr bool defined = true; \
	static constexpr auto names = _names; \
}

//! Check if an option has a valid range
template<typename Option>
constexpr
enable_if_t<option_range<Option>::defined, bool>
is_in_range(Option const& value)
{
	return value >= option_range<Option>::min && value <= option_range<Option>::max;
}

//! Check if an option has a valid range
/** When the option range is not defined, assume yes
 */
template<typename Option>
constexpr
enable_if_t<not option_range<Option>::defined, bool>
is_in_range(Option const& value)
{
	return true;
}

//! Workaround for buggy MSVC
template<typename Option, Option arg>
struct is_in_range_t
{
	static constexpr bool value = is_in_range(arg);
};

template<typename Option>
void throw_if_out_of_range(Option const& value)
{
	if (!is_in_range(value))
		throw std::out_of_range(std::to_string(value) + " not in "
			+ " [" + std::to_string(option_range<Option>::min)
			+ " " + std::to_string(option_range<Option>::max) + "]");
}

//! Parse a string as an option value
/*! This does a case-insenstive prefix comparison with the names for the
 * allowed values for the option, and throws if no match is found.
 *
 * \todo we should also throw if ambiguous (common prefix)
 * \todo we should allow multiple variants for string values
 */
template<typename Option>
Option parse_option_string(std::string const& val)
{
	const auto names = option_range<Option>::names;

	const char* str = val.c_str();
	const size_t sz = val.size();
	const auto from = names + option_range<Option>::min;
	const auto to = names + option_range<Option>::max + 1;
	const auto found = std::find_if(from, to, [&](const char* candidate) {
		return !strncasecmp(str, candidate, sz);
	});
	if (found < to)
		return Option(found - from);

	/* error out: use the implementation-specific DEMANGLE
	 * to get a user-visible name for the type */
	char * tname = DEMANGLE(typeid(Option).name());
	const std::string errmsg = val + " is not a valid " + tname;
	free(tname);

	throw std::invalid_argument(errmsg);
}


template<typename Option>
enable_if_t<option_range<Option>::defined, std::istream&>
operator>>(std::istream& in, Option& out)
{
	// Skip whitespace
	in >> std::ws;

	// Look at the next character
	int next = in.peek();

	// If it's a digit, assume the user is giving us a numeric value
	// between the minimum and maximum allowed value for the option
	if (isdigit(next)) {
		unsigned int v;
		in >> v;
		out = (Option)v;
		throw_if_out_of_range(out);
	} else {
		// parse as a string: compare the provided value
		std::string s;
		in >> s;
		out = parse_option_string<Option>(s);
	}
	return in;
}

#endif
