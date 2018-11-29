/*  Copyright 2018 Giuseppe Bilotta, Alexis Herault, Robert A.
 	Dalrymple, Eugenio Rustico, Ciro Del Negro

	Conservatoire National des Arts et Metiers, Paris, France

	Istituto Nazionale di Geofisica e Vulcanologia,
    Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

/*! \file
 * Option range structure definition
 */

#ifndef _OPTION_RANGE_H
#define	_OPTION_RANGE_H

#include <stdexcept>
#include <istream>

#include "cpp11_missing.h"

//! 'Traits'-like structure to determine the first and last valid value for each option
/** This can be used to programmatically iterate over each possible value
 * of a given option
 */

template<typename Option>
struct option_range
{
	static constexpr Option min = Option();
	static constexpr Option max = Option();
	static constexpr bool defined = false;
};

#define DEFINE_OPTION_RANGE(_option, _min, _max) \
template<> \
struct option_range<_option> \
{ \
	static constexpr _option min = _min; \
	static constexpr _option max = _max; \
	static constexpr bool defined = true; \
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

template<typename Option>
void throw_if_out_of_range(Option const& value)
{
	if (!is_in_range(value))
		throw std::out_of_range(std::to_string(value) + " not in "
			+ " [" + std::to_string(option_range<Option>::min)
			+ " " + std::to_string(option_range<Option>::max) + "]");
}

template<typename Option>
enable_if_t<option_range<Option>::defined, std::istream&>
operator>>(std::istream& in, Option& out)
{
	unsigned int v;
	in >> v;
	out = (Option)v;
	throw_if_out_of_range(out);
	return in;
}

#endif
