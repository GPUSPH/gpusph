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

/* Specializations of option extractors for the problem class */

#include <Options.h>

#include <stdexcept>
#include <algorithm>

using namespace std;

template<>
string
Options::get(string const& key, string const& _default) const
{
	OptionMap::const_iterator found(m_options.find(key));
	if (found != m_options.end()) {
		return found->second;
	}
	return _default;
}

static const string true_values[] = {
	"yes", "true", "1"
};
static const string *true_values_end = true_values + sizeof(true_values)/sizeof(*true_values);

static bool is_true_value(std::string const& value)
{
	return find(true_values, true_values_end, value) != true_values_end;
}

static const string false_values[] = {
	"no", "false", "0"
};
static const string *false_values_end = false_values + sizeof(false_values)/sizeof(*false_values);

static bool is_false_value(std::string const& value)
{
	return find(false_values, false_values_end, value) != false_values_end;
}


template<>
bool
Options::get(string const& key, bool const& _default) const
{
	OptionMap::const_iterator found(m_options.find(key));
	if (found != m_options.end()) {
		string const& value = found->second;
		if (is_true_value(value))
			return true;
		if (is_false_value(value))
			return false;
		stringstream error;
		error << "invalid boolean value '" << value << "' for key '" << key << "'";
		throw invalid_argument(error.str());
	}
	return _default;
}
