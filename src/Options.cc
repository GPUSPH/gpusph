/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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
 * Specializations of option extractors for the problem class
 */

#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include <thread>

#include "Options.h"
#include "backend_select.opt"

using namespace std;

vector<int> parse_devices_string(const char *argv)
{
	vector<int> ret;

	istringstream tokenizer(argv);

	string dev_string;
	while (getline(tokenizer, dev_string, ',')) {
		int first_dev = 0, last_dev = 0;
		int cvt = sscanf(dev_string.c_str(), "%u-%u", &first_dev, &last_dev);
		if (cvt == 0 /* no specification */ || first_dev < 0 || last_dev < 0) {
			throw invalid_argument("device specification " + dev_string + " is not a (range of) non-negative integers");
		} else if (cvt == 1) {
			ret.push_back(first_dev);
		} else {
			if (last_dev < first_dev) {
				std::swap(first_dev, last_dev);
			}
			while (first_dev <= last_dev) {
				ret.push_back(first_dev++);
			}
		}
	}
	return ret;
}

std::vector<int> get_default_devices()
{
	const char *env_spec = getenv("GPUSPH_DEVICE");
	if (!env_spec || !*env_spec)
		env_spec = "0";
	printf(" * No devices specified, falling back to default (%s)...\n", env_spec);
	return parse_devices_string(env_spec);
}

OptionMap::const_iterator
Options::find_option(string const& key) const
{
	OptionMap::const_iterator found(m_options.find(key));
	if (found != m_options.end()) return found;
	// not found: replace all _ with -
	string norm = key;
	string::size_type pos = 0;
	while ( (pos = norm.find("_", pos)) != string::npos ) {
		norm.replace(pos, 1, "-");
		++pos;
	}
	found = m_options.find(norm);
	if (found != m_options.end()) return found;
	// not found: replace all - with _
	pos = 0;
	while ( (pos = norm.find("-", pos)) != string::npos ) {
		norm.replace(pos, 1, "_");
		++pos;
	}
	return m_options.find(norm);
}

//! get a string value
template<>
string
Options::get(string const& key, string const& _default) const
{
	OptionMap::const_iterator found(find_option(key));
	if (found != m_options.end()) {
		return found->second;
	}
	if ((key == "dem") && !dem.empty())
		return dem;
	return _default;
}

//! values accepted to mean 'true'
static const string true_values[] = {
	"yes", "true", "1"
};
static const string *true_values_end = true_values + sizeof(true_values)/sizeof(*true_values);

//! check if a string value represents a true value
static bool is_true_value(string const& value)
{
	return find(true_values, true_values_end, value) != true_values_end;
}

//! values accepted to mean 'false'
static const string false_values[] = {
	"no", "false", "0"
};
static const string *false_values_end = false_values + sizeof(false_values)/sizeof(*false_values);

//! check if a string value represents a true value
static bool is_false_value(string const& value)
{
	return find(false_values, false_values_end, value) != false_values_end;
}


//! get a boolean value, throwing if the string representation is not a known boolean value
template<>
bool
Options::get(string const& key, bool const& _default) const
{
	OptionMap::const_iterator found(find_option(key));
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

void
Options::add_writers(const char *spec)
{
	istringstream tokenizer(spec);

	string writer_spec;
	while (getline(tokenizer, writer_spec, ',')) {
		auto semicolon = writer_spec.find(':');

		if (semicolon == string::npos)
			throw invalid_argument("writer specification " + writer_spec + " is not in the form WRITER:FREQ");

		auto writer = writer_spec.substr(0, semicolon);
		auto freq_str = writer_spec.substr(semicolon+1);

		WriterType wt = Writer::Type(writer);
		if (wt == WRITERTYPE_END)
			throw invalid_argument("unknown writer " + writer);

		size_t conv;
		double freq = stod(freq_str, &conv);
		if (conv != freq_str.size())
			throw invalid_argument("frequency specification " + freq_str + " is not a floating-point value");

		writers_extra.push_back(make_pair(wt, freq));
	}

}
