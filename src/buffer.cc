/*  Copyright 2018 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/*! \file
 * Implementation of methods for the Buffer (and related) classes
 */

#include <sstream>
#include <iomanip>

#include "buffer.h"

using namespace std;

string MultiBufferList::inspect() const
{
	stringstream _desc;
	size_t count;
	size_t key_width;
	flag_t max_key = 0;

	_desc << "MultiBufferList. ";
	_desc << "Registered buffer keys: ";

	count = 0;
	for (flag_t k : m_buffer_keys) {
		if (count > 0) _desc << ", ";
		if (k > max_key) max_key = k;
		_desc << k;
		++count;
	}
	_desc << "\n";

	key_width = to_string(max_key).size();

	for (auto const& sv : m_state) {
		_desc << "\tState " << sv.first << "\n";
		count = 0;
		for (auto const& pair : sv.second) {
			_desc << "\t\t\t";
			_desc << setw(key_width) << right << pair.first;
			_desc << "\t" + pair.second->inspect() << "\n";
			++count;
		}
	}

	_desc << "\tPool:\n";

	for (auto const& sv : m_pool) {
		_desc << "\t\t" << setw(key_width) << right << sv.first << ":\n";
		for (auto const& b : sv.second) {
			_desc << "\t\t\t" + b->inspect() << "\n";
			++count;
		}
	}

	return _desc.str();
}

