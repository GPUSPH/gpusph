/*  Copyright (c) 2015-2018 INGV, EDF, UniCT, JHU

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
#include <limits.h> // UINT_MAX
#include <stdexcept>
#include <fstream>

#include "Reader.h"

Reader::Reader(void) :
	filename(),
	npart(SIZE_MAX),
	buf(NULL)
{}

Reader::~Reader(void) {
	empty();
}

void
Reader::empty()
{
	if (buf != NULL){
		delete [] buf;
		buf = NULL;
	}
}

void
Reader::reset()
{
	empty();
	filename = "";
	npart = UINT_MAX;
}

void
Reader::setFilename(std::string const& fn)
{
	// reset npart
	npart = UINT_MAX;
	// copy filename
	filename = fn;
	// check whether file exists
	std::ifstream f(filename.c_str());
	if(!f.good())
		throw std::invalid_argument(std::string("could not open ") + fn);
	f.close();
}
