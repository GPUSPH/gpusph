/*  Copyright (c) 2013-2019 INGV, EDF, UniCT, JHU

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
/*
  This file has been extracted from "Sphynx" code.
  Originally developed by Arno Mayrhofer (2013), Christophe Kassiotis (2013), Martin Ferrand (2013).
  It contains a class for reading *.h5sph files - input files in hdf5 format.
*/

#ifndef _HDF5SPHREADER_H
#define _HDF5SPHREADER_H

#include <string>
#include <iostream>

#include "Reader.h"

class HDF5SphReader : public Reader
{
public:
	// returns the number of particles in the h5sph file
	size_t getNParts(void) override;

	// allocates the buffer and reads the data from the h5sph file
	void read(void) override;
};

#endif
