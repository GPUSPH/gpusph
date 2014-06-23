/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _VTKWRITER_H
#define	_VTKWRITER_H

#include "Writer.h"

using namespace std;

class VTKWriter : public Writer
{
public:
	VTKWriter(const GlobalData *_gdata);
	~VTKWriter();

	virtual void write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);
	virtual void write_WaveGage(double t, GageList const& gage);

private:
	// open a file whose name is built from the given base and sequence number
	// returns FILE object and stores the filename (without the dirname) into
	// `filename` if it's not NULL
	FILE *open_data_file(const char* base, string const& num, string *filename);
};

#endif	/* _VTKWRITER_H */
