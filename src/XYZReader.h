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

#include <string>
#include "Point.h"

class XYZReader {
private:

	std::string		filename;
	unsigned int	npart;

public:
	// constructor
	XYZReader();
	~XYZReader();

	// returns the number of particles in the XYZ file
	int getNParts();

	// allocates the buffer and reads the data from the XYZ file; optionally, returns bbox
	void read(Point *bbox_min = NULL, Point *bbox_max = NULL);

	// counts the points in the file, without allocating nor loading anything
	uint count();

	// frees the buffer
	void empty();

	// free the buffer, reset npart and filename
	void reset();

	// sets the filename
	void setFilename(std::string const&);

	PointVect points;
};