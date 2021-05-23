/*  Copyright (c) 2011-2018 INGV, EDF, UniCT, JHU

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

#include "XYZReader.h"
#include <fstream> // ifstream
#include <sstream> // stringstream
#include <limits> // UINT_MAX, numeric_limits

#include <stdexcept>

using namespace std;

size_t XYZReader::getNParts()
{
	// if npart != UINT_MAX, file was already opened (for loading or counting only)
	if (npart != UINT_MAX)
		return npart;

	// otherwise, we just count the lines
	ifstream xyzFile(filename.c_str());

	// basic I/O check
	if (!xyzFile.good()) {
		stringstream err_msg;
		err_msg	<< "failed to open XYZ file " << filename;
		throw runtime_error(err_msg.str());
	}

	unsigned int numLines = 0;

	string unused;
	while ( getline(xyzFile, unused) )
		++numLines;

	npart = numLines; // cache for future usage
	return numLines;
}

void XYZReader::read()
{
	read(NULL, NULL);
}

void XYZReader::read(Point *bbox_min, Point *bbox_max)
{
	// read npart if it was yet uninitialized
	if (npart == UINT_MAX)
		getNParts();

	cout << "Reading particle data from the input: " << filename << endl;

	// allocating read buffer
	if(buf == NULL)
		buf = new ReadParticles[npart];
	else{
		delete [] buf;
		buf = new ReadParticles[npart];
	}

	ifstream xyzFile(filename.c_str());

	// basic I/O check
	if (!xyzFile.good()) {
		stringstream err_msg;
		err_msg	<< "failed to open XYZ " << filename;
		throw runtime_error(err_msg.str());
	}

	// reset the bounding box
	// NOTE: using NAN instead of DBL_MAX/-DBL_MAX to leave a "correct"
	// bbox in case the file contains no points
	if (bbox_min) *bbox_min = Point(NAN, NAN, NAN);
	if (bbox_max) *bbox_max = Point(NAN, NAN, NAN);

	double x, y, z;
	ReadParticles *part = buf;
	while( !xyzFile.eof() ) {
		// read point coordinates
		xyzFile >> x >> y >> z;

		part->Coords_0 = x;
		part->Coords_1 = y;
		part->Coords_2 = z;

		// ignore the rest of the line (e.g. vertex normals might be given)
		// TODO read normals if present
		xyzFile.ignore ( numeric_limits<streamsize>::max(), '\n' );

		// update the bounding box ends, if given
		Point p(x, y, z);
		if (bbox_min) setMinPerElement(*bbox_min, p);
		if (bbox_max) setMaxPerElement(*bbox_max, p);

		++part;

    }

	xyzFile.close();
}
