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

#include "XYZReader.h"
#include <fstream> // ifstream
#include <sstream> // stringstream
#include <limits> // UINT_MAX, numeric_limits

#include <stdexcept>

using namespace std;

XYZReader::XYZReader(void) {
	filename = "";
	npart = UINT_MAX;
}

XYZReader::~XYZReader() {
	empty();
}

int XYZReader::getNParts()
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

	std::string unused;
	while ( std::getline(xyzFile, unused) )
		++numLines;

	return numLines;
}

void XYZReader::read(Point *bbox_min, Point *bbox_max)
{
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

	while( !xyzFile.eof() ) {
		// read point coordinates
		xyzFile >> x >> y >> z;
		// ignore the rest of the line (e.g. vertex normals might be given)
		xyzFile.ignore ( std::numeric_limits<std::streamsize>::max(), '\n' );
		// we have a point, here
		Point p = Point(x,y,z);
		// update the bounding box ends, if given
		if (bbox_min) setMinPerElement(*bbox_min, p);
		if (bbox_max) setMaxPerElement(*bbox_max, p);
		// append to the list
		points.push_back(p);
    }

	xyzFile.close();
}

void XYZReader::empty()
{
	points.empty();
}

void XYZReader::reset()
{
	empty();
	filename = "";
	npart = UINT_MAX;
}

void XYZReader::setFilename(std::string const& fn)
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
