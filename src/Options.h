/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
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

#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include <cmath>
#include <string>

using namespace std;

struct Options {
	string	problem; // problem name
	int		device;  // which device to use
	string	dem; // DEM file to use
	string	dir; // directory where data will be saved
	bool	console; // run in console (no GUI)
	double	deltap; // deltap
	float	tend; // simulation end
	Options(void) :
		problem(),
		device(-1),
		dem(),
		dir(),
		console(false),
		deltap(NAN),
		tend(NAN)
	{};
};

#endif
