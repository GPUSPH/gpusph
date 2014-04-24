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
	double	deltap; // deltap
	float	tend; // simulation end
	bool	nosave; // disable saving
	bool	striping; // enable striping (i.e. compute/transfer overlap)
	unsigned int num_hosts; // number of physical hosts to which the processes are being assigned
	bool byslot_scheduling; // by slot scheduling across MPI nodes (not round robin)
	Options(void) :
		problem(),
		device(-1),
		dem(),
		dir(),
		deltap(NAN),
		tend(NAN),
		nosave(false),
		striping(false),
		num_hosts(0),
		byslot_scheduling(false)
	{};
};

#endif
