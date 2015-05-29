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

struct Options {
	std::string	problem; // problem name
	std::string	resume_fname; // file to resume simulation from
	int		device;  // which device to use
	std::string	dem; // DEM file to use
	std::string	dir; // directory where data will be saved
	double	deltap; // deltap
	float	tend; // simulation end
	float	checkpoint_freq; // frequency of hotstart checkpoints (in simulated seconds)
	int	checkpoints; // number of hotstart checkpoints to keep
	bool	nosave; // disable saving
	bool	gpudirect; // enable GPUDirect
	bool	striping; // enable striping (i.e. compute/transfer overlap)
	bool	asyncNetworkTransfers; // enable asynchronous network transfers
	unsigned int num_hosts; // number of physical hosts to which the processes are being assigned
	bool byslot_scheduling; // by slot scheduling across MPI nodes (not round robin)
	Options(void) :
		problem(),
		resume_fname(),
		device(-1),
		dem(),
		dir(),
		deltap(NAN),
		checkpoint_freq(NAN),
		checkpoints(-1),
		tend(NAN),
		nosave(false),
		gpudirect(false),
		striping(false),
		asyncNetworkTransfers(false),
		num_hosts(0),
		byslot_scheduling(false)
	{};
};

#endif
