/*  Copyright (c) 2014-2019 INGV, EDF, UniCT, JHU

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
#ifndef H_HOTWRITER_H
#define H_HOTWRITER_H

#include "Writer.h"
#include "HotFile.h"

/**
A Writer class used for hot-starting a simulation.
Hot-starting means restarting a stopped simulation from a known point.  The
advantages include restarting a job that may have stopped due to program
crash or intentionally stopped by the user, etc.

This writer will save a user-defined number of state files, from which the
simulation may be later restarted.  Note, *only* this number of files will be
retained, all previous state files are removed as the simulation progresses.
The disk usage depends on the particle count times number of state files to
save, plus some additional overhead.

To use this writer, first include this in your model code constructor:

    add_writer(HOTWRITER, 5);

After you have run the model, inspect data/ directory and verify that files
such as 'hot_00063.bin' are present.

To hot-start a subsequent simulation, identify the point in time from which you
want to start (associated with a particular hot start file)
e.g. 'hot_00063.bin' and invoke GPUSPH with the --resume command line option:

    ./GPUSPH --resume `pwd`/tests/DamBreak3dnn/data/hot_00063.bin

If the hotstart file is found and valid, the simulation will start from that
point. GPUSPH will abort otherwise.
*/
class HotWriter : public Writer {
public:
	HotWriter(const GlobalData *_gdata);
	~HotWriter();

	void write(uint numParts, const BufferList &buffers,
		uint node_offset, double t, const bool testpoints);

	void set_num_files_to_save(int num_files) {
		_num_files_to_save = num_files;
	}

	int get_num_files_to_save() const {
		return _num_files_to_save;
	}

private:
	int					_num_files_to_save;
	std::vector<std::string>	_current_filenames;
	uint				_particle_count;
};

/** Determines how far back in simulation time we can restart a simulation */
#define DEFAULT_NUM_FILES_TO_SAVE 8

#endif
