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
#include <sstream>

#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <errno.h>

#include "HotWriter.h"
#include "GlobalData.h"

using namespace std;

HotWriter::HotWriter(const GlobalData *_gdata): Writer(_gdata) {

	m_fname_sfx = ".bin";

	_num_files_to_save = DEFAULT_NUM_FILES_TO_SAVE;
	_particle_count = 0;
}

HotWriter::~HotWriter() {
}

void HotWriter::write(uint numParts, const BufferList &buffers,
	uint node_offset, double t, const bool testpoints) {

	// generate filename with iterative integer
	ofstream out;
	string filename;
	if (gdata->run_mode == REPACK) {
		filename = open_data_file(out, "repack", current_filenum());
		gdata->clOptions->resume_fname = m_dirname + "/" + filename;
	}
	else
		filename = open_data_file(out, "hot", current_filenum());

	// save the filename in order to manage removing unwanted files
	_current_filenames.push_back(m_dirname + "/" + filename);

	// create and save the hot file
	HotFile *hf = new HotFile(out, gdata, numParts, node_offset, t, testpoints);
	hf->save();
	delete hf;

	out.close();

	// remove unwanted files, we only keep the last _num_files_to_save ones
	if(_num_files_to_save > 0 && _current_filenames.size() > _num_files_to_save) {
		int num_to_remove = _current_filenames.size() - _num_files_to_save;
		for(int i = 0; i < num_to_remove; i++) {
			string to_remove = _current_filenames.at(i);
			if(unlink(to_remove.c_str())) {
				perror(to_remove.c_str());
			}
		}
		_current_filenames.erase (_current_filenames.begin(),
			_current_filenames.begin() + num_to_remove);
	}

}

