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

bool HotWriter::need_write(double t) const {
	/* check if we would write according to the common Writer logic */
	bool would_write = Writer::need_write(t);
	/* but then delay write until the next buildneibs */
	if (would_write)
		return ((gdata->iterations % gdata->problem->simparams()->buildneibsfreq) == 0);
	return false;
}

void HotWriter::write(uint numParts, const BufferList &buffers,
	uint node_offset, double t, const bool testpoints) {

	// generate filename with iterative integer
	ofstream out;
	string filename = open_data_file(out, "hot", current_filenum());

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

