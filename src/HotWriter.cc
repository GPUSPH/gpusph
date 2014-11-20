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
	_num_files_to_save = DEFAULT_NUM_FILES_TO_SAVE;
	_particle_count = 0;
	_file_count = 0;
	_write_next_time = false;
}

HotWriter::~HotWriter() {
}

bool HotWriter::need_write(float t) const {
	if (m_writefreq == 0)
		return false;

	if ((gdata->iterations % gdata->problem->m_simparams.buildneibsfreq) == 0) {
		return true;
	}
	return false;
}

string HotWriter::next_filename() {
	stringstream ss;

	ss.width(Writer::FNUM_WIDTH);
	ss.fill('0');
	ss << _file_count;

	_file_count++;
	return ss.str();
}

void HotWriter::write(uint numParts, const BufferList &buffers,
	uint node_offset, float t, const bool testpoints) {

	cout << "KAG: write()" << endl;

	// generate filename with iterative integer
	string filename = m_dirname + "/hot_" + next_filename() + ".bin";

	// save the filename in order to manage removing unwanted files
	_current_filenames.push_back(filename);

	// remove unwanted files, we only keep the last _num_files_to_save ones
	if(_current_filenames.size() > _num_files_to_save) {
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

	// create and save the hot file
	HotFile *hf = new HotFile(filename, gdata, numParts, buffers, node_offset,
		t, testpoints);
	hf->save();
}

