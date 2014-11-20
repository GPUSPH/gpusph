#include <sstream>
#include <unistd.h>
#include <iostream>
#include <stdexcept>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
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

HotFile::HotFile(const string &filename, const GlobalData *gdata, uint numParts,
	const BufferList &buffers, uint node_offset, float t,
	const bool testpoints) {
	_gdata = gdata;
	_filename = filename;
	_particle_count = numParts;
	_buffers = buffers;
	_node_offset = node_offset;
	_t = t;
	_testpoints = testpoints;
}

HotFile::HotFile(const string &filename, const GlobalData *gdata) {
	_gdata = gdata;
	_filename = filename;
}

bool HotFile::save() {
	FILE * fp;
	if((fp = fopen(_filename.c_str(), "wb")) == NULL) {
		perror(_filename.c_str());
		return false;
	}
	// write a header
	writeHeader(fp, VERSION_1);
	BufferList::const_iterator iter = _gdata->s_hBuffers.begin();
	while (iter != _gdata->s_hBuffers.end()) {
		writeBuffer(fp, (AbstractBuffer*)iter->second, VERSION_1);
		iter++;
	}
	fclose(fp);
	return true;
}

bool HotFile::load() {
	FILE * fp;
	if((fp = fopen(_filename.c_str(), "rb")) == NULL) {
		perror(_filename.c_str());
		return false;
	}
	// read header
	if(readHeader(fp) == false) {
		return false;
	}

	BufferList::const_iterator iter = _gdata->s_hBuffers.begin();
	while (iter != _gdata->s_hBuffers.end()) {
		cout << "Will load buffer here..." << endl;
		if(readBuffer(fp, (AbstractBuffer*)iter->second, VERSION_1) == false) {
			cout << "failed to read buffer" << endl;
		}
		iter++;
	}

	fclose(fp);
	return true;
}

HotFile::~HotFile() {
}

void HotFile::writeHeader(FILE *fp, version_t version) {
	if(version == VERSION_1) {
		memset(&_header, 0, sizeof(header_t));
		_header.version = 1;
		_header.buffer_count = _gdata->s_hBuffers.size();
		_header.particle_count = _particle_count;
		_header.t = _gdata->t;
		uint wrote = fwrite((char *)(&_header), sizeof(header_t), 1, fp);
	}
}

bool HotFile::readHeader(FILE *fp) {
	memset(&_header, 0, sizeof(header_t));

	// read and check version
	uint v;
	uint read = fread((char *)&v, sizeof(uint), 1, fp);
	if(v != 1) {
		cerr << "Unsupported hot file version: " << v << endl;
		return false;
	}

	// read header
	rewind(fp);
	read = fread((char *)(&_header), sizeof(header_t), 1, fp);
	_particle_count = _header.particle_count;

	return true;
}

void HotFile::writeBuffer(FILE *fp, AbstractBuffer *buffer, version_t version) {
	if(version == VERSION_1) {
		encoded_buffer_t eb;
		memset(&eb, 0, sizeof(encoded_buffer_t));
		eb.name_length = strlen(buffer->get_buffer_name());
		strcpy(eb.name, buffer->get_buffer_name());
		eb.element_size = buffer->get_element_size();
		eb.array_count = buffer->get_array_count();
		uint objects_wrote = fwrite((unsigned char *)(&eb), sizeof(
				encoded_buffer_t), 1, fp);

		// Guiseppe says we need only write the first buffer
		for(int i = 0; i < 1 /*buffer->get_array_count()*/; i++) {
			const void *data = buffer->get_buffer(i);
			if(data == NULL) {
				cout << "NULL buffer in HotWriter" << endl;
				continue;
			}
			objects_wrote = fwrite((unsigned char *)data,
				buffer->get_element_size(), _particle_count, fp);
		}
	}
}

bool HotFile::readBuffer(FILE *fp, AbstractBuffer *buffer, version_t version) {
	if(version == VERSION_1) {
		encoded_buffer_t eb;
		memset(&eb, 0, sizeof(encoded_buffer_t));
		uint objects_read = fread((unsigned char *)(&eb), sizeof(
				encoded_buffer_t), 1, fp);
		cout << "read buffer header: " << eb.name << endl;
		if(strcmp(buffer->get_buffer_name(), eb.name)) {
			cerr << "Error, names do not match: " << buffer->get_buffer_name()
				<< ", " << eb.name << endl;
			exit(1);
		}
		objects_read = fread((unsigned char *)buffer->get_buffer(),
			buffer->get_element_size(), _particle_count, fp);
		cout << "buffer read " << (objects_read * buffer->get_element_size())
			<< endl;
	}
	return true;
}

std::ostream& operator<<(std::ostream &strm, const HotFile &h) {
	return strm << "HotFile( version=" << h._header.version << ", pc=" <<
		h._header.particle_count << ")";
}

