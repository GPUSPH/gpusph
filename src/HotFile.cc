/**
HotFile represents a hot-start file used to restart simulations.
The file represents particle and other state.
*/

#include <stdexcept>
#include "HotFile.h"

/**
HotFile buffer encoding.
*/
typedef struct {
	uint	name_length;
	char	name[64];
	uint	element_size;
	uint	array_count;
} encoded_buffer_t;

HotFile::HotFile(ofstream &fp, const GlobalData *gdata, uint numParts,
	const BufferList &buffers, uint node_offset, double t,
	const bool testpoints) {
	_fp.out = &fp;
	_gdata = gdata;
	_particle_count = numParts;
	_buffers = buffers;
	_node_offset = node_offset;
	_t = t;
	_testpoints = testpoints;
}

HotFile::HotFile(ifstream &fp, const GlobalData *gdata) {
	_fp.in = &fp;
	_gdata = gdata;
}

void HotFile::save() {
	// write a header
	writeHeader(_fp.out, VERSION_1);

	// TODO FIXME multinode should take into account _node_offset
	BufferList::const_iterator iter = _gdata->s_hBuffers.begin();
	while (iter != _gdata->s_hBuffers.end()) {
		writeBuffer(_fp.out, (AbstractBuffer*)iter->second, VERSION_1);
		iter++;
	}
}

// auxiliary method that checks that two values are the same, and throws an
// exception otherwise
static void
check_counts_match(const char* what, size_t hf_count, size_t sim_count)
{
	if (hf_count == sim_count)
		return;

	ostringstream os;
	os << "mismatched " << what << " count; HotFile has " << hf_count
		<< ", simulation has " << sim_count;
	throw runtime_error(os.str());
}


void HotFile::load() {
	// read header
	readHeader(_fp.in);

	// TODO FIXME multinode should take into account per-rank particles
	check_counts_match("particle", _particle_count, _gdata->totParticles);

	// TODO FIXME would it be possible to restore from a situation with a
	// different number of arrays?
	check_counts_match("buffer", _header.buffer_count, _gdata->s_hBuffers.size());

	// TODO FIXME multinode should take into account _node_offset
	BufferList::const_iterator iter = _gdata->s_hBuffers.begin();
	while (iter != _gdata->s_hBuffers.end()) {
		cout << "Will load buffer here..." << endl;
		readBuffer(_fp.in, (AbstractBuffer*)iter->second, VERSION_1);
		iter++;
	}
}

HotFile::~HotFile() {
}

// auxiliary method that throws an exception about an unsupported
// HotFile version
static void
unsupported_version(uint version)
{
	ostringstream os;
	os << "unsupported HotFile version " << version;
	throw out_of_range(os.str());
}

void HotFile::writeHeader(ofstream *fp, version_t version) {
	switch (version) {
	case VERSION_1:
		memset(&_header, 0, sizeof(header_t));
		_header.version = 1;
		_header.buffer_count = _gdata->s_hBuffers.size();
		_header.particle_count = _particle_count;
		_header.iterations = _gdata->iterations;
		_header.dt = _gdata->dt;
		_header.t = _gdata->t;
		fp->write((char *)&_header, sizeof(_header));
		break;
	default:
		unsupported_version(version);
	}
}

void HotFile::readHeader(ifstream *fp) {
	memset(&_header, 0, sizeof(header_t));

	// read and check version
	uint v;
	fp->read((char*)&v, sizeof(v));
	if (v != 1)
		unsupported_version(v);

	fp->seekg(0); // rewind
	fp->read((char*)&_header, sizeof(_header));
	_particle_count = _header.particle_count;
}

void HotFile::writeBuffer(ofstream *fp, AbstractBuffer *buffer, version_t version) {
	switch (version) {
	case VERSION_1:
		encoded_buffer_t eb;
		memset(&eb, 0, sizeof(encoded_buffer_t));
		eb.name_length = strlen(buffer->get_buffer_name());
		strcpy(eb.name, buffer->get_buffer_name());
		eb.element_size = buffer->get_element_size();
		eb.array_count = buffer->get_array_count();
		fp->write((char*)&eb, sizeof(eb));

		// Guiseppe says we need only write the first buffer
		for(int i = 0; i < 1 /*buffer->get_array_count()*/; i++) {
			const void *data = buffer->get_buffer(i);
			const size_t size = buffer->get_element_size()*_particle_count;
			if(data == NULL) {
				cerr << "NULL buffer " << i << " for " << buffer->get_buffer_name()
					<< " in HotWriter" << endl;
				continue;
			}
			fp->write((const char*)data, size);
		}
		break;
	default:
		unsupported_version(version);
	}
}

// auxiliary method that throw an exception about a
// buffer name mismatch
static void
bufname_mismatch(const char *expected, const char *got)
{
	ostringstream os;
	os	<< "Buffer name mismatch: '"
		<< expected << "' expected, got '"
		<< got << "'";
	throw runtime_error(os.str());
}

void HotFile::readBuffer(ifstream *fp, AbstractBuffer *buffer, version_t version) {
	size_t sz = buffer->get_element_size()*_particle_count;
	switch (version) {
	case VERSION_1:
		encoded_buffer_t eb;
		memset(&eb, 0, sizeof(encoded_buffer_t));
		fp->read((char*)&eb, sizeof(eb));
		cout << "read buffer header: " << eb.name << endl;
		if (strcmp(buffer->get_buffer_name(), eb.name))
			bufname_mismatch(buffer->get_buffer_name(), eb.name);
		fp->read((char*)buffer->get_buffer(), sz);
		break;
	default:
		unsupported_version(version);
	}
}

std::ostream& operator<<(std::ostream &strm, const HotFile &h) {
	return strm << "HotFile( version=" << h._header.version << ", pc=" <<
		h._header.particle_count << ")";
}

