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

/**
HotFile object encoding

Note that ODE does some massaging of the quaternion, so resuming will not give
exactly the same results as running the simulation continuously.

There are ways around this, such as hacking directly the ODE internals or
keeping a history of the motions of the objects, but this is currently deemed
overkill.
*/
typedef struct {
	uint	index;
	float	gravity_center[3];
	float	quaternion[4];
	float	linvel[3];
	float	angvel[3];
	float	reserved[10];
} encoded_body_t;

HotFile::HotFile(ofstream &fp, const GlobalData *gdata, uint numParts,
	uint node_offset, double t, const bool testpoints) {
	_fp.out = &fp;
	_gdata = gdata;
	_particle_count = numParts;
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
		writeBuffer(_fp.out, iter->second, VERSION_1);
		iter++;
	}

	// TODO FIXME for new moving/floating bodies treatment
	/*const float3 *cgs = _gdata->problem->get_bodies_cg();
	const dQuaternion *quats = _gdata->problem->get_bodies_quaternion();
	const float3 *linvels = _gdata->problem->get_bodies_linearvel();
	const float3 *angvels = _gdata->problem->get_bodies_angularvel();
	for (int b = 0; b < _header.body_count; ++b)
		writeBody(_fp.out, b, cgs + b, quats[b],
			linvels + b, angvels + b, VERSION_1);*/
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

	//// TODO FIXME multinode should take into account per-rank particles
	//check_counts_match("particle", _particle_count, _gdata->totParticles);

	// TODO FIXME would it be possible to restore from a situation with a
	// different number of arrays?
	check_counts_match("buffer", _header.buffer_count, _gdata->s_hBuffers.size());

	// NOTE: simulation with ODE bodies cannot be resumed identically due to
	// the way ODE handles its internal state.
	// TODO FIXME/ should be num ODE bodies
	check_counts_match("body", _header.body_count, _gdata->problem->simparams()->numbodies);

	BufferList::const_iterator iter = _gdata->s_hBuffers.begin();
	while (iter != _gdata->s_hBuffers.end()) {
		cout << "Will load buffer here..." << endl;
		readBuffer(_fp.in, (AbstractBuffer*)iter->second, VERSION_1);
		iter++;
	}

	for (uint b = 0; b < _header.body_count; ++b) {
		cout << "Restoring body #" << b << " ..." << endl;
		readBody(_fp.in, VERSION_1);
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
		memset(&_header, 0, sizeof(_header));
		_header.version = 1;
		_header.buffer_count = _gdata->s_hBuffers.size();
		_header.particle_count = _particle_count;
		//TODO FIXME
		_header.body_count = _gdata->problem->simparams()->numbodies;
		_header.numOpenBoundaries = _gdata->problem->simparams()->numOpenBoundaries;
		_header.iterations = _gdata->iterations;
		_header.dt = _gdata->dt;
		_header.t = _gdata->t;
		fp->write((char *)&_header, sizeof(_header));
		break;
	default:
		unsupported_version(version);
	}
}

void HotFile::readHeader(uint &part_count, uint &numOpenBoundaries) {
	memset(&_header, 0, sizeof(_header));

	// read and check version
	uint v;
	_fp.in->read((char*)&v, sizeof(v));
	if (v != 1)
		unsupported_version(v);

	_fp.in->seekg(0); // rewind
	_fp.in->read((char*)&_header, sizeof(_header));
	_particle_count = _header.particle_count;
	numOpenBoundaries = _header.numOpenBoundaries;
	_node_offset = part_count;
	part_count += _particle_count;
}

void HotFile::writeBuffer(ofstream *fp, const AbstractBuffer *buffer, version_t version) {
	switch (version) {
	case VERSION_1:
		encoded_buffer_t eb;
		memset(&eb, 0, sizeof(eb));
		eb.name_length = strlen(buffer->get_buffer_name());
		strcpy(eb.name, buffer->get_buffer_name());
		eb.element_size = buffer->get_element_size();
		eb.array_count = buffer->get_array_count();
		fp->write((char*)&eb, sizeof(eb));

		// Guiseppe says we need only write the first buffer
		// TODO FIXME this is not true anymore since we now properly support
		// multi-component buffers vs doulbe-buffering properly, so if one
		// buffer has an array_count > 1 we should dump/restore all of them;
		// but this will lead to a hotfile version change, and we only need to
		// implement it when we will have multi-component buffers on host.
		// The only caveat is that presently the number of components
		// may be 1 or 2 for the same buffer depending on whether the hotfile
		// was saved after or before the introduction of the multibufferlist.
		for(int i = 0; i < 1 /*buffer->get_array_count()*/; i++) {
			const void *data = buffer->get_offset_buffer(i, _node_offset);
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
		memset(&eb, 0, sizeof(eb));
		fp->read((char*)&eb, sizeof(eb));
		cout << "read buffer header: " << eb.name << endl;
		if (strcmp(buffer->get_buffer_name(), eb.name))
			bufname_mismatch(buffer->get_buffer_name(), eb.name);
		fp->read((char*)buffer->get_offset_buffer(0, _node_offset), sz);
		break;
	default:
		unsupported_version(version);
	}
}

void HotFile::writeBody(ofstream *fp, uint index, const float3 *cg, const dQuaternion quaternion,
	const float3 *linvel, const float3 *angvel, version_t version)
{
	switch (version) {
	case VERSION_1:
		encoded_body_t eb;
		memset(&eb, 0, sizeof(eb));

		eb.index = index;

		eb.gravity_center[0] = cg->x;
		eb.gravity_center[1] = cg->y;
		eb.gravity_center[2] = cg->z;

		eb.quaternion[0] = quaternion[0];
		eb.quaternion[1] = quaternion[1];
		eb.quaternion[2] = quaternion[2];
		eb.quaternion[3] = quaternion[3];

		eb.linvel[0] = linvel->x;
		eb.linvel[1] = linvel->y;
		eb.linvel[2] = linvel->z;

		eb.angvel[0] = angvel->x;
		eb.angvel[1] = angvel->y;
		eb.angvel[2] = angvel->z;

		fp->write((const char *)&eb, sizeof(eb));
		break;
	default:
		unsupported_version(version);
	}
}

void HotFile::readBody(ifstream *fp, version_t version)
{
	switch (version) {
	case VERSION_1:
		encoded_body_t eb;
		memset(&eb, 0, sizeof(eb));
		fp->read((char *)&eb, sizeof(eb));
		// TODO FIXME
		//_gdata->problem->restore_ODE_body(eb.index, eb.gravity_center, eb.quaternion,
		//	eb.linvel, eb.angvel);
		break;
	default:
		unsupported_version(version);
	}
}


std::ostream& operator<<(std::ostream &strm, const HotFile &h) {
	return strm << "HotFile( version=" << h._header.version << ", pc=" <<
		h._header.particle_count << ", bc=" << h._header.body_count << ")" << endl;
}

