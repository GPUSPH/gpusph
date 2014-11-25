/**
HotFile represents a hot-start file used to restart simulations.
The file represents particle and other state.
*/

#ifndef H_HOTFILE_H

#include <string>
#include <fstream>

#include "GlobalData.h"

typedef unsigned int uint;
typedef unsigned long ulong;

using namespace std;

/**
HotFile header encoding.
*/
typedef struct {
	uint	version;
	uint	buffer_count;
	uint	particle_count;
	uint	reserved[13];
	ulong	iterations;
	double	t;
	float	dt;
	uint	_reserved[3];
} header_t;

/** HotFile version. */
typedef enum {
	VERSION_1,
} version_t;

class HotFile {
public:
	HotFile(ifstream &fp, const GlobalData *gdata);
	HotFile(ofstream &fp, const GlobalData *gdata, uint numParts,
		const BufferList &buffers, uint node_offset, double t,
		const bool testpoints);
	~HotFile();
	ulong get_iterations() { return _header.iterations; }
	float get_dt() { return _header.dt; }
	double get_t() { return _header.t; }
	void save();
	void load();
private:
	union {
		ifstream		*in;
		ofstream		*out;
	}					_fp;
	uint				_particle_count;
	BufferList			_buffers;
	uint				_node_offset;
	double				_t;
	bool				_testpoints;
	const GlobalData	*_gdata;
	header_t			_header;

	void writeBuffer(ofstream *fp, AbstractBuffer *buffer, version_t version);
	void writeHeader(ofstream *fp, version_t version);
	void readBuffer(ifstream *fp, AbstractBuffer *buffer, version_t version);
	void readHeader(ifstream *fp);

	friend std::ostream& operator<<(std::ostream&, const HotFile&);
};

#endif
