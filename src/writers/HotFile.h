/**
HotFile represents a hot-start file used to restart simulations.
The file represents particle and other state.
*/

#ifndef H_HOTFILE_H

#include <string>
#include <fstream>

#include "GlobalData.h"
#include "MovingBody.h"

typedef unsigned int uint;
typedef unsigned long ulong;

/**
HotFile header encoding.
*/
typedef struct {
	uint	version;
	uint	buffer_count;
	uint	particle_count;
	uint	body_count;
	uint	numOpenBoundaries;
	uint	reserved[12];
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
	HotFile(std::ifstream &fp, const GlobalData *gdata);
	HotFile(std::ofstream &fp, const GlobalData *gdata, uint numParts,
		uint node_offset, double t, const bool testpoints);
	~HotFile();
	ulong get_iterations() { return _header.iterations; }
	float get_dt() { return _header.dt; }
	double get_t() { return _header.t; }
	void save();
	void load();
	void readHeader(uint &part_count, uint &numOpenBoundaries);
private:
	union {
		std::ifstream		*in;
		std::ofstream		*out;
	}					_fp;
	uint				_particle_count;
	uint				_node_offset;
	double				_t;
	bool				_testpoints;
	const GlobalData	*_gdata;
	header_t			_header;

	void writeBuffer(std::ofstream *fp, const AbstractBuffer *buffer, version_t version);
	void writeBody(std::ofstream *fp, const MovingBodyData *mbdata, const uint numparts, version_t version);
	void writeHeader(std::ofstream *fp, version_t version);
	void readBuffer(std::ifstream *fp, AbstractBuffer *buffer, version_t version);
	void readBody(std::ifstream *fp, version_t version);

	friend std::ostream& operator<<(std::ostream&, const HotFile&);
};

#endif
