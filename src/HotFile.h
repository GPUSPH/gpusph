/**
HotFile represents a hot-start file used to restart simulations.
The file represents particle and other state.
*/

#ifndef H_HOTFILE_H

#include <string>

#include "GlobalData.h"

typedef unsigned int uint;

using namespace std;

/**
HotFile header encoding.
*/
typedef struct {
	uint	version;
	uint	buffer_count;
	uint	particle_count;
	uint	reserved[16];
	float	t;
} header_t;

/** HotFile version. */
typedef enum { VERSION_1 } version_t;

class HotFile {
public:
	HotFile(const string &filename, const GlobalData *gdata);
	HotFile(const string &filename, const GlobalData *gdata, uint numParts,
		const BufferList &buffers, uint node_offset, float t,
		const bool testpoints);
	~HotFile();
	float get_t() { return _header.t; }
	bool save();
	bool load();
private:
	string				_filename;
	uint				_particle_count;
	BufferList			_buffers;
	uint				_node_offset;
	float				_t;
	bool				_testpoints;
	const GlobalData	*_gdata;
	header_t			_header;

	void writeBuffer(FILE *fp, AbstractBuffer *buffer, version_t version);
	void writeHeader(FILE *fp, version_t version);
	bool readBuffer(FILE *fp, AbstractBuffer *buffer, version_t version);
	bool readHeader(FILE *fp);

	friend std::ostream& operator<<(std::ostream&, const HotFile&);
};

#endif
