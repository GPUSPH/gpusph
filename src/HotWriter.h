#ifndef H_HOTWRITER_H
#define H_HOTWRITER_H

#include "Writer.h"
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <pthread.h>

using namespace std;

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
e.g. 'hot_00063.bin'.  Set an environment variable named GPUSPH_HOTSTART_FILE
to the path of the desired hot start file:

    export GPUSPH_HOTSTART_FILE=`pwd`/tests/DamBreak3dnn/data/hot_00063.bin

When the simulation starts, if the GPUSPH_HOTSTART_FILE environment variable
is set *and* points to a valid file, the simulation will start from that point.
Otherwise, the simulation will start from the beginning.
*/
class HotWriter : public Writer {
public:
	HotWriter(const GlobalData *_gdata);
	~HotWriter();

	bool need_write(float t) const;

	void write(uint numParts, const BufferList &buffers,
		uint node_offset, float t, const bool testpoints);

	void set_num_files_to_save(int num_files) {
		_num_files_to_save = num_files;
	}

	int get_num_files_to_save() {
		return _num_files_to_save;
	}

private:
	int					_num_files_to_save;
	std::vector<string>	_current_filenames;
	uint				_particle_count;
	uint				_file_count;
	bool				_write_next_time;
	string				next_filename();
};

/** Determines how far back in simulation time we can restart a simulation */
#define DEFAULT_NUM_FILES_TO_SAVE 8

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

/**
HotFile buffer encoding.
*/
typedef struct {
	uint	name_length;
	char	name[64];
	uint	element_size;
	uint	array_count;
} encoded_buffer_t;

/** HotFile version. */
typedef enum { VERSION_1 } version_t;

/**
HotFile represents a hot-start file used to restart simulations.
The file represents particle and other state.
*/
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
