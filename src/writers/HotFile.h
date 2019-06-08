/*  Copyright (c) 2014-2017 INGV, EDF, UniCT, JHU

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
