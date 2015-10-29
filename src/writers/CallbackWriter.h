/*  Copyright 2015 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

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

#ifndef _CALLBACKWRITER_H
#define	_CALLBACKWRITER_H

#include "Writer.h"

#include <vector>
#include <fstream>

class CallbackWriter : public Writer
{
	std::vector<std::ofstream*> m_streams;

	ConstWriterMap m_last_writers;

public:
	CallbackWriter(const GlobalData *_gdata);
	~CallbackWriter();

	virtual void write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints);

	// the CallbackWriter gets called after all other writers,
	// and it holds a list of all writers that have written when it got called
	void set_writers_list(ConstWriterMap const& writers);
	ConstWriterMap const& get_writers_list() const;

	// return pointer to wt writer if it was in the last writers list, NULL otherwise
	const Writer *get_other_writer(WriterType wt) const;

	// we have our own open_data_file that can be called by the Problem::writer_callback,
	// and we keep track of all opened stream and close them on destruction. The syntax
	// is slightly different from the one used in Writer::open_data_file():
	// we return the stream to allow the user to do:
	// 	(open_data_file() << data << data).close()
	// and optionally return the filename in the last parameter
	std::ofstream& open_data_file(const char* base, std::string const& num, std::string const& sfx, std::string *fname = NULL);
};

#endif

