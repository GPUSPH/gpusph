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

#include "CallbackWriter.h"
#include "Problem.h"

using namespace std;

CallbackWriter::CallbackWriter(const GlobalData *_gdata) : Writer(_gdata)
{
}

CallbackWriter::~CallbackWriter()
{
	vector<ofstream*>::iterator stream(m_streams.begin());
	while (stream != m_streams.end()) {
		ofstream *os = *stream;
		os->close();
		delete *stream;
		*stream = NULL;
		stream = m_streams.erase(stream);
	}
}

void
CallbackWriter::set_writers_list(ConstWriterMap const& writers)
{
	m_last_writers = writers;
}

ConstWriterMap const&
CallbackWriter::get_writers_list() const
{
	return m_last_writers;
}

// return pointer to wt writer if it was in the last writers list, NULL otherwise
const Writer *
CallbackWriter::get_other_writer(WriterType wt) const
{
	ConstWriterMap::const_iterator wpair = m_last_writers.find(wt);
	if (wpair != m_last_writers.end())
		return wpair->second;
	else
		return NULL;
}

void
CallbackWriter::write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints)
{
	m_problem->writer_callback(this, numParts, buffers, node_offset, t, testpoints);

	vector<ofstream*>::iterator stream(m_streams.begin());
	while (stream != m_streams.end()) {
		ofstream *os = *stream;
		if (!os->is_open()) {
			delete *stream;
			*stream = NULL;
			stream = m_streams.erase(stream);
		} else {
			++stream;
		}
	}
}

ofstream&
CallbackWriter::open_data_file(const char* base, std::string const& num, std::string const& sfx, std::string *fname)
{
	m_streams.push_back(new ofstream());

	string _fname = Writer::open_data_file(*(m_streams.back()), base, num, sfx);

	if (fname)
		*fname = _fname;

	return *(m_streams.back());
}


