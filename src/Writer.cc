/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#include <sstream>
#include <stdexcept>

#include "Writer.h"
#include "GlobalData.h"

#include "CustomTextWriter.h"
#include "TextWriter.h"
#include "UDPWriter.h"
#include "VTKLegacyWriter.h"
#include "VTKWriter.h"
#include "Writer.h"
#include "HotWriter.h"

vector<Writer*> Writer::m_writers = vector<Writer*>();
bool Writer::m_forced = false;

void
Writer::Create(GlobalData *_gdata)
{
	const Problem *problem = _gdata->problem;
	WriterList const& wl = problem->get_writers();
	WriterList::const_iterator it(wl.begin());
	WriterList::const_iterator end(wl.end());

	for (; it != end; ++it) {
		Writer *writer = NULL;
		WriterType wt = it->first;
		float freq = it->second;
		switch (wt) {
		case TEXTWRITER:
			writer = new TextWriter(_gdata);
			break;
		case VTKWRITER:
			writer = new VTKWriter(_gdata);
			break;
		case VTKLEGACYWRITER:
			writer = new VTKLegacyWriter(_gdata);
			break;
		case CUSTOMTEXTWRITER:
			writer = new CustomTextWriter(_gdata);
			break;
		case UDPWRITER:
			writer = new UDPWriter(_gdata);
			break;
		case HOTWRITER:
			writer = new HotWriter(_gdata);
			break;
		default:
			stringstream ss;
			ss << "Unknown writer type " << wt;
			throw runtime_error(ss.str());
		}
		writer->set_write_freq(freq);
		m_writers.push_back(writer);
	}
}

bool
Writer::NeedWrite(float t)
{
	bool need_write = false;
	vector<Writer*>::iterator it(m_writers.begin());
	vector<Writer*>::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = *it;
		need_write |= writer->need_write(t);
	}
	return need_write;
}

void
Writer::MarkWritten(float t, bool force)
{
	vector<Writer*>::iterator it(m_writers.begin());
	vector<Writer*>::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = *it;
		if (writer->need_write(t) || force || m_forced)
			writer->mark_written(t);
	}
}

void
Writer::Write(uint numParts, BufferList const& buffers,
	uint node_offset, float t, const bool testpoints)
{
	vector<Writer*>::iterator it(m_writers.begin());
	vector<Writer*>::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = *it;
		if (writer->need_write(t) || m_forced)
			writer->write(numParts, buffers, node_offset, t, testpoints);
	}
}

void
Writer::WriteWaveGage(float t, GageList const& gage)
{
	vector<Writer*>::iterator it(m_writers.begin());
	vector<Writer*>::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = *it;
		if (writer->need_write(t) || m_forced)
			writer->write_WaveGage(t, gage);
	}
}

void
Writer::Destroy()
{
	vector<Writer*>::iterator it(m_writers.begin());
	vector<Writer*>::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = *it;
		delete writer;
	}
	m_writers.clear();
}

/**
 *  Default Constructor; makes sure the file output format starts at PART_00000
 */
Writer::Writer(const GlobalData *_gdata) :
	m_FileCounter(0), gdata(_gdata),
	m_writefreq(0), m_last_write_time(-1)
{
	m_problem = _gdata->problem;

	m_dirname = m_problem->get_dirname() + "/data";
	mkdir(m_dirname.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

	if(m_problem->get_simparams()->testpoints){
		string testpointsDir = m_dirname + "/testpoints";
		mkdir(testpointsDir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
	}

	string energy_fn = open_data_file(m_energyfile, "energy", "", ".txt");
	if (m_energyfile) {
		m_energyfile << "#\ttime";
		uint fluid = 0;
		for (; fluid < m_problem->get_physparams()->numFluids; ++fluid)
			m_energyfile	<< "\tkinetic" << fluid
							<< "\tpotential" << fluid
							<< "\telastic" << fluid;
		m_energyfile << endl;
	}

	//WaveGage
	string WaveGage_fn = open_data_file(m_WaveGagefile, "WaveGage", "", ".txt");
	if (m_WaveGagefile) {
		m_WaveGagefile << "#\ttime";
		uint gage = 0;
		for (; gage < m_problem->get_simparams()->gage.size(); ++gage)
			m_WaveGagefile << "\tzgage" << gage;
		m_WaveGagefile << endl;
	}
}

Writer::~Writer()
{
	// hi
}

void
Writer::set_write_freq(float f)
{
	m_writefreq = f;
}

bool
Writer::need_write(float t) const
{
	if (m_writefreq == 0)
		return false;

	if (m_last_write_time < 0 || t - m_last_write_time >= m_writefreq)
		return true;

	return false;
}

void
Writer::write_energy(float t, float4 *energy)
{
	if (m_energyfile) {
		m_energyfile << t;
		uint fluid = 0;
		for (; fluid < m_problem->get_physparams()->numFluids; ++fluid)
			m_energyfile	<< "\t" << energy[fluid].x
							<< "\t" << energy[fluid].y
							<< "\t" << energy[fluid].z;
		m_energyfile << endl;
	}
}

//WaveGage
void
Writer::write_WaveGage(float t, GageList const& gage)
{
	if (m_WaveGagefile) {
		m_WaveGagefile << t;
		for (size_t i=0; i < gage.size(); i++) {
			m_WaveGagefile << "\t" << gage[i].z;
		}
		m_WaveGagefile << endl;
	}
}

string
Writer::current_filenum() {
	stringstream ss;

	ss.width(FNUM_WIDTH);
	ss.fill('0');
	ss << m_FileCounter;

	return ss.str();
}

string
Writer::next_filenum()
{
	string ret = current_filenum();

	if (m_FileCounter >= MAX_FILES) {
		stringstream ss;
		ss << "too many files created (> " << MAX_FILES;
		throw runtime_error(ss.str());
	}

	m_FileCounter++;
	return ret;
}

uint Writer::getLastFilenum()
{
	return m_FileCounter;
}

string
Writer::open_data_file(ofstream &out, const char* base, string const& num, string const& sfx)
{
	string filename(base), full_filename;

	if (gdata && gdata->mpi_nodes > 1)
		filename += "_n" + gdata->rankString();

	if (!num.empty())
		filename += "_" + num;

	filename += sfx;

	full_filename = m_dirname + "/" + filename;

	out.open(full_filename.c_str());

	if (!out) {
		stringstream ss;
		ss << "Cannot open data file " << full_filename;
		throw runtime_error("Cannot open data file " + full_filename);
	}

	out.exceptions(ofstream::failbit | ofstream::badbit);

	return filename;
}

