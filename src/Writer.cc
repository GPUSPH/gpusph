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

WriterMap Writer::m_writers = WriterMap();
bool Writer::m_forced = false;

static const char* WriterName[] = {
	"TextWriter",
	"VTKWriter",
	"VTKLegacyWriter",
	"CustomTextWriter",
	"UDPWriter",
	"HotWriter"
};

void
Writer::Create(GlobalData *_gdata)
{
	const Problem *problem = _gdata->problem;
	const Options *options = _gdata->clOptions;

	WriterList const& wl = problem->get_writers();
	WriterList::const_iterator it(wl.begin());
	WriterList::const_iterator end(wl.end());

	/* average writer frequency, used as default frequency for HOTWRITER,
	 * unless otherwise specified */
	float avg_freq = 0;
	int avg_count = 0;

	for (; it != end; ++it) {
		Writer *writer = NULL;
		WriterType wt = it->first;
		float freq = it->second;

		/* Check if the writer is in there already */
		WriterMap::iterator wm = m_writers.find(wt);
		if (wm != m_writers.end()) {
			writer = wm->second;
			cerr << "Overriding " << WriterName[wt] << " writing frequency" << endl;
		} else {
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
			m_writers[wt] = writer;
		}
		writer->set_write_freq(freq);
		if (freq != 0)
			cout << WriterName[wt] << " will write every " << freq << " seconds" << endl;
		else
			cout << WriterName[wt] << " disabled" << endl;

		avg_freq += freq;
		++avg_count;
	}

	avg_freq /= avg_count;

	/* Checkpoint setup: we setup a HOTWRITER if it's missing,
	 * change its frequency if present, and set the number of checkpoints
	 * as appropriate
	 */
	HotWriter *htwr = NULL;
	const WriterType wt = HOTWRITER;
	WriterMap::iterator wm = m_writers.find(wt);
	float freq = options->checkpoint_freq;
	int chkpts = options->checkpoints;

	if (wm != m_writers.end()) {
		htwr = static_cast<HotWriter*>(wm->second);
		/* found */
		if (isfinite(freq)) {
			cerr << "Command-line overrides " << WriterName[wt] << " writing frequency" << endl;
		}
	} else if (freq == 0) {
		cerr << "Command-line disables " << WriterName[wt] << endl;
		/* don't set htwr, checkpointing is disabled */
	} else {
		/* ok, generate a new one */
		htwr = new HotWriter(_gdata);
		m_writers[wt] = htwr;

		/* if frequency is not defined, used the average of the writers */
		if (!isfinite(freq))
			freq = avg_freq;

		/* still not defined? assume 0.1s TODO FIXME compute from tend or whatever */
		if (!isfinite(freq))
			freq = 0.1f;
	}

	if (htwr) {
		if (isfinite(freq))
			htwr->set_write_freq(freq);
		if (chkpts >= 0)
			htwr->set_num_files_to_save(chkpts);

		/* retrieve the actual values used, to select message */
		freq  = htwr->get_write_freq();
		chkpts = htwr->get_num_files_to_save();
		if (freq != 0) {
			cout << "HotStart checkpoints every " << freq << " (simulated) seconds" << endl;
			if (chkpts > 0)
				cout << "\twill keep the last " << chkpts << " checkpoints" << endl;
			else
				cout << "\twill keep ALL checkpoints" << endl;
		} else {
			cout << "HotStart checkpoints DISABLED" << endl;
		}
	}

}

bool
Writer::NeedWrite(float t)
{
	bool need_write = false;
	WriterMap::iterator it(m_writers.begin());
	WriterMap::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = it->second;
		need_write |= writer->need_write(t);
	}
	return need_write;
}

void
Writer::MarkWritten(float t, bool force)
{
	WriterMap::iterator it(m_writers.begin());
	WriterMap::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = it->second;
		if (writer->need_write(t) || force || m_forced)
			writer->mark_written(t);
	}
}

void
Writer::Write(uint numParts, BufferList const& buffers,
	uint node_offset, float t, const bool testpoints)
{
	WriterMap::iterator it(m_writers.begin());
	WriterMap::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = it->second;
		if (writer->need_write(t) || m_forced)
			writer->write(numParts, buffers, node_offset, t, testpoints);
	}
}

void
Writer::WriteWaveGage(float t, GageList const& gage)
{
	WriterMap::iterator it(m_writers.begin());
	WriterMap::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = it->second;
		if (writer->need_write(t) || m_forced)
			writer->write_WaveGage(t, gage);
	}
}

void
Writer::Destroy()
{
	WriterMap::iterator it(m_writers.begin());
	WriterMap::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = it->second;
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

	if (floor(t/m_writefreq) > floor(m_last_write_time/m_writefreq))
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

