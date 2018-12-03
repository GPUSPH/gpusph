/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

/*! \file
 * Writer dispatch implementation
 */

#include <sstream>
#include <stdexcept>

#include "Writer.h"
#include "GlobalData.h"

#include "CommonWriter.h"
#include "CustomTextWriter.h"
#include "CallbackWriter.h"
#include "TextWriter.h"
#ifndef _MSC_VER
 #include "UDPWriter.h"
#endif
#include "VTKLegacyWriter.h"
#include "VTKWriter.h"
#include "Writer.h"
#include "HotWriter.h"

#include "catalyst_select.opt"
#if USE_CATALYST == 1
#include "DisplayWriter.h"
#endif

using namespace std;

WriterMap Writer::m_writers = WriterMap();
WriteFlags Writer::m_write_flags = WriteFlags();
bool Writer::m_pending_hotwriter = false;

static const char* WriterName[] = {
	"CommonWriter",
	"TextWriter",
	"VTKWriter",
	"VTKLegacyWriter",
	"CallbackWriter",
	"CustomTextWriter",
	"UDPWriter",
	"HotWriter",
	"DisplayWriter"
};

const char* Writer::Name(WriterType key)
{
	return WriterName[key];
}

void
Writer::Create(GlobalData *_gdata)
{
	const ProblemCore *problem = _gdata->problem;
	const Options *options = _gdata->clOptions;

	WriterList const& wl = problem->get_writers();
	WriterList::const_iterator it(wl.begin());
	WriterList::const_iterator end(wl.end());

	/* average writer frequency, used as default frequency for HOTWRITER,
	 * unless otherwise specified */
	double avg_freq = 0;
	int avg_count = 0;

	for (; it != end; ++it) {
		Writer *writer = NULL;
		WriterType wt = it->first;
		double freq = it->second;
		// previous frequency, in case of override. used to correct the
		// avg_freq
		double old_freq = 0;

		/* Check if the writer is in there already */
		WriterMap::iterator wm = m_writers.find(wt);
		if (wm != m_writers.end()) {
			writer = wm->second;
			old_freq = writer->get_write_freq();
			cerr << "Overriding " << WriterName[wt] << " writing frequency" << endl;
		} else {
			switch (wt) {
			case COMMONWRITER:
				writer = new CommonWriter(_gdata);
				break;
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
				#ifdef _MSC_VER				
					throw std::invalid_argument("UDPWriter is not supported in Windows");
				#else
 					writer = new UDPWriter(_gdata);
				#endif
				break;
			case HOTWRITER:
				writer = new HotWriter(_gdata);
				break;
			case CALLBACKWRITER:
				writer = new CallbackWriter(_gdata);
				break;
#if USE_CATALYST == 1
			case DISPLAYWRITER:
				writer = new DisplayWriter(_gdata);
				break;
#endif
			default:
				stringstream ss;
				ss << "Unknown writer type " << wt;
				throw runtime_error(ss.str());
			}
			m_writers[wt] = writer;
		}
		writer->set_write_freq(freq);

		if (freq > 0)
			cout << WriterName[wt] << " will write every " << freq << " (simulated) seconds" << endl;
		else if (freq == 0)
			cout << WriterName[wt] << " will write every iteration" << endl;
		else if (freq < 0)
			cout << WriterName[wt] << " has been disabled" << endl;
		else if (std::isnan(freq))
			cout << WriterName[wt] << " has special treatment" << endl;
		else
			cerr << WriterName[wt] << " has unknown writing frequency " << freq << endl;

		// add current frequency for the average computation,
		// keeping in mind it might be an override of a previously set frequency
		avg_freq += (freq - old_freq);

		// increment the average divisor when this wasn't an override
		if (freq > 0 && old_freq == 0)
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
	double freq = options->checkpoint_freq;
	int chkpts = options->checkpoints;

	if (wm != m_writers.end()) {
		htwr = static_cast<HotWriter*>(wm->second);
		/* found */
		if (isfinite(freq)) {
			cerr << "Command-line overrides " << WriterName[wt] << " writing frequency" << endl;
		}
	} else if (freq < 0) {
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
		if (freq >= 0) {
			cout << "HotStart checkpoints every " << freq << " (simulated) seconds" << endl;
			if (chkpts > 0)
				cout << "\twill keep the last " << chkpts << " checkpoints" << endl;
			else
				cout << "\twill keep ALL checkpoints" << endl;
		} else {
			cout << "HotStart checkpoints DISABLED" << endl;
		}
	}

	// If there is no CommonWriter, create it. It will have the default settings
	// of writing whenever any other writer writes
	if (m_writers.find(COMMONWRITER) == m_writers.end())
		m_writers[COMMONWRITER] = new CommonWriter(_gdata);
}

ConstWriterMap
Writer::NeedWrite(double t)
{
	ConstWriterMap need_write;
	WriterMap::iterator it(m_writers.begin());
	WriterMap::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		const Writer *writer = it->second;
		if (writer->need_write(t)) {
			if (it->first == HOTWRITER)
				m_pending_hotwriter = true;
			else
				need_write[it->first] = it->second;
		}
	}
	return need_write;
}

WriterMap
Writer::StartWriting(double t, WriteFlags const& write_flags)
{
	WriterMap started;

	m_write_flags = write_flags;

	// is this a forced write?
	const bool forced = write_flags.forced_write;
	// is this a hot write?
	const bool hot = write_flags.hot_write;

	// is the common writer special?
	// (the common writer is not considered special during a hot write)
	const bool common_special =
		m_writers[COMMONWRITER]->is_special()
		&& !hot;

	WriterMap::iterator it(m_writers.begin());
	WriterMap::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		// The HotWriter only actually writes during a hot write (unless forced)
		// and conversely it's the only writer that writes during a hot write
		if (hot && it->first != HOTWRITER)
			continue;
		if (it->first == HOTWRITER && !hot && !forced)
			continue;

		// skip COMMONWRITER if special
		if (common_special && it->first == COMMONWRITER)
			continue;

		Writer *writer = it->second;
		if (writer->need_write(t) || forced) {
			writer->start_writing(t, write_flags);
			started[it->first] = it->second;
		}
	}

	if (common_special && !started.empty()) {
		m_writers[COMMONWRITER]->start_writing(t, write_flags);
	}

	return started;
}

void
Writer::MarkWritten(WriterMap writers, double t)
{
	// is this a hot write?
	const bool hot = m_write_flags.hot_write;
	// is the common writer special?
	const bool common_special =
		m_writers[COMMONWRITER]->is_special()
		&& !hot;

	WriterMap::iterator it(writers.begin());
	WriterMap::iterator end(writers.end());
	for ( ; it != end; ++it) {
		// skip COMMONWRITER if special
		if (common_special && it->first == COMMONWRITER)
			continue;

		it->second->mark_written(t);
	}

	if (common_special && !writers.empty())
		m_writers[COMMONWRITER]->mark_written(t);

	// clear the write flags
	m_write_flags.clear();
	if (hot)
		m_pending_hotwriter = false;
}

void
Writer::FakeMarkWritten(ConstWriterMap writers, double t)
{
	// is this a hot write?
	const bool hot = m_write_flags.hot_write;
	// is the common writer special?
	const bool common_special =
		m_writers[COMMONWRITER]->is_special()
		&& !hot;

	ConstWriterMap::iterator it(writers.begin());
	ConstWriterMap::iterator end(writers.end());
	for ( ; it != end; ++it) {
		// skip COMMONWRITER if special
		if (common_special && it->first == COMMONWRITER)
			continue;

		Writer *writer = const_cast<Writer*>(it->second);
		// Only call the default mark_written (which simply resets the last write
		// time), since we didn't actually do any of the writer-specific start_writing
		// stuff
		writer->Writer::mark_written(t);
	}

	if (common_special && !writers.empty())
		m_writers[COMMONWRITER]->mark_written(t);

	// clear the write flags
	m_write_flags.clear();
	if (hot)
		m_pending_hotwriter = false;
}

/* TODO FIXME C++11
 * All of the Write* delegates have the exact same structure,
 * wish we could use C++11 variadic templates and code them as a single
 * function.
 */

void
Writer::Write(WriterMap writers, uint numParts, BufferList const& buffers,
	uint node_offset, double t, const bool testpoints)
{
	// is this a hot write?
	const bool hot = m_write_flags.hot_write;
	// is the common writer special?
	const bool common_special =
		m_writers[COMMONWRITER]->is_special()
		&& !hot;

	// save it because it writes last
	CallbackWriter *cbwriter = NULL;

	ConstWriterMap have_written;

	WriterMap::iterator it(writers.begin());
	WriterMap::iterator end(writers.end());
	for ( ; it != end; ++it) {
		// skip COMMONWRITER if special
		if (common_special && it->first == COMMONWRITER)
			continue;

		// skip CALLBACKWRITER, it'll be called after all other writers
		if (it->first == CALLBACKWRITER) {
			cbwriter = static_cast<CallbackWriter*>(it->second);
			continue;
		}

		it->second->write(numParts, buffers, node_offset, t, testpoints);

		have_written[it->first] = it->second;
	}

	if (common_special && !writers.empty())
		m_writers[COMMONWRITER]->write(numParts, buffers, node_offset, t, testpoints);

	if (cbwriter) {
		cbwriter->set_writers_list(have_written);
		cbwriter->write(numParts, buffers, node_offset, t, testpoints);
	}
}

void
Writer::WriteWaveGage(WriterMap writers, double t, GageList const& gage)
{
	// is this a hot write?
	const bool hot = m_write_flags.hot_write;
	// is the common writer special?
	const bool common_special =
		m_writers[COMMONWRITER]->is_special()
		&& !hot;

	WriterMap::iterator it(writers.begin());
	WriterMap::iterator end(writers.end());
	for ( ; it != end; ++it) {
		// skip COMMONWRITER if special
		if (common_special && it->first == COMMONWRITER)
			continue;

		it->second->write_WaveGage(t, gage);
	}

	if (common_special && !writers.empty())
		m_writers[COMMONWRITER]->write_WaveGage(t, gage);
}

void
Writer::WriteObjects(WriterMap writers, double t)
{
	// is this a hot write?
	const bool hot = m_write_flags.hot_write;
	// is the common writer special?
	const bool common_special =
		m_writers[COMMONWRITER]->is_special()
		&& !hot;

	WriterMap::iterator it(writers.begin());
	WriterMap::iterator end(writers.end());
	for ( ; it != end; ++it) {
		// skip COMMONWRITER if special
		if (common_special && it->first == COMMONWRITER)
			continue;

		it->second->write_objects(t);
	}

	if (common_special && !writers.empty())
		m_writers[COMMONWRITER]->write_objects(t);
}

void
Writer::WriteEnergy(WriterMap writers, double t, double4 *energy)
{
	// is this a hot write?
	const bool hot = m_write_flags.hot_write;
	// is the common writer special?
	const bool common_special =
		m_writers[COMMONWRITER]->is_special()
		&& !hot;

	WriterMap::iterator it(writers.begin());
	WriterMap::iterator end(writers.end());
	for ( ; it != end; ++it) {
		// skip COMMONWRITER if special
		if (common_special && it->first == COMMONWRITER)
			continue;

		it->second->write_energy(t, energy);
	}

	if (common_special && !writers.empty())
		m_writers[COMMONWRITER]->write_energy(t, energy);
}

void
Writer::WriteObjectForces(WriterMap writers, double t, uint numobjects,
		const float3* computedforces, const float3* computedtorques,
		const float3* appliedforces, const float3* appliedtorques)
{
	// is this a hot write?
	const bool hot = m_write_flags.hot_write;
	// is the common writer special?
	const bool common_special =
		m_writers[COMMONWRITER]->is_special()
		&& !hot;

	WriterMap::iterator it(writers.begin());
	WriterMap::iterator end(writers.end());
	for ( ; it != end; ++it) {
		// skip COMMONWRITER if special
		if (common_special && it->first == COMMONWRITER)
			continue;

		it->second->write_objectforces(t, numobjects,
			computedforces, computedtorques,
			appliedforces, appliedtorques);
	}

	if (common_special && !writers.empty())
		m_writers[COMMONWRITER]->write_objectforces(t, numobjects,
				computedforces, computedtorques,
				appliedforces, appliedtorques);
}

void
Writer::WriteFlux(WriterMap writers, double t, float* fluxes)
{
	// is this a hot write?
	const bool hot = m_write_flags.hot_write;
	// is the common writer special?
	const bool common_special =
		m_writers[COMMONWRITER]->is_special()
		&& !hot;

	WriterMap::iterator it(writers.begin());
	WriterMap::iterator end(writers.end());
	for ( ; it != end; ++it) {
		// skip COMMONWRITER if special
		if (common_special && it->first == COMMONWRITER)
			continue;

		it->second->write_flux(t, fluxes);
	}

	if (common_special && !writers.empty())
		m_writers[COMMONWRITER]->write_flux(t, fluxes);
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
	m_last_write_time(-1),
	m_writefreq(0),
	m_FileCounter(0),
	gdata(_gdata)
{
	m_problem = _gdata->problem;

	m_dirname = m_problem->get_dirname() + "/data";
	mkdir(m_dirname.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

	if (_gdata->simframework->hasPostProcessEngine(TESTPOINTS)) {
		string testpointsDir = m_dirname + "/testpoints";
		mkdir(testpointsDir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
	}

}

Writer::~Writer()
{
	// hi
}

void
Writer::set_write_freq(double f)
{
	m_writefreq = f;
}

bool
Writer::need_write(double t) const
{
	// negative frequency: writer disabled
	if (m_writefreq < 0)
		return false;

	// null frequency: write always
	if (m_writefreq == 0)
		return true;

	if (floor(t/m_writefreq) > floor(m_last_write_time/m_writefreq))
		return true;

	return false;
}

string
Writer::current_filenum() const {
	stringstream ss;

	ss.width(FNUM_WIDTH);
	ss.fill('0');
	ss << m_FileCounter;

	return ss.str();
}

string
Writer::last_filenum() const {
	stringstream ss;

	ss.width(FNUM_WIDTH);
	ss.fill('0');
	ss << m_FileCounter-1;

	return ss.str();
}

uint Writer::getFilenum() const
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

	out.open(full_filename.c_str(), fstream::binary);

	if (!out) {
		stringstream ss;
		ss << "Cannot open data file " << full_filename;
		throw runtime_error("Cannot open data file " + full_filename);
	}

	out.exceptions(ofstream::failbit | ofstream::badbit);

	return filename;
}

