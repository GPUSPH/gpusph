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

vector<Writer*> Writer::m_writers = vector<Writer*>();
float Writer::m_timer_tick = 0;
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
		int freq = it->second;
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
Writer::WriteObjectForces(float t, uint numobjects, float3* forces, float3* momentums)
{
	vector<Writer*>::iterator it(m_writers.begin());
	vector<Writer*>::iterator end(m_writers.end());
	for ( ; it != end; ++it) {
		Writer *writer = *it;
		writer->write_objectforces(t, numobjects, forces, momentums);
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

	string energy_fn = m_dirname + "/energy.txt";
	m_energyfile = fopen(energy_fn.c_str(), "w");
	/*if (!m_energyfile) {
		stringstream ss;
		ss << "Cannot open data file " << energy_fn;
		throw runtime_error(ss.str());
	} else*/
	if (m_energyfile) {
		fputs("#\ttime", m_energyfile);
		uint fluid = 0;
		for (; fluid < m_problem->get_physparams()->numFluids; ++fluid)
			fprintf(m_energyfile, "\tkinetic%u\tpotential%u\telastic%u",
					fluid, fluid, fluid);
		fputs("\n", m_energyfile);
	}

	//WaveGage
	string WaveGage_fn = m_dirname + "/WaveGage.txt";
	m_WaveGagefile = fopen(WaveGage_fn.c_str(), "w");
	/*if (!m_WaveGagefile) {
		stringstream ss;
		ss << "Cannot open data file " << WaveGage_fn;
		throw runtime_error(ss.str());
	} else */
	if (m_WaveGagefile)
	{
		fputs("#\ttime", m_WaveGagefile);
		uint gage = 0;
		for (; gage < m_problem->get_simparams()->gage.size(); ++gage)
			fprintf(m_WaveGagefile, "\tzgage%u",
					gage);
		fputs("\n", m_WaveGagefile);
	}

	// Forces on objects
	//WaveGage
	string objectforces_fn = m_dirname + "/objectforces.txt";
	m_objectforcesfile = fopen(objectforces_fn.c_str(), "w");
}

Writer::~Writer()
{
	// hi
}

void
Writer::set_write_freq(int f)
{
	m_writefreq = f;
}

bool
Writer::need_write(float t) const
{
	if (m_writefreq == 0)
		return false;

	if (m_last_write_time < 0 || t - m_last_write_time >= m_timer_tick*m_writefreq)
		return true;

	return false;
}

void
Writer::write_energy(float t, float4 *energy)
{
	if (m_energyfile) {
		fprintf(m_energyfile, "%g", t);
		uint fluid = 0;
		for (; fluid < m_problem->get_physparams()->numFluids; ++fluid)
			fprintf(m_energyfile, "\t%g\t%g\t%g",
					energy[fluid].x, energy[fluid].y, energy[fluid].z);
		fputs("\n", m_energyfile);
		fflush(m_energyfile);
	}
}

//WaveGage
void
Writer::write_WaveGage(float t, GageList const& gage)
{
	if (m_WaveGagefile) {
		fprintf(m_WaveGagefile, "%g", t);
		for (size_t i=0; i < gage.size(); i++) {
			fprintf(m_WaveGagefile, "\t%g",
				gage[i].z);
		}
		fputs("\n", m_WaveGagefile);
		fflush(m_WaveGagefile);
	}
}


// Object forces
void
Writer::write_objectforces(float t, uint numobjects, float3* forces, float3* momentums)
{
	if (m_objectforcesfile) {
		fprintf(m_objectforcesfile, "%g", t);
		for (int i=0; i < numobjects; i++) {
			fprintf(m_objectforcesfile, "\t%d", i);
			fprintf(m_objectforcesfile, "\t%e\t%e\t%e", forces[i].x, forces[0].y, forces[0].z);
			fprintf(m_objectforcesfile, "\t%e\t%e\t%e", momentums[i].x, momentums[0].y, momentums[0].z);
		}
		fputs("\n", m_objectforcesfile);
		fflush(m_objectforcesfile);
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
