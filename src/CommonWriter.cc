/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#include <fstream>

#include "CommonWriter.h"

#include "GlobalData.h"

using namespace std;

CommonWriter::CommonWriter(const GlobalData *_gdata)
	: Writer(_gdata)
{
	m_fname_sfx = ".txt";

	// special value denoting default behavior of writing every time
	// any other writer does
	m_writefreq = -1;

	// TODO only do this if energy writing is enabled
	string energy_fn = open_data_file(m_energyfile, "energy");
	if (m_energyfile) {
		m_energyfile << "#\ttime";
		uint fluid = 0;
		for (; fluid < m_problem->get_physparams()->numFluids; ++fluid)
			m_energyfile	<< "\tkinetic" << fluid
							<< "\tpotential" << fluid
							<< "\telastic" << fluid;
		m_energyfile << endl;
	}

	size_t ngages = m_problem->get_simparams()->gage.size();
	if (ngages > 0) {
		string WaveGage_fn = open_data_file(m_WaveGagefile, "WaveGage");
		if (m_WaveGagefile) {
			m_WaveGagefile << "#\ttime";
			for (size_t gage = 0; gage < ngages; ++gage)
				m_WaveGagefile << "\tzgage" << gage;
			m_WaveGagefile << endl;
		}
	}
}

CommonWriter::~CommonWriter()
{
	if (m_energyfile)
		m_energyfile.close();
	if (m_WaveGagefile)
		m_WaveGagefile.close();
}

void
CommonWriter::write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints)
{ /* do nothing */ }

void
CommonWriter::write_energy(double t, float4 *energy)
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

void
CommonWriter::write_WaveGage(double t, GageList const& gage)
{
	if (m_WaveGagefile) {
		m_WaveGagefile << t;
		for (size_t i=0; i < gage.size(); i++) {
			m_WaveGagefile << "\t" << gage[i].z;
		}
		m_WaveGagefile << endl;
	}
}

bool
CommonWriter::need_write(double t) const
{
	if (m_writefreq < 0)
		return false; // special
	return Writer::need_write(t);
}

