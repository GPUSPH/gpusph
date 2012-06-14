/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

#include "Writer.h"
#include <sstream>
#include <stdexcept>

/**
 *  Default Constructor; makes sure the file output format starts at PART_00000
 */
Writer::Writer(const Problem *problem)
  : m_FileCounter(0), m_problem(problem)
{
	m_dirname = problem->get_dirname() + "/data";
	mkdir(m_dirname.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);

	string energy_fn = m_dirname + "/energy.txt";
	m_energyfile = fopen(energy_fn.c_str(), "w");
	if (!m_energyfile) {
		stringstream ss;
		ss << "Cannot open data file " << energy_fn;
		throw runtime_error(ss.str());
	} else {
		fputs("#\ttime", m_energyfile);
		uint fluid = 0;
		for (; fluid < problem->get_physparams().numFluids; ++fluid)
			fprintf(m_energyfile, "\tkinetic%u\tpotential%u\telastic%u",
					fluid, fluid, fluid);
		fputs("\n", m_energyfile);
	}
}

Writer::~Writer()
{
	// hi
}

void
Writer::write_energy(float t, float4 *energy)
{
	fprintf(m_energyfile, "%g", t);
	uint fluid = 0;
	for (; fluid < m_problem->get_physparams().numFluids; ++fluid)
		fprintf(m_energyfile, "\t%g\t%g\t%g",
				energy[fluid].x, energy[fluid].y, energy[fluid].z);
	fputs("\n", m_energyfile);
	fflush(m_energyfile);
}

string
Writer::next_filenum()
{
	stringstream ss;

	if (m_FileCounter >= MAX_FILES) {
		stringstream ss;
		ss << "too many files created (> " << MAX_FILES;
		throw runtime_error(ss.str());
	}
	ss.width(FNUM_WIDTH);
	ss.fill('0');
	ss << m_FileCounter;

	m_FileCounter++;
	return ss.str();
}

