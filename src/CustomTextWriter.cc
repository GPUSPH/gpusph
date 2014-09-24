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

#include "CustomTextWriter.h"
#include "GlobalData.h"

using namespace std;

CustomTextWriter::CustomTextWriter(const GlobalData *_gdata)
  : Writer(_gdata)
{
	m_fname_sfx = ".txt";
	string time_filename = m_dirname + "/time.txt";
    m_timefile = NULL;
    m_timefile = fopen(time_filename.c_str(), "w");

	if (m_timefile == NULL) {
		stringstream ss;
		ss << "Cannot open data file " << time_filename;
		throw runtime_error(ss.str());
		}
}


CustomTextWriter::~CustomTextWriter()
{
    if (m_timefile != NULL) {
        fclose(m_timefile);
		m_timefile = NULL;
    }
}

void
CustomTextWriter::write(uint numParts, BufferList const& buffers, uint node_offset, float t, const bool testpoints)
{
	const double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();
	const float4 *vel = buffers.getData<BUFFER_VEL>();
	const particleinfo *info = buffers.getData<BUFFER_INFO>();
	const float3 *vort = buffers.getData<BUFFER_VORTICITY>();

	ofstream fid;
	string filename = open_data_file(fid, "PART_", next_filenum());

	// Modify this part to match your requirements
	// Writing datas
	for (uint i=0; i < numParts; i++) {
		// id, type, object, position
		fid << id(info[i]) << "\t" << type(info[i]) << "\t" << object(info[i]) << "\t";
		fid << pos[i].x << "\t" << pos[i].y << "\t" << pos[i].z << "\t";

		// velocity
		if (FLUID(info[i]))
			fid << vel[i].x << "\t" << vel[i].y << "\t" << vel[i].z << "\t";
		else
			fid << "0.0\t0.0\t0.0\t";

		// mass
		fid << pos[i].w << "\t";

		// density
		if (FLUID(info[i]))
			fid << vel[i].w << "\t";
		else
			fid << "0.0\t";

		// pressure
		if (FLUID(info[i]))
			fid << m_problem->pressure(vel[i].w, object(info[i])) << "\t";
		else
			fid << "0.0\t";

		// vorticity
		if (vort) {
			if (FLUID(info[i]))
				fid << vort[i].x << "\t" << vort[i].y << "\t" << vort[i].z << "\t";
			else
				fid << "0.0\t0.0\t0.0\t";
		}

		fid << endl;

	}

	fid.close();

	//Writing time to VTUinp.pvd file
	if (m_timefile != NULL) {
		fprintf(m_timefile,"%d\t%f\n", m_FileCounter, t);
	}

}

