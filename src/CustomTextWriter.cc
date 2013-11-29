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

using namespace std;

CustomTextWriter::CustomTextWriter(const Problem *problem)
  : Writer(problem)
{
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
CustomTextWriter::write(	uint 				numParts,
							const double4		*pos,
							const float4		*vel,
							const particleinfo	*info,
							const float3		*vort,
							float				t,
							const bool			testpoints,
							const float4		*normals,
							const float4		*gradGamma,
							const float			*tke,
							const float			*turbvisc)
{
	string filename, full_filename;

	filename = "PART_" + next_filenum() + ".txt";
	full_filename = m_dirname + "/" + filename;

	FILE *fid = fopen(full_filename.c_str(), "w");

	if (fid == NULL) {
		stringstream ss;
		ss << "Cannot open data file " <<full_filename;
		throw runtime_error(ss.str());
		}

	// Modify this part to match your requirements
	// Writing datas
	for (int i=0; i < numParts; i++) {
		// position
		  fprintf(fid,"%d\t%d\t%d\t%f\t%f\t%f\t", id(info[i]), type(info[i]), object(info[i])
												, pos[i].x, pos[i].y, pos[i].z);
		// velocity
		if (FLUID(info[i]))
			fprintf(fid,"%f\t%f\t%f\t",vel[i].x, vel[i].y, vel[i].z);
		else
			fprintf(fid,"%f\t%f\t%f\t",0.0, 0.0, 0.0);

		// mass
		fprintf(fid,"%f\t",pos[i].w);

		// density
		if (FLUID(info[i]))
			fprintf(fid,"%f\t",vel[i].w);
		else
			fprintf(fid,"%f\t", 0.0);

		// pressure
		if (FLUID(info[i]))
			fprintf(fid,"%f\t",m_problem->pressure(vel[i].w, object(info[i])));  //Tony
		else
			fprintf(fid,"%f\t", 0.0);

		// vorticity
		if (vort) {
			if (FLUID(info[i]) > 0.0)
				fprintf(fid,"%f\t%f\t%f\t",vort[i].x, vort[i].y, vort[i].z);
			else
				fprintf(fid,"%f\t%f\t%f\t",0.0, 0.0, 0.0);
			}

		fprintf(fid,"\n");

		}

	fclose(fid);

	//Writing time to VTUinp.pvd file
	if (m_timefile != NULL) {
		fprintf(m_timefile,"%d\t%f\n", m_FileCounter, t);
		}

}

