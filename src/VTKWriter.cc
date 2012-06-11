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

#include <sstream>
#include <stdexcept>

#include "VTKWriter.h"

using namespace std;

VTKWriter::VTKWriter(const Problem *problem)
  : Writer(problem)
{
	string time_filename = m_dirname + "/VTUinp.pvd";
    m_timefile = NULL;
    m_timefile = fopen(time_filename.c_str(), "w");

	if (m_timefile == NULL) {
		stringstream ss;
		ss << "Cannot open data file " << time_filename;
		throw runtime_error(ss.str());
		}

	// Writing header of VTUinp.pvd file
	fprintf(m_timefile,"<?xml version='1.0'?>\n");
	fprintf(m_timefile," <VTKFile type='Collection' version='0.1'>\n");
	fprintf(m_timefile,"  <Collection>\n");
}


VTKWriter::~VTKWriter()
{
	if (m_timefile != NULL) {
		fprintf(m_timefile,"   </Collection>\n");
		fprintf(m_timefile,"  </VTKFile>\n");
		fclose(m_timefile);
	}
}

/* Endianness check: (char*)&endian_int reads the first byte of the int,
 * which is 0 on big-endian machines, and 1 in little-endian machines */
static int endian_int=1;
static const char* endianness[2] = { "BigEndian", "LittleEndian" };

static float zeroes[4];

/* auxiliary functions to write data array entrypoints */
inline void
scalar_array(FILE *fid, const char *type, const char *name, size_t offset)
{
	fprintf(fid, "	<DataArray type='%s' Name='%s' "
			"format='appended' offset='%zu'/>\n",
			type, name, offset);
}

inline void
vector_array(FILE *fid, const char *type, const char *name, uint dim, size_t offset)
{
	fprintf(fid, "	<DataArray type='%s' Name='%s' NumberOfComponents='%u' "
			"format='appended' offset='%zu'/>\n",
			type, name, dim, offset);
}

inline void
vector_array(FILE *fid, const char *type, uint dim, size_t offset)
{
	fprintf(fid, "	<DataArray type='%s' NumberOfComponents='%u' "
			"format='appended' offset='%zu'/>\n",
			type, dim, offset);
}

void VTKWriter::write(uint numParts, const float4 *pos, const float4 *vel,
				const particleinfo *info, const float3 *vort, float t, const bool testpoints,
				const float4 *normals)
{
	string filename, full_filename;

	filename = "PART_" + next_filenum() + ".vtu";
	full_filename = m_dirname + "/" + filename;

	FILE *fid = fopen(full_filename.c_str(), "w");

	if (fid == NULL) {
		stringstream ss;
		ss << "Cannot open data file " << full_filename;
		throw runtime_error(ss.str());
		}

	// Header
	fprintf(fid,"<?xml version='1.0'?>\n");
	fprintf(fid,"<VTKFile type= 'UnstructuredGrid'  version= '0.1'  byte_order= '%s'>\n",
		endianness[*(char*)&endian_int & 1]);
	fprintf(fid," <UnstructuredGrid>\n");
	fprintf(fid,"  <Piece NumberOfPoints='%d' NumberOfCells='%d'>\n", numParts, numParts);

	fprintf(fid,"   <PointData Scalars='Pressure' Vectors='Velocity'>\n");

	size_t offset = 0;

	// pressure
	scalar_array(fid, "Float32", "Pressure", offset);
	offset += sizeof(float)*numParts+sizeof(int);

	// density
	scalar_array(fid, "Float32", "Density", offset);
	offset += sizeof(float)*numParts+sizeof(int);

	// mass
	scalar_array(fid, "Float32", "Mass", offset);
	offset += sizeof(float)*numParts+sizeof(int);

	// particle info
	if (info) {
		scalar_array(fid, "Int16", "Part type", offset);
		offset += sizeof(ushort)*numParts+sizeof(int);
		scalar_array(fid, "Int16", "Part flag", offset);
		offset += sizeof(ushort)*numParts+sizeof(int);
		scalar_array(fid, "Int16", "Fluid number", offset);
		offset += sizeof(ushort)*numParts+sizeof(int);
		scalar_array(fid, "Int16", "Part object", offset);
		offset += sizeof(ushort)*numParts+sizeof(int);
		scalar_array(fid, "UInt32", "Part id", offset);
		offset += sizeof(uint)*numParts+sizeof(int);
	}

	// velocity
	vector_array(fid, "Float32", "Velocity", 3, offset);
	offset += sizeof(float)*3*numParts+sizeof(int);

	// vorticity
	if (vort) {
		vector_array(fid, "Float32", "Vorticity", 3, offset);
		offset += sizeof(float)*3*numParts+sizeof(int);
	}

	// normals
	if (normals) {
		vector_array(fid, "Float32", "Normals", 3, offset);
		offset += sizeof(float)*3*numParts+sizeof(int);

		scalar_array(fid, "Float32", "Criteria", offset);
		offset += sizeof(float)*numParts+sizeof(int);
	}

	fprintf(fid,"   </PointData>\n");

	// position
	fprintf(fid,"   <Points>\n");
	vector_array(fid, "Float32", 3, offset);
	offset += sizeof(float)*3*numParts+sizeof(int);
	fprintf(fid,"   </Points>\n");

	// Cells data
	fprintf(fid,"   <Cells>\n");
	scalar_array(fid, "Int32", "connectivity", offset);
	offset += sizeof(uint)*numParts+sizeof(int);
	scalar_array(fid, "Int32", "offsets", offset);
	offset += sizeof(uint)*numParts+sizeof(int);
	fprintf(fid,"	<DataArray type='Int32' Name='types' format='ascii'>\n");
	for (int i = 0; i < numParts; i++)
		fprintf(fid,"%d\t", 1);
	fprintf(fid,"\n");
	fprintf(fid,"	</DataArray>\n");
	fprintf(fid,"   </Cells>\n");
	fprintf(fid,"  </Piece>\n");

	fprintf(fid," </UnstructuredGrid>\n");
	fprintf(fid," <AppendedData encoding='raw'>\n_");

	int numbytes=sizeof(float)*numParts;

	// pressure
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (int i=0; i < numParts; i++) {
		float value = 0.0;
		if (FLUID(info[i]))
			value = m_problem->pressure(vel[i].w, object(info[i]));
		else if (TESTPOINTS(info[i]))
			value = vel[i].w;
		fwrite(&value, sizeof(value), 1, fid);
	}

	// density
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (int i=0; i < numParts; i++) {
		float value = 0.0;
		if (FLUID(info[i]))
			value = vel[i].w;
		fwrite(&value, sizeof(value), 1, fid);
	}

	// mass
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (int i=0; i < numParts; i++) {
		float value = pos[i].w;
		fwrite(&value, sizeof(value), 1, fid);
	}

	// particle info
	if (info) {
		numbytes=sizeof(ushort)*numParts;

		// type
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (int i=0; i < numParts; i++) {
			ushort value = PART_TYPE(info[i]);
			fwrite(&value, sizeof(value), 1, fid);
		}

		// flag
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (int i=0; i < numParts; i++) {
			ushort value = PART_FLAG(info[i]);
			fwrite(&value, sizeof(value), 1, fid);
		}

		// fluid number
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (int i=0; i < numParts; i++) {
			ushort value = PART_FLUID_NUM(info[i]);
			fwrite(&value, sizeof(value), 1, fid);
		}

		// object
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (int i=0; i < numParts; i++) {
			ushort value = object(info[i]);
			fwrite(&value, sizeof(value), 1, fid);
		}

		numbytes=sizeof(uint)*numParts;

		// id
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (int i=0; i < numParts; i++) {
			uint value = id(info[i]);
			fwrite(&value, sizeof(value), 1, fid);
		}
	}

	numbytes=sizeof(float)*3*numParts;

	// velocity
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (int i=0; i < numParts; i++) {
		float *value = zeroes;
		if (FLUID(info[i]) || TESTPOINTS(info[i])) {
			value = (float*)(vel + i);
		}
		fwrite(value, sizeof(*value), 3, fid);
	}

	// vorticity
	if (vort) {
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (int i=0; i < numParts; i++) {
			float *value = zeroes;
			if (FLUID(info[i])) {
				value = (float*)(vort + i);
			}
			fwrite(value, sizeof(*value), 3, fid);
		}
	}

	// normals
	if (normals) {
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (int i=0; i < numParts; i++) {
			float *value = zeroes;
			if (FLUID(info[i])) {
				value = (float*)(normals + i);
			}
			fwrite(value, sizeof(*value), 3, fid);
		}

		numbytes=sizeof(float)*numParts;
		// criteria
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (int i=0; i < numParts; i++) {
			float value = 0;
			if (FLUID(info[i]))
				value = normals[i].w;
			fwrite(&value, sizeof(value), 1, fid);
		}
	}

	numbytes=sizeof(float)*3*numParts;

	// position
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (int i=0; i < numParts; i++) {
		float *value = (float*)(pos + i);
		fwrite(value, sizeof(*value), 3, fid);
	}

	numbytes=sizeof(int)*numParts;
	// connectivity
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (int i=0; i < numParts; i++) {
		uint value = i;
		fwrite(&value, sizeof(value), 1, fid);
	}
	// offsets
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (int i=0; i < numParts; i++) {
		uint value = i+1;
		fwrite(&value, sizeof(value), 1, fid);
	}

	fprintf(fid," </AppendedData>\n");
	fprintf(fid,"</VTKFile>");

	fclose(fid);

	// Writing time to VTUinp.pvd file
	if (m_timefile != NULL) {
		fprintf(m_timefile,"<DataSet timestep='%f' group='' part='%d' file='%s'/>\n",
			t, 0, filename.c_str());
		}
}
