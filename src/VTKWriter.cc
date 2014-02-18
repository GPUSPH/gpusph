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

#include "VTKWriter.h"
// GlobalData is required for writing the device index. With some order
// of inclusions, a forward declaration might be required
#include "GlobalData.h"

using namespace std;

VTKWriter::VTKWriter(const Problem *problem)
  : Writer(problem)
{
	string time_filename = m_dirname + "/VTUinp.pvd";
    m_timefile = NULL;
    m_timefile = fopen(time_filename.c_str(), "w");

	/*if (m_timefile == NULL) {
		stringstream ss;
		ss << "Cannot open data file " << time_filename;
		throw runtime_error(ss.str());
		}*/

	// Writing header of VTUinp.pvd file
	if (m_timefile) {
		fprintf(m_timefile,"<?xml version='1.0'?>\n");
		fprintf(m_timefile," <VTKFile type='Collection' version='0.1'>\n");
		fprintf(m_timefile,"  <Collection>\n");
	}
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

void
VTKWriter::write(uint numParts, BufferList const& buffers, uint node_offset, float t, const bool testpoints)
{
	const double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();
	const hashKey *particleHash = buffers.getData<BUFFER_HASH>();
	const float4 *vel = buffers.getData<BUFFER_VEL>();
	const particleinfo *info = buffers.getData<BUFFER_INFO>();
	const float3 *vort = buffers.getData<BUFFER_VORTICITY>();
	const float4 *normals = buffers.getData<BUFFER_NORMALS>();
	const float4 *gradGamma = buffers.getData<BUFFER_GRADGAMMA>();
	const float *tke = buffers.getData<BUFFER_TKE>();
	const float *eps = buffers.getData<BUFFER_EPSILON>();
	const float *turbvisc = buffers.getData<BUFFER_TURBVISC>();
	const float *priv = buffers.getData<BUFFER_PRIVATE>();

	// CSV file for tespoints
	string testpoints_fname = m_dirname + "/testpoints/testpoints_" + current_filenum() + ".csv";
	FILE *testpoints_file = NULL;
	if (m_gdata->problem->get_simparams()->csvtestpoints) {
		testpoints_file = fopen(testpoints_fname.c_str(), "w");
		if (testpoints_file == NULL) {
			stringstream ss;
			ss << "Cannot open testpoints file " << testpoints_fname;
			throw runtime_error(ss.str());
		}
		// write CSV header
		if (testpoints_file)
			fprintf(testpoints_file,"T,ID,Pressure,Object,CellIndex,PosX,PosY,PosZ,VelX,VelY,VelZ\n");
	}

	string filename;

	FILE *fid = open_data_file("PART", next_filenum(), &filename);

	// Header
	//====================================================================================
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

	// gamma
	if (gradGamma) {
		scalar_array(fid, "Float32", "Gamma", offset);
		offset += sizeof(float)*numParts+sizeof(int);
	}

	// turbulent kinetic energy
	if (tke) {
		scalar_array(fid, "Float32", "TKE", offset);
		offset += sizeof(float)*numParts+sizeof(int);
	}

	// turbulent epsilon
	if (eps) {
		scalar_array(fid, "Float32", "Epsilon", offset);
		offset += sizeof(float)*numParts+sizeof(int);
	}

	// eddy viscosity
	if (turbvisc) {
		scalar_array(fid, "Float32", "Eddy viscosity", offset);
		offset += sizeof(float)*numParts+sizeof(int);
	}

	// particle info
	if (info) {
		scalar_array(fid, "Int16", "Part type", offset);
		offset += sizeof(ushort)*numParts+sizeof(int);
//		scalar_array(fid, "Int16", "Part flag", offset);
//		offset += sizeof(ushort)*numParts+sizeof(int);
//		scalar_array(fid, "Int16", "Fluid number", offset);
//		offset += sizeof(ushort)*numParts+sizeof(int);
//		scalar_array(fid, "Int16", "Part object", offset);
//		offset += sizeof(ushort)*numParts+sizeof(int);
		scalar_array(fid, "UInt32", "Part id", offset);
		offset += sizeof(uint)*numParts+sizeof(int);
	}

	// device index
	if (m_gdata) {
		scalar_array(fid, "UInt32", "DeviceIndex", offset);
		offset += sizeof(uint)*numParts+sizeof(int);
	}

	// cell index
	scalar_array(fid, "UInt32", "CellIndex", offset);
	offset += sizeof(uint)*numParts+sizeof(int);

	// velocity
	vector_array(fid, "Float32", "Velocity", 3, offset);
	offset += sizeof(float)*3*numParts+sizeof(int);

	// gradient gamma
	if (gradGamma) {
		vector_array(fid, "Float32", "Gradient Gamma", 3, offset);
		offset += sizeof(float)*3*numParts+sizeof(int);
	}

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

	// private
	if (priv) {
		scalar_array(fid, "Float32", "Private", offset);
		offset += sizeof(float)*numParts+sizeof(int);
	}

	fprintf(fid,"   </PointData>\n");

	// position
	fprintf(fid,"   <Points>\n");
	vector_array(fid, "Float64", 3, offset);
	offset += sizeof(double)*3*numParts+sizeof(int);
	fprintf(fid,"   </Points>\n");

	// Cells data
	fprintf(fid,"   <Cells>\n");
	scalar_array(fid, "Int32", "connectivity", offset);
	offset += sizeof(uint)*numParts+sizeof(int);
	scalar_array(fid, "Int32", "offsets", offset);
	offset += sizeof(uint)*numParts+sizeof(int);
	fprintf(fid,"	<DataArray type='Int32' Name='types' format='ascii'>\n");
	for (uint i = node_offset; i < node_offset + numParts; i++)
		fprintf(fid,"%d\t", 1);
	fprintf(fid,"\n");
	fprintf(fid,"	</DataArray>\n");
	fprintf(fid,"   </Cells>\n");
	fprintf(fid,"  </Piece>\n");

	fprintf(fid," </UnstructuredGrid>\n");
	fprintf(fid," <AppendedData encoding='raw'>\n_");
	//====================================================================================

	int numbytes=sizeof(float)*numParts;

	// pressure
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		float value = 0.0;
		if (TESTPOINTS(info[i]))
			value = vel[i].w;
		else
			value = m_problem->pressure(vel[i].w, object(info[i]));
		fwrite(&value, sizeof(value), 1, fid);
	}

	// density
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		float value = 0.0;
		//if (FLUID(info[i]))
			value = vel[i].w;
		fwrite(&value, sizeof(value), 1, fid);
	}

	// mass
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		float value = pos[i].w;
		fwrite(&value, sizeof(value), 1, fid);
	}

	// gamma
	if (gradGamma) {
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = gradGamma[i].w;
			fwrite(&value, sizeof(value), 1, fid);
		}
	}

	// turbulent kinetic energy
	if (tke) {
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = tke[i];
			fwrite(&value, sizeof(value), 1, fid);
		}
	}

	// turbulent epsilon
	if (eps) {
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=0; i < numParts; i++) {
			float value = eps[i];
			fwrite(&value, sizeof(value), 1, fid);
		}
	}

	// eddy viscosity
	if (turbvisc) {
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = turbvisc[i];
			fwrite(&value, sizeof(value), 1, fid);
		}
	}

	// particle info
	if (info) {
		numbytes=sizeof(ushort)*numParts;

		// type
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			ushort value = PART_TYPE(info[i]);
			if (m_gdata->problem->get_simparams()->csvtestpoints && value == (TESTPOINTSPART >> MAX_FLUID_BITS)) {
				fprintf(testpoints_file,"%g,%u,%g,%u,%u,%g,%g,%g,%g,%g,%g\n",
					t, id(info[i]),
					vel[i].w, object(info[i]), cellHashFromParticleHash( particleHash[i] ),
					pos[i].x, pos[i].y, pos[i].z,
					vel[i].x, vel[i].y, vel[i].z);
			}
			fwrite(&value, sizeof(value), 1, fid);
		}

//		// flag
//		fwrite(&numbytes, sizeof(numbytes), 1, fid);
//		for (uint i=node_offset; i < node_offset + numParts; i++) {
//			ushort value = PART_FLAG(info[i]);
//			fwrite(&value, sizeof(value), 1, fid);
//		}

//		// fluid number
//		fwrite(&numbytes, sizeof(numbytes), 1, fid);
//		for (uint i=node_offset; i < node_offset + numParts; i++) {
//			ushort value = PART_FLUID_NUM(info[i]);
//			fwrite(&value, sizeof(value), 1, fid);
//		}

//		// object
//		fwrite(&numbytes, sizeof(numbytes), 1, fid);
//		for (uint i=node_offset; i < node_offset + numParts; i++) {
//			ushort value = object(info[i]);
//			fwrite(&value, sizeof(value), 1, fid);
//		}

		numbytes=sizeof(uint)*numParts;

		// id
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			uint value = id(info[i]);
			fwrite(&value, sizeof(value), 1, fid);
		}
	}

	// device index
	if (m_gdata) {
		numbytes = sizeof(uint)*numParts;
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		// The previous way was to compute the theoretical containing cell solely according on the particle position. This, however,
		// was inconsistent with the actual particle distribution among the devices, since one particle can be physically out of the
		// containing cell until next calchash/reorder.
		// The current policy is: just list the particles according to how the global array is partitioned. In other words, we rely
		// on the particle index to understad which device downloaded the particle data.
		for (uint d = 0; d < m_gdata->devices; d++) {
			// compute the global device ID for each device
			uint value = m_gdata->GLOBAL_DEVICE_ID(m_gdata->mpi_rank, d);
			// write one for each particle (no need for the "absolute" particle index)
			for (uint p = 0; p < m_gdata->s_hPartsPerDevice[d]; p++)
				fwrite(&value, sizeof(value), 1, fid);
		}
		// There two alternate policies: 1. use particle hash or 2. compute belonging device.
		// To use the particle hash, instead of just relying on the particle index, use the following code:
		/*
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			uint value = m_gdata->s_hDeviceMap[ cellHashFromParticleHash(particleHash[i]) ];
			fwrite(&value, sizeof(value), 1, fid);
		}
		*/
		// This should be equivalent to the current "listing" approach. If for any reason (e.g. debug) one needs to write the
		// device index according to the current spatial position, it is enough to compute the particle hash from its position
		// instead of reading it from the particlehash array. Please note that this would reflect the spatial split but not the
		// actual assignments: until the next calchash is performed, one particle remains in the containing device even if it
		// it is slightly outside the domain.
	}

	// linearized cell index (NOTE: particles might be slightly off the belonging cell)
	numbytes = sizeof(uint)*numParts;
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (int i=node_offset; i < node_offset + numParts; i++) {
		uint value = cellHashFromParticleHash( particleHash[i] );
		fwrite(&value, sizeof(value), 1, fid);
	}

	numbytes=sizeof(float)*3*numParts;

	// velocity
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		float *value = zeroes;
		//if (FLUID(info[i]) || TESTPOINTS(info[i]))
			value = (float*)(vel + i);
		fwrite(value, sizeof(*value), 3, fid);
	}

	// gradient gamma
	if (gradGamma) {
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			value = (float*)(gradGamma + i);
			fwrite(value, sizeof(*value), 3, fid);
		}
	}

	// vorticity
	if (vort) {
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
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
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			if (FLUID(info[i])) {
				value = (float*)(normals + i);
			}
			fwrite(value, sizeof(*value), 3, fid);
		}

		numbytes=sizeof(float)*numParts;
		// criteria
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = 0;
			if (FLUID(info[i]))
				value = normals[i].w;
			fwrite(&value, sizeof(value), 1, fid);
		}
	}

	numbytes=sizeof(float)*numParts;

	// private
	if (priv) {
		fwrite(&numbytes, sizeof(numbytes), 1, fid);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = priv[i];
			fwrite(&value, sizeof(value), 1, fid);
		}
	}

	numbytes=sizeof(double)*3*numParts;

	// position
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		double *value = (double*)(pos + i);
		fwrite(value, sizeof(*value), 3, fid);
	}

	numbytes=sizeof(int)*numParts;
	// connectivity
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (uint i=0; i < numParts; i++) {
		uint value = i;
		fwrite(&value, sizeof(value), 1, fid);
	}
	// offsets
	fwrite(&numbytes, sizeof(numbytes), 1, fid);
	for (uint i=0; i < numParts; i++) {
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
		fflush(m_timefile);
	}

	// close testpoints file
	if (testpoints_file != NULL)
		fclose(testpoints_file);
}

void
VTKWriter::write_WaveGage(float t, GageList const& gage)
{
	// call the generic write_WaveGage first
	Writer::write_WaveGage(t, gage);

	FILE *fp = open_data_file("WaveGage", current_filenum(), NULL);
	size_t num = gage.size();

	// Header
	fprintf(fp,"<?xml version=\"1.0\"?>\r\n");
	fprintf(fp,"<VTKFile type= \"UnstructuredGrid\"  version= \"0.1\"  byte_order= \"BigEndian\">\r\n");
	fprintf(fp," <UnstructuredGrid>\r\n");
	fprintf(fp,"  <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\r\n", num, num);

	//Writing Position
	fprintf(fp,"   <Points>\r\n");
	fprintf(fp,"	<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\r\n");
	for (int i=0; i <  num; i++)
		fprintf(fp,"%f\t%f\t%f\t",gage[i].x, gage[i].y, gage[i].z);
	fprintf(fp,"\r\n");
	fprintf(fp,"	</DataArray>\r\n");
	fprintf(fp,"   </Points>\r\n");

	// Cells data
	fprintf(fp,"   <Cells>\r\n");
	fprintf(fp,"	<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\r\n");
	for (int i = 0; i < num; i++)
		fprintf(fp,"%d\t", i);
	fprintf(fp,"\r\n");
	fprintf(fp,"	</DataArray>\r\n");
	fprintf(fp,"\r\n");

	fprintf(fp,"	<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\r\n");
	for (int i = 0; i < num; i++)
		fprintf(fp,"%d\t", i + 1);
	fprintf(fp,"\r\n");
	fprintf(fp,"	</DataArray>\r\n");

	fprintf(fp,"\r\n");
	fprintf(fp,"	<DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">\r\n");
	for (int i = 0; i < num; i++)
		fprintf(fp,"%d\t", 1);
	fprintf(fp,"\r\n");
	fprintf(fp,"	</DataArray>\r\n");

	fprintf(fp,"   </Cells>\r\n");

	fprintf(fp,"  </Piece>\r\n");
	fprintf(fp," </UnstructuredGrid>\r\n");
	fprintf(fp,"</VTKFile>");
	fclose(fp);
}

FILE *
VTKWriter::open_data_file(const char* base, string const& num, string *fname)
{
	string filename(base), full_filename;

	if (m_gdata && m_gdata->mpi_nodes > 1)
		filename += "n" + m_gdata->rankString();

	filename += "_" + num + ".vtu";
	full_filename = m_dirname + "/" + filename;

	FILE *fid = fopen(full_filename.c_str(), "w");

	if (fid == NULL) {
		stringstream ss;
		ss << "Cannot open data file " << full_filename;
		throw runtime_error(ss.str());
	}

	if (fname)
		*fname = filename;
	return fid;
}
