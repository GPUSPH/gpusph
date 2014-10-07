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
#include <fstream>
#include <stdexcept>

#include "VTKWriter.h"
// GlobalData is required for writing the device index. With some order
// of inclusions, a forward declaration might be required
#include "GlobalData.h"

using namespace std;

// TODO for the time being, we assume no more than 256 devices
// upgrade to UInt16 / ushort if it's ever needed

typedef unsigned char dev_idx_t;
static const char dev_idx_str[] = "UInt8";

VTKWriter::VTKWriter(const GlobalData *_gdata)
  : Writer(_gdata)
{
	m_fname_sfx = ".vtu";

	string time_fname = open_data_file(m_timefile, "VTUinp", "", ".pvd");

	// Writing header of VTUinp.pvd file
	if (m_timefile) {
		m_timefile << "<?xml version='1.0'?>\n";
		m_timefile << "<VTKFile type='Collection' version='0.1'>\n";
		m_timefile << " <Collection>\n";
	}
}


VTKWriter::~VTKWriter()
{
	mark_timefile();
	m_timefile.close();
}

/* Endianness check: (char*)&endian_int reads the first byte of the int,
 * which is 0 on big-endian machines, and 1 in little-endian machines */
static int endian_int=1;
static const char* endianness[2] = { "BigEndian", "LittleEndian" };

static float zeroes[4];

/* auxiliary functions to write data array entrypoints */
inline void
scalar_array(ofstream &out, const char *type, const char *name, size_t offset)
{
	out << "	<DataArray type='" << type << "' Name='" << name
		<< "' format='appended' offset='" << offset << "'/>" << endl;
}

inline void
vector_array(ofstream &out, const char *type, const char *name, uint dim, size_t offset)
{
	out << "	<DataArray type='" << type << "' Name='" << name
		<< "' NumberOfComponents='" << dim
		<< "' format='appended' offset='" << offset << "'/>" << endl;
}

inline void
vector_array(ofstream &out, const char *type, uint dim, size_t offset)
{
	out << "	<DataArray type='" << type
		<< "' NumberOfComponents='" << dim
		<< "' format='appended' offset='" << offset << "'/>" << endl;
}

// Binary dump a single variable of a given type
template<typename T>
inline void
write_var(ofstream &out, T const& var)
{
	out.write(reinterpret_cast<const char *>(&var), sizeof(T));
}

// Binary dump an array of variables of given type and size
template<typename T>
inline void
write_arr(ofstream &out, T const *var, size_t len)
{
	out.write(reinterpret_cast<const char *>(var), sizeof(T)*len);
}


void
VTKWriter::write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints)
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
	const float4 *eulervel = buffers.getData<BUFFER_EULERVEL>();
	const float *priv = buffers.getData<BUFFER_PRIVATE>();

	// CSV file for tespoints
	string testpoints_fname = m_dirname + "/testpoints/testpoints_" + current_filenum() + ".csv";
	ofstream testpoints_file;
	if (gdata->problem->get_simparams()->csvtestpoints) {
		testpoints_file.open(testpoints_fname.c_str());
		if (!testpoints_file) {
			stringstream ss;
			ss << "Cannot open testpoints file " << testpoints_fname;
			throw runtime_error(ss.str());
		}
		// write CSV header
		testpoints_file << "T,ID,Pressure,Object,CellIndex,PosX,PosY,PosZ,VelX,VelY,VelZ,Tke,Eps" << endl;
	}

	string filename;

	ofstream fid;
	filename = open_data_file(fid, "PART", next_filenum());

	// Header
	//====================================================================================
	fid << "<?xml version='1.0'?>" << endl;
	fid << "<VTKFile type='UnstructuredGrid'  version='0.1'  byte_order='" <<
		endianness[*(char*)&endian_int & 1] << "'>" << endl;
	fid << " <UnstructuredGrid>" << endl;
	fid << "  <Piece NumberOfPoints='" << numParts << "' NumberOfCells='" << numParts << "'>" << endl;
	fid << "   <PointData Scalars='Pressure' Vectors='Velocity'>" << endl;

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

	/* Fluid number is only included if there are more than 1 */
	bool write_fluid_num = (gdata->problem->get_physparams()->numFluids > 1);

	/* Object number is only included if there are any */
	// TODO a better way would be for GPUSPH to expose the highest
	// object number ever associated with any particle, so that we
	// could check that
	bool write_part_obj = (gdata->problem->get_simparams()->numODEbodies > 0);

	// particle info
	// TODO check the highest part type/flag/fluid/object and select the type
	// appropriately; presently none of it is > 256, so assume UInt8 suffices
	if (info) {
		scalar_array(fid, "UInt8", "Part type", offset);
		offset += sizeof(uchar)*numParts+sizeof(int);
		// TODO don't write Part flag unless it's needed
		scalar_array(fid, "UInt8", "Part flag", offset);
		offset += sizeof(uchar)*numParts+sizeof(int);
		if (write_fluid_num) {
			scalar_array(fid, "UInt8", "Fluid number", offset);
			offset += sizeof(uchar)*numParts+sizeof(int);
		}
		if (write_part_obj) {
			scalar_array(fid, "UInt8", "Part object", offset);
			offset += sizeof(uchar)*numParts+sizeof(int);
		}
		scalar_array(fid, "UInt32", "Part id", offset);
		offset += sizeof(uint)*numParts+sizeof(int);
	}

	// device index
	if (MULTI_DEVICE) {
		scalar_array(fid, dev_idx_str, "DeviceIndex", offset);
		offset += sizeof(dev_idx_t)*numParts+sizeof(int);
	}

	// cell index
	scalar_array(fid, "UInt32", "CellIndex", offset);
	offset += sizeof(uint)*numParts+sizeof(int);

	// velocity
	vector_array(fid, "Float32", "Velocity", 3, offset);
	offset += sizeof(float)*3*numParts+sizeof(int);

	if (eulervel) {
		// Eulerian velocity
		vector_array(fid, "Float32", "Eulerian velocity", 3, offset);
		offset += sizeof(float)*3*numParts+sizeof(int);

		// Eulerian pressure
		scalar_array(fid, "Float32", "Eulerian pressure", offset);
		offset += sizeof(float)*numParts+sizeof(int);
	}

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

	fid << "   </PointData>" << endl;

	// position
	fid << "   <Points>" << endl;
	vector_array(fid, "Float64", 3, offset);
	offset += sizeof(double)*3*numParts+sizeof(int);
	fid << "   </Points>" << endl;

	// Cells data
	fid << "   <Cells>" << endl;
	scalar_array(fid, "Int32", "connectivity", offset);
	offset += sizeof(uint)*numParts+sizeof(int);
	scalar_array(fid, "Int32", "offsets", offset);
	offset += sizeof(uint)*numParts+sizeof(int);
	scalar_array(fid, "UInt8", "types", offset);
	offset += sizeof(uchar)*numParts+sizeof(int);
	fid << "   </Cells>" << endl;
	fid << "  </Piece>" << endl;

	fid << " </UnstructuredGrid>" << endl;
	fid << " <AppendedData encoding='raw'>\n_";
	//====================================================================================

	int numbytes=sizeof(float)*numParts;

	// pressure
	write_var(fid, numbytes);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		float value = 0.0;
		if (TESTPOINTS(info[i]))
			value = vel[i].w;
		else
			value = m_problem->pressure(vel[i].w, object(info[i]));
		write_var(fid, value);
	}

	// density
	write_var(fid, numbytes);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		float value = 0.0;
		if (TESTPOINTS(info[i]))
			// TODO FIXME: Testpoints compute pressure only
			// In the future we would like to have a density here
			// but this needs to be done correctly for multifluids
			value = NAN;
		else
			value = vel[i].w;
		write_var(fid, value);
	}

	// mass
	write_var(fid, numbytes);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		float value = pos[i].w;
		write_var(fid, value);
	}

	// gamma
	if (gradGamma) {
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = gradGamma[i].w;
			write_var(fid, value);
		}
	}

	// turbulent kinetic energy
	if (tke) {
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = tke[i];
			write_var(fid, value);
		}
	}

	// turbulent epsilon
	if (eps) {
		write_var(fid, numbytes);
		for (uint i=0; i < numParts; i++) {
			float value = eps[i];
			write_var(fid, value);
		}
	}

	// eddy viscosity
	if (turbvisc) {
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = turbvisc[i];
			write_var(fid, value);
		}
	}

	// particle info
	if (info) {
		numbytes=sizeof(uchar)*numParts;

		// type
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			uchar value = PART_TYPE(info[i]);
			if (gdata->problem->get_simparams()->csvtestpoints && value == (TESTPOINTSPART >> MAX_FLUID_BITS)) {
				float tkeVal = 0.0f;
				float epsVal = 0.0f;
				if(tke)
					tkeVal = tke[i];
				if(eps)
					epsVal = eps[i];

				testpoints_file << t << ","
					<< id(info[i]) << ","
					<< vel[i].w << ","
					<< object(info[i]) << ","
					<< cellHashFromParticleHash( particleHash[i] ) << ","
					<< pos[i].x << ","
					<< pos[i].y << ","
					<< pos[i].z << ","
					<< vel[i].x << ","
					<< vel[i].y << ","
					<< vel[i].z << ","
					<< tkeVal << ","
					<< epsVal << endl;
			}
			write_var(fid, value);
		}

		// flag
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			uchar value = PART_FLAG(info[i]);
			write_var(fid, value);
		}

		// fluid number
		if (write_fluid_num) {
			write_var(fid, numbytes);
			for (uint i=node_offset; i < node_offset + numParts; i++) {
				uchar value = PART_FLUID_NUM(info[i]);
				write_var(fid, value);
			}
		}

		// object
		if (write_part_obj) {
			write_var(fid, numbytes);
			for (uint i=node_offset; i < node_offset + numParts; i++) {
				uchar value = object(info[i]);
				write_var(fid, value);
			}
		}

		numbytes=sizeof(uint)*numParts;

		// id
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			uint value = id(info[i]);
			write_var(fid, value);
		}
	}

	// device index
	if (MULTI_DEVICE) {
		numbytes = sizeof(dev_idx_t)*numParts;
		write_var(fid, numbytes);
		// The previous way was to compute the theoretical containing cell solely according on the particle position. This, however,
		// was inconsistent with the actual particle distribution among the devices, since one particle can be physically out of the
		// containing cell until next calchash/reorder.
		// The current policy is: just list the particles according to how the global array is partitioned. In other words, we rely
		// on the particle index to understad which device downloaded the particle data.
		for (uint d = 0; d < gdata->devices; d++) {
			// compute the global device ID for each device
			dev_idx_t value = gdata->GLOBAL_DEVICE_ID(gdata->mpi_rank, d);
			// write one for each particle (no need for the "absolute" particle index)
			for (uint p = 0; p < gdata->s_hPartsPerDevice[d]; p++)
				write_var(fid, value);
		}
		// There two alternate policies: 1. use particle hash or 2. compute belonging device.
		// To use the particle hash, instead of just relying on the particle index, use the following code:
		/*
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			uint value = gdata->s_hDeviceMap[ cellHashFromParticleHash(particleHash[i]) ];
			write_var(fid, value);
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
	write_var(fid, numbytes);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		uint value = cellHashFromParticleHash( particleHash[i] );
		write_var(fid, value);
	}

	numbytes=sizeof(float)*3*numParts;

	// velocity
	write_var(fid, numbytes);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		float *value = zeroes;
		//if (FLUID(info[i]) || TESTPOINTS(info[i]))
			value = (float*)(vel + i);
		write_arr(fid, value, 3);
	}

	if (eulervel) {
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			value = (float*)(eulervel + i);
			write_arr(fid, value, 3);
		}

		numbytes=sizeof(float)*numParts;

		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = eulervel[i].w;
			write_var(fid, value);
		}

		numbytes=sizeof(float)*3*numParts;
	}

	// gradient gamma
	if (gradGamma) {
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			value = (float*)(gradGamma + i);
			write_arr(fid, value, 3);
		}
	}

	// vorticity
	if (vort) {
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			if (FLUID(info[i])) {
				value = (float*)(vort + i);
			}
			write_arr(fid, value, 3);
		}
	}

	// normals
	if (normals) {
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float *value = zeroes;
			if (FLUID(info[i])) {
				value = (float*)(normals + i);
			}
			write_arr(fid, value, 3);
		}

		numbytes=sizeof(float)*numParts;
		// criteria
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = 0;
			if (FLUID(info[i]))
				value = normals[i].w;
			write_var(fid, value);
		}
	}

	numbytes=sizeof(float)*numParts;

	// private
	if (priv) {
		write_var(fid, numbytes);
		for (uint i=node_offset; i < node_offset + numParts; i++) {
			float value = priv[i];
			write_var(fid, value);
		}
	}

	numbytes=sizeof(double)*3*numParts;

	// position
	write_var(fid, numbytes);
	for (uint i=node_offset; i < node_offset + numParts; i++) {
		double *value = (double*)(pos + i);
		write_arr(fid, value, 3);
	}

	numbytes=sizeof(int)*numParts;
	// connectivity
	write_var(fid, numbytes);
	for (uint i=0; i < numParts; i++) {
		uint value = i;
		write_var(fid, value);
	}
	// offsets
	write_var(fid, numbytes);
	for (uint i=0; i < numParts; i++) {
		uint value = i+1;
		write_var(fid, value);
	}

	// types (currently all cells type=1, single vertex, the particle)
	numbytes=sizeof(uchar)*numParts;
	write_var(fid, numbytes);
	for (uint i=0; i < numParts; i++) {
		uchar value = 1;
		write_var(fid, value);
	}

	fid << " </AppendedData>" << endl;
	fid << "</VTKFile>" << endl;

	// Writing time to VTUinp.pvd file
	if (m_timefile) {
		// TODO should node info for multinode be stored in group or part?
		m_timefile << "<DataSet timestep='" << t << "' group='' part='0' "
			<< "file='" << filename << "'/>" << endl;
		mark_timefile();
	}

	// close testpoints file
	if (testpoints_file)
		testpoints_file.close();
}

void
VTKWriter::write_WaveGage(double t, GageList const& gage)
{
	// call the generic write_WaveGage first
	Writer::write_WaveGage(t, gage);

	ofstream fp;
	open_data_file(fp, "WaveGage", current_filenum());
	size_t num = gage.size();

	// Header
	fp << "<?xml version='1.0'?>" << endl;
	fp << "<VTKFile type= 'UnstructuredGrid'  version= '0.1'  byte_order= 'BigEndian'>" << endl;
	fp << " <UnstructuredGrid>" << endl;
	fp << "  <Piece NumberOfPoints='" << num << "' NumberOfCells='" << num << "'>" << endl;

	//Writing Position
	fp << "   <Points>" << endl;
	fp << "	<DataArray type='Float32' NumberOfComponents='3' format='ascii'>" << endl;
	for (size_t i=0; i <  num; i++)
		fp << gage[i].x << "\t" << gage[i].y << "\t" << gage[i].z << "\t";
	fp << endl;
	fp << "	</DataArray>" << endl;
	fp << "   </Points>" << endl;

	// Cells data
	fp << "   <Cells>" << endl;
	fp << "	<DataArray type='Int32' Name='connectivity' format='ascii'>" << endl;
	for (size_t i = 0; i < num; i++)
		fp << i << "\t" ;
	fp << endl;
	fp << "	</DataArray>" << endl;
	fp << "" << endl;

	fp << "	<DataArray type='Int32' Name='offsets' format='ascii'>" << endl;
	for (size_t i = 0; i < num; i++)
		fp << (i+1) << "\t" ;
	fp << endl;
	fp << "	</DataArray>" << endl;

	fp << "" << endl;
	fp << "	<DataArray type='Int32' Name='types' format='ascii'>" << endl;
	for (size_t i = 0; i < num; i++)
		fp << 1 << "\t" ;
	fp << endl;
	fp << "	</DataArray>" << endl;

	fp << "   </Cells>" << endl;

	fp << "  </Piece>" << endl;
	fp << " </UnstructuredGrid>" << endl;
	fp << "</VTKFile>" <<endl;

	fp.close();
}

void
VTKWriter::mark_timefile()
{
	if (!m_timefile)
		return;
	// Mark the current position, close the XML, go back
	// to the marked position
	ofstream::pos_type mark = m_timefile.tellp();
	m_timefile << " </Collection>\n";
	m_timefile << "</VTKFile>" << endl;
	m_timefile.seekp(mark);
}
