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

#include "VTKLegacyWriter.h"
#include "GlobalData.h"

using namespace std;

static inline void print_lookup(ofstream &fid)
{
	fid << "LOOKUP_TABLE default" << endl;
}

VTKLegacyWriter::VTKLegacyWriter(const GlobalData *_gdata)
  : Writer(_gdata)
{
	m_fname_sfx = ".vtk";

	string time_fname = open_data_file(m_timefile, "VTUinp", "", ".pvd");

	// Writing header of VTUinp.pvd file
	if (m_timefile) {
		m_timefile << "<?xml version='1.0'?>\n";
		m_timefile << "<VTKFile type='Collection' version='0.1'>\n";
		m_timefile << " <Collection>\n";
	}
}


VTKLegacyWriter::~VTKLegacyWriter()
{
	mark_timefile();
	m_timefile.close();
}

void
VTKLegacyWriter::write(uint numParts, BufferList const& buffers, uint node_offset, double t, const bool testpoints)
{
	const double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();
	const float4 *vel = buffers.getData<BUFFER_VEL>();
	const particleinfo *info = buffers.getData<BUFFER_INFO>();
	const float3 *vort = buffers.getData<BUFFER_VORTICITY>();

	ofstream fid;
	string filename = open_data_file(fid, "PART", current_filenum());

	// Header
	fid << "# vtk DataFile Version 2.0\n" << m_dirname << endl;
	fid << "ASCII\nDATASET UNSTRUCTURED_GRID" << endl;

	fid << "POINTS " << numParts << "double" << endl;

	// Start with particle positions
	for (uint i=0; i < numParts; ++i)
		fid << pos[i].x << " " << pos[i].y << " " << pos[i].z << endl;
	fid << endl;

	// Cells = particles
	fid << "CELLS " << numParts << " " << (2*numParts) << endl;
	for (uint i=0; i < numParts; ++i)
		fid << "1  " << i << endl;
	fid << endl;

	fid << "CELL_TYPES " << numParts << endl;
	for (uint i=0; i < numParts; ++i)
		fid << "1\n";
	fid << endl;

	// Now, the data
	fid << "POINT_DATA " << numParts << endl;

	// Velocity
	fid << "VECTORS Velocity float" << endl;
	for (uint i=0; i < numParts; ++i)
		fid << vel[i].x << " " << vel[i].y << " " << vel[i].z << endl;
	fid << endl;

	// Pressure
	fid << "SCALARS Pressure float" << endl;
	print_lookup(fid);
	for (uint i=0; i < numParts; ++i) {
		float value = 0.0;
		if (TESTPOINT(info[i]))
			value = vel[i].w;
		else
			value = m_problem->pressure(vel[i].w, fluid_num(info[i]));

		fid << value << endl;
	}
	fid << endl;

	// Density
	fid << "SCALARS Density float" << endl;
	print_lookup(fid);
	for (uint i=0; i < numParts; ++i) {
		float value = 0.0;
		if (TESTPOINT(info[i]))
			// TODO FIXME: Testpoints compute pressure only
			// In the future we would like to have a density here
			// but this needs to be done correctly for multifluids
			value = NAN;
		else
			value = vel[i].w;
		fid << value << endl;
	}
	fid << endl;

	// Mass
	fid << "SCALARS Mass float" << endl;
	print_lookup(fid);
	for (uint i=0; i < numParts; ++i)
		fid << pos[i].w << endl;
	fid << endl;

	// Vorticity
	if (vort) {
		fid << "VECTORS Vorticity float" << endl;
		for (uint i=0; i < numParts; ++i) {
			if (FLUID(info[i]))
				fid << vort[i].x << " " << vort[i].y << " " << vort[i].z << endl;
			else
				fid << "0.0 0.0 0.0" << endl;
		}
		fid << endl;
	}

	// Info
	/* Fluid number is only included if there are more than 1 */
	bool write_fluid_num = (gdata->problem->physparams()->numFluids() > 1);

	/* Object number is only included if there are any */
	// TODO a better way would be for GPUSPH to expose the highest
	// object number ever associated with any particle, so that we
	// could check that
	bool write_part_obj = (gdata->problem->simparams()->numbodies > 0);
	if (info) {
		fid << "SCALARS Type+flags int" << endl;
		print_lookup(fid);
		for (uint i=0; i < numParts; ++i)
			fid << type(info[i]) << endl;
		fid << endl;

		if (write_fluid_num || write_part_obj) {
			if (write_fluid_num)
				fid << "SCALARS Fluid int" << endl;
			else
				fid << "SCALARS Object int" << endl;
			print_lookup(fid);
			for (uint i=0; i < numParts; ++i)
				fid << object(info[i]) << endl;
			fid << endl;
		}

		fid << "SCALARS ParticleId int" << endl;
		print_lookup(fid);
		for (uint i=0; i < numParts; ++i)
			fid << id(info[i]) << endl;
		fid << endl;

	}

	fid.close();

	// Writing time to VTUinp.pvd file
	if (m_timefile) {
		// TODO should node info for multinode be stored in group or part?
		m_timefile << "<DataSet timestep='" << t << "' group='' part='0' "
			<< "file='" << filename << "'/>" << endl;
		mark_timefile();
	}
}

void
VTKLegacyWriter::mark_timefile()
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
