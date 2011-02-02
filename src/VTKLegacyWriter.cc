#include <sstream>
#include <stdexcept>

#include "VTKLegacyWriter.h"

using namespace std;

static inline void print_lookup(FILE *fid)
{
	fprintf(fid, "LOOKUP_TABLE default\n");
}

VTKLegacyWriter::VTKLegacyWriter(const Problem *problem)
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
	fprintf(m_timefile,"<?xml version=\"1.0\"?>\r\n");
	fprintf(m_timefile," <VTKFile type=\"Collection\" version=\"0.1\">\r\n");
	fprintf(m_timefile,"  <Collection>\r\n");
}


VTKLegacyWriter::~VTKLegacyWriter()
{
	if (m_timefile != NULL) {
		fprintf(m_timefile,"   </Collection>\r\n");
		fprintf(m_timefile,"  </VTKFile>\r\n");
		fclose(m_timefile);
	}
}

void VTKLegacyWriter::write(uint numParts, const float4 *pos, const float4 *vel,
							const particleinfo *info, const float3 *vort, float t)
{
	string filename, full_filename;

	filename = "PART_" + next_filenum() + ".vtk";
	full_filename = m_dirname + "/" + filename;

	FILE *fid = fopen(full_filename.c_str(), "w");

	if (fid == NULL) {
		stringstream ss;
		ss << "Cannot open data file " <<full_filename;
		throw runtime_error(ss.str());
		}

	// Header
	fprintf(fid, "# vtk DataFile Version 2.0\n%s\n", m_dirname.c_str());
	fprintf(fid, "ASCII\nDATASET UNSTRUCTURED_GRID\n");

	fprintf(fid, "POINTS %u float\n", numParts);
	// Start with particle positions
	for (int i=0; i < numParts; ++i)
		fprintf(fid, "%f %f %f\n", pos[i].x, pos[i].y, pos[i].z);
	fprintf(fid, "\n");

	// Cells = particles
	fprintf(fid, "CELLS %u %u\n", numParts, 2*numParts);
	for (int i=0; i < numParts; ++i)
		fprintf(fid, "1 %u\n", i);
	fprintf(fid, "\n");

	fprintf(fid, "CELL_TYPES %u\n", numParts);
	for (int i=0; i < numParts; ++i)
		fprintf(fid, "1\n");
	fprintf(fid, "\n");

	// Now, the data
	fprintf(fid, "POINT_DATA %u\n", numParts);

	// Velocity
	fprintf(fid, "VECTORS Velocity float\n");
	for (int i=0; i < numParts; ++i) {
		if (pos[i].w > 0.0)
			fprintf(fid, "%f %f %f\n", vel[i].x, vel[i].y, vel[i].z);
		else
			fprintf(fid, "%f %f %f\n", 0.0, 0.0, 0.0);
	}
	fprintf(fid, "\n\n");

	// Pressure
	fprintf(fid, "SCALARS Pressure float\n");
	print_lookup(fid);
	for (int i=0; i < numParts; ++i) {
		if (FLUID(info[i]) )
			fprintf(fid, "%f\n", m_problem->pressure(vel[i].w, object(info[i])));
		else
			fprintf(fid, "0.0\n");
	}
	fprintf(fid, "\n\n");

	// Density
	fprintf(fid, "SCALARS Density float\n");
	print_lookup(fid);
	for (int i=0; i < numParts; ++i) {
		if (FLUID(info[i]) > 0.0)
			fprintf(fid, "%f\n", vel[i].w);
		else
			fprintf(fid, "0.0\n");
	}
	fprintf(fid, "\n\n");

	// Mass
	fprintf(fid, "SCALARS Mass float\n");
	print_lookup(fid);
	for (int i=0; i < numParts; ++i)
		fprintf(fid, "%f\n", pos[i].w);
	fprintf(fid, "\n\n");

	// Vorticity
	if (vort) {
		fprintf(fid, "VECTORS Vorticity float\n");
		for (int i=0; i < numParts; ++i) {
			if (pos[i].w > 0.0)
				fprintf(fid, "%f %f %f\n", vort[i].x, vort[i].y, vort[i].z);
			else
				fprintf(fid, "%f %f %f\n", 0.0, 0.0, 0.0);
		}
		fprintf(fid, "\n\n");
	}


	// Info
	if (info) {
		fprintf(fid, "SCALARS Type int\n");
		print_lookup(fid);
		for (int i=0; i < numParts; ++i) {
			fprintf(fid, "%d\n", type(info[i]));
		}
		fprintf(fid, "\n\n");

		fprintf(fid, "SCALARS Object int\n");
		print_lookup(fid);
		for (int i=0; i < numParts; ++i) {
			fprintf(fid, "%d\n", type(info[i]));
		}
		fprintf(fid, "\n\n");

		fprintf(fid, "SCALARS ParticleId int\n");
		print_lookup(fid);
		for (int i=0; i < numParts; ++i) {
			fprintf(fid, "%u\n", id(info[i]));
		}
		fprintf(fid, "\n\n");
	}

	fclose(fid);

	// Writing time to VTUinp.pvd file
	// Writing time to VTUinp.pvd file
	if (m_timefile != NULL) {
		fprintf(m_timefile,"<DataSet timestep=\"%f\" group=\"\" part=\"%d\" file=\"%s\"/>\r\n",
			t, 0, filename.c_str());
		}}
