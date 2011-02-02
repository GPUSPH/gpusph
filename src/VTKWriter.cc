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
	fprintf(m_timefile,"<?xml version=\"1.0\"?>\r\n");
	fprintf(m_timefile," <VTKFile type=\"Collection\" version=\"0.1\">\r\n");
	fprintf(m_timefile,"  <Collection>\r\n");
}


VTKWriter::~VTKWriter()
{
	if (m_timefile != NULL) {
		fprintf(m_timefile,"   </Collection>\r\n");
		fprintf(m_timefile,"  </VTKFile>\r\n");
		fclose(m_timefile);
	}
}

void VTKWriter::write(uint numParts, const float4 *pos, const float4 *vel,
				const particleinfo *info, const float3 *vort, float t)
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
	fprintf(fid,"<?xml version=\"1.0\"?>\r\n");
	fprintf(fid,"<VTKFile type= \"UnstructuredGrid\"  version= \"0.1\"  byte_order= \"BigEndian\">\r\n");
	fprintf(fid," <UnstructuredGrid>\r\n");
	fprintf(fid,"  <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\r\n", numParts, numParts);

	fprintf(fid,"   <PointData Scalars=\"Pressure\" Vectors=\"Velocity\">\r\n");

	// Writing pressure
	fprintf(fid,"	<DataArray type=\"Float32\" Name=\"Pressure\" format=\"ascii\">\r\n");
	for (int i=0; i < numParts; i++)
		if (FLUID(info[i]))
			fprintf(fid,"%f\t",m_problem->pressure(vel[i].w, object(info[i])));
		else
			fprintf(fid,"%f\t", 0.0);
	fprintf(fid,"\r\n");
	fprintf(fid,"	</DataArray>\r\n");

	// Writing density
	fprintf(fid,"	<DataArray type=\"Float32\" Name=\"Density\" format=\"ascii\">\r\n");
	for (int i=0; i < numParts; i++)
		if (FLUID(info[i]))
			fprintf(fid,"%f\t",vel[i].w);
		else
			fprintf(fid,"%f\t", 0.0);
	fprintf(fid,"\r\n");
	fprintf(fid,"	</DataArray>\r\n");

	 // Writing mass
	fprintf(fid,"	<DataArray type=\"Float32\" Name=\"Mass\" format=\"ascii\">\r\n");
	for (int i=0; i < numParts; i++)
		fprintf(fid,"%f\t",pos[i].w);
	fprintf(fid,"\r\n");
	fprintf(fid,"	</DataArray>\r\n");

	// Writing particle info
	if (info) {
		fprintf(fid,"	<DataArray type=\"Int16\" Name=\"Part type\" format=\"ascii\">\r\n");
		for (int i=0; i < numParts; i++)
			fprintf(fid,"%d\t", type(info[i]));
		fprintf(fid,"\r\n");
		fprintf(fid,"	</DataArray>\r\n");

		fprintf(fid,"	<DataArray type=\"Int16\" Name=\"Part object\" format=\"ascii\">\r\n");
		for (int i=0; i < numParts; i++)
			fprintf(fid,"%d\t", object(info[i]));
		fprintf(fid,"\r\n");
		fprintf(fid,"	</DataArray>\r\n");

		fprintf(fid,"	<DataArray type=\"UInt32\" Name=\"Part id\" format=\"ascii\">\r\n");
		for (int i=0; i < numParts; i++)
			fprintf(fid,"%u\t", id(info[i]));
		fprintf(fid,"\r\n");
		fprintf(fid,"	</DataArray>\r\n");
		fprintf(fid, "\n\n");
	}

	// Writing velocity
	fprintf(fid,"	<DataArray type=\"Float32\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"ascii\">\r\n");
	for (int i=0; i < numParts; i++)
		if (FLUID(info[i]))
			fprintf(fid,"%f\t%f\t%f\t",vel[i].x, vel[i].y, vel[i].z);
		else
			fprintf(fid,"%f\t%f\t%f\t",0.0, 0.0, 0.0);
	fprintf(fid,"\r\n");
	fprintf(fid,"	</DataArray>\r\n");

	// Writing vorticity
	if (vort) {
		fprintf(fid,"	<DataArray type=\"Float32\" Name=\"Vorticity\" NumberOfComponents=\"3\" format=\"ascii\">\r\n");
		for (int i=0; i < numParts; i++)
			if (FLUID(info[i]))
				fprintf(fid,"%f\t%f\t%f\t",vort[i].x, vort[i].y, vort[i].z);
			else
				fprintf(fid,"%f\t%f\t%f\t",0.0, 0.0, 0.0);
		fprintf(fid,"\r\n");
		fprintf(fid,"	</DataArray>\r\n");
	}

	fprintf(fid,"   </PointData>\r\n");

	// Writing position
	fprintf(fid,"   <Points>\r\n");
	fprintf(fid,"	<DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"ascii\">\r\n");
	for (int i=0; i < numParts; i++)
		fprintf(fid,"%f\t%f\t%f\t",pos[i].x, pos[i].y, pos[i].z);
	fprintf(fid,"\r\n");
	fprintf(fid,"	</DataArray>\r\n");
	fprintf(fid,"   </Points>\r\n");

	// Cells data
	fprintf(fid,"   <Cells>\r\n");
	fprintf(fid,"	<DataArray type=\"Int32\" Name=\"connectivity\" format=\"ascii\">\r\n");
	for (int i = 0; i < numParts; i++)
		fprintf(fid,"%d\t", i);
	fprintf(fid,"\r\n");
	fprintf(fid,"	</DataArray>\r\n");
	fprintf(fid,"\r\n");

	fprintf(fid,"	<DataArray type=\"Int32\" Name=\"offsets\" format=\"ascii\">\r\n");
	for (int i = 0; i < numParts; i++)
		fprintf(fid,"%d\t", i + 1);
	fprintf(fid,"\r\n");
	fprintf(fid,"	</DataArray>\r\n");

	fprintf(fid,"\r\n");
	fprintf(fid,"	<DataArray type=\"Int32\" Name=\"types\" format=\"ascii\">\r\n");
	for (int i = 0; i < numParts; i++)
		fprintf(fid,"%d\t", 1);
	fprintf(fid,"\r\n");
	fprintf(fid,"	</DataArray>\r\n");

	fprintf(fid,"   </Cells>\r\n");

	fprintf(fid,"  </Piece>\r\n");
	fprintf(fid," </UnstructuredGrid>\r\n");
	fprintf(fid,"</VTKFile>");

	fclose(fid);

	// Writing time to VTUinp.pvd file
	if (m_timefile != NULL) {
		fprintf(m_timefile,"<DataSet timestep=\"%f\" group=\"\" part=\"%d\" file=\"%s\"/>\r\n",
			t, 0, filename.c_str());
		}
}
