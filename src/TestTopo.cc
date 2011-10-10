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

#include <cmath>
#include <fstream>
#include <string>
#include <iostream>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "TestTopo.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include <cstdlib> // for exit(int)


TestTopo::TestTopo(const Options &options) : Problem(options)
{
	const char* dem_file;
	if (options.dem.empty())
		dem_file = "../half_wave0.1m.txt";
	else
		dem_file = options.dem.c_str();

	// Reading DEM
	ifstream fdem(dem_file);
	if (!fdem.good())
	{
		cerr << "Unable to open " << dem_file << endl;
		exit(1);
	}

	string s;

	for (int i = 1; i <= 6; i++) {
			fdem >> s;
			if (s.find("north:") != string::npos) fdem >> north;
			else if (s.find("south:") != string::npos) fdem >> south;
			else if (s.find("east:") != string::npos) fdem >> east;
			else if (s.find("west:") != string::npos) fdem >> west;
			else if (s.find("cols:") != string::npos) fdem >> m_ncols;
			else if (s.find("rows:") != string::npos) fdem >> m_nrows;
			}
	double zmin = 1e6, zmax = 0;
	nsres = (north - south)/(m_nrows - 1);
	ewres = (east - west)/(m_ncols - 1);
	m_dem = new float[m_ncols*m_nrows];

	// Reading dem data
	for (int i = 0; i < m_ncols*m_nrows; i++) {
		double z;
		fdem >> z;
		//z /= 2.0;
		zmax = std::max(z, zmax);
		zmin = std::min(z, zmin);
		m_dem[i] = z;
		}
	fdem.close();
	for (int i = 0; i < m_ncols*m_nrows; i++) {
		m_dem[i] -= zmin;
		}
	std::cout << "zmin=" << zmin << "\n";
	std::cout << "zmax=" << zmax << "\n";
	std::cout << "nsres=" << nsres << "\n";
	std::cout << "ewres=" << ewres << "\n";

	// Size and origin of the simulation domain
	m_writerType = VTKWRITER;

	// SPH parameters
	set_deltap(0.2);
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.00001f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = KINEMATICVISC;
	m_simparams.mbcallback = false;
	m_simparams.usedem = true;

	// Physical parameters
	H = 2.0;
	//nsres = 1; ewres = 1;
	m_size = make_float3(ewres*((float) m_ncols), nsres*((float) m_nrows), H);
	cout << "m_size: " << m_size.x << " " << m_size.y << " " << m_size.z << "\n";

	m_origin = make_float3(0.0f, 0.0f, 0.0f);
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0, 1000.0f, 7.0f, 200*H);

	m_physparams.dcoeff = 50.47;
    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.r0 = m_deltap;
	m_physparams.artvisccoeff = 0.05f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	m_physparams.epsxsph = 0.5f;
	m_physparams.ewres = ewres;
	m_physparams.nsres = nsres;
	m_physparams.demdx = ewres/5.0;
	m_physparams.demdy = nsres/5.0;
	m_physparams.demdx = ewres/5.0;
	m_physparams.demdxdy = m_physparams.demdx*m_physparams.demdy;
	m_physparams.demzmin = 5.0*m_deltap;
	// Scales for drawing
	m_maxrho = density(H,0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	//m_maxvel = sqrt(m_physparams.gravity*H);
	m_maxvel = 18.0f;

	// Drawing and saving times
	m_displayinterval = 0.001f;
	m_writefreq = 0;
	m_screenshotfreq = 10;

	// Name of problem used for directory creation
	m_name = "TestTopo";
	create_problem_dir();
}


TestTopo::~TestTopo(void)
{
	release_memory();
}


void TestTopo::release_memory(void)
{
	delete [] m_dem;
	parts.clear();
	boundary_parts.clear();
	piston_parts.clear();
}


int TestTopo::fill_parts()
{
	experiment_box.SetCubeDem(H, m_dem, m_ncols, m_nrows, nsres, ewres, false);
	parts.reserve(14000);
	boundary_parts.reserve(14000);

	experiment_box.SetPartMass(m_deltap, m_physparams.rho0[0]);
	//experiment_box.FillDem(boundary_parts, m_physparams.r0);
	experiment_box.FillBorder(boundary_parts, m_physparams.r0, 0, false);
	experiment_box.FillBorder(boundary_parts, m_physparams.r0, 1, true);
	experiment_box.FillBorder(boundary_parts, m_physparams.r0, 2, false);
	experiment_box.FillBorder(boundary_parts, m_physparams.r0, 3, true);
	experiment_box.Fill(parts, 0.8, m_deltap, true);

	return boundary_parts.size() + parts.size();
}


void TestTopo::draw_boundary(float t)
{
	experiment_box.GLDraw();
}


void TestTopo::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
Point  p ;
	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,0,i);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		pos[i] = make_float4(parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}
