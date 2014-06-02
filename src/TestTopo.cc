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

#include <cmath>

#include <iostream>
#include <stdexcept>

#include "TestTopo.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"

// set to 0 to use boundary particles, 1 to use boundary planes
#define USE_PLANES 1

#define EB experiment_box

TestTopo::TestTopo(const GlobalData *_gdata) : Problem(_gdata)
{
	const char* dem_file;
	if (m_options->dem.empty())
		dem_file = "half_wave0.1m.txt";
	else
		dem_file = m_options->dem.c_str();

	EB = TopoCube::load_ascii_grid(dem_file);

	std::cout << "zmin=" << -EB->get_voff() << "\n";
	std::cout << "zmax=" << EB->get_H() << "\n";
	std::cout << "ncols=" << EB->get_ncols() << "\n";
	std::cout << "nrows=" << EB->get_nrows() << "\n";
	std::cout << "nsres=" << EB->get_nsres() << "\n";
	std::cout << "ewres=" << EB->get_ewres() << "\n";

	// Size and origin of the simulation domain
	set_dem(EB->get_dem(), EB->get_ncols(), EB->get_nrows());

	// SPH parameters
	set_deltap(0.05);
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

	EB->SetCubeHeight(H);

	// TODO FIXME adapt DEM to homogeneous precision
	m_size = make_double3(
			EB->get_vx()(0), // x component of vx
			EB->get_vy()(1), // y component of vy
			H);
	cout << "m_size: " << m_size.x << " " << m_size.y << " " << m_size.z << "\n";

	m_origin = make_double3(0.0, 0.0, 0.0);
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);
	m_physparams.set_density(0, 1000.0f, 7.0f, 20.f);

	m_physparams.dcoeff = 50.47;
    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.r0 = m_deltap;
	m_physparams.artvisccoeff = 0.05f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	m_physparams.epsxsph = 0.5f;

	m_physparams.ewres = EB->get_ewres();
	m_physparams.nsres = EB->get_nsres();
	m_physparams.demdx = EB->get_ewres()/5.0;
	m_physparams.demdy = EB->get_nsres()/5.0;
	m_physparams.demdx = EB->get_ewres()/5.0;
	m_physparams.demdxdy = m_physparams.demdx*m_physparams.demdy;
	m_physparams.demzmin = 5.0*m_deltap;

#undef EB

	// Drawing and saving times
	set_timer_tick(0.001f);
	add_writer(VTKWRITER, 100);

	// Name of problem used for directory creation
	m_name = "TestTopo";
}


TestTopo::~TestTopo(void)
{
	release_memory();
}


void TestTopo::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
	piston_parts.clear();
}


int TestTopo::fill_parts()
{
	parts.reserve(1000);
	boundary_parts.reserve(1000);

	experiment_box->SetPartMass(m_deltap, m_physparams.rho0[0]);
	//experiment_box->FillDem(boundary_parts, m_physparams.r0);
#if !USE_PLANES
	experiment_box->FillBorder(boundary_parts, m_physparams.r0, 0, false);
	experiment_box->FillBorder(boundary_parts, m_physparams.r0, 1, true);
	experiment_box->FillBorder(boundary_parts, m_physparams.r0, 2, false);
	experiment_box->FillBorder(boundary_parts, m_physparams.r0, 3, true);
#endif
	experiment_box->Fill(parts, 0.8, m_deltap, true);

	return boundary_parts.size() + parts.size();
}

uint TestTopo::fill_planes()
{
#if USE_PLANES
	return 4;
#else
	return 0;
#endif
}

void TestTopo::copy_planes(float4 *planes, float *planediv)
{
	// planes are defined as a x + by +c z + d= 0

	experiment_box->get_planes(planes, planediv);
}

void TestTopo::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,0,i);
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}

void TestTopo::fillDeviceMap()
{
	// force split along Y axis: X is longer but part of the domain is only
	// DEM, so it is not convenient to split evenly along X
	fillDeviceMapByAxis(Y_AXIS);
}
