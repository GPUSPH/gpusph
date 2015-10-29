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
#include "cudasimframework.cu"

// set to 0 to use boundary particles, 1 to use boundary planes
#define USE_PLANES 1

#define EB experiment_box

TestTopo::TestTopo(GlobalData *_gdata) : Problem(_gdata)
{
	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		//viscosity<KINEMATICVISC>,
		add_flags<ENABLE_DEM | (USE_PLANES ? ENABLE_PLANES : ENABLE_NONE)>
	);

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
	simparams()->dt = 0.00001f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;

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
	physparams()->gravity = make_float3(0.0, 0.0, -9.81f);

	add_fluid(1000.0f);
	set_equation_of_state(0,  7.0f, 20.f);

	physparams()->dcoeff = 50.47;
    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	physparams()->r0 = m_deltap;
	physparams()->artvisccoeff = 0.05f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	physparams()->epsxsph = 0.5f;

	physparams()->ewres = EB->get_ewres();
	physparams()->nsres = EB->get_nsres();
	physparams()->demdx = EB->get_ewres()/5.0;
	physparams()->demdy = EB->get_nsres()/5.0;
	physparams()->demdx = EB->get_ewres()/5.0;
	physparams()->demdxdy = physparams()->demdx*physparams()->demdy;
	physparams()->demzmin = 5.0*m_deltap;

#undef EB

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);

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

	experiment_box->SetPartMass(m_deltap, physparams()->rho0[0]);
	//experiment_box->FillDem(boundary_parts, physparams()->r0);
#if !USE_PLANES
	experiment_box->FillBorder(boundary_parts, physparams()->r0, 0, false);
	experiment_box->FillBorder(boundary_parts, physparams()->r0, 1, true);
	experiment_box->FillBorder(boundary_parts, physparams()->r0, 2, false);
	experiment_box->FillBorder(boundary_parts, physparams()->r0, 3, true);
#endif
	experiment_box->Fill(parts, 0.8, m_deltap, true);

	return boundary_parts.size() + parts.size();
}

void TestTopo::copy_planes(PlaneList& planes)
{
#if USE_PLANES
	std::vector<double4> box_plane( experiment_box->get_planes() );
	for (size_t i = 0; i < box_plane.size(); ++i)
		planes.push_back(implicit_plane(box_plane[i]));
#endif
}

void TestTopo::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i]= make_particleinfo(PT_BOUNDARY,0,i);
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i]= make_particleinfo(PT_FLUID,0,i);
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
