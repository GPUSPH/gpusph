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

#include "DamBreak3D.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

// Set to 0 for uniform initial density
#define HYDROSTATIC_INIT 1

#define CENTER_DOMAIN 1
// set to coords (x,y,z) if more accuracy is needed in such point
// (waiting for relative coordinates)
#if CENTER_DOMAIN
#define OFFSET_X (-lx/2)
#define OFFSET_Y (-ly/2)
#define OFFSET_Z (-lz/2)
#else
#define OFFSET_X 0
#define OFFSET_Y 0
#define OFFSET_Z 0
#endif

DamBreak3D::DamBreak3D(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 1.6;
	ly = 0.67;
	lz = 0.6;
	H = 0.4;
	wet = get_option("wet", false);

	m_usePlanes = get_option("use-planes", false);

	// density diffusion terms: 0 none, 1 Molteni & Colagrossi, 2 Ferrari
	const int rhodiff = get_option("density-diffusion", 1);

	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		boundary<DYN_BOUNDARY>
	).select_options(
		rhodiff, FlagSwitch<ENABLE_NONE, ENABLE_DENSITY_DIFFUSION, ENABLE_FERRARI>(),
		m_usePlanes, add_flags<ENABLE_PLANES>()
	);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled or Ferrari correction is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		(simparams()->simflags & (ENABLE_DENSITY_DIFFUSION | ENABLE_FERRARI)) ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// SPH parameters
	set_deltap(0.02); //0.008

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(OFFSET_X, OFFSET_Y, OFFSET_Z);

	// enlarge the domain to take into account the extra layers of particles
	// of the boundary
	if (simparams()->boundarytype == DYN_BOUNDARY && !m_usePlanes) {
		// number of layers
		dyn_layers = ceil(simparams()->kernelradius*simparams()->sfactor);
		// extra layers are one less (since other boundary types still have
		// one layer)
		double3 extra_offset = make_double3((dyn_layers-1)*m_deltap);
		m_origin -= extra_offset;
		m_size += 2*extra_offset;
	}


	simparams()->dt = 2.5e-4f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 1.5f;

	// Physical parameters
	physparams()->gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(physparams()->gravity);
	float max_hydro_vel = sqrt(2*g*H);
	float c0 = 10*ceil(max_hydro_vel);
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, c0);

	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	physparams()->dcoeff = 5.0f*g*H;
	physparams()->r0 = m_deltap;

	// BC when using MK boundary condition: Coupled with m_simsparams->boundarytype=MK_BOUNDARY
	#define MK_par 2
	physparams()->MK_K = g*H;
	physparams()->MK_d = 1.1*m_deltap/MK_par;
	physparams()->MK_beta = MK_par;
	#undef MK_par

	set_kinematic_visc(0, 1.0e-6f);
	physparams()->artvisccoeff = 0.3f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	physparams()->smagfactor = 0.12*0.12*m_deltap*m_deltap;
	physparams()->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);

	// Name of problem used for directory creation
	m_name = "DamBreak3D";
}


DamBreak3D::~DamBreak3D(void)
{
	release_memory();
}


void DamBreak3D::release_memory(void)
{
	parts.clear();
	obstacle_parts.clear();
	boundary_parts.clear();
}


int DamBreak3D::fill_parts()
{
	float r0 = physparams()->r0;

	Cube fluid, fluid1;

	experiment_box = Cube(Point(m_origin), m_size.x, m_size.y, m_size.z);

	if (!m_usePlanes) {
		experiment_box.SetPartMass(r0, physparams()->rho0[0]);
		if (simparams()->boundarytype == DYN_BOUNDARY)
			experiment_box.FillIn(boundary_parts, m_deltap, dyn_layers, false);
		else
			experiment_box.FillBorder(boundary_parts, r0, false);
	}

	obstacle = Cube(Point(m_origin + make_double3(0.9, 0.24, r0)),
		0.12, 0.12, lz - r0);

	m_fluidOrigin = m_origin;
	if (simparams()->boundarytype == DYN_BOUNDARY) // shift by the extra offset of the experiment box
		m_fluidOrigin += make_double3((dyn_layers-1)*m_deltap);
	m_fluidOrigin += make_double3(r0); // one wd space from the boundary
	fluid = Cube(Point(m_fluidOrigin), 0.4, ly - 2*r0, H - r0);

	if (wet) {
		fluid1 = Cube(Point(m_fluidOrigin + make_double3(H + m_deltap, 0, 0)),
			lx - H - m_deltap - 2*r0, 0.67 - 2*r0, 0.1);
	}

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	obstacle.SetPartMass(r0, physparams()->rho0[0]);
	obstacle.FillBorder(obstacle_parts, r0, true);

	fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid.Fill(parts, m_deltap, true);
	if (wet) {
		fluid1.SetPartMass(m_deltap, physparams()->rho0[0]);
		fluid1.Fill(parts, m_deltap, true);
		obstacle.Unfill(parts, r0);
	}

	return parts.size() + boundary_parts.size() + obstacle_parts.size();
}

void DamBreak3D::copy_planes(PlaneList &planes)
{
	if (!m_usePlanes) return;

	// bottom
	planes.push_back( implicit_plane(0, 0, 1, -m_origin.z) );
	// back
	planes.push_back( implicit_plane(1.0, 0, 0, -m_origin.x) );
	// front
	planes.push_back( implicit_plane(-1.0, 0, 0, m_origin.x + lx) );
	// side with smaller Y ("left")
	planes.push_back( implicit_plane(0, 1.0, 0, -m_origin.y) );
	// side with greater Y ("right")
	planes.push_back( implicit_plane(0, -1.0, 0, m_origin.y + ly) );
}


void DamBreak3D::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	int j = 0;

	if(boundary_parts.size()){
		std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
		for (uint i = 0; i < boundary_parts.size(); i++) {
#if HYDROSTATIC_INIT
			double water_column = m_fluidOrigin.z + H - boundary_parts[i](2);
			if (water_column < 0)
				water_column = 0;
			float rho = density(water_column, 0);
#else
			float rho = physparams()->rho0[0];
#endif
			vel[i] = make_float4(0, 0, 0, rho);
			info[i]= make_particleinfo(PT_BOUNDARY, 0, i);
			calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
		}
		j = boundary_parts.size();
		std::cout << "Boundary part mass:" << pos[j-1].w << "\n";
	}

	//Testpoints
	if (test_points.size()) {
		std::cout << "\nTest points: " << test_points.size() << "\n";
		for (uint i = 0; i < test_points.size(); i++) {
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
			info[i]= make_particleinfo(PT_TESTPOINT, 0, i);
			calc_localpos_and_hash(test_points[i], info[i], pos[i], hash[i]);
		}
		j += test_points.size();
		std::cout << "Test point mass:" << pos[j-1].w << "\n";
	}

	std::cout << "Obstacle parts: " << obstacle_parts.size() << "\n";
	for (uint i = j; i < j + obstacle_parts.size(); i++) {
#if HYDROSTATIC_INIT
		double water_column = m_fluidOrigin.z + H - obstacle_parts[i-j](2);
		if (water_column < 0)
			water_column = 0;
		float rho = density(water_column, 0);
#else
		float rho = physparams()->rho0[0];
#endif
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_BOUNDARY, 0, i);
		calc_localpos_and_hash(obstacle_parts[i-j], info[i], pos[i], hash[i]);
	}
	j += obstacle_parts.size();
	std::cout << "Obstacle part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
#if HYDROSTATIC_INIT
		double water_column = m_fluidOrigin.z + H - parts[i-j](2);
		if (water_column < 0)
			water_column = 0;
		float rho = density(water_column, 0);
#else
		float rho = physparams()->rho0[0];
#endif
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
	std::flush(std::cout);
}

void DamBreak3D::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}
