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

#include "Spheric2LJ.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

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

Spheric2LJ::Spheric2LJ(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 3.22;
	ly = 1.0;
	lz = 1.0;
	H = 0.55;
	wet = false;
	m_usePlanes = get_option("use-planes", true);

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(OFFSET_X, OFFSET_Y, OFFSET_Z);

	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		viscosity<ARTVISC>,
		//viscosity<SPSVISC>,
		//viscosity<DYNAMICVISC>,
		boundary<LJ_BOUNDARY>,
		add_flags<ENABLE_FERRARI>
	).select_options(
		m_usePlanes, add_flags<ENABLE_PLANES>()
	);

	// SPH parameters
	// ratio h / deltap (needs to be defined before calling set_deltap)
	simparams()->sfactor = 1.3;
	// set deltap (automatically computes h based on sfactor * deltap)
	set_deltap(0.02); //0.008
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->ferrari = 0.1;
	simparams()->tend = 1.0f;

	// Free surface detection
	addPostProcess(SURFACE_DETECTION);

	// Test points
	addPostProcess(TESTPOINTS);

	// Physical parameters
	physparams()->gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(physparams()->gravity);

	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 20.f);

    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	physparams()->dcoeff = 5.0f*g*H;
	physparams()->r0 = m_deltap;

	// BC when using MK boundary condition: Coupled with m_simsparams->boundarytype=MK_BOUNDARY
	#define MK_par 2
	physparams()->MK_K = g*H;
	physparams()->MK_d = 1.1*m_deltap/MK_par;
	physparams()->MK_beta = MK_par;
	#undef MK_par

	set_kinematic_visc(0, 1.0e-2f);
	physparams()->artvisccoeff = 0.3f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.05);

	// Name of problem used for directory creation
	m_name = "Spheric2LJ";
}


Spheric2LJ::~Spheric2LJ(void)
{
	release_memory();
}


void Spheric2LJ::release_memory(void)
{
	parts.clear();
	obstacle_parts.clear();
	boundary_parts.clear();
}


int Spheric2LJ::fill_parts()
{
	float r0 = physparams()->r0;

	Cube fluid, fluid1;

	experiment_box = Cube(Point(m_origin), lx, ly, lz);

	obstacle = Cube(Point(m_origin + make_double3(2.3955, 0.295, 0.0)), 0.161, 0.403, 0.161);


	fluid = Cube(Point(m_origin + r0), 0.4, ly - 2*r0, H - r0);

	if (wet) {
		fluid1 = Cube(Point(m_origin + r0 + make_double3(H + m_deltap, 0, 0)),
			lx - H - m_deltap - 2*r0, 0.67 - 2*r0, 0.1);
	}

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	if (!m_usePlanes) {
		experiment_box.SetPartMass(r0, physparams()->rho0[0]);
		experiment_box.FillBorder(boundary_parts, r0, false);
	}

	obstacle.SetPartMass(r0, physparams()->rho0[0]);
	obstacle.FillBorder(obstacle_parts, r0, true);

	fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid.Fill(parts, m_deltap, true);
	if (wet) {
		fluid1.SetPartMass(m_deltap, physparams()->rho0[0]);
		fluid1.Fill(parts, m_deltap, true);
		obstacle.Unfill(parts, r0);
	}

	// Setting probes for Spheric2 test case
	//*******************************************************************
	// Wave gages
	add_gage(m_origin + make_double3(2.724, 0.5, 0.0));
	add_gage(m_origin + make_double3(2.228, 0.5, 0.0));
	add_gage(m_origin + make_double3(1.732, 0.5, 0.0));
	add_gage(m_origin + make_double3(0.582, 0.5, 0.0));
	// Pressure probes
	if (m_simframework->hasPostProcessEngine(TESTPOINTS)) {
		test_points.push_back(m_origin + make_double3(2.3955, 0.529, 0.021));
		test_points.push_back(m_origin + make_double3(2.3955, 0.529, 0.061));
		test_points.push_back(m_origin + make_double3(2.3955, 0.529, 0.101));
		test_points.push_back(m_origin + make_double3(2.3955, 0.529, 0.141));
		test_points.push_back(m_origin + make_double3(2.4165, 0.471, 0.161));
		test_points.push_back(m_origin + make_double3(2.4565, 0.471, 0.161));
		test_points.push_back(m_origin + make_double3(2.4965, 0.471, 0.161));
		test_points.push_back(m_origin + make_double3(2.5365, 0.471, 0.161));
	}
	//*******************************************************************

	return parts.size() + boundary_parts.size() + obstacle_parts.size() + test_points.size();
}

void Spheric2LJ::copy_planes(PlaneList& planes)
{
	if (!m_usePlanes) return;

	// bottom
	planes.push_back( implicit_plane(0, 0, 1.0, -m_origin.z) );
	// back
	planes.push_back( implicit_plane(1.0, 0, 0, -m_origin.x) );
	// front
	planes.push_back( implicit_plane(-1.0, 0, 0, m_origin.x + lx) );
	// side with smaller Y ("left")
	planes.push_back( implicit_plane(0, 1.0, 0, -m_origin.y) );
	// side with greater Y ("right")
	planes.push_back( implicit_plane(0, -1.0, 0, m_origin.y + ly) );
}

void Spheric2LJ::fillDeviceMap()
{
	// TODO: test which split performs better, if Y (not many particles passing) or X (smaller section)
	fillDeviceMapByAxis(Y_AXIS);
	//fillDeviceMapByEquation();
}

void Spheric2LJ::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	for (uint i = 0; i < boundary_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i]= make_particleinfo(PT_BOUNDARY,0,i);
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	uint j = boundary_parts.size();
	if (boundary_parts.size() > 0)
		std::cout << "Boundary part mass:" << pos[j-1].w << "\n";
	else
		std::cout << "No boundary parts" << std::endl;

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
	else
		std::cout << "No test points" << std::endl;

	std::cout << "Obstacle parts: " << obstacle_parts.size() << "\n";
	for (uint i = j; i < j + obstacle_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i]= make_particleinfo(PT_BOUNDARY,1,i);
		calc_localpos_and_hash(obstacle_parts[i-j], info[i], pos[i], hash[i]);
	}
	j += obstacle_parts.size();
	if (obstacle_parts.size() > 0)
		std::cout << "Obstacle part mass:" << pos[j-1].w << "\n";
	else
		std::cout << "No obstacle parts" << std::endl;

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i]= make_particleinfo(PT_FLUID,0,i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	if (parts.size() > 0)
		std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
	else
		std::cout << "No fluid parts" << std::endl;

	std::flush(std::cout);
}
