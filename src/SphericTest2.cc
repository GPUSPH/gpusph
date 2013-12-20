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

#include "SphericTest2.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"

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

SphericTest2::SphericTest2(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	lx = 3.22;
	ly = 1.0;
	lz = 1.0;
	H = 0.55;
	wet = false;
	m_usePlanes = true;

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(OFFSET_X, OFFSET_Y, OFFSET_Z);

	m_writerType = VTKWRITER;
	//m_writerType = UDPWRITER;

	// SPH parameters
	set_deltap(0.02); //0.008
	m_simparams.dt = 0.0003f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.ferrari = 0.1;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = SPSVISC;
	//m_simparams.visctype = DYNAMICVISC;
	m_simparams.boundarytype= LJ_BOUNDARY;
	m_simparams.tend = 5.0f;

	// Free surface detection
	m_simparams.surfaceparticle = true;
	m_simparams.savenormals = false;

	// Test points
	m_simparams.testpoints = true;

	// Vorticity
	m_simparams.vorticity = false;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0, 1000.0, 7.0f, 20.f);

    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.dcoeff = 5.0f*g*H;
	m_physparams.r0 = m_deltap;

	// BC when using MK boundary condition: Coupled with m_simsparams.boundarytype=MK_BOUNDARY
	#define MK_par 2
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;
	#undef MK_par

	m_physparams.kinematicvisc = 1.0e-2f;
	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;

	// Drawing and saving times
	m_displayinterval = 0.01f;
	m_writefreq = 5;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "SphericTest2";
}


SphericTest2::~SphericTest2(void)
{
	release_memory();
}


void SphericTest2::release_memory(void)
{
	parts.clear();
	obstacle_parts.clear();
	boundary_parts.clear();
}


int SphericTest2::fill_parts()
{
	float r0 = m_physparams.r0;

	Cube fluid, fluid1;

	experiment_box = Cube(Point(m_origin), Vector(lx, 0, 0),
						Vector(0, ly, 0), Vector(0, 0, lz));

	obstacle = Cube(Point(m_origin + make_double3(0.9, 0.24, r0)), Vector(0.12, 0, 0),
					Vector(0, 0.12, 0), Vector(0, 0, lz - r0));

	fluid = Cube(Point(m_origin + r0), Vector(0.4, 0, 0),
				Vector(0, ly - 2*r0, 0), Vector(0, 0, H - r0));

	if (wet) {
		fluid1 = Cube(Point(m_origin + r0 + make_double3(H + m_deltap, 0, 0)), Vector(lx - H - m_deltap - 2*r0, 0, 0),
					Vector(0, 0.67 - 2*r0, 0), Vector(0, 0, 0.1));
	}

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	if (!m_usePlanes) {
		experiment_box.SetPartMass(r0, m_physparams.rho0[0]);
		experiment_box.FillBorder(boundary_parts, r0, false);
	}

	obstacle.SetPartMass(r0, m_physparams.rho0[0]);
	obstacle.FillBorder(obstacle_parts, r0, true);

	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	fluid.Fill(parts, m_deltap, true);
	if (wet) {
		fluid1.SetPartMass(m_deltap, m_physparams.rho0[0]);
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
	if (m_simparams.testpoints) {
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

uint SphericTest2::fill_planes()
{
	return (m_usePlanes ? 5 : 0);
}

void SphericTest2::copy_planes(float4 *planes, float *planediv)
{
	if (!m_usePlanes) return;

	// bottom
	planes[0] = make_float4(0, 0, 1.0, -m_origin.z);
	planediv[0] = 1.0;
	// back
	planes[1] = make_float4(1.0, 0, 0, -m_origin.x);
	planediv[1] = 1.0;
	// front
	planes[2] = make_float4(-1.0, 0, 0, m_origin.x + lx);
	planediv[2] = 1.0;
	// side with smaller Y ("left")
	planes[3] = make_float4(0, 1.0, 0, -m_origin.y);
	planediv[3] = 1.0;
	// side with greater Y ("right")
	planes[4] = make_float4(0, -1.0, 0, m_origin.y + ly);
	planediv[4] = 1.0;
}

void SphericTest2::fillDeviceMap(GlobalData* gdata)
{
	// TODO: test which split performs better, if Y (not many particles passing) or X (smaller section)
	fillDeviceMapByAxis(gdata, Y_AXIS);
	//fillDeviceMapByEquation(gdata);
}

void SphericTest2::copy_to_array(float4 *pos, float4 *vel, particleinfo *info, uint* hash)
{
	float4 localpos;
	uint hashvalue;

	for (uint i = 0; i < boundary_parts.size(); i++) {
		calc_localpos_and_hash(boundary_parts[i], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,0,i);
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
			calc_localpos_and_hash(test_points[i], pos[i], hash[i]);
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			info[i]= make_particleinfo(TESTPOINTSPART, 0, i);
		}
		j += test_points.size();
		std::cout << "Test point mass:" << pos[j-1].w << "\n";
	}
	else
		std::cout << "No test points" << std::endl;

	std::cout << "Obstacle parts: " << obstacle_parts.size() << "\n";
	for (uint i = j; i < j + obstacle_parts.size(); i++) {
		calc_localpos_and_hash(obstacle_parts[i-j], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,1,i);
	}
	j += obstacle_parts.size();
	if (obstacle_parts.size() > 0)
		std::cout << "Obstacle part mass:" << pos[j-1].w << "\n";
	else
		std::cout << "No obstacle parts" << std::endl;

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		calc_localpos_and_hash(parts[i-j], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
	}
	j += parts.size();
	if (parts.size() > 0)
		std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
	else
		std::cout << "No fluid parts" << std::endl;
	std::flush(std::cout);
}
