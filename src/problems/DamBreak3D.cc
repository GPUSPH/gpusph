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

DamBreak3D::DamBreak3D(const GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 1.6;
	ly = 0.67;
	lz = 0.6;
	H = 0.4;
	wet = true;

	m_usePlanes = false;
	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(OFFSET_X, OFFSET_Y, OFFSET_Z);

	SETUP_FRAMEWORK(
		viscosity<SPSVISC>,
		boundary<LJ_BOUNDARY>
	);

	addFilter(MLS_FILTER, 10);

	// SPH parameters
	set_deltap(0.02); //0.008
	m_simparams.dt = 0.0003f;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.tend = 1.5f;

	// Free surface detection
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;

	// Vorticity
	m_simparams.vorticity = false;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	H = 0.4f;
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

	m_physparams.kinematicvisc[0] = 1.0e-6f;
	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	m_physparams.smagfactor = 0.12*0.12*m_deltap*m_deltap;
	m_physparams.kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;

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
	float r0 = m_physparams.r0;

	Cube fluid, fluid1;

	experiment_box = Cube(Point(m_origin), lx, ly, lz);

	obstacle = Cube(Point(m_origin + make_double3(0.9, 0.24, r0)),
		0.12, 0.12, lz - r0);

	fluid = Cube(Point(m_origin + r0), 0.4, ly - 2*r0, H - r0);

	if (wet) {
		fluid1 = Cube(Point(m_origin + r0 + make_double3(H + m_deltap, 0, 0)),
			lx - H - m_deltap - 2*r0, 0.67 - 2*r0, 0.1);
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

	return parts.size() + boundary_parts.size() + obstacle_parts.size();
}

uint DamBreak3D::fill_planes()
{
	return (m_usePlanes ? 5 : 0);
}

void DamBreak3D::copy_planes(float4 *planes, float *planediv)
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
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
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
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			info[i]= make_particleinfo(PT_TESTPOINT, 0, i);
			calc_localpos_and_hash(test_points[i], info[i], pos[i], hash[i]);
		}
		j += test_points.size();
		std::cout << "Test point mass:" << pos[j-1].w << "\n";
	}

	std::cout << "Obstacle parts: " << obstacle_parts.size() << "\n";
	for (uint i = j; i < j + obstacle_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(PT_BOUNDARY, 1, i);
		calc_localpos_and_hash(obstacle_parts[i-j], info[i], pos[i], hash[i]);
	}
	j += obstacle_parts.size();
	std::cout << "Obstacle part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
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
