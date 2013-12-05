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
#include <iostream>

#include "ConeSplashDown.h"
#include "Point.h"
#include "RigidBody.h"


ConeSplashDown::ConeSplashDown(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	lx = 1.6;
	ly = 0.67;
	lz = 0.6;
	H = 0.4;

	m_size = make_float3(lx, ly, lz);
	m_origin = make_float3(0.0, 0.0, 0.0);

	m_writerType = VTKWRITER;

	// SPH parameters
	set_deltap(0.025f);
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.0001f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 17;
	m_simparams.mlsfreq = 0;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = DYNAMICVISC;
    m_simparams.boundarytype= LJ_BOUNDARY;
	m_simparams.tend = 1.5;

	// Free surface detection
	m_simparams.surfaceparticle = true;
	m_simparams.savenormals =true;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0, 1000.0, 7.0, 50.0);

    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.dcoeff = 5.0*g*H;
	m_physparams.r0 = m_deltap;

	// BC when using MK boundary condition: Coupled with m_simsparams.boundarytype=MK_BOUNDARY
	#define MK_par 2
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;
	#undef MK_par

	m_physparams.kinematicvisc = 1.0e-6;
	m_physparams.artvisccoeff = 0.3;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;

	// Allocate data for floating bodies
	allocate_bodies(1);

	// Drawing and saving times
	m_displayinterval = 0.01f;
	m_writefreq = 5;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "ConeSplashDown";
	create_problem_dir();
}


ConeSplashDown::~ConeSplashDown(void)
{
	release_memory();
}


void ConeSplashDown::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}


int ConeSplashDown::fill_parts()
{
	float r0 = m_physparams.r0;

	Cube fluid;

	experiment_box = Cube(Point(0, 0, 0), Vector(lx, 0, 0),
						Vector(0, ly, 0), Vector(0, 0, lz));

	fluid = Cube(Point(r0, r0, r0), Vector(lx-2*r0, 0, 0),
				Vector(0, ly - 2*r0, 0), Vector(0, 0, H - r0));

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	experiment_box.SetPartMass(r0, m_physparams.rho0[0]);
	experiment_box.FillBorder(boundary_parts, r0, false);

	fluid.SetPartMass(m_deltap*m_deltap*m_deltap*m_physparams.rho0[0]);
	fluid.Fill(parts, m_deltap, true);

	// Rigid body #1
	double rb = 0.1;
	double rt = 0.05;
	double h = 0.1;

	object1 = Cone(Point(0.9*lx, ly/2, 1.4*H), rb, rt, h, EulerParameters(M_PI/2.0, -M_PI/4.0, 0));
	//object1 = Sphere(Point(0.9*lx, ly/2, 1.4*H), rb);
	//object1 = Cube(Point(0.9*lx, ly/2, 1.4*H), rb, rb, rb, EulerParameters(M_PI/4.0, -M_PI/4.0, 0));
	std::cout << "Cone part mass: " << object1.SetPartMass(r0, m_physparams.rho0[0]*0.4) << "\n";
	std::cout << "Cone mass: " << object1.SetMass(r0, m_physparams.rho0[0]*0.4) << "\n";
	object1.SetInertia(r0);

	RigidBody* rigid_body = get_body(0);
	rigid_body->AttachObject(&object1);
	object1.FillBorder(rigid_body->GetParts(), r0);
	rigid_body->GetParts()[10].print();
	//  Set the initial velocity and angular velocity
	rigid_body->SetInitialValues(Vector(-5., 0.0, -5.), Vector(0.0, 0.0, 0.0));

	return parts.size() + boundary_parts.size() + get_bodies_numparts();
}


void ConeSplashDown::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,0,i);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	for (int k = 0; k < m_simparams.numbodies; k++) {
		PointVect & rbparts = get_body(k)->GetParts();
		std::cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = j; i < j + rbparts.size(); i++) {
			pos[i] = make_float4(rbparts[i - j]);
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			info[i]= make_particleinfo(OBJECTPART, k, i - j);
		}
		j += rbparts.size();
		std::cout << ", part mass: " << pos[j-1].w << "\n";
	}

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		pos[i] = make_float4(parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}
