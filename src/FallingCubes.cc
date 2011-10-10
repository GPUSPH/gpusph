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
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "FallingCubes.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "RigidBody.h"
#include "EulerParameters.h"


FallingCubes::FallingCubes(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	lx = 1.6;
	ly = 0.67;
	lz = 0.6;	
	H = 0.4;
	
	m_size = make_float3(lx, ly, lz);
	m_origin = make_float3(0.0f, 0.0f, 0.0f);

	m_writerType = VTKWRITER;

	// SPH parameters
	set_deltap(0.02f);
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.0001f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = DYNAMICVISC;
    m_simparams.boundarytype= LJ_BOUNDARY;
	m_simparams.tend = 1.5f;

	// Free surface detection
	m_simparams.surfaceparticle = true;
	m_simparams.savenormals =true;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0, 1000.0, 7.0, 30.0);

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
	allocate_bodies(3);

	// Scales for drawing
	m_maxrho = density(H, 0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	//m_maxvel = sqrt(m_physparams.gravity*H);
	m_maxvel = 1.0f;

	// Drawing and saving times
	m_displayinterval = 0.01;
	m_rbdata_writeinterval = 0.01;
	m_writefreq = 50;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "FallingCubes";
	create_problem_dir();
}


FallingCubes::~FallingCubes(void)
{
	release_memory();
}


void FallingCubes::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}


int FallingCubes::fill_parts()
{
	float r0 = m_physparams.r0;

	Cube fluid;

	experiment_box = Cube(Point(0, 0, 0), Vector(lx, 0, 0),
						Vector(0, ly, 0), Vector(0, 0, lz));

	fluid = Cube(Point(r0, r0, r0), Vector(lx - 2*r0, 0, 0),
				Vector(0, ly - 2*r0, 0), Vector(0, 0, H - r0));

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	experiment_box.SetPartMass(r0, m_physparams.rho0[0]);
	experiment_box.FillBorder(boundary_parts, r0, false);

	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	fluid.Fill(parts, m_deltap, true);

	// Rigid body #1
	Point rb_cg = Point(0.4, 0.4, H + 0.15);
	double l = 0.1, w = 0.1, h = 0.1;
	cube[0] = Cube(rb_cg - Vector(l/2, w/2, h/2), Vector(l, 0, 0),
					Vector(0, w, 0), Vector(0, 0, h));
	l += m_deltap/2.0;
	w += m_deltap/2.0;
	h += m_deltap/2.0;
	double rb_density = m_physparams.rho0[0];
	double rb_mass = l*w*h*rb_density;
	double SetInertia[3] = {rb_mass*(w*w + h*h)/12.0, rb_mass*(l*l + h*h)/12.0, rb_mass*(w*w + l*l)/12.0};

	RigidBody * rigid_body = get_body(0);
	PointVect & rbparts1 = rigid_body->GetParts();
	cube[0].FillBorder(rbparts1, r0, true);

	// Setting inertial frame data
	rigid_body->SetInertialFrameData(rb_cg, SetInertia, rb_mass, EulerParameters());
	rigid_body->SetInitialValues(Vector(0.0, 0.0, -2.0), Vector(0.0, 0.0, 0.0));
	
	// Rigid body #2
	rb_cg = Point(0.7, 0.4, H + 0.15);
	l = 0.1, w = 0.2, h = 0.1;
	cube[1] = Cube(rb_cg - Vector(l/2, w/2, h/2), Vector(l, 0, 0),
					Vector(0, w, 0), Vector(0, 0, h));
	cube[1].SetPartMass(r0, m_physparams.rho0[0]*0.9);
	cube[1].SetMass(r0, m_physparams.rho0[0]*0.9);
	cube[1].SetInertia(r0);

	rigid_body = get_body(1);
	rigid_body->AttachObject(&cube[1]);
	cube[1].FillBorder(rigid_body->GetParts(), r0, true);
	rigid_body->SetInitialValues(Vector(0.0, 0.0, 0.0), Vector(5.0, -3.0, 4.0));

	// Rigid body #3
	l = 0.1, w = 0.1, h = 0.2;
	rb_cg = Point(1.3, 0.2, H + 0.15);
	cube[2] = Cube(rb_cg - Vector(l/2, w/2, h/2), Vector(l, 0, 0),
					Vector(0, w, 0), Vector(0, 0, h));
	cube[2].SetPartMass(r0, m_physparams.rho0[0]*0.2);
	cube[2].SetMass(r0, m_physparams.rho0[0]*0.2);
	cube[2].SetInertia(r0);
	cube[2].Unfill(parts, r0);

	rigid_body = get_body(2);
	rigid_body->AttachObject(&cube[2]);
	cube[2].FillBorder(rigid_body->GetParts(), r0, true);
	rigid_body->SetInitialValues(Vector(0.0, 0.0, -1.0), Vector(0.0, 0.0, 3.0));

	return parts.size() + boundary_parts.size() + get_bodies_numparts();
}


void FallingCubes::draw_boundary(float t)
{
	glColor3f(0.0, 1.0, 0.0);
	experiment_box.GLDraw();	
	glColor3f(1.0, 0.0, 0.0);
	cube[0].GLDraw();
	for (int i = 0; i < m_simparams.numbodies; i++)
		get_body(i)->GLDraw();
}


void FallingCubes::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART, 0, i);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass: " << pos[j-1].w << "\n";
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
		info[i]= make_particleinfo(FLUIDPART, 0, i);
	}
	j += parts.size();
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";
}
