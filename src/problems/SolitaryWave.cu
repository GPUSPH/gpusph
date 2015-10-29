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

#include "SolitaryWave.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define MK_par 2

SolitaryWave::SolitaryWave(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 9.0;
	ly = 0.4;
	lz = 3.0;

	// Data for problem setup
	slope_length = 8.5f;
	h_length = 0.5f;
	height = .63f;
	beta = 4.2364*M_PI/180.0;

	m_size = make_double3(lx, ly, lz + 1.2*height);
	m_origin = make_double3(0.0, 0.0, -1.2*height);

	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		//viscosity<KINEMATICVISC>,
		//viscosity<SPSVISC>,
		boundary<LJ_BOUNDARY>,
		//boundary<MK_BOUNDARY>,
		flags<ENABLE_DTADAPT | ENABLE_PLANES>
	);

	//addFilter(SHEPARD_FILTER, 20);


	// Add objects to the tank
	icyl = 1;	// icyl = 0 means no cylinders
	icone = 0;	// icone = 0 means no cone

	i_use_bottom_plane = 1; // 1 for real plane instead of boundary parts

	// SPH parameters
	set_deltap(0.04f);  //0.005f;
	simparams()->dt = 0.00013f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 10.0;

	addPostProcess(VORTICITY);

	// Physical parameters
	H = 0.45f;
	physparams()->gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(physparams()->gravity);

	add_fluid(1000.0f);
	set_equation_of_state(0,  7.0f, 20.f);
	float r0 = m_deltap;
	physparams()->r0 = r0;

	physparams()->artvisccoeff = 0.3f;
	set_kinematic_visc(0, 1.0e-6f);
	physparams()->smagfactor = 0.12*0.12*m_deltap*m_deltap;
	physparams()->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	// BC when using LJ
	physparams()->dcoeff = 5.0f*g*H;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// BC when using MK
	physparams()->MK_K = g*H;
	physparams()->MK_d = 1.1*m_deltap/MK_par;
	physparams()->MK_beta = MK_par;

	// Compute parameters for piston movement
	// The velocity will be c/(cosh(a*t+b)^2), so keep it simple
	// and have only a, b, c as calss variables.
	const double amplitude = 0.2;
	const double Hoh = amplitude/H;
	const double kappa = sqrt(3*Hoh)/(2.0*H);
	const double cel = sqrt(g*(H + amplitude));
	const double S = sqrt(16.0*amplitude*H/3.0);
	const double tau = 2.0*(3.8 + Hoh)/(kappa*cel);
	piston_tend = tau;
	piston_tstart = 0.2;
	piston_initial_crotx = r0;
	a = 2.0*(3.8 + Hoh)/tau;
	b = 2.0*((3.8 + Hoh)*(-piston_tstart/tau - 0.5) - 2.0*Hoh*(piston_initial_crotx/S - 0.5));
	c = (3.8 + Hoh)*S/tau;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);

	// Name of problem used for directory creation
	m_name = "SolitaryWave";
}


SolitaryWave::~SolitaryWave(void)
{
	release_memory();
}


void SolitaryWave::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}


void
SolitaryWave::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{
	dx = make_double3(0.0);
	if (object == &piston) {
		const double ti = min(piston_tend, max(piston_tstart, t0));
		const double tf = min(piston_tend, max(piston_tstart, t1));

		if (t1 >= piston_tstart && t1 <= piston_tend) {
			kdata.lvel.x = c/(cosh(a*t1 + b)*cosh(a*t1 + b));
			if (tf != ti)
				dx.x = c/a*(tanh(a*tf+b) - tanh(a*ti + b));
			kdata.crot.x += dx.x;
		}
		else
			kdata.lvel.x = 0.0f;
	}
	else {
		const double tstart = 0.0;
		const double tend = 1.0;
		const double velz = 0.5;

		const double ti = min(tend, max(tstart, t0));
		const double tf = min(tend, max(tstart, t1));

		// Setting postion of center of rotation
		if (t1 >= tstart && t1 <= tend) {
			kdata.crot.z = initial_kdata.crot.z + velz*(t1 - tstart);
			// Setting linear velocity
			kdata.lvel.z = velz;
		}
		else
			kdata.lvel.z = 0.0f;
		// Computing the displacement of center of rotation between t = t0 and t = t1
		dx.z = (tf - ti)*velz;
	}

	// Setting angular velocity at t = t1 and the rotation between t = t0 and t = 1.
	// Here we have a simple translation movement so the angular velocity is null and
	// the rotation between t0 and t1 equal to identity.
	kdata.avel = make_double3(0.0f);
	dr.Identity();
}


int SolitaryWave::fill_parts()
{
	const float r0 = physparams()->r0;
	const float width = ly;

	const float br = (simparams()->boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);

	experiment_box = Cube(Point(0, 0, 0), h_length + slope_length, width, height);

	boundary_parts.reserve(100);
	parts.reserve(34000);

	piston = Rect(Point(piston_initial_crotx, 0, 0), Vector(0, width, 0), Vector(0, 0, height));
	piston.SetPartMass(m_deltap, physparams()->rho0[0]);
	piston.Fill(piston.GetParts(), br, true);
	add_moving_body(&piston, MB_MOVING);

	if (i_use_bottom_plane == 0) {
	   experiment_box1 = Rect(Point(h_length, 0, 0), Vector(0, width, 0),
			Vector(slope_length/cos(beta), 0.0, slope_length*tan(beta)));
	   experiment_box1.SetPartMass(m_deltap, physparams()->rho0[0]);
	   experiment_box1.Fill(boundary_parts,br,true);
	   std::cout << "bottom rectangle defined" <<"\n";
	   }

	if (icyl == 1) {
		Point p[10];
		p[0] = Point(h_length + slope_length/(cos(beta)*10), width/2, -height);
		p[1] = Point(h_length + slope_length/(cos(beta)*10), width/6,  -height);
		p[2] = Point(h_length + slope_length/(cos(beta)*10), 5*width/6, -height);
		p[3] = Point(h_length + slope_length/(cos(beta)*5), 0, -height);
		p[4] = Point(h_length + slope_length/(cos(beta)*5),  width/3, -height);
		p[5] = Point(h_length + slope_length/(cos(beta)*5), 2*width/3, -height);
		p[6] = Point(h_length + slope_length/(cos(beta)*5),  width, -height);
		p[7] = Point(h_length + 3*slope_length/(cos(beta)*10),  width/6, -height);
		p[8] = Point(h_length + 3*slope_length/(cos(beta)*10),  width/2, -height);
		p[9] = Point(h_length+ 3*slope_length/(cos(beta)*10), 5*width/6, -height);
		//p[]  = Point(h_length+ 4*slope_length/(cos(beta)*10), width/2, -height*.75);

		for (int i = 0; i < 10; i++) {
			double radius = 0.025;
			if (i == 0)
				radius = 0.05;
			cyl[i] = Cylinder(p[i], radius, height);
		    cyl[i].SetPartMass(m_deltap, physparams()->rho0[0]);
		    cyl[i].FillBorder(cyl[i].GetParts(), br, false, false);
			add_moving_body(&(cyl[i]), MB_MOVING);
		}
	}
	if (icone == 1) {
		Point p1 = Point(h_length + slope_length/(cos(beta)*10), width/2, -height);
		cone = Cone(p1, width/4, width/10, height);
		cone.SetPartMass(m_deltap, physparams()->rho0[0]);
		cone.FillBorder(cone.GetParts(), br, false, true);
		add_moving_body(&cone, MB_MOVING);
    }

	Rect fluid;
	float z = 0;
	int n = 0;
	while (z < H) {
		z = n*m_deltap + 1.5*r0;    //z = n*m_deltap + 1.5*r0;
		float x = piston_initial_crotx + r0;
		float l = h_length + z/tan(beta) - 1.5*r0/sin(beta) - x;
		fluid = Rect(Point(x,  r0, z),
				Vector(0, width-2.0*r0, 0), Vector(l, 0, 0));
		fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
		fluid.Fill(parts, m_deltap, true);
		n++;
	 }

    return parts.size() + boundary_parts.size() + get_bodies_numparts();
}

void SolitaryWave::copy_planes(PlaneList &planes)
{
	const double w = m_size.y;
	const double l = h_length + slope_length;

	//  plane is defined as a x + by +c z + d= 0
	planes.push_back( implicit_plane(0, 0, 1.0, 0) );   //bottom, where the first three numbers are the normal, and the last is d.
	planes.push_back( implicit_plane(0, 1.0, 0, 0) );   //wall
	planes.push_back( implicit_plane(0, -1.0, 0, w) ); //far wall
	planes.push_back( implicit_plane(1.0, 0, 0, 0) );  //end
	planes.push_back( implicit_plane(-1.0, 0, 0, l) );  //one end
	if (i_use_bottom_plane == 1)  {
		planes.push_back( implicit_plane(-sin(beta),0,cos(beta), h_length*sin(beta)) );  //sloping bottom starting at x=h_length
	}
}


void SolitaryWave::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
		std::cout << "      "<< 0  <<"--"<< boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i]= make_particleinfo(PT_BOUNDARY, 0, i);  // first is type, object, 3rd id
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	uint object_particle_counter = 0;
	for (uint k = 0; k < m_bodies.size(); k++) {
		PointVect & rbparts = m_bodies[k]->object->GetParts();
		std::cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = 0; i < rbparts.size(); i++) {
			uint ij = i + j;
			float ht = H - rbparts[i](2);
			if (ht < 0)
				ht = 0.0;
			float rho = density(ht, 0);
			rho = physparams()->rho0[0];
			vel[ij] = make_float4(0, 0, 0, rho);
			uint ptype = (uint) PT_BOUNDARY;
			switch (m_bodies[k]->type) {
				case MB_ODE:
					ptype |= FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE;
					break;
				case MB_FORCES_MOVING:
					ptype |= FG_COMPUTE_FORCE | FG_MOVING_BOUNDARY;
					break;
				case MB_MOVING:
					ptype |= FG_MOVING_BOUNDARY;
					break;
			}
			info[ij] = make_particleinfo(ptype, k, ij);
			calc_localpos_and_hash(rbparts[i], info[ij], pos[ij], hash[ij]);
		}
		if (k < simparams()->numforcesbodies) {
			gdata->s_hRbFirstIndex[k] = -j + object_particle_counter;
			gdata->s_hRbLastIndex[k] = object_particle_counter + rbparts.size() - 1;
			object_particle_counter += rbparts.size();
		}
		j += rbparts.size();
		std::cout << ", part mass: " << pos[j-1].w << "\n";
		std::cout << ", part type: " << type(info[j-1])<< "\n";
	}

	std::cout << "\nFluid parts: " << parts.size() << "\n";
	std::cout << "      "<< j  <<"--"<< j+ parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
	    info[i]= make_particleinfo(PT_FLUID,0,i);
		calc_localpos_and_hash(parts[i - j], info[i], pos[i], hash[i]);
		// initializing density
		//       float rho = physparams()->rho0*pow(1.+g*(H-pos[i].z)/physparams()->bcoeff,1/physparams()->gammacoeff);
		//        vel[i] = make_float4(0, 0, 0, rho);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";

	std::cout << " Everything uploaded" <<"\n";
}
#undef MK_par
