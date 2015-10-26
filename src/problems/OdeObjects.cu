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

#include "OdeObjects.h"
#include "Point.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

OdeObjects::OdeObjects(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 1.6;
	ly = 0.67;
	lz = 0.6;
	H = 0.4;
	wet = false;

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0.0, 0.0, 0.0);

	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		boundary<LJ_BOUNDARY>
	);

	// SPH parameters
	set_deltap(0.015f);
	simparams()->dt = 0.0001f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 1.5;

	// Physical parameters
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(physparams()->gravity);
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0, 10);

	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	physparams()->dcoeff = 5.0*g*H;
	physparams()->r0 = m_deltap;

	// BC when using MK boundary condition: Coupled with m_simsparams->boundarytype=MK_BOUNDARY
	#define MK_par 2
	physparams()->MK_K = g*H;
	physparams()->MK_d = 1.1*m_deltap/MK_par;
	physparams()->MK_beta = MK_par;
	#undef MK_par

	set_kinematic_visc(0, 1.0e-6);
	physparams()->artvisccoeff = 0.3;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	// Initialize ODE
	dInitODE();
	m_ODEWorld = dWorldCreate();	// Create a dynamic world
	m_ODESpace = dHashSpaceCreate(0);
	m_ODEJointGroup = dJointGroupCreate(0);
	dWorldSetGravity(m_ODEWorld, physparams()->gravity.x, physparams()->gravity.y, physparams()->gravity.z);	// Set gravity (x, y, z)

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);
	add_writer(COMMONWRITER, 0.0);

	// Name of problem used for directory creation
	m_name = "OdeObjects";
}


OdeObjects::~OdeObjects(void)
{
	release_memory();
	dWorldDestroy(m_ODEWorld);
	dCloseODE();
}


void OdeObjects::release_memory(void)
{
	parts.clear();
	obstacle_parts.clear();
	boundary_parts.clear();
}


int OdeObjects::fill_parts()
{
	float r0 = physparams()->r0;

	Cube fluid, fluid1;

	experiment_box = Cube(Point(0, 0, 0), lx, ly, lz);
	planes[0] = dCreatePlane(m_ODESpace, 0.0, 0.0, 1.0, 0.0);
	planes[1] = dCreatePlane(m_ODESpace, 1.0, 0.0, 0.0, 0.0);
	planes[2] = dCreatePlane(m_ODESpace, -1.0, 0.0, 0.0, -lx);
	planes[3] = dCreatePlane(m_ODESpace, 0.0, 1.0, 0.0, 0.0);
	planes[4] = dCreatePlane(m_ODESpace, 0.0, -1.0, 0.0, -ly);

	obstacle = Cube(Point(0.6, 0.24, 2*r0), 0.12, 0.12, 0.7*lz - 2*r0);

	fluid = Cube(Point(r0, r0, r0), 0.4, ly - 2*r0, H - r0);

	if (wet) {
		fluid1 = Cube(Point(H + m_deltap + r0 , r0, r0),
			lx - H - m_deltap - 2*r0, 0.67 - 2*r0, 0.1);
	}

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	experiment_box.SetPartMass(r0, physparams()->rho0[0]);
	experiment_box.FillBorder(boundary_parts, r0, false);

	obstacle.SetPartMass(r0, physparams()->rho0[0]*0.1);
	obstacle.SetMass(r0, physparams()->rho0[0]*0.1);
	obstacle.FillBorder(obstacle.GetParts(), r0, true);
	obstacle.ODEBodyCreate(m_ODEWorld, m_deltap);
	obstacle.ODEGeomCreate(m_ODESpace, m_deltap);
	add_moving_body(&obstacle, MB_ODE);

	fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid.Fill(parts, m_deltap, true);
	if (wet) {
		fluid1.SetPartMass(m_deltap, physparams()->rho0[0]);
		fluid1.Fill(parts, m_deltap, true);
		obstacle.Unfill(parts, r0);
	}

	// Rigid body #1 : sphere
	Point rb_cg = Point(0.6, 0.15*ly, 0.05 + r0);
	sphere = Sphere(rb_cg, 0.05);
	sphere.SetPartMass(r0, physparams()->rho0[0]*0.6);
	sphere.SetMass(r0, physparams()->rho0[0]*0.6);
	sphere.Unfill(parts, r0);
	sphere.FillBorder(sphere.GetParts(), r0);
	sphere.ODEBodyCreate(m_ODEWorld, m_deltap);
	sphere.ODEGeomCreate(m_ODESpace, m_deltap);
	add_moving_body(&sphere, MB_ODE);

	// Rigid body #2 : cylinder
	cylinder = Cylinder(Point(0.9, 0.7*ly, r0), 0.05, Vector(0, 0, 0.2));
	cylinder.SetPartMass(r0, physparams()->rho0[0]*0.3);
	cylinder.SetMass(r0, physparams()->rho0[0]*0.05);
	cylinder.Unfill(parts, r0);
	cylinder.FillBorder(cylinder.GetParts(), r0);
	cylinder.ODEBodyCreate(m_ODEWorld, m_deltap);
	cylinder.ODEGeomCreate(m_ODESpace, m_deltap);
	add_moving_body(&cylinder, MB_ODE);

	joint = dJointCreateHinge(m_ODEWorld, 0);				// Create a hinge joint
	dJointAttach(joint, obstacle.m_ODEBody, 0);		// Attach joint to bodies
	dJointSetHingeAnchor(joint, 0.7, 0.24, 2*r0);	// Set a joint anchor
	dJointSetHingeAxis(joint, 0, 1, 0);

	return parts.size() + boundary_parts.size() + get_bodies_numparts();
}


void OdeObjects::ODE_near_callback(void *data, dGeomID o1, dGeomID o2)
{
	const int N = 10;
	dContact contact[N];

	int n = dCollide(o1, o2, N, &contact[0].geom, sizeof(dContact));
	if ((o1 == cube.m_ODEGeom && o2 == sphere.m_ODEGeom) || (o2 == cube.m_ODEGeom && o1 == sphere.m_ODEGeom)) {
		cout << "Collision between cube and obstacle " << n << "contact points\n";
	}
	for (int i = 0; i < n; i++) {
		contact[i].surface.mode = dContactBounce;
		contact[i].surface.mu   = dInfinity;
		contact[i].surface.bounce     = 0.0; // (0.0~1.0) restitution parameter
		contact[i].surface.bounce_vel = 0.0; // minimum incoming velocity for bounce
		dJointID c = dJointCreateContact(m_ODEWorld, m_ODEJointGroup, &contact[i]);
		dJointAttach (c, dGeomGetBody(contact[i].geom.g1), dGeomGetBody(contact[i].geom.g2));
	}
}


void OdeObjects::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i] = make_particleinfo(PT_BOUNDARY, 0, i);
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
	}

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i] = make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}
