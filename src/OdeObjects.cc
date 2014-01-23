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


OdeObjects::OdeObjects(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	lx = 1.6;
	ly = 0.67;
	lz = 0.6;
	H = 0.4;
	wet = false;

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0.0, 0.0, 0.0);

	m_writerType = VTKWRITER;

	// SPH parameters
	set_deltap(0.015f);
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
	m_simparams.tend = 1.5;

	// Free surface detection
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0, 1000.0, 7.0, 10);

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
	allocate_ODE_bodies(2);
	dInitODE();				// Initialize ODE
	m_ODEWorld = dWorldCreate();	// Create a dynamic world
	m_ODESpace = dHashSpaceCreate(0);
	m_ODEJointGroup = dJointGroupCreate(0);
	dWorldSetGravity(m_ODEWorld, m_physparams.gravity.x, m_physparams.gravity.y, m_physparams.gravity.z);	// Set gravity（x, y, z)

	// Drawing and saving times
	m_displayinterval = 0.01f;
	m_writefreq = 10;
	m_screenshotfreq = 0;

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
	float r0 = m_physparams.r0;

	Cube fluid, fluid1;

	experiment_box = Cube(Point(0, 0, 0), Vector(lx, 0, 0),
						Vector(0, ly, 0), Vector(0, 0, lz));
	planes[0] = dCreatePlane(m_ODESpace, 0.0, 0.0, 1.0, 0.0);
	planes[1] = dCreatePlane(m_ODESpace, 1.0, 0.0, 0.0, 0.0);
	planes[2] = dCreatePlane(m_ODESpace, -1.0, 0.0, 0.0, -lx);
	planes[3] = dCreatePlane(m_ODESpace, 0.0, 1.0, 0.0, 0.0);
	planes[4] = dCreatePlane(m_ODESpace, 0.0, -1.0, 0.0, -ly);

	obstacle = Cube(Point(0.6, 0.24, 2*r0), Vector(0.12, 0, 0),
					Vector(0, 0.12, 0), Vector(0, 0, 0.7*lz - 2*r0));


	fluid = Cube(Point(r0, r0, r0), Vector(0.4, 0, 0),
				Vector(0, ly - 2*r0, 0), Vector(0, 0, H - r0));

	if (wet) {
		fluid1 = Cube(Point(H + m_deltap + r0 , r0, r0), Vector(lx - H - m_deltap - 2*r0, 0, 0),
					Vector(0, 0.67 - 2*r0, 0), Vector(0, 0, 0.1));
	}

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	experiment_box.SetPartMass(r0, m_physparams.rho0[0]);
	experiment_box.FillBorder(boundary_parts, r0, false);

	obstacle.SetPartMass(r0, m_physparams.rho0[0]*0.1);
	obstacle.SetMass(r0, m_physparams.rho0[0]*0.1);
	//obstacle.FillBorder(obstacle.GetParts(), r0, true);
	//obstacle.ODEBodyCreate(m_ODEWorld, m_deltap);
	//obstacle.ODEGeomCreate(m_ODESpace, m_deltap);
	//add_ODE_body(&obstacle);

	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	fluid.Fill(parts, m_deltap, true);
	if (wet) {
		fluid1.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid1.Fill(parts, m_deltap, true);
		obstacle.Unfill(parts, r0);
	}

	// Rigid body #1 : sphere
	Point rb_cg = Point(0.6, 0.15*ly, 0.05 + r0);
	sphere = Sphere(rb_cg, 0.05);
	sphere.SetPartMass(r0, m_physparams.rho0[0]*0.6);
	sphere.SetMass(r0, m_physparams.rho0[0]*0.6);
	sphere.Unfill(parts, r0);
	sphere.FillBorder(sphere.GetParts(), r0);
	sphere.ODEBodyCreate(m_ODEWorld, m_deltap);
	sphere.ODEGeomCreate(m_ODESpace, m_deltap);
	add_ODE_body(&sphere);

	// Rigid body #2 : cylinder
	cylinder = Cylinder(Point(0.9, 0.7*ly, r0), 0.05, Vector(0, 0, 0.2));
	cylinder.SetPartMass(r0, m_physparams.rho0[0]*0.3);
	cylinder.SetMass(r0, m_physparams.rho0[0]*0.3);
	cylinder.Unfill(parts, r0);
	cylinder.FillBorder(cylinder.GetParts(), r0);
	cylinder.ODEBodyCreate(m_ODEWorld, m_deltap);
	cylinder.ODEGeomCreate(m_ODESpace, m_deltap);
	add_ODE_body(&cylinder);

	/*joint = dJointCreateHinge(m_ODEWorld, 0);				// Create a hinge joint
	dJointAttach(joint, obstacle.m_ODEBody, 0);		// Attach joint to bodies
	dJointSetHingeAnchor(joint, 0.7, 0.24, 2*r0);	// Set a joint anchor
	dJointSetHingeAxis(joint, 0, 1, 0);*/

	return parts.size() + boundary_parts.size() + obstacle_parts.size() + get_ODE_bodies_numparts();
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


void OdeObjects::copy_to_array(float4 *pos, float4 *vel, particleinfo *info, hashKey* hash)
{
	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(BOUNDPART, 0, i);
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	for (uint k = 0; k < m_simparams.numODEbodies; k++) {
		PointVect & rbparts = get_ODE_body(k)->GetParts();
		std::cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = j; i < j + rbparts.size(); i++) {
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			info[i] = make_particleinfo(OBJECTPART, k, i - j);
			calc_localpos_and_hash(rbparts[i - j], info[i], pos[i], hash[i]);
		}
		j += rbparts.size();
		std::cout << ", part mass: " << pos[j-1].w << "\n";
	}

	std::cout << "Obstacle parts: " << obstacle_parts.size() << "\n";
	for (uint i = j; i < j + obstacle_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(BOUNDPART, 1, i);
		calc_localpos_and_hash(obstacle_parts[i-j], info[i], pos[i], hash[i]);
	}
	j += obstacle_parts.size();
	std::cout << "Obstacle part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}
