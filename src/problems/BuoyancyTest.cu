/*
 * BuyancyTest.cc
 *
 *  Created on: 20 juin 2014
 *      Author: alexisherault
 */

#include "BuoyancyTest.h"
#include <iostream>

#include "GlobalData.h"
#include "cudasimframework.cu"
#include "Cube.h"
#include "Sphere.h"
#include "Point.h"
#include "Vector.h"

BuoyancyTest::BuoyancyTest(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 1.0;
	ly = 1.0;
	lz = 1.0;
	H = 0.7;

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0.0, 0.0, 0.0);

	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		viscosity<ARTVISC>,
		//viscosity<SPSVISC>,
		//viscosity<KINEMATICVISC>,
		boundary<DYN_BOUNDARY>
	);

	//addFilter(SHEPARD_FILTER, 37);

	// SPH parameters
	set_deltap(0.02); //0.008
	simparams()->dt = 0.0003f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 5.0f; //0.00036f

	// Physical parameters
	H = 0.6f;
	physparams()->gravity = make_float3(0.0, 0.0, -9.81f);
	double g = length(physparams()->gravity);
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 40.f);

    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	physparams()->dcoeff = 5.0f*g*H;
	physparams()->r0 = m_deltap;

	set_kinematic_visc(0, 1.0e-6f);
	physparams()->artvisccoeff = 0.3f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	physparams()->smagfactor = 0.12*0.12*m_deltap*m_deltap;
	physparams()->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;

	// Initialize Chrono
	InitChrono();

	//add_writer(VTKWRITER, 0.005);
	add_writer(VTKWRITER, 0.1);
	add_writer(COMMONWRITER, 0.0);

	// Name of problem used for directory creation
	m_name = "BuoyancyTest";
}


BuoyancyTest::~BuoyancyTest(void)
{
	release_memory();
}


void BuoyancyTest::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}


int BuoyancyTest::fill_parts()
{
	const double dp = m_deltap;
	const int layers = 4;

	Cube experiment_box = Cube(Point(0, 0, 0), lx, ly, lz);
	//experiment_box.BodyCreate(m_bodies_physical_system, 0, true);
	//experiment_box.GetBody()->SetBodyFixed(true);

	Cube fluid = Cube(Point(dp*layers, dp*layers, dp*layers),
		lx - 2.0*dp*layers, ly - 2.0*dp*layers, H);

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	experiment_box.SetPartMass(m_deltap, physparams()->rho0[0]);
	experiment_box.FillIn(boundary_parts, m_deltap, layers, false);
	fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid.Fill(parts, m_deltap, true);

	const int object_type = 0;
	Object *floating;
	switch (object_type) {
		case 0: {
			double olx = 10.0*m_deltap;
			double oly = 10.0*m_deltap;
			double olz = 10.0*m_deltap;
			cube  = Cube(Point(lx/2.0 - olx/2.0, ly/2.0 - oly/2.0, H/2.0 - olz/2.0), olx, oly, olz);
			floating = &cube;
			}
			break;

		case 1: {
			double r = 6.0*m_deltap;
			sphere = Sphere(Point(lx/2.0, ly/2.0, H/2.0 - r/4.0), r);
			floating = &sphere;
			}
			break;

		case 2: {
			double R = lx*0.2;
			double r = 4.0*m_deltap;
			torus = Torus(Point(lx/2.0, ly/2.0, H/2.0), Vector(0, 0, 1), R, r);
			floating = &torus;
			}
			break;
	}

	floating->SetMass(m_deltap, physparams()->rho0[0]*0.5);
	floating->SetInertia(m_deltap);
	floating->SetPartMass(m_deltap, physparams()->rho0[0]);
	floating->FillIn(floating->GetParts(), m_deltap, layers);
	floating->Unfill(parts, m_deltap*0.85);

	bool collide = true;
	if (object_type != 2)
		collide = false;
	floating->BodyCreate(m_bodies_physical_system, dp, collide);
	add_moving_body(floating, MB_ODE);
	floating->BodyPrintInformation(collide);

	PointVect & rbparts = get_mbdata(uint(0))->object->GetParts();
	cout << "Rigid body " << 1 << ": " << rbparts.size() << " particles \n";
	cout << "totl rb parts:" << get_bodies_numparts() << "\n";
	return parts.size() + boundary_parts.size() + get_bodies_numparts();
}


void
BuoyancyTest::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	cout << "Boundary parts: " << boundary_parts.size() << endl;
	for (uint i = 0; i < boundary_parts.size(); ++i) {
		float ht = H - boundary_parts[i](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i] = make_particleinfo(PT_BOUNDARY, 0, i);
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	uint j = boundary_parts.size();
	cout << "Boundary part mass: " << pos[j-1].w << endl;

	uint object_particle_counter = 0;
	for (uint k = 0; k < m_bodies.size(); k++) {
		PointVect & rbparts = m_bodies[k]->object->GetParts();
		cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = 0; i < rbparts.size(); i++) {
			uint ij = i + j;
			float ht = H - rbparts[i](2);
			if (ht < 0)
				ht = 0.0;
			// Test density 1
			//			float rho = density(ht, 0);
			float rho = physparams()->rho0[0];
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
		cout << ", part mass: " << pos[j-1].w << "\n";
	}

	cout << "Fluid parts: " << parts.size() << endl;
	for (uint i = 0; i < parts.size(); ++i) {
		uint ij = i+j;
		float ht = H - parts[i](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht, 0);
		vel[ij] = make_float4(0, 0, 0, rho);
		info[ij] = make_particleinfo(PT_FLUID, 0, ij);
		calc_localpos_and_hash(parts[i], info[ij], pos[ij], hash[ij]);
	}
	j += parts.size();

	cout << "Fluid part mass: " << pos[j-1].w << endl;

	flush(cout);
}
