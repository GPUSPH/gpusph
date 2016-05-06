/*
 * OilJet.cu
 *
 *  Created on: 25 juin 2015
 *      Author: alexisherault
 */
#include <iostream>
#include <stdexcept>

#include "OilJet.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"


OilJet::OilJet(GlobalData *_gdata) : Problem(_gdata)
{
	// Data for problem setup
	layers = 5;

	/* SETUP_FRAMEWORK(
		kernel<WENDLAND>,
	    //viscosity<ARTVISC>,
		viscosity<KINEMATICVISC>,
		//viscosity<SPSVISC>,
		//boundary<LJ_BOUNDARY>
		//boundary<MK_BOUNDARY>
		boundary<DYN_BOUNDARY>,
		periodicity<PERIODIC_XY>
	); */

	SETUP_FRAMEWORK(
		formulation<SPH_GRENIER>,
		viscosity<DYNAMICVISC>,
		boundary<DYN_BOUNDARY>
	);

	set_deltap(0.05f);  // 0.05 is minimum to have 3 layers of particles in the cylinder

	// Water and oil level
	water_level = 2.;
	pipe_length = 2.;
	inner_diam = 0.4;

	// Size and origin of the simulation domain
	lx = 2;
	ly = 2;
	lz = water_level + pipe_length + layers*m_deltap;

	m_size = make_double3(lx , ly, 1.1*lz);
	m_origin = make_double3(0, 0, - pipe_length - layers*m_deltap);

	//addFilter(SHEPARD_FILTER, 20);
	  //MLS_FILTER

	// SPH parameters
	simparams()->dt = 0.00013;
	simparams()->dtadaptfactor = 0.2;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 2.; //seconds
	simparams()->maxneibsnum = 512;

	// Physical parameters
	physparams()->gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(physparams()->gravity);

	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 10.f);
	set_kinematic_visc(0, 1.0e-6);

	physparams()->artvisccoeff =  0.3;
	physparams()->smagfactor = 0.12*0.12*m_deltap*m_deltap;
	physparams()->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	//Wave piston definition:  location, start & stop times, stroke and frequency (2 \pi/period)
	piston_tstart = 0.0;
	piston_tend = simparams()->tend;
	piston_vel = 1.0;

	// Drawing and saving times
	add_writer(VTKWRITER, .01);  //second argument is saving time in seconds

	// Name of problem used for directory creation
	m_name = "OilJet";
}


OilJet::~OilJet(void)
{
	release_memory();
}


void OilJet::release_memory(void)
{
	parts_f1.clear();
	boundary_parts_f1.clear();
}

void
OilJet::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
		const float3& force, const float3& torque, const KinematicData& initial_kdata,
		KinematicData& kdata, double3& dx, EulerParameters& dr)
{
    dx = make_double3(0.0);
    dr.Identity();
    kdata.avel = make_double3(0.0);
    if (t0 >= piston_tstart & t1 <= piston_tend) {
    	kdata.lvel = make_double3(0.0, 0.0, piston_vel);
    	dx.z = -piston_vel*(t0 - t1);
	} else {
		kdata.lvel = make_double3(0.0);
	}
}


int OilJet::fill_parts()
{
	const int layersm1 = layers - 1;

	Cube fluid1 = Cube(Point(m_deltap/2, m_deltap/2, m_deltap/2), lx - m_deltap, ly - m_deltap, water_level - m_deltap);
	fluid1.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid1.Fill(parts_f1, m_deltap, true);

	double deltap_wall = m_deltap;
	Cube bottom = Cube(Point(m_deltap/2, m_deltap/2, -(layersm1 + 0.5)*m_deltap),
			lx - m_deltap, ly - m_deltap, layersm1*m_deltap);
	bottom.SetPartMass(deltap_wall, physparams()->rho0[0]);
	bottom.Fill(boundary_parts_f1, deltap_wall, true);

	double plength = pipe_length + layersm1*m_deltap - m_deltap/2.;
	Point corigin = Point(lx/2., ly/2., - plength - m_deltap/2.);
	Cylinder pipe = Cylinder(corigin, (inner_diam - m_deltap)/2. + layersm1*m_deltap, plength );
	pipe.Unfill(boundary_parts_f1, 0.4*deltap_wall);
	pipe.SetPartMass(deltap_wall, physparams()->rho0[0]);
	pipe.Fill(boundary_parts_f2, deltap_wall);

	Cylinder oil = Cylinder(corigin, (inner_diam - m_deltap)/2., plength);
	oil.SetPartMass(m_deltap, physparams()->rho0[0]);
	oil.Unfill(boundary_parts_f2, m_deltap);
	oil.Fill(parts_f2, m_deltap);

	piston_origin = make_double3(lx/2., ly/2., - plength + layersm1*m_deltap/2.);
	piston = Cylinder(corigin, (inner_diam - m_deltap)/2. + layersm1*m_deltap, layersm1*m_deltap);
	piston.Unfill(parts_f2, m_deltap);
	piston.SetPartMass(deltap_wall, physparams()->rho0[0]);
	piston.Fill(piston.GetParts(), deltap_wall);
	add_moving_body(&piston, MB_MOVING);
	set_body_cg(&piston, piston_origin);

	return parts_f1.size() + boundary_parts_f1.size() +
			parts_f2.size() + boundary_parts_f2.size() + get_bodies_numparts();
}


void OilJet::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	int j = 0;

	cout << "\nBoundary parts fluid1: " << boundary_parts_f1.size() << "\n";
	cout << "      " << j  << "--" << boundary_parts_f1.size() << "\n";
	for (uint i = j; i < j + boundary_parts_f1.size(); i++) {
		float ht = water_level - boundary_parts_f1[i-j](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_BOUNDARY, 0, i);  // first is type, object, 3rd id
		calc_localpos_and_hash(boundary_parts_f1[i-j], info[i], pos[i], hash[i]);
	}
	j += boundary_parts_f1.size();
	cout << "Boundary part fluid1 mass:" << pos[j-1].w << "\n";

	cout << "\nBoundary parts fluid2: " << boundary_parts_f2.size() << "\n";
	cout << "      " << j  << "--" << boundary_parts_f2.size() << "\n";
	for (uint i = j; i < j + boundary_parts_f2.size(); i++) {
		float ht = water_level - boundary_parts_f2[i-j](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht + 2*0.5*piston_vel/abs(physparams()->gravity.z) , 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_BOUNDARY, 0, i);  // first is type, object, 3rd id
		calc_localpos_and_hash(boundary_parts_f2[i-j], info[i], pos[i], hash[i]);
	}
	j += boundary_parts_f2.size();
	cout << "Boundary part fluid2 mass:" << pos[j-1].w << "\n";

	cout << "\nFluid1 parts: " << parts_f1.size() << "\n";
	cout << "      "<< j  << "--" << j + parts_f1.size() << "\n";
	for (uint i = j; i < j + parts_f1.size(); i++) {
		float ht = water_level - parts_f1[i-j](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(parts_f1[i-j], info[i], pos[i], hash[i]);
	}
	j += parts_f1.size();
	cout << "Fluid1 part mass:" << pos[j-1].w << "\n";

	cout << "\nFluid2 parts: " << parts_f2.size() << "\n";
	cout << "      "<< j  << "--" << j + parts_f2.size() << "\n";
	for (uint i = j; i < j + parts_f2.size(); i++) {
		float ht = water_level - parts_f2[i-j](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht, 0);
		vel[i] = make_float4(0, 0, piston_vel, rho);
		info[i]= make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(parts_f2[i-j], info[i], pos[i], hash[i]);
	}
	j += parts_f2.size();
	cout << "Fluid2 part mass:" << pos[j-1].w << "\n";

	uint object_particle_counter = 0;
	for (uint k = 0; k < m_bodies.size(); k++) {
		PointVect & rbparts = m_bodies[k]->object->GetParts();
		cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = 0; i < rbparts.size(); i++) {
			uint ij = i + j;
			float ht = water_level - rbparts[i](2);
			if (ht < 0)
				ht = 0.0;
			float rho = density(ht + 2*0.5*piston_vel/abs(physparams()->gravity.z), 0);
			//rho = physparams()->rho0[0];
			vel[ij] = make_float4(0, 0, piston_vel, rho);
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

	cout << "Everything uploaded" <<"\n";
}



