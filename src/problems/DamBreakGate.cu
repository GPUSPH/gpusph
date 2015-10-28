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

#include "DamBreakGate.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define SIZE_X		(1.60)
#define SIZE_Y		(0.67)
#define SIZE_Z		(0.40)

// default: origin in 0,0,0
#define ORIGIN_X	(0)
#define ORIGIN_Y	(0)
#define ORIGIN_Z	(0)


DamBreakGate::DamBreakGate(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	m_size = make_double3(SIZE_X, SIZE_Y, SIZE_Z + 0.7);
	m_origin = make_double3(ORIGIN_X, ORIGIN_Y, ORIGIN_Z);

	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,//DYNAMICVISC//SPSVISC
		boundary<LJ_BOUNDARY>
	);

	//addFilter(MLS_FILTER, 10);

	// SPH parameters
	set_deltap(0.015f);
	simparams()->dt = 0.0001f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 10.f;

	// Physical parameters
	H = 0.4f;
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

	set_kinematic_visc(0, 1.0e-6f);
	physparams()->artvisccoeff = 0.3f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);
	add_writer(COMMONWRITER, 0.0);

	// Name of problem used for directory creation
	m_name = "DamBreakGate";
}


DamBreakGate::~DamBreakGate(void)
{
	release_memory();
}


void DamBreakGate::release_memory(void)
{
	parts.clear();
	gate_parts.clear();
	obstacle_parts.clear();
	boundary_parts.clear();
}

void
DamBreakGate::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{
	const double tstart = 0.1;
	const double tend = 0.4;

	// Computing, at t = t1, new position of center of rotation (here only translation)
	// along with linear velocity
	if (t1 >= tstart && t1 <= tend) {
		kdata.lvel = make_double3(0.0, 0.0, 4.*(t1 - tstart));
		kdata.crot.z = initial_kdata.crot.z + 2.*(t1 - tstart)*(t1 - tstart);
		}
	else
		kdata.lvel = make_double3(0.0f);

	// Computing the displacement of center of rotation between t = t0 and t = t1
	double ti = min(tend, max(tstart, t0));
	double tf = min(tend, max(tstart, t1));
	dx.z = 2.*(tf - tstart)*(tf - tstart) - 2.*(ti - tstart)*(ti - tstart);

	// Setting angular velocity at t = t1 and the rotation between t = t0 and t = 1.
	// Here we have a simple translation movement so the angular velocity is null and
	// the rotation between t0 and t1 equal to identity.
	kdata.avel = make_double3(0.0f);
	dr.Identity();
}


int DamBreakGate::fill_parts()
{
	float r0 = physparams()->r0;

	Cube fluid, fluid1, fluid2, fluid3, fluid4;

	experiment_box = Cube(Point(ORIGIN_X, ORIGIN_Y, ORIGIN_Z), 1.6, 0.67, 0.4);

	float3 gate_origin = make_float3(0.4 + 2*physparams()->r0, 0, 0);
	gate = Rect (Point(gate_origin) + Point(ORIGIN_X, ORIGIN_Y, ORIGIN_Z), Vector(0, 0.67, 0),
				Vector(0,0,0.4));

	obstacle = Cube(Point(0.9 + ORIGIN_X, 0.24 + ORIGIN_Y, r0 + ORIGIN_Z), 0.12, 0.12, 0.4 - r0);

	fluid = Cube(Point(r0 + ORIGIN_X, r0 + ORIGIN_Y, r0 + ORIGIN_Z), 0.4, 0.67 - 2*r0, 0.4 - r0);

	bool wet = false;	// set wet to true have a wet bed experiment
	if (wet) {
		fluid1 = Cube(Point(0.4 + m_deltap + r0 + ORIGIN_X, r0 + ORIGIN_Y, r0 + ORIGIN_Z),
			0.5 - m_deltap - 2*r0, 0.67 - 2*r0, 0.03);

		fluid2 = Cube(Point(1.02 + r0  + ORIGIN_X, r0 + ORIGIN_Y, r0 + ORIGIN_Z),
			0.58 - 2*r0, 0.67 - 2*r0, 0.03);

		fluid3 = Cube(Point(0.9 + ORIGIN_X , m_deltap  + ORIGIN_Y, r0 + ORIGIN_Z),
			0.12, 0.24 - 2*r0, 0.03);

		fluid4 = Cube(Point(0.9 + ORIGIN_X , 0.36 + m_deltap  + ORIGIN_Y, r0 + ORIGIN_Z),
			0.12, 0.31 - 2*r0, 0.03);
	}

	boundary_parts.reserve(2000);
	parts.reserve(14000);
	gate_parts.reserve(2000);

	experiment_box.SetPartMass(r0, physparams()->rho0[0]);
	experiment_box.FillBorder(boundary_parts, r0, false);

	gate.SetPartMass(r0, physparams()->rho0[0]);
	gate.Fill(gate.GetParts(), r0, true);
	add_moving_body(&gate, MB_MOVING);

	obstacle.SetPartMass(r0, physparams()->rho0[0]);
	obstacle.FillBorder(obstacle_parts, r0, true);

	fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid.Fill(parts, m_deltap, true);

	if (wet) {
		fluid1.SetPartMass(m_deltap, physparams()->rho0[0]);
		fluid1.Fill(parts, m_deltap, true);
		fluid2.SetPartMass(m_deltap, physparams()->rho0[0]);
		fluid2.Fill(parts, m_deltap, true);
		fluid3.SetPartMass(m_deltap, physparams()->rho0[0]);
		fluid3.Fill(parts, m_deltap, true);
		fluid4.SetPartMass(m_deltap, physparams()->rho0[0]);
		fluid4.Fill(parts, m_deltap, true);
	}

	return parts.size() + boundary_parts.size() + obstacle_parts.size() + get_bodies_numparts();
}

void DamBreakGate::copy_to_array(BufferList &buffers)
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
		std::cout << ", part type: " << type(info[j-1])<< "\n";
	}

	std::cout << "Obstacle parts: " << obstacle_parts.size() << "\n";
	for (uint i = j; i < j + obstacle_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i] = make_particleinfo(PT_BOUNDARY, 1, i);
		calc_localpos_and_hash(obstacle_parts[i-j], info[i], pos[i], hash[i]);
	}
	j += obstacle_parts.size();
	std::cout << "Obstacle part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i] = make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}

