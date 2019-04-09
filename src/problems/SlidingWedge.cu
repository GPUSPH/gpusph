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

#include "SlidingWedge.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define MK_par 2

SlidingWedge::SlidingWedge(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 10;
	ly = 3.7;
	lz = 3.1;

	// Data for problem setup
	H = 2.44; //Water depth
	tan_beta = 0.5;
	beta = atan(0.5); //slope angle in radian: atan(0.025) or atan(0.05)
	slope_length = 2*H/tan_beta; //slope length projected on x
	x0 = 0.5;
	layers = 1;

	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
	    //viscosity<ARTVISC>,
		viscosity<KINEMATICVISC>,
		//viscosity<SPSVISC>,
		boundary<LJ_BOUNDARY>,
		//boundary<MK_BOUNDARY>
		//boundary<DYN_BOUNDARY>,
		densitydiffusion<FERRARI>
	);

	addPostProcess(SURFACE_DETECTION);

	set_deltap(0.1f);  // 0.05 is minimum to have 3 layers of particles in the cylinder
	m_size = make_double3(lx + 2*layers*m_deltap , ly + 2*layers*m_deltap, lz + layers*m_deltap);
	m_origin = make_double3(-x0 - layers*m_deltap, - ly/2. - layers*m_deltap, -H);

	// SPH parameters
	resize_neiblist(128);
	simparams()->dt = 0.00013;
	simparams()->dtadaptfactor = 0.2;
	simparams()->buildneibsfreq = 10;
	t0 = 0.4;
	simparams()->tend = 4.0 + t0; //seconds
	simparams()->densityDiffCoeff = 1.0;

	// Physical parameters
	set_gravity(-9.81f);
	float g = get_gravity_magnitude();

	add_fluid(1000.0);
	set_equation_of_state(0, 7.0f, 30.f);
	set_kinematic_visc(0, 1.0e-6);

	physparams()->artvisccoeff =  0.3;
	physparams()->smagfactor = 0.12*0.12*m_deltap*m_deltap;
	physparams()->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	//WaveGage
	const double wg1x = 1.83, wg1y = 0;
	const double wg2x = 1.2446, wg2y = 0.635;
	const float slength = simparams()->slength;
	add_gage(wg1x, wg1y, slength);
	add_gage(wg1x, wg1y, 0.5*slength);
	add_gage(wg1x, wg1y, 0.25*slength);
	add_gage(wg1x, wg1y, 0);
	add_gage(wg2x, wg2y, slength);
	add_gage(wg2x, wg2y, 0.5*slength);
	add_gage(wg2x, wg2y, 0.25*slength);
	add_gage(wg2x, wg2y, 0);

	// Allocate data for bodies

	// Drawing and saving times
	add_writer(VTKWRITER, .1);  //second argument is saving time in seconds

	// Name of problem used for directory creation
	m_name = "SlidingWedge";
}


SlidingWedge::~SlidingWedge(void)
{
	release_memory();
}


void SlidingWedge::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}

void
SlidingWedge::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
		const float3& force, const float3& torque, const KinematicData& initial_kdata,
		KinematicData& kdata, double3& dx, EulerParameters& dr)
{
	const double a = -0.097588;
	const double b = 0.759361;
	const double c = 0.078776;
    dx = make_double3(0.0);
    dr.Identity();
    kdata.avel = make_double3(0.0);
    if (t0 >= t0 & t1 <= t0 + 2.6714) {
    	const double f0 = a*t0*t0*t0 + b*t0*t0 + c*t0;
    	const double f1 = a*t1*t1*t1 + b*t1*t1 + c*t1;
    	const double v1 = 3*a*t1*t1 + 2*b*t1 + t1;
    	kdata.lvel = make_double3(v1*cos(beta), 0.0, -v1*sin(beta));
    	dx.x = (f1 - f0)*cos(beta);
    	dx.z = -(f1 - f0)*sin(beta);
	} else {
		kdata.lvel = make_double3(0.0);
	}
}


int SlidingWedge::fill_parts(bool fill)
{
	boundary_parts.reserve(55000);
	parts.reserve(140000);

	Cube water =  Cube(Point(-x0, -ly/2., - H), lx, ly, H);
	water.SetPartMass(m_deltap, physparams()->rho0[0]);
	water.InnerFill(parts, m_deltap);
	Cube experiment_box =  Cube(Point(-x0, -ly/2., - H), lx, ly, lz);
	experiment_box.SetPartMass(m_deltap, physparams()->rho0[0]);
	PlaneCut(parts, 1, 0, 2, 0);
	Cube slope = Cube(Point(2*H + layers*m_deltap, -ly/2., -H - m_deltap/2.),
			(2*H + x0 + 2*layers*m_deltap)/cos(beta), ly, layers*m_deltap,
			EulerParameters(Vector(0, 1, 0), M_PI + beta));
	slope.SetPartMass(m_deltap, physparams()->rho0[0]);
	slope.InnerFill(boundary_parts, m_deltap);
	PlaneCut(boundary_parts, 1, 0, 0, x0);
	PlaneCut(boundary_parts, -1, 0, 0, 2*H);
	experiment_box.FillOut(boundary_parts, m_deltap, layers, false);
	PlaneCut(boundary_parts, 1, 0, 2, 2*layers*m_deltap);
	const double hw = 0.61;
	const double lw = 0.91;
	const double ww = 0.455;
	const double D = 0.1;
	wedge = Cube(Point(D/tan_beta, -ww/2., - D - hw), lw, ww, hw);
	wedge.SetPartMass(m_deltap, physparams()->rho0[0]);
	wedge.FillIn(wedge.GetParts(), m_deltap, layers);
	//PlaneCut(wedge.GetParts(), 1, 0, 2, 0);
	add_moving_body(&wedge, MB_MOVING);
	wedge.Unfill(parts, m_deltap/2);

	return parts.size() + boundary_parts.size() + get_bodies_numparts();
}


void SlidingWedge::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	int j = 0;

	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
	std::cout << "      " << j  << "--" << boundary_parts.size() << "\n";
	for (uint i = j; i < j + boundary_parts.size(); i++) {
		float ht =  - boundary_parts[i-j](2);
		if (ht < 0)
			ht = 0.0;
		float rho = hydrostatic_density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_BOUNDARY, 0, i);  // first is type, object, 3rd id
		calc_localpos_and_hash(boundary_parts[i-j], info[i], pos[i], hash[i]);
	}
	j += boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	uint object_particle_counter = 0;
	for (uint k = 0; k < m_bodies.size(); k++) {
		PointVect & rbparts = m_bodies[k]->object->GetParts();
		std::cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = 0; i < rbparts.size(); i++) {
			uint ij = i + j;
			float ht = - rbparts[i](2);
			if (ht < 0)
				ht = 0.0;
			float rho = hydrostatic_density(ht, 0);
			//rho = physparams()->rho0[0];
			vel[ij] = make_float4(0, 0, 0, rho);
			uint ptype = (uint) PT_BOUNDARY;
			switch (m_bodies[k]->type) {
				case MB_FLOATING:
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

	std::cout << "\nFluid parts: " << parts.size() << "\n";
	std::cout << "      "<< j  << "--" << j + parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		float ht =  - parts[i-j](2);
		if (ht < 0)
			ht = 0.0;
		float rho = hydrostatic_density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";

	std::cout << "Everything uploaded" <<"\n";
}

#undef MK_par
