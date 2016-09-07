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

#include <iostream>
#include <stdexcept>

#include "OffshorePile.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define MK_par 2

OffshorePile::OffshorePile(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 60;
	ly = 1.5;
	lz = 3.0;

	// Data for problem setup
	H = 1.0; //max water depth
	double tan_beta = 0.025;
	beta = atan(0.025); //slope angle in radian: atan(0.025) or atan(0.05)
	slope_length = H/tan_beta; //slope length projected on x
	h_length = 4.5; //72-slope_length; //flat bottom length
	height = 1.1+0.4; //cylinder height + cylinder z position
	layers = 3;

	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
	    //viscosity<ARTVISC>,
		viscosity<KINEMATICVISC>,
		//viscosity<SPSVISC>,
		//boundary<LJ_BOUNDARY>
		//boundary<MK_BOUNDARY>
		boundary<DYN_BOUNDARY>,
		periodicity<PERIODIC_Y>
	);

	set_deltap(0.05f);  // 0.05 is minimum to have 3 layers of particles in the cylinder
	x0 = -1.;
	periodic_offset_y = m_deltap/2.;
	m_size = make_double3(lx - x0 , ly + m_deltap, lz + 1.5*layers*m_deltap);
	m_origin = make_double3(x0, 0., -1.5*layers*m_deltap);

	addFilter(SHEPARD_FILTER, 20);
	  //MLS_FILTER

	addPostProcess(SURFACE_DETECTION);

	// SPH parameters
	simparams()->dt = 0.00013;
	simparams()->dtadaptfactor = 0.2;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 160; //seconds

	// Physical parameters
	physparams()->gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(physparams()->gravity);

	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 25.f);
	set_kinematic_visc(0, 1.0e-6);

	physparams()->artvisccoeff =  0.3;
	physparams()->smagfactor = 0.12*0.12*m_deltap*m_deltap;
	physparams()->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;


	//Wave piston definition:  location, start & stop times, stroke and frequency (2 \pi/period)
	piston_height = 2*H;
	piston_width = ly;
	piston_tstart = 0.2;
	piston_tend = simparams()->tend;
	float stroke = 0.399; // EOL37: 0.145; // EOL95: 0.344; // EOL96: 0.404;
	//float period = 2.4;
	piston_amplitude = stroke/2.;
	piston_omega = 2.0*M_PI/2.4;		// period T = 2.4 s
	piston_origin = make_double3(-(layers + 0.5)*m_deltap, periodic_offset_y, -m_deltap);

	// Cylinder data
	cyl_diam = 0.2 + m_deltap;
	cyl_height = 2*H;
	cyl_xpos = h_length + 0.4/tan_beta;
	cyl_rho = 607.99;

	//WaveGage
	const float slength = simparams()->slength;
	add_gage(cyl_xpos, ly/2 + periodic_offset_y + 0.5, slength);
	add_gage(cyl_xpos, ly/2 + periodic_offset_y + 0.5, 0.5*slength);
	add_gage(cyl_xpos, ly/2 + periodic_offset_y + 0.5, 0.25*slength);
	add_gage(1.0, ly/2 + periodic_offset_y, m_deltap);
	add_gage(h_length, ly/2 + periodic_offset_y, m_deltap);
	add_gage(h_length-h_length/4, ly/2 + periodic_offset_y, m_deltap);
	add_gage(h_length-h_length/2, ly/2 + periodic_offset_y, m_deltap);
	add_gage(h_length-h_length*3/4, ly/2 + periodic_offset_y, m_deltap);

	// Allocate data for bodies
	dInitODE();
	m_ODEWorld = dWorldCreate();	// Create a dynamic world
	m_ODESpace = dHashSpaceCreate(0);
	m_ODEJointGroup = dJointGroupCreate(0);
	dWorldSetGravity(m_ODEWorld, physparams()->gravity.x, physparams()->gravity.y, physparams()->gravity.z);	// Set gravity（x, y, z)

	// Drawing and saving times
	add_writer(VTKWRITER, .1);  //second argument is saving time in seconds
	add_writer(COMMONWRITER, 0.0);

	// Name of problem used for directory creation
	m_name = "OffshorePile";
}


OffshorePile::~OffshorePile(void)
{
	release_memory();
	dWorldDestroy(m_ODEWorld);
	dCloseODE();
}


void OffshorePile::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}

void
OffshorePile::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
		const float3& force, const float3& torque, const KinematicData& initial_kdata,
		KinematicData& kdata, double3& dx, EulerParameters& dr)
{
    dx= make_double3(0.0);
    dr.Identity();
    kdata.avel = make_double3(0.0);
    if (t0 >= piston_tstart & t1 <= piston_tend) {
    	const double arg0 = piston_omega*(t0 - piston_tstart);
    	const double arg1 = piston_omega*(t1 - piston_tstart);
    	kdata.lvel = make_double3(-piston_amplitude*piston_omega*sin(arg1), 0.0, 0.0);
    	dx.x = piston_amplitude*(cos(arg1)-cos(arg0));
	} else {
		kdata.lvel = make_double3(0.0);
	}
}


int OffshorePile::fill_parts()
{
	boundary_parts.reserve(55000);
	parts.reserve(140000);

	const int layersm1 = layers - 1;

	piston = Cube(Point(piston_origin), layersm1*m_deltap, piston_width, piston_height);
    piston.SetPartMass(m_deltap, physparams()->rho0[0]);
	piston.Fill(piston.GetParts(), m_deltap);
	add_moving_body(&piston, MB_MOVING);
	set_body_cg(&piston, piston_origin);

	Cube bottom_flat = Cube(Point(x0, periodic_offset_y, -(layersm1 + 0.5)*m_deltap),
			h_length - x0 + 5*m_deltap , ly, layersm1*m_deltap);
	Cube bottom_slope = Cube(Point(h_length, periodic_offset_y, -(layersm1 + 0.5)*m_deltap),
			lx - h_length, ly, layersm1*m_deltap, EulerParameters(Vector(0, 1, 0), -beta));
	bottom_flat.SetPartMass(m_deltap, physparams()->rho0[0]);
	bottom_flat.Fill(boundary_parts, m_deltap, true);
	bottom_slope.Unfill(boundary_parts, m_deltap*0.9);
	bottom_slope.SetPartMass(m_deltap, physparams()->rho0[0]);
	bottom_slope.Fill(boundary_parts, m_deltap, true);
	double zfw = (lx - h_length)*tan(beta) - layersm1*m_deltap;
	Cube far_wall = Cube(Point(lx - layersm1*m_deltap, periodic_offset_y, zfw), layersm1*m_deltap, ly, H);
	far_wall.SetPartMass(m_deltap, physparams()->rho0[0]);
	far_wall.Unfill(boundary_parts, 0.9*m_deltap);
	far_wall.Fill(boundary_parts, m_deltap, true);

	Cube fluid1 = Cube(Point(m_deltap/2., periodic_offset_y, m_deltap/2.), h_length, ly, H - m_deltap);
	Cube fluid2 = Cube(Point(h_length + m_deltap, periodic_offset_y,  m_deltap/2.),
			lx - h_length - m_deltap, ly, H - m_deltap/2, EulerParameters(Vector(0, 1, 0), -beta));
	fluid1.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid2.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid2.Fill(parts, m_deltap);
	fluid1.Unfill(parts, m_deltap);
	double hu = 1.2*(lx - h_length)*tan(beta);
	Cube unfill_top = Cube(Point(h_length + m_deltap, periodic_offset_y, H + m_deltap/2.), lx - h_length, ly, H + hu);
	unfill_top.Unfill(parts, m_deltap);
	fluid1.Fill(parts, m_deltap);

	/*fluid1.Fill(parts, m_deltap);
	double hu = 1.2*(lx - h_length)*tan(beta);
	Cube unfill_slope = Cube(Point(h_length, 0, -hu), 1.05*(lx - h_length), ly, hu, EulerParameters(Vector(0, 1, 0), -beta));
	unfill_slope.Unfill(parts, m_deltap/2.);*/

	// Rigid body : cylinder
	cyl = Cylinder(Point(cyl_xpos, ly/2. + periodic_offset_y, 0), (cyl_diam - m_deltap)/2., cyl_height);
	cyl.SetPartMass(m_deltap, physparams()->rho0[0]);
	cyl.SetMass(m_deltap, cyl_rho);
	cyl.FillIn(cyl.GetParts(), m_deltap, layers);
	cyl.Unfill(parts, m_deltap);
	cyl.Unfill(boundary_parts, 0.8*m_deltap);

	cyl.ODEBodyCreate(m_ODEWorld, m_deltap);
	cyl.ODEGeomCreate(m_ODESpace, m_deltap);
	dBodySetLinearVel(cyl.ODEGetBody(), 0.0, 0.0, 0.0);
	dBodySetAngularVel(cyl.ODEGetBody(), 0.0, 0.0, 0.0);
	add_moving_body(&cyl, MB_FLOATING);

	joint = dJointCreateFixed(m_ODEWorld, 0);				// Create a fixed joint
	dJointAttach(joint, cyl.ODEGetBody(), 0);				// Attach joint to cylinder and fixed frame
	dJointSetFixed(joint);

	return parts.size() + boundary_parts.size() + get_bodies_numparts();
}


void OffshorePile::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	int j = 0;

	cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
	cout << "      " << j  << "--" << boundary_parts.size() << "\n";
	for (uint i = j; i < j + boundary_parts.size(); i++) {
		float ht = H - boundary_parts[i-j](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_BOUNDARY, 0, i);  // first is type, object, 3rd id
		calc_localpos_and_hash(boundary_parts[i-j], info[i], pos[i], hash[i]);
	}
	j += boundary_parts.size();
	cout << "Boundary part mass:" << pos[j-1].w << "\n";

	uint object_particle_counter = 0;
	for (uint k = 0; k < m_bodies.size(); k++) {
		PointVect & rbparts = m_bodies[k]->object->GetParts();
		cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = 0; i < rbparts.size(); i++) {
			uint ij = i + j;
			float ht = H - rbparts[i](2);
			if (ht < 0)
				ht = 0.0;
			float rho = density(ht, 0);
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
		cout << ", part mass: " << pos[j-1].w << "\n";
	}

	cout << "\nFluid parts: " << parts.size() << "\n";
	cout << "      "<< j  << "--" << j + parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		float ht = H - parts[i-j](2);
		if (ht < 0)
			ht = 0.0;
		float rho = density(ht, 0);
		vel[i] = make_float4(0, 0, 0, rho);
		info[i]= make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	cout << "Fluid part mass:" << pos[j-1].w << "\n";

	cout << "Everything uploaded" <<"\n";
}

#undef MK_par
