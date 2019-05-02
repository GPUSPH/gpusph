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

#include "SolitaryWave.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

#define MK_par 2

SolitaryWave::SolitaryWave(GlobalData *_gdata) : XProblem(_gdata)
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
		add_flags<ENABLE_PLANES>
	);

	//addFilter(SHEPARD_FILTER, 20);


	// Add objects to the tank
	icyl = 1;	// icyl = 0 means no cylinders

	i_use_bottom_plane = 1; // 1 for real plane instead of boundary parts

	// SPH parameters
	set_deltap(0.01f);  //0.005f;
	set_timestep(0.00013f);
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 10.0;

	addPostProcess(VORTICITY);

	// Physical parameters
	H = 0.45f;
	set_gravity(-9.81f);
	setMaxFall(H);
	float g = get_gravity_magnitude();

	add_fluid(1000.0f);
	set_equation_of_state(0,  7.0f, 20.f);
	const float r0 = m_deltap;

	physparams()->artvisccoeff = 0.3f;
	set_kinematic_visc(0, 1.0e-6f);
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

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

	// Building the geometry
	setPositioning(PP_CORNER);
	const float width = ly;

	const float br = (simparams()->boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);

	GeometryID fluid;
	float z = 0;
	int n = 0;
	while (z < H) {
		z = n*m_deltap + 1.5*r0;
		float x = piston_initial_crotx + r0;
		float l = h_length + z/tan(beta) - 1.5*r0/sin(beta) - x;
		fluid = addRect(GT_FLUID, FT_SOLID, Point(x, r0, z),
				l, width-2.0*r0);
		n++;
	 }
	GeometryID piston = addBox(GT_MOVING_BODY, FT_BORDER, Point(piston_initial_crotx, 0, 0), 0, width, height);
	//piston.SetPartMass(m_deltap, physparams()->rho0[0]);
	//piston.Fill(piston.GetParts(), br, true);
	disableCollisions(piston);
	if (i_use_bottom_plane == 0) {
		GeometryID experiment_box1 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
				Point(h_length, 0, 0),
				slope_length/cos(beta), width, slope_length*tan(beta));
		disableCollisions(experiment_box1);
	}

	if (icyl == 1) {
		setPositioning(PP_BOTTOM_CENTER);
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
			cyl[i] = addCylinder(GT_MOVING_BODY, FT_BORDER, p[i], radius, height);
			disableCollisions(cyl[i]);
		}
	}
}

void
SolitaryWave::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{
	dx = make_double3(0.0);
	if (index == 0) { // piston index
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

#undef MK_par
