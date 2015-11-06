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

#include "WaveTank.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"


#define MK_par 2

WaveTank::WaveTank(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 9.0;
	ly = 0.6;
	lz = 1.0;


	// Data for problem setup
	slope_length = 8.5;
	h_length = 0.5;
	height = .63;
	beta = 4.2364*M_PI/180.0;

	// Add objects to the tank
	use_cyl = false;
	use_cone = false;

	SETUP_FRAMEWORK(
	    //viscosity<ARTVISC>,
		//viscosity<KINEMATICVISC>,
		viscosity<SPSVISC>,
		boundary<LJ_BOUNDARY>,
		//boundary<MK_BOUNDARY>,
		flags<ENABLE_DTADAPT | ENABLE_PLANES>
	);

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0, 0, 0);
	if (use_cyl || use_cone) {
		m_origin.z -= 2.0*height;
		m_size.z += 2.0*height;
	}

	addFilter(SHEPARD_FILTER, 20); // or MLS_FILTER

	if (get_option("testpoints", false)) {
		addPostProcess(TESTPOINTS);
	}

	// use a plane for the bottom
	use_bottom_plane = 1;  //1 for plane; 0 for particles

	// SPH parameters
	set_deltap(0.03f);  //0.005f;
	simparams()->dt = 0.0001;
	simparams()->dtadaptfactor = 0.2;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 10.0f; //seconds

	//WaveGage
	if (get_option("gages", false)) {
		add_gage(1, 0.3);
		add_gage(0.5, 0.3);
	}

	// Physical parameters
	H = 0.45;
	physparams()->gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(physparams()->gravity);

	float r0 = m_deltap;
	physparams()->r0 = r0;

	add_fluid( 1000.0f);
	set_equation_of_state(0,  7.0f, 20.f);
	set_kinematic_visc(0,1.0e-6);

	physparams()->artvisccoeff =  0.2;
	physparams()->smagfactor = 0.12*0.12*m_deltap*m_deltap;
	physparams()->kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

	// BC when using LJ
	physparams()->dcoeff = 0.5*g*H;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// BC when using MK
	physparams()->MK_K = g*H;
	physparams()->MK_d = 1.1*m_deltap/MK_par;
	physparams()->MK_beta = MK_par;

	//Wave paddle definition:  location, start & stop times, stroke and frequency (2 \pi/period)

	paddle_length = .7f;
	paddle_width = m_size.y - 2*r0;
	paddle_tstart=0.5f;
	paddle_origin = make_double3(0.25f, r0, 0.0f);
	paddle_tend = 30.0f;//seconds
	// The stroke value is given at free surface level H
	float stroke = 0.2;
	// m_mbamplitude is the maximal angular value for paddle angle
	// Paddle angle is in [-m_mbamplitude, m_mbamplitude]
	paddle_amplitude = atan(stroke/(2.0*(H - paddle_origin.z)));
	std::cout << "\npaddle_amplitude (radians): " << paddle_amplitude << "\n";
	paddle_omega = 2.0*M_PI/0.8;		// period T = 0.8 s

	// Drawing and saving times

	add_writer(VTKWRITER, .1);  //second argument is saving time in seconds

	// Name of problem used for directory creation
	m_name = "WaveTank";
}


WaveTank::~WaveTank(void)
{
	release_memory();
}


void WaveTank::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}

void
WaveTank::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{

    dx= make_double3(0.0);
    kdata.lvel=make_double3(0.0f, 0.0f, 0.0f);
    if (t1> paddle_tstart & t1 < paddle_tend){
       kdata.avel = make_double3(0.0, paddle_amplitude*paddle_omega*sin(paddle_omega*(t1-paddle_tstart)),0.0);
       EulerParameters dqdt = 0.5*EulerParameters(kdata.avel)*kdata.orientation;
       dr = EulerParameters::Identity() + (t1-t0)*dqdt*kdata.orientation.Inverse();
       dr.Normalize();
	   kdata.orientation = kdata.orientation + (t1 - t0)*dqdt;
	   kdata.orientation.Normalize();
	   }
	else {
	   kdata.avel = make_double3(0.0,0.0,0.0);
	   kdata.orientation = kdata.orientation;
	   dr.Identity();
	}
}


int WaveTank::fill_parts()
{
	const float r0 = physparams()->r0;
	const float br = (simparams()->boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);

	experiment_box = Cube(Point(0, 0, 0), h_length + slope_length,ly, height);

	boundary_parts.reserve(100);
	parts.reserve(34000);

    const float amplitude = -paddle_amplitude ;
	paddle = Rect(Point(paddle_origin), Vector(0, paddle_width, 0),
				Vector(paddle_length*sin(amplitude), 0, paddle_length*cos(amplitude)));
    paddle.SetPartMass(m_deltap, physparams()->rho0[0]);
	paddle.Fill(paddle.GetParts(), br, true);
	add_moving_body(&paddle, MB_MOVING);
	set_body_cg(&paddle, paddle_origin);

	bottom_rect = Rect(Point(h_length, 0, 0), Vector(0, ly, 0),
		//	Vector(slope_length/cos(beta), 0.0, slope_length*tan(beta)));
		 Vector(0.0,0.0,paddle_length));
	if (!use_bottom_plane) {
	   bottom_rect.SetPartMass(m_deltap, physparams()->rho0[0]);
	   bottom_rect.Fill(boundary_parts,br,true);
	   }

	Rect fluid;
	float z = 0;
	int n = 0;
	while (z < H) {
		z = n*m_deltap + 1.5*r0;    //z = n*m_deltap + 1.5*r0;
		float x = paddle_origin.x + (z - paddle_origin.z)*tan(amplitude) + 1.0*r0/cos(amplitude);
		float l = h_length + z/tan(beta) - 1.5*r0/sin(beta) - x;
		fluid = Rect(Point(x,  r0, z),
				Vector(0, ly-2.0*r0, 0), Vector(l, 0, 0));
		fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
		fluid.Fill(parts, m_deltap, true);
		n++;
	 }

	if (m_simframework->hasPostProcessEngine(TESTPOINTS)) {
		Point pos = Point(0.5748, 0.1799, 0.2564, 0.0);
		test_points.push_back(pos);
		pos = Point(0.5748, 0.2799, 0.2564, 0.0);
		test_points.push_back(pos);
		pos = Point(1.5748, 0.2799, 0.2564, 0.0);
		test_points.push_back(pos);
	}

	if (use_cyl) {
		Point p[10];
		p[0] = Point(h_length + slope_length/(cos(beta)*10), ly/2., 0);
		p[1] = Point(h_length + slope_length/(cos(beta)*10), ly/6.,  0);
		p[2] = Point(h_length + slope_length/(cos(beta)*10), 5*ly/6, 0);
		p[3] = Point(h_length + slope_length/(cos(beta)*5), 0, 0);
		p[4] = Point(h_length + slope_length/(cos(beta)*5), ly/3, 0);
		p[5] = Point(h_length + slope_length/(cos(beta)*5), 2*ly/3, 0);
		p[6] = Point(h_length + slope_length/(cos(beta)*5), ly, 0);
		p[7] = Point(h_length + 3*slope_length/(cos(beta)*10), ly/6, 0);
		p[8] = Point(h_length + 3*slope_length/(cos(beta)*10), ly/2, 0);
		p[9] = Point(h_length+ 3*slope_length/(cos(beta)*10), 5*ly/6, 0);
		p[10] = Point(h_length+ 4*slope_length/(cos(beta)*10), ly/2, 0);

		for (int i = 0; i < 11; i++) {
			cyl[i] = Cylinder(p[i], Vector(.025, 0, 0), Vector(0, 0, height));
			cyl[i].SetPartMass(m_deltap, physparams()->rho0[0]);
			cyl[i].FillBorder(boundary_parts, br, false, false);
			cyl[i].Unfill(parts, br);
		}
	}
	if (use_cone) {
		Point p1 = Point(h_length + slope_length/(cos(beta)*10), ly/2, 0);
		cone = Cone(p1,Vector(ly/4, 0.0, 0.0), Vector(ly/10, 0., 0.), Vector(0, 0, height));
		cone.SetPartMass(m_deltap, physparams()->rho0[0]);
		cone.FillBorder(boundary_parts, br, false, true);
		cone.Unfill(parts, br);
    }

	return  boundary_parts.size() + get_bodies_numparts() + parts.size() + test_points.size();
}

void WaveTank::copy_planes(PlaneList &planes)
{
	const double w = m_size.y;
	const double l = h_length + slope_length;

	//  plane is defined as a x + by +c z + d= 0
	planes.push_back( implicit_plane(0, 0, 1.0, 0) );   //bottom, where the first three numbers are the normal, and the last is d.
	planes.push_back( implicit_plane(0, 1.0, 0, 0) );   //wall
	planes.push_back( implicit_plane(0, -1.0, 0, w) ); //far wall
	planes.push_back( implicit_plane(1.0, 0, 0, 0) );  //end
	planes.push_back( implicit_plane(-1.0, 0, 0, l) );  //one end
	if (use_bottom_plane)  {
		planes.push_back( implicit_plane(-sin(beta),0,cos(beta), h_length*sin(beta)) );  //sloping bottom starting at x=h_length
	}
}

void WaveTank::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	int j = 0;

	if (test_points.size()) {
		//Testpoints
		std::cout << "\nTest points: " << test_points.size() << "\n";
		std::cout << "      " << j << "--" << test_points.size() << "\n";
		for (uint i = 0; i < test_points.size(); i++) {
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
			info[i]= make_particleinfo(PT_TESTPOINT, 0, i);  // first is type, object, 3rd id
			calc_localpos_and_hash(test_points[i], info[i], pos[i], hash[i]);
		}
		j += test_points.size();
		std::cout << "Test point mass:" << pos[j-1].w << "\n";
	}

	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
	std::cout << "      " << j  << "--" << boundary_parts.size() << "\n";
	for (uint i = j; i < j + boundary_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
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
				float ht = H - rbparts[i](2);
				if (ht < 0)
					ht = 0.0;
				float rho = density(ht, 0);
				rho = physparams()->rho0[0];
				vel[ij] = make_float4(0, 0, 0, rho);
				uint ptype = (uint) PT_BOUNDARY;
				switch (m_bodies[k]->type) {
					case MB_ODE:
						ptype |= FG_COMPUTE_FORCE | FG_MOVING_BOUNDARY;
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
	std::cout << "      "<< j  << "--" << j + parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		info[i]= make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";

	std::cout << "Everything uploaded" <<"\n";
}


#undef MK_par
