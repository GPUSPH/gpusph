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

#include "WaveTank.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"


#define MK_par 2

WaveTank::WaveTank(GlobalData *_gdata) : XProblem(_gdata)
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

	SETUP_FRAMEWORK(
	    //viscosity<ARTVISC>,
		//viscosity<KINEMATICVISC>,
		viscosity<SPSVISC>,
		boundary<LJ_BOUNDARY>,
		//boundary<MK_BOUNDARY>,
		add_flags<ENABLE_PLANES>
	);

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0, 0, 0);
	if (use_cyl) {
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
	set_timestep(0.0001);
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
	set_gravity(-9.81f);
	setMaxFall(H);

	float r0 = m_deltap;
	physparams()->r0 = r0;

	add_fluid( 1000.0f);
	set_equation_of_state(0,  7.0f, 20.f);
	set_kinematic_visc(0,1.0e-6);

	physparams()->artvisccoeff =  0.2;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;

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
	cout << "\npaddle_amplitude (radians): " << paddle_amplitude << "\n";
	paddle_omega = 2.0*M_PI/0.8;		// period T = 0.8 s

	// Drawing and saving times

	add_writer(VTKWRITER, .1);  //second argument is saving time in seconds

	// Name of problem used for directory creation
	m_name = "WaveTank";

	// Building the geometry
	const float br = (simparams()->boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);
	setPositioning(PP_CORNER);

	GeometryID experiment_box = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
	Point(0, 0, 0), h_length + slope_length,ly, height);
	disableCollisions(experiment_box);

  const float amplitude = -paddle_amplitude ;
	GeometryID paddle = addBox(GT_MOVING_BODY, FT_BORDER,
		Point(paddle_origin),	0, paddle_width, paddle_length);
	rotate(paddle, 0,-amplitude, 0);
	disableCollisions(paddle);

	if (!use_bottom_plane) {
		GeometryID bottom = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
				Point(h_length, 0, 0), 0, ly, paddle_length);
		//	Vector(slope_length/cos(beta), 0.0, slope_length*tan(beta)));
		disableCollisions(bottom);
	}

	GeometryID fluid;
	float z = 0;
	int n = 0;
	while (z < H) {
		z = n*m_deltap + 1.5*r0;    //z = n*m_deltap + 1.5*r0;
		float x = paddle_origin.x + (z - paddle_origin.z)*tan(amplitude) + 1.0*r0/cos(amplitude);
		float l = h_length + z/tan(beta) - 1.5*r0/sin(beta) - x;
		fluid = addRect(GT_FLUID, FT_SOLID, Point(x,  r0, z),
				l, ly-2.0*r0);
		n++;
	 }

	if (hasPostProcess(TESTPOINTS)) {
		Point pos = Point(0.5748, 0.1799, 0.2564, 0.0);
		addTestPoint(pos);
		pos = Point(0.5748, 0.2799, 0.2564, 0.0);
		addTestPoint(pos);
		pos = Point(1.5748, 0.2799, 0.2564, 0.0);
		addTestPoint(pos);
	}

	if (use_cyl) {
		setPositioning(PP_BOTTOM_CENTER);
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
			GeometryID cyl = addCylinder(GT_FIXED_BOUNDARY, FT_BORDER,
				p[i], .025, height);
			disableCollisions(cyl);
			setEraseOperation(cyl, ET_ERASE_FLUID);
		}
	}
}


void
WaveTank::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{

    dx= make_double3(0.0);
    kdata.lvel=make_double3(0.0f, 0.0f, 0.0f);
    if (t1> paddle_tstart && t1 < paddle_tend){
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

#undef MK_par
