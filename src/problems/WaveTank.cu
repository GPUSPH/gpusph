/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

WaveTank::WaveTank(GlobalData *_gdata) : Problem(_gdata)
{
	// use a plane for the bottom
	const bool use_bottom_plane = get_option("bottom-plane", true);
	// Add objects to the tank
	const bool use_cyl = get_option("cylinder", false);

	// Size and origin of the simulation domain
	lx = 9.0;
	ly = 0.6;
	lz = 1.0;

	// Data for problem setup
	slope_length = 8.5;
	h_length = 0.5;
	height = .63;
	beta = atan(height/slope_length);

	SETUP_FRAMEWORK(
		viscosity<SPSVISC>,
		boundary<LJ_BOUNDARY>,
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

	// SPH parameters
	set_deltap(1.0/64.0);
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

	float r0 = m_deltap;

	auto water = add_fluid( 1000.0f);
	set_equation_of_state(water, 7.0f, NAN);
	set_kinematic_visc(water, 1.0e-6);
	set_artificial_visc(0.2f);

	//Wave paddle definition:  location, start & stop times, stroke and frequency (2 \pi/period)
	paddle_length = .7f;
	paddle_width = m_size.y - 2*r0;
	paddle_tstart = 0.5f;
	paddle_origin = make_double3(0.25f, r0, 0.0f);
	paddle_tend = 30.0f;//seconds
	// The stroke value is given at free surface level H
	float stroke = 0.2;
	float period = 0.8;
	// m_mbamplitude is the maximal angular value for paddle angle
	// Paddle angle is in [-m_mbamplitude, m_mbamplitude]
	paddle_amplitude = atan(stroke/(2.0*(H - paddle_origin.z)));
	cout << "\npaddle_amplitude (radians): " << paddle_amplitude << "\n";
	paddle_omega = 2.0*M_PI/period;

	// set max fall as at-rest height + (half) wave height, see e.g. Ch. 6 in
	// Dean & Dalrymple, Water Waves Mechanics for Engineers and Scientists
	float wave_height = H*stroke/4;
	setMaxFall(H+wave_height);

	// set maximum speed from the stroke speed, times a safety factor
	float stroke_speed_safety_factor = 2.0f;
	float stroke_speed = 2.0f*stroke/period;
	setMaxParticleSpeed(stroke_speed*stroke_speed_safety_factor);

	// Drawing and saving times

	add_writer(VTKWRITER, .1);  //second argument is saving time in seconds

	// Building the geometry
	const float br = (simparams()->boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);
	setPositioning(PP_CORNER);


	GeometryID paddle = addRect(GT_MOVING_BODY, FT_SOLID,
		Point(paddle_origin), paddle_length, paddle_width);
	rotate(paddle, 0, M_PI/2+paddle_amplitude, 0);
	disableCollisions(paddle);

	if (!use_bottom_plane) {
		GeometryID bottom = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
				Point(h_length, 0, 0), lx, ly);
		rotate(bottom, 0, beta, 0);
		disableCollisions(bottom);
	}

	GeometryID fluid = addBox(GT_FLUID, FT_SOLID, m_origin, lx, ly, H);

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

	{
		const double w = m_size.y;
		const double l = h_length + slope_length;

		addPlane(0, 0, 1, 0);  //bottom, where the first three numbers are the normal, and the last is d.
		addPlane(0, 1, 0, 0);  //wall
		addPlane(0, -1, 0, w); //far wall
		addPlane(1.0, 0, 0, 0);   //end
		addPlane(-1.0, 0, 0, l);  //one end

		// sloping bottom starting at x=h_length
		// this is only used to unfill if !use_bottom_plane
		addPlane(-sin(beta), 0, cos(beta), h_length*sin(beta),
			use_bottom_plane ? FT_NOFILL : FT_UNFILL);

		// this plane corresponds to the initial paddle position, and is only used to cut out
		// the fluid behind the paddle
		const double pcx = cos(paddle_amplitude);
		const double pcz = sin(paddle_amplitude);
		const double pcd = paddle_origin.x*pcx + paddle_origin.z*pcz;
		addPlane(pcx, 0, pcz, -pcd, FT_UNFILL);
	}
}


void
WaveTank::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{

    dx = make_double3(0.0);
    kdata.lvel = make_double3(0.0f, 0.0f, 0.0f);
    kdata.crot = make_double3(0.25f, m_deltap, 0.0f);
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

#undef MK_par
