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

#include "WaveTank2D.h"
#include "particledefine.h"
#include "GlobalData.h"
#include "cudasimframework.cu"


#define MK_par 2

WaveTank2D::WaveTank2D(GlobalData *_gdata) : Problem(_gdata)
{
	// use planes in general
	const bool use_planes = get_option("use_planes", false);
	// use a plane for the bottom
	const bool use_bottom_plane = get_option("bottom-plane", use_planes);
	// Add objects to the tank
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", COLAGROSSI);

	if (use_bottom_plane && !use_planes)
		throw std::invalid_argument("cannot use bottom plane if not using planes");

	// Use conservative corrected SPH
	const bool USE_CCSPH = get_option("use_ccsph", true);


	// Data for problem setup
	slope_length = 8.5;
	h_length = 2.5;
	H = 0.45; // Still Water depth

	// Size of the simulation domain
	lx = slope_length + h_length;
	ly = 1.0;
	lz = 1.0;

	SETUP_FRAMEWORK(
		space_dimensions<R2>,
		viscosity<ARTVISC>,
		boundary<DUMMY_BOUNDARY>
	).select_options(
		RHODIFF,
		use_planes, add_flags<ENABLE_PLANES>(),
		USE_CCSPH, add_flags<ENABLE_CCSPH>()
	);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		(simparams()->densitydiffusiontype != DENSITY_DIFFUSION_NONE) ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);


	//m_size = make_double3(lx, ly, lz);

	if (get_option("testpoints", false)) {
		addPostProcess(TESTPOINTS);
	}

	// SPH parameters
	set_deltap(1.0/64.0);
	simparams()->dtadaptfactor = 0.2;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 10.0f; // Simulation time in seconds

	set_ccsph_min_det(0.4);

	//WaveGage
	double influence_radius = simparams()->influenceRadius;
	if (get_option("gages", true)) {
		add_gage(make_double2(1, 0), influence_radius);
		add_gage(make_double2(2, 0), influence_radius);
	}

	// Physical parameters
	set_gravity(-9.81f);

	float r0 = m_deltap;

	auto water = add_fluid(1000.0f);
	set_equation_of_state(water, 7.0f, NAN);

	//Wave paddle definition:  location, start & stop times, stroke and frequency (2 \pi/period)
	paddle_length = .7f;
	paddle_width = m_size.y - 2*r0;
	paddle_tstart = 0.5f;
	paddle_origin = make_double3(0.0, 0.0, 0.0);
	paddle_tend = simparams()->tend;//seconds

	// The stroke value is given at free surface level H
	float stroke = 0.1;
	float period = 0.8;

	// m_mbamplitude is the maximal angular value for paddle angle
	// Paddle angle is in [-m_mbamplitude, m_mbamplitude]
	paddle_amplitude = atan(stroke/(2.0*(H - paddle_origin.z)));
	cout << "\npaddle_amplitude (radians): " << paddle_amplitude << "\n";
	paddle_omega = 2.0*M_PI/period;

	// set max fall as at-rest H + (half) wave height, see e.g. Ch. 6 in
	// Dean & Dalrymple, Water Waves Mechanics for Engineers and Scientists
	float wave_height = H*stroke/4;
	setMaxFall(H+wave_height);

	// set maximum speed from the stroke speed, times a safety factor
	float stroke_speed_safety_factor = 2.0f;
	float stroke_speed = 2.0f*stroke/period;
	setMaxParticleSpeed(stroke_speed*stroke_speed_safety_factor);

	set_artificial_visc(0.01f);
	//set_artificial_visc(1e-6*8.0/(physparams()->sscoeff[0]*simparams()->slength));

	// Drawing and saving times
	add_writer(VTKWRITER, .1);  //second argument is saving time in seconds
	add_writer(COMMONWRITER, .01);  //second argument is saving time in seconds

	/*
	 * Building the geometry
	 */

	// number of layers of thickness for the boxes that implement the walls and slopes;
	// this depends on the boundary model: for anything up to SA a single layer is used,
	// for higher values (e.g. dynamic boundaries) we need to fill the kernel influence
	// radius
	const int num_layers = (simparams()->boundarytype > SA_BOUNDARY) ?
		simparams()->get_influence_layers() : 1;
	const double box_thickness = (num_layers - 1)*m_deltap;
	const double3 slope_origin = make_double3(paddle_origin.x + h_length, -box_thickness, 0);
	setDynamicBoundariesLayers(num_layers);

	setPositioning(PP_CORNER);

	// place the fluid box. Fill the whole domain in the X and Y direction,
	// up to the at-rest height. This will be “carved out” by the wall and floor
	// geometries placed afterwards
	addRect(GT_FLUID, FT_SOLID, Point(-stroke, 0.0, 0.0), lx, H);

	// place the paddle
	GeometryID paddle = addRect(GT_MOVING_BODY, FT_SOLID,
		Point(paddle_origin - make_double3(box_thickness, 0, 0)),
		box_thickness, paddle_length);
	rotate(paddle, 0, 0, -paddle_amplitude);

	// sloped bottom, if not a plane
	beta = atan(H/slope_length);
	double rot_correction = sin(beta)*box_thickness;

	const double extra_beach_length = 2.0;

	if (!use_bottom_plane) {
		GeometryID bottom = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
				slope_origin + make_double3(rot_correction, (1-cos(beta))*box_thickness, 0),
				lx + extra_beach_length - h_length - rot_correction, box_thickness);
		rotate(bottom, 0, 0, -beta);
	}

	// place the walls: as planes, if required; otherwise, as boxes
	if (use_planes) {
		const double l = h_length + slope_length;

		addPlane(0, 1, 0, 0);  //bottom, where the first three numbers are the normal, and the last is d.
		addPlane(-1.0, 0, 0, l);  // beach end
	} else {
		// flat bottom rectangle (before the slope begins)
		GeometryID bottom = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
			Point(paddle_origin - make_double3(box_thickness, box_thickness, 0)),
			h_length + box_thickness + rot_correction, box_thickness);
		setUnfillRadius(bottom, 0.5*m_deltap);
	}

	// these planes are used at least for cutting, so they are always defined
	{
		// sloping bottom as a plane. if use_bottom_plane, then it will be
		// an actual geometry; if !use_bottom_plane, it will only be used
		// to unfill the fluid (since the sloping box would not be sufficient
		// to remove all of the fluid below)
		GeometryID plane = addPlane(-sin(beta), cos(beta), 0, slope_origin.x*sin(beta),
			use_bottom_plane ? FT_NOFILL : FT_UNFILL);
		setEraseOperation(plane, ET_ERASE_FLUID);


		// this plane corresponds to the initial paddle position, and is only used to cut out
		// the fluid behind the paddle. it will not be an actual geometry
		const double pcx = cos(paddle_amplitude);
		const double pcz = sin(paddle_amplitude);
		const double pcd = paddle_origin.x*pcx + paddle_origin.z*pcz;
		plane = addPlane(pcx, pcz, 0, -pcd, FT_UNFILL);

		setEraseOperation(plane, ET_ERASE_FLUID);
	}
}


void
WaveTank2D::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
			const float3& force, const float3& torque, const KinematicData& initial_kdata,
			KinematicData& kdata, double3& dx, EulerParameters& dr)
{

    dx = make_double3(0.0);
    kdata.lvel = make_double3(0.0f, 0.0f, 0.0f);
    kdata.crot = make_double3(0.0f, 0.0, 0.0f);
    if (t1> paddle_tstart && t1 < paddle_tend){
	    kdata.avel = make_double3(0.0, 0.0, -paddle_amplitude*paddle_omega*sin(paddle_omega*(t1-paddle_tstart)));
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
