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

#include "OffshorePile.h"
#include "Point.h"
#include "Cylinder.h"
#include "Vector.h"
#include "Cube.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

OffshorePile::OffshorePile(GlobalData *_gdata) : XProblem(_gdata)
{

	SETUP_FRAMEWORK(
			kernel<WENDLAND>,
			viscosity<KINEMATICVISC>,
			boundary<DYN_BOUNDARY>,
			periodicity<PERIODIC_Y>
			);

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
	// Explicitly set number of layers. Also, prevent having undefined number of layers before the constructor ends.
	setDynamicBoundariesLayers(3);

	set_deltap(0.05f);  // 0.05 is minimum to have 3 layers of particles in the cylinder
	x0 = -1.;
	periodic_offset_y = m_deltap/2.;
	m_size = make_double3(lx - x0 , ly + m_deltap, lz + 1.5*layers*m_deltap);
	m_origin = make_double3(x0, 0., -1.5*layers*m_deltap);

	// Shepard filter
	addFilter(SHEPARD_FILTER, 20);

	addPostProcess(SURFACE_DETECTION);

	// SPH parameters
	set_timestep(0.00013);
	simparams()->dtadaptfactor = 0.2;
	simparams()->tend = 120; //seconds

	// Physical parameters
	set_gravity(-9.81f);
	float g = get_gravity_magnitude();
	physparams()->dcoeff = 5.0f * g * H;

	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 40.f);
	set_kinematic_visc(0, 1.0e-6);
	setWaterLevel(H);

	resize_neiblist(256 + 64, 32); // 352
	simparams()->buildneibsfreq = 1;

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
	cyl_diam = 0.2;
	cyl_height = 2*H;
	cyl_xpos = h_length + 0.4/tan_beta;
	cyl_rho = 607.99;

	//WaveGage
	const float slength = simparams()->slength;
	add_gage(cyl_xpos, ly/2 + periodic_offset_y + 0.5, 0.0);
	add_gage(cyl_xpos, ly/2 + periodic_offset_y + 0.5, slength);
	add_gage(cyl_xpos, ly/2 + periodic_offset_y + 0.5, 0.5*slength);
	add_gage(cyl_xpos, ly/2 + periodic_offset_y + 0.5, 0.25*slength);
	add_gage(cyl_xpos, ly/2 + periodic_offset_y + 0.5, 2.0*slength);
	add_gage(1.0, ly/2 + periodic_offset_y, m_deltap);
	add_gage(h_length, ly/2 + periodic_offset_y, m_deltap);
	add_gage(h_length-h_length/4, ly/2 + periodic_offset_y, m_deltap);
	add_gage(h_length-h_length/2, ly/2 + periodic_offset_y, m_deltap);
	add_gage(h_length-h_length*3/4, ly/2 + periodic_offset_y, m_deltap);

	// Drawing and saving times
	add_writer(VTKWRITER, 1.);  //second argument is saving time in seconds
	add_writer(COMMONWRITER, 0.);

	// Name of problem used for directory creation
	m_name = "OffshorePile";

  // Create the geometry
	setPositioning(PP_CORNER);

  const int layersm1 = layers - 1;

  // Fluid
	GeometryID fluid1 = addBox(GT_FLUID, FT_SOLID,
        Point(m_deltap/2., periodic_offset_y, m_deltap/2.),
        h_length, ly, H - m_deltap);
	GeometryID fluid2 = addBox(GT_FLUID, FT_SOLID,
        make_double3(h_length + m_deltap, periodic_offset_y,  m_deltap/2.),
			  lx - h_length - m_deltap, ly, H - m_deltap/2);
  rotate(fluid2,EulerParameters(Vector(0, 1, 0), -beta));

  double hu = 1.2*(lx - h_length)*tan(beta);
	GeometryID unfill_top = addBox(GT_FLUID, FT_NOFILL,
        make_double3(h_length + m_deltap, periodic_offset_y, H + m_deltap/2.),
        lx - h_length, ly, H + hu);
	setEraseOperation(unfill_top,ET_ERASE_FLUID);

	// Rigid body: cylinder
	setPositioning(PP_BOTTOM_CENTER);
  GeometryID cyl = addCylinder(GT_MOVING_BODY, FT_BORDER,
        make_double3(cyl_xpos, ly/2., 0),
        (cyl_diam - m_deltap)/2., cyl_height);
	disableCollisions(cyl);
  enableFeedback(cyl);
	setEraseOperation(cyl,ET_ERASE_FLUID);
	setUnfillRadius(cyl,m_deltap*0.8);
	setEraseOperation(cyl,ET_ERASE_BOUNDARY);

	setPositioning(PP_CORNER);
	// Rigid body: piston
  GeometryID piston =
    addBox(GT_MOVING_BODY, FT_BORDER, piston_origin,
    layersm1*m_deltap, piston_width, piston_height);
	disableCollisions(piston);
	disableFeedback(piston);

  // Walls
	GeometryID bottom_flat = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
    make_double3(x0, periodic_offset_y, -(layersm1 + 0.5)*m_deltap),
    h_length - x0 + 5*m_deltap , ly, layersm1*m_deltap);
	disableCollisions(bottom_flat);

  GeometryID bottom_slope = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
    make_double3(h_length, periodic_offset_y, -(layersm1 + 0.5)*m_deltap),
		lx - h_length, ly, layersm1*m_deltap);
	disableCollisions(bottom_slope);
  setOrientation(bottom_slope,EulerParameters(Vector(0, 1, 0), -beta));
	setUnfillRadius(bottom_slope,m_deltap*0.9);
	setEraseOperation(bottom_slope,ET_ERASE_BOUNDARY);

  double zfw = (lx - h_length)*tan(beta) - layersm1*m_deltap;
	GeometryID far_wall = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
        Point(lx - layersm1*m_deltap, periodic_offset_y, zfw),
        layersm1*m_deltap, ly, H);
	disableCollisions(far_wall);
	setUnfillRadius(far_wall,m_deltap*0.9);
	setEraseOperation(far_wall,ET_ERASE_BOUNDARY);

}

// Piston's motion
  void
OffshorePile::moving_bodies_callback(const uint index, Object* object, const double t0, const double t1,
    const float3& force, const float3& torque, const KinematicData& initial_kdata,
    KinematicData& kdata, double3& dx, EulerParameters& dr)
{
	if (index == 1) { // we only want to impose the piston's (index=1) motion
		dx= make_double3(0.0);
		dr.Identity();
		kdata.avel = make_double3(0.0);
		if (t0 >= piston_tstart && t1 <= piston_tend) {
			const double arg0 = piston_omega*(t0 - piston_tstart);
			const double arg1 = piston_omega*(t1 - piston_tstart);
			kdata.lvel = make_double3(-piston_amplitude*piston_omega*sin(arg1), 0.0, 0.0);
			dx.x = piston_amplitude*(cos(arg1)-cos(arg0));
		} else {
			kdata.lvel = make_double3(0.0);
		}
	}
}
