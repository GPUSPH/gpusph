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

#include "XDamBreak3D.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

XDamBreak3D::XDamBreak3D(GlobalData *_gdata) : XProblem(_gdata)
{
	// *** user parameters from command line
	const bool WET = get_option("wet", false);
	const bool USE_PLANES = get_option("use_planes", true);
	const uint NUM_OBSTACLES = get_option("num_obstacles", 1);
	const bool ROTATE_OBSTACLE = get_option("rotate_obstacle", true);
	const uint NUM_TESTPOINTS = get_option("num_testpoints", 3);
	// density diffusion terms: 0 none, 1 Molteni & Colagrossi, 2 Ferrari
	const int RHODIFF = get_option("density-diffusion", 1);

	// ** framework setup
	// viscosities: ARTVISC*, KINEMATICVISC*, DYNAMICVISC*, SPSVISC, KEPSVISC
	// boundary types: LJ_BOUNDARY*, MK_BOUNDARY, SA_BOUNDARY, DYN_BOUNDARY*
	// * = tested in thsi problem
	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		boundary<LJ_BOUNDARY>
	).select_options(
		RHODIFF, FlagSwitch<ENABLE_NONE, ENABLE_DENSITY_DIFFUSION, ENABLE_FERRARI>(),
		USE_PLANES, add_flags<ENABLE_PLANES>()
	);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled or Ferrari correction is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		(simparams()->simflags & (ENABLE_DENSITY_DIFFUSION | ENABLE_FERRARI)) ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// Explicitly set number of layers. Also, prevent having undefined number of layers before the constructor ends.
	setDynamicBoundariesLayers(3);

	// *** Initialization of minimal physical parameters
	set_deltap(0.02f);
	physparams()->r0 = m_deltap;
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);
	const float g = length(physparams()->gravity);
	const double H = 0.4;
	physparams()->dcoeff = 5.0f * g * H;
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 20.0f);
	//set_kinematic_visc(0, 1.0e-2f);
	set_dynamic_visc(0, 1.0e-4f);

	// *** Initialization of minimal simulation parameters
	simparams()->maxneibsnum = 128 + 64;

	// *** Other parameters and settings
	add_writer(VTKWRITER, 0.005f);
	m_name = "XDamBreak3D";

	// *** Geometrical parameters, starting from the size of the domain
	const double dimX = 1.6;
	const double dimY = 0.67;
	const double dimZ = 0.6;
	const double obstacle_side = 0.12;
	const double obstacle_xpos = 0.9;
	const double water_length = 0.4;
	const double water_height = H;
	const double water_bed_height = 0.1;

	// If we used only makeUniverseBox(), origin and size would be computed automatically
	m_origin = make_double3(0, 0, 0);
	m_size = make_double3(dimX, dimY, dimZ);

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);

	// main container
	if (USE_PLANES) {
		// limit domain with 6 planes
		makeUniverseBox(m_origin, m_origin + m_size);
	} else {
		GeometryID box =
			addBox(GT_FIXED_BOUNDARY, FT_BORDER, m_origin, dimX, dimY, dimZ);
		// we simulate inside the box, so do not erase anything
		setEraseOperation(box, ET_ERASE_NOTHING);
	}

	// Planes unfill automatically but the box won't, to void deleting all the water. Thus,
	// we define the water at already the right distance from the walls.
	double BOUNDARY_DISTANCE = m_deltap;
	if (simparams()->boundarytype == DYN_BOUNDARY && !USE_PLANES)
			BOUNDARY_DISTANCE *= getDynamicBoundariesLayers();

	// Add the main water part
	addBox(GT_FLUID, FT_SOLID, Point(BOUNDARY_DISTANCE, BOUNDARY_DISTANCE, BOUNDARY_DISTANCE),
		water_length - BOUNDARY_DISTANCE, dimY - 2 * BOUNDARY_DISTANCE, water_height - BOUNDARY_DISTANCE);
	// Add the water bed if wet. After we'll implement the unfill with custom dx, it will be possible to declare
	// the water bed overlapping with the main part.
	if (WET) {
		addBox(GT_FLUID, FT_SOLID,
			Point(water_length + m_deltap, BOUNDARY_DISTANCE, BOUNDARY_DISTANCE),
			dimX - water_length - BOUNDARY_DISTANCE - m_deltap,
			dimY - 2 * BOUNDARY_DISTANCE,
			water_bed_height - BOUNDARY_DISTANCE);
	}

	// set positioning policy to PP_BOTTOM_CENTER: given point will be the center of the base
	setPositioning(PP_BOTTOM_CENTER);

	// add one or more obstacles
	const double Y_DISTANCE = dimY / (NUM_OBSTACLES + 1);
	// rotation angle
	const double Z_ANGLE = M_PI / 4;

	for (uint i = 0; i < NUM_OBSTACLES; i++) {
		// Obstacle is of type GT_MOVING_BODY, although the callback is not even implemented, to
		// make the forces feedback available
		GeometryID obstacle = addBox(GT_MOVING_BODY, FT_BORDER,
			Point(obstacle_xpos, Y_DISTANCE * (i+1) + (ROTATE_OBSTACLE ? obstacle_side/2 : 0), 0),
				obstacle_side, obstacle_side, dimZ );
		if (ROTATE_OBSTACLE) {
			rotate(obstacle, 0, 0, Z_ANGLE);
			// until we'll fix it, the rotation centers are always the corners
			// shift(obstacle, 0, obstacle_side/2, 0);
		}
		// enable force feedback to measure forces
		enableFeedback(obstacle);
	}

	// Optionally, add a floating objects
	/*
	// set positioning policy to PP_CENTER: given point will be the geometrical center of the object
	setPositioning(PP_CENTER);
	GeometryID floating_obj =
		addSphere(GT_FLOATING_BODY, FT_BORDER, Point(water_length, dimY/2, water_height), obstacle_side);
	// half water density to make it float
	setMassByDensity(floating_obj, m_physparams->rho0[0] / 2);
	setParticleMassByDensity(floating_obj, m_physparams->rho0[0] / 2);
	// disable collisions: will only interact with fluid
	// disableCollisions(floating_obj);
	*/

	// add testpoints
	const float TESTPOINT_DISTANCE = dimZ / (NUM_TESTPOINTS + 1);
	for (uint t = 0; t < NUM_TESTPOINTS; t++)
		addTestPoint(Point(dimX, dimY/2.0, t * TESTPOINT_DISTANCE));
}

// since the fluid topology is roughly symmetric along Y through the whole simulation, prefer Y split
void XDamBreak3D::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}
