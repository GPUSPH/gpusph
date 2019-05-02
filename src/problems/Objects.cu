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

#include "chrono_select.opt"
#if USE_CHRONO == 1
#include "chrono/physics/ChLinkDistance.h"
//#include "chrono/core/ChCoordsys.h"
#include "chrono/physics/ChLinkLock.h"
#endif

#include "Objects.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

Objects::Objects(GlobalData *_gdata) : XProblem(_gdata)
{
	// *** user parameters from command line
	const bool WET = get_option("wet", false);
	const bool USE_PLANES = get_option("use_planes", true);
	const uint NUM_OBSTACLES = get_option("num_obstacles", 1);
	const bool ROTATE_OBSTACLE = get_option("rotate_obstacle", true);
	const uint NUM_TESTPOINTS = get_option("num_testpoints", 3);
	// density diffusion terms, see DensityDiffusionType
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", FERRARI);

	// ** framework setup TODO newvisc update
	// viscosities: KINEMATICVISC*, DYNAMICVISC*
	// turbulence models: ARTVISC*, SPSVISC, KEPSVISC
	// boundary types: LJ_BOUNDARY*, MK_BOUNDARY, SA_BOUNDARY, DYN_BOUNDARY*
	// * = tested in thsi problem
	SETUP_FRAMEWORK(
		viscosity<DYNAMICVISC>,
		boundary<LJ_BOUNDARY>
	).select_options(
		RHODIFF,
		USE_PLANES, add_flags<ENABLE_PLANES>()
	);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled or Ferrari correction is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		RHODIFF != 0 ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// Explicitly set number of layers. Also, prevent having undefined number of layers before the constructor ends.
	setDynamicBoundariesLayers(3);

	// *** Initialization of minimal physical parameters
	set_deltap(0.02f);
	set_gravity(-9.81);
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 20.0f);
	//set_kinematic_visc(0, 1.0e-2f);
	set_dynamic_visc(0, 1.0e-4f);

	// default tend 2s
	simparams()->tend=2.0f;

	// *** Initialization of minimal simulation parameters
	resize_neiblist(256, 64);

	// *** Other parameters and settings
	add_writer(VTKWRITER, 0.01f);
	m_name = "Objects";

	// *** Geometrical parameters, starting from the size of the domain
	const double dimX = 1.6;
	const double dimY = 0.8;
	const double dimZ = 0.8;
	const double obstacle_side = 0.1;
	const double objects_side = 0.08;
	const double obstacle_xpos = 1.0;
	const double water_length = 0.5;
	const double water_height = 0.5;

	setMaxFall(water_height);

	// If we used only makeUniverseBox(), origin and size would be computed automatically
	m_origin = make_double3(0, 0, 0);
	m_size = make_double3(dimX, dimY, dimZ);

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);

	// main container
	GeometryID box =
		addBox(GT_FIXED_BOUNDARY, FT_BORDER, m_origin, dimX, dimY, dimZ ); //m_deltap);
	// we simulate inside the box, so do not erase anything
	setEraseOperation(box, ET_ERASE_NOTHING);
	disableCollisions(box);

	// floor for collisions only. Note FT_NOFILL
	GeometryID floor =
		addBox(GT_FIXED_BOUNDARY, FT_NOFILL, make_double3(m_origin.x, m_origin.y, m_origin.z - m_deltap), dimX, dimY, m_deltap);
	// do not erase anything
	setEraseOperation(floor, ET_ERASE_NOTHING);
	// need to set a density to have an inertia and collide
	setMassByDensity(floor, physparams()->rho0[0]);
	//disableCollisions(floor);

	// We define the water at already the right distance from the walls.
	double BOUNDARY_DISTANCE = m_deltap;
	if (simparams()->boundarytype == DYN_BOUNDARY)
			BOUNDARY_DISTANCE *= getDynamicBoundariesLayers();

	// Add the main water part
	addBox(GT_FLUID, FT_SOLID, Point(BOUNDARY_DISTANCE, BOUNDARY_DISTANCE, BOUNDARY_DISTANCE),
		water_length - BOUNDARY_DISTANCE, dimY - 2 * BOUNDARY_DISTANCE, water_height - BOUNDARY_DISTANCE);

	// set positioning policy to PP_BOTTOM_CENTER: given point will be the center of the base
	setPositioning(PP_BOTTOM_CENTER);

	// add one or more obstacles
	const double Y_DISTANCE = dimY / (NUM_OBSTACLES + 1);
	// rotation angle
	const double Z_ANGLE = M_PI / 4;

	for (uint i = 0; i < NUM_OBSTACLES; i++) {
		// Obstacle is of type GT_MOVING_BODY, although the callback is not even implemented, to
		// make the forces feedback available
		GeometryID obstacle = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
			Point(obstacle_xpos, Y_DISTANCE * (i+1) + (ROTATE_OBSTACLE ? obstacle_side/2 : 0), BOUNDARY_DISTANCE),
				obstacle_side, obstacle_side, dimZ/2.0 );
		if (ROTATE_OBSTACLE) {
			rotate(obstacle, 0, 0, Z_ANGLE);
			// until we'll fix it, the rotation centers are always the corners
			// shift(obstacle, 0, obstacle_side/2, 0);
		}
		// enable force feedback to measure forces
		//enableFeedback(obstacle);
		// debug
		//disableCollisions(obstacle);
	}

	// Add a floating objects
	// set positioning policy to PP_CENTER: given point will be the geometrical center of the object
	setPositioning(PP_CENTER);
	/*
	GeometryID floating_obj =
		addSphere(GT_FLOATING_BODY, FT_BORDER, Point(water_length, dimY/2, water_height), obstacle_side);
	*/
	floating_obj_1 =
		addCube(GT_FLOATING_BODY, FT_BORDER, Point(water_length, dimY/5.0*1.5, water_height), objects_side);
	floating_obj_2 =
		addSphere(GT_FLOATING_BODY, FT_BORDER, Point(water_length, dimY/5.0*2.5, water_height), objects_side / 2.0);
	// half water density to make it float
	const double density = physparams()->rho0[0] / 4;
	setMassByDensity(floating_obj_1, density);
	setParticleMassByDensity(floating_obj_1, physparams()->rho0[0]);
	setMassByDensity(floating_obj_2, density);
	setParticleMassByDensity(floating_obj_2, physparams()->rho0[0]);
	// play with rotations
	const double angle = M_PI / 4;
	rotate(floating_obj_1, 0, angle, angle);
	// disable collisions: will only interact with fluid
	//disableCollisions(floating_obj_1);
	//disableCollisions(floating_obj_2);
}

// since the fluid topology is roughly symmetric along Y through the whole simulation, prefer Y split
void Objects::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}

void Objects::initializeObjectJoints() {
#if USE_CHRONO == 1
	// Make a new ChLinkDistance
	auto joint1 = std::make_shared< ::chrono::ChLinkDistance >();
	auto a1 = getGeometryInfo(floating_obj_1)->ptr->GetBody();
	auto a2 = getGeometryInfo(floating_obj_2)->ptr->GetBody();
	joint1->SetName("DistanceConstraint");
	// distance link params: body1, body1, bool (true if pos are relative), endpoint1, endpoint2, bool (true if distance is auto)
	joint1->Initialize(	a1, a2, false, a1->GetPos(), a2->GetPos(), true);
	// Add the link to the physical system
	m_bodies_physical_system->AddLink( joint1 );
#else
	throw runtime_error("Chrono disabled, no object joints available");
#endif
}
