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

// TODO FIXME domain box walls don't contain the floating objects,
// despite having collisions enabled (neither w/ planes nor with box)

#include "chrono_select.opt"
#if USE_CHRONO == 1
#include "chrono/physics/ChSystem.h"
#include "chrono/physics/ChLinkDistance.h"
#include "chrono/physics/ChLinkLock.h"
#endif

#include "Objects.h"
#include "GlobalData.h"

Objects::Objects(GlobalData *_gdata) : Problem(_gdata)
{
	// *** user parameters from command line
	const bool USE_PLANES = get_option("use_planes", true);
	const uint NUM_OBSTACLES = get_option("num_obstacles", 1);
	const bool ROTATE_OBSTACLE = get_option("rotate_obstacle", true);
	// density diffusion terms, see DensityDiffusionType
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", FERRARI);

	setup_framework(RHODIFF, USE_PLANES);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled or Ferrari correction is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		RHODIFF != 0 ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// *** Initialization of minimal physical parameters
	set_deltap(0.02f);
	set_gravity(-9.81);

	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, NAN);
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

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);
	// set filling method to BORDER_TANGENT to simplify geometry definition
	setFillingMethod(Object::BORDER_TANGENT);

	const Point corner(0, 0, 0);

	// main container
	if (USE_PLANES) {
		// limit domain with 6 planes. Due to our filling method, using:
		//   makeUniverseBox(corner, corner + Vector(dimX, dimY, dimZ));
		// would place the planes half a dp from the fluid,
		// which would be correct if we used ghost particles,
		// but with the LJ planes we have currently, we should have a full dp.
		// As an alternative, we could set the LJ r0 to half dp,
		// but this would cause issues with the obstacles,
		// which are filled with particles instead.
		// Instead, we shift the corners of the universe box out by half a dp.
		// TODO remeber to fix this when we implement ghost particles!
		const double half_dp = m_deltap/2;
		const Vector half_dp_vec = Vector(half_dp, half_dp, half_dp);
		const Vector dim_vec = Vector(dimX, dimY, dimZ);
		auto planes = makeUniverseBox(corner - half_dp_vec, corner + dim_vec + half_dp_vec);
	} else {
		GeometryID domain_box =
			addBox(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, corner, dimX, dimY, dimZ);
	}

	// We define the water at already the right distance from the walls.
	// Add the main water part
	addBox(GT_FLUID, FT_SOLID, corner, water_length, dimY, water_height);

	// set positioning policy to PP_BOTTOM_CENTER: given point will be the center of the base
	setPositioning(PP_BOTTOM_CENTER);

	// add one or more obstacles
	const double Y_DISTANCE = dimY / (NUM_OBSTACLES + 1);
	// rotation angle
	const double Z_ANGLE = M_PI / 4;

	for (uint i = 0; i < NUM_OBSTACLES; i++) {
		// Obstacle is of type GT_MOVING_BODY, although the callback is not even implemented, to
		// make the forces feedback available
		GeometryID obstacle = addBox(GT_FIXED_BOUNDARY, FT_INNER_BORDER,
			corner + Vector(obstacle_xpos, dimY/2, 0), obstacle_side, obstacle_side, dimZ/2.0);
		if (ROTATE_OBSTACLE) {
			rotate(obstacle, 0, 0, Z_ANGLE);
			// until we'll fix it, the rotation centers are always the corners
			shift(obstacle, 0, obstacle_side/2, 0);
		}
		// enable force feedback to measure forces
		//enableFeedback(obstacle);
	}

	// Add a couple of floating objects
	// set positioning policy to PP_CENTER: given point will be the geometrical center of the object
	setPositioning(PP_CENTER);
	floating_obj_1 =
		addCube(GT_FLOATING_BODY, FT_INNER_BORDER, Point(water_length, dimY/5.0*1.5, water_height), objects_side);
	floating_obj_2 =
		addSphere(GT_FLOATING_BODY, FT_INNER_BORDER, Point(water_length, dimY/5.0*2.5, water_height), objects_side / 2.0);
	// quarter water density to make it float
	const double density = physparams()->rho0[0] / 4;
	setMassByDensity(floating_obj_1, density);
	//setParticleMassByDensity(floating_obj_1, physparams()->rho0[0]);
	setMassByDensity(floating_obj_2, density);
	//setParticleMassByDensity(floating_obj_2, physparams()->rho0[0]);
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
	auto joint1 = chrono_types::make_shared< ::chrono::ChLinkDistance >();
	auto a1 = getGeometryBody(floating_obj_1);
	auto a2 = getGeometryBody(floating_obj_2);
	joint1->SetName("DistanceConstraint");
	// distance link params: body1, body1, bool (true if pos are relative), endpoint1, endpoint2, bool (true if distance is auto)
	joint1->Initialize(	a1, a2, false, a1->GetPos(), a2->GetPos(), true);
	// Add the link to the physical system
	m_chrono_system->AddLink( joint1 );
#else
	throw std::runtime_error("Chrono disabled, no object joints available");
#endif
}
