/*  Copyright (c) 2019 INGV, EDF, UniCT, JHU, NU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA
    Northwestern University, Evanston (IL), USA

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

#include "DamBreak3DFEA.h"
#include "cudasimframework.cu"

DamBreak3DFEA::DamBreak3DFEA(GlobalData *_gdata) : Problem(_gdata)
{
	// *** user parameters from command line
	const bool WET = get_option("wet", false);
	const bool USE_PLANES = get_option("use_planes", false);
	const bool USE_CCSPH = get_option("use_ccsph", false);
	const uint NUM_OBSTACLES = get_option("num_obstacles", 0);
	const bool ROTATE_OBSTACLE = get_option("rotate_obstacle", true);
	const uint ppH = get_option("ppH", 64);
	const uint NUM_TESTPOINTS = get_option("num_testpoints", 3);
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", COLAGROSSI);

	// *** Geometrical parameters, starting from the size of the domain
	const double H = 0.4;
	const double dimX = 1.6;
	const double dimY = 0.3;
	const double dimZ = 0.7;
	const double obstacle_side = 0.12;
	const double obstacle_xpos = 0.9;
	const double water_length = 0.4;
	const double water_height = H;
	const double water_bed_height = 0.1;
	const double beam_h = 0.6;

	// ** framework setup
	// viscosities: KINEMATICVISC*, DYNAMICVISC*
	// turbulence models: ARTVISC*, SPSVISC, KEPSVISC
	// boundary types: LJ_BOUNDARY*, MK_BOUNDARY, SA_BOUNDARY, DYN_BOUNDARY*
	// * = tested in this problem
	SETUP_FRAMEWORK(
		rheology<NEWTONIAN>,
		turbulence_model<ARTIFICIAL>,
		boundary<DUMMY_BOUNDARY>,
		add_flags<ENABLE_FEA>
	).select_options(
		RHODIFF,
		USE_PLANES, add_flags<ENABLE_PLANES>(),
		USE_CCSPH, add_flags<ENABLE_CCSPH>()
	);

	// will dump testpoints separately
	addPostProcess(TESTPOINTS);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		(simparams()->densitydiffusiontype != DENSITY_DIFFUSION_NONE) ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// *** Initialization of minimal physical parameters
	set_deltap(H/ppH);
	simparams()->tend = 6;
	set_gravity(-9.81);
	auto water = add_fluid(1000.0);

	set_equation_of_state(water,  7.0f, 20.0f);
	set_kinematic_visc(water, 1.0e-6f);

	simparams()->tend=1.0f;
	simparams()->densityDiffCoeff = 0.1f;
	set_artificial_visc(0.05f);
	/*
	set_sps_parameters(0.12, 0.0066); // default values
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;*/
	simparams()->repack_a = 0.1f;
	simparams()->repack_alpha = 0.01f;
	simparams()->repack_maxiter = 10;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.01f);
	//addPostProcess(VORTICITY);

	// set filling method to BORDER_TANGENT, so that geometries automatically get filled
	// half a m_deltap inside/outside
	setFillingMethod(Object::BORDER_TANGENT);
	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);
	const Point corner = Point(0, 0, 0);

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
		makeUniverseBox(corner - half_dp_vec, corner + dim_vec + half_dp_vec);
	} else {
		addBox(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, corner, dimX, dimY, dimZ);
	}

	// Add the main water part
	addBox(GT_FLUID, FT_SOLID, corner, water_length, dimY, water_height);

	// Add the water bed if wet.
	if (WET) {
		addBox(GT_FLUID, FT_SOLID, corner + Vector(water_length, 0, 0),
			dimX - water_length, dimY, water_bed_height);
	}

	// addTetFile(GT_DEFORMABLE_BODY, FT_INNER_BORDER, Point(0,0,0), "dambreak.1.node", "dambreak.1.ele", 0.021);
	// set positioning policy to PP_BOTTOM_CENTER: given point will be the center of the base

	//	set_fea_ground(0, 0, 1, 0.05); // a, b, c and d parameters of a plane equation. Grounding nodes in the negative side of the plane

	// Define beams
	setPositioning(PP_BOTTOM_CENTER);

	GeometryID beam0 = addCylinder(GT_DEFORMABLE_BODY, FT_INNER_BORDER, Point(0.5, dimY/2.0, 0), 0.04, 0.04 - 0.005, beam_h, 2);

	setYoungModulus(beam0, 30e5);
	setPoissonRatio(beam0, 0.001);
	setDensity(beam0, 1000);
	setAlphaDamping(beam0, 0.5);

	setEraseOperation(beam0, ET_ERASE_FLUID);

	setPositioning(PP_CENTER);
	// node writer
	const double box_side = 0.1;
	GeometryID writer_box = addBox(GT_FEA_WRITE, FT_NOFILL, Point(0.5, dimY/2.0, beam_h), box_side, box_side, box_side);


	const double dynamometer_side = 0.1;
	GeometryID dynamometer = addBox(GT_FEA_RIGID_JOINT, FT_NOFILL, Point(0.5, dimY/2.0, 0), dynamometer_side, dynamometer_side, dynamometer_side);
	setEraseOperation(dynamometer, ET_ERASE_NOTHING);
	setUnfillRadius(dynamometer, 0.5*m_deltap);
	setDynamometer(dynamometer, true);

	simparams()->fea_write_every = 0.01f;

	setPositioning(PP_BOTTOM_CENTER);
	// add one or more obstacles
	const double Y_DISTANCE = dimY / (NUM_OBSTACLES + 1);
	// rotation angle
	const double Z_ANGLE = M_PI / 4;

	// activate the solid obstacle
	for (uint i = 0; i < NUM_OBSTACLES; i++) {
		// Obstacle is of type GT_MOVING_BODY, although the callback is not even implemented, to
		// make the forces feedback available
		GeometryID obstacle = addBox(GT_MOVING_BODY, FT_INNER_BORDER,
			Point(obstacle_xpos, Y_DISTANCE * (i+1) + (ROTATE_OBSTACLE ? obstacle_side/2 : 0), 0),
				obstacle_side, obstacle_side, dimZ);
		setEraseOperation(obstacle, ET_ERASE_NOTHING);
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
		addSphere(GT_FLOATING_BODY, FT_INNER_BORDER, Point(water_length, dimY/2, water_height), obstacle_side);
	// half water density to make it float
	setMassByDensity(floating_obj, physparams()->rho0[0] / 2);
	setParticleMassByDensity(floating_obj, physparams()->rho0[0] / 2);
	// disable collisions: will only interact with fluid
	// disableCollisions(floating_obj);
	*/

	// add testpoints
	const float TESTPOINT_DISTANCE = dimZ / (NUM_TESTPOINTS + 1);
	for (uint t = 0; t < NUM_TESTPOINTS; t++)
		addTestPoint(Point(0.25*dimX, dimY/2.0, (t+1) * TESTPOINT_DISTANCE/2.0));

	for (uint t = 0; t < NUM_TESTPOINTS; t++)
		addTestPoint(Point(0.4*dimX, dimY/2.0, (t+1) * TESTPOINT_DISTANCE/2.0));

	for (uint t = 0; t < NUM_TESTPOINTS; t++)
		addTestPoint(Point(0.75*dimX, dimY/2.0, (t+1) * TESTPOINT_DISTANCE/2.0));

	for (uint t = 0; t < NUM_TESTPOINTS; t++)
		addTestPoint(Point(0.9*dimX, dimY/2.0, (t+1) * TESTPOINT_DISTANCE/2.0));
}

// since the fluid topology is roughly symmetric along Y through the whole simulation, prefer Y split
void DamBreak3DFEA::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}

void DamBreak3DFEA::initializeParticles(BufferList &buffer, const uint numParticle)
{
	float4 *pos = buffer.getData<BUFFER_POS>();
	const float4 *vel = buffer.getData<BUFFER_VEL>();
	const ushort4 *info= buffer.getData<BUFFER_INFO>();

	// TODO FIXME the particle mass should be assigned from the mesh. We should 
	// understand why GetMass on the fea mesh gives 0
/*
	for (uint i = 0; i < numParticle; i++) {
		if (DEFORMABLE(info[i]))
			pos[i].w = physical_density(vel[i].w, 0)*m_deltap*m_deltap*m_deltap;
	}
	*/
}

bool DamBreak3DFEA::need_write(double t) const
{
	return false;
}
