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

#include "XCompleteSaExample.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

XCompleteSaExample::XCompleteSaExample(GlobalData *_gdata) : XProblem(_gdata)
{
	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		formulation<SPH_F1>,
		viscosity<DYNAMICVISC>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_NONE>,
		add_flags<ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM | ENABLE_MOVING_BODIES | ENABLE_FERRARI>
	);

	// *** Initialization of minimal physical parameters
	set_deltap(0.02f);
	physparams()->r0 = m_deltap;
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);

	// *** Initialization of minimal simulation parameters
	simparams()->maxneibsnum = 256 + 64 + 32; // 352
	// ferrari correction
	simparams()->ferrariLengthScale = 0.25f;

	// buildneibs at every iteration
	simparams()->buildneibsfreq = 1;

	// *** Other parameters and settings
	add_writer(VTKWRITER, 1e-2f);
	m_name = "XCompleteSaExample";

		m_origin = make_double3(-1, -1, -1);
	m_size = make_double3(3, 3, 3);

	// Set world size and origin like CompleteSaExample, instead of computing automatically.
	// Also, HDF5 file loading does not support bounding box detection yet
	const double MARGIN = 0.1;
	const double INLET_BOX_LENGTH = 0.25;
	// size of the main cube, excluding the inlet and any margin
	double box_l, box_w, box_h;
	box_l = box_w = box_h = 1.0;
	// world size
	double world_l = box_l + INLET_BOX_LENGTH + 2 * MARGIN; // length is 1 (box) + 0.2 (inlet box length)
	double world_w = box_w + 2 * MARGIN;
	double world_h = box_h + 2 * MARGIN;
	m_origin = make_double3(- INLET_BOX_LENGTH - MARGIN, - MARGIN, - MARGIN);
	m_size = make_double3(world_l, world_w ,world_h);

	// set max_fall 5 for sspeed =~ 70
	//setMaxFall(5);
	setWaterLevel(0.5);
	setMaxParticleSpeed(7.0);

	add_fluid(1000.0);
	// explicitly set sspeed (not necessary since using setMaxParticleSpeed();
	//set_equation_of_state(7.0f, 70.0f);
	// also possible:
	//set_equation_of_state(7.0f, NAN);
	// to set the adjabatic exponent, but no the sound speed
	set_kinematic_visc(0, 1.0e-2f);

	// add "universe box" of planes
	//makeUniverseBox(m_origin, m_origin + m_size );


	// fluid
	addHDF5File(GT_FLUID, Point(0,0,0), "./sa/0.xcomplete_sa_example.fluid.h5sph", NULL);

	// main container
	GeometryID container =
		addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./sa/0.xcomplete_sa_example.boundary.kent0.h5sph", NULL);
	disableCollisions(container);

	// Inflow square. Load it as GT_FIXED_BOUNDARY to disable it.
	GeometryID inlet =
		addHDF5File(GT_OPENBOUNDARY, Point(0,0,0), "./sa/0.xcomplete_sa_example.boundary.kent1.h5sph", NULL);
	disableCollisions(inlet);

	// set velocity or pressure driven (see define in header)
	// TODO call this function setInflowType with enum VELOCITY_DRIVEN, PRESSURE_DRIVEN
	setVelocityDriven(inlet, VELOCITY_DRIVEN);

	// Floating box, with STL mesh for collision detection
	// GT_FLOATING_BODY for floating, GT_MOVING_BODY for force measurement only
	GeometryID cube =
		addHDF5File(GT_FLOATING_BODY, Point(0,0,0), "./sa/0.xcomplete_sa_example.boundary.kent2.h5sph",
			"./meshes/CompleteSaExample_cube_coarse.stl");

	enableFeedback(cube);

	// NOTE: physparams()->rho0[0] is not available yet if set_density() was not explicitly called,
	// so we use an absolute value instead (half water density)
	setMassByDensity(cube, 500);
}

/*
void XCompleteSaExample::init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash)
{
	const float k0 = 1.0f/sqrtf(0.09f);

	for (uint i = 0; i < numpart; i++) {
		k[i] = k0;
		e[i] = 2.874944542f*k0*0.01f;
	}
} // */

/*
void XCompleteSaExample::imposeForcedMovingObjects(
			float3	&centerOfGravity,
			float3	&translation,
			float*	rotationMatrix,
	const	uint	ob,
	const	double	t,
	const	float	dt)
{
	switch (ob) {
		case 2:
			centerOfGravity = make_float3(0.0f, 0.0f, 0.0f);
			translation = make_float3(0.2f*dt, 0.0f, 0.0f);
			for (uint i=0; i<9; i++)
				rotationMatrix[i] = (i%4==0) ? 1.0f : 0.0f;
			break;
		default:
			break;
	}
}
// */

uint XCompleteSaExample::max_parts(uint numpart)
{
	return (uint)((float)numpart*2.0f);
}

void XCompleteSaExample::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}
