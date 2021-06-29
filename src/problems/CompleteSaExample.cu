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

#include "CompleteSaExample.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

CompleteSaExample::CompleteSaExample(GlobalData *_gdata) : Problem(_gdata)
{
	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		formulation<SPH_F1>,
		viscosity<DYNAMICVISC>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_NONE>,
		densitydiffusion<BREZZI>,
		add_flags<ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM | ENABLE_MOVING_BODIES>
	);

	// *** Initialization of minimal physical parameters
	set_deltap(0.02f);
	set_gravity(-9.81);

	// *** Initialization of minimal simulation parameters
	resize_neiblist(256 + 64, 128); // 352
	// Density diffusion coefficient
	if (simparams()->densitydiffusiontype == FERRARI)
		simparams()->ferrariLengthScale = 0.25f;
	else
		simparams()->densityDiffCoeff = 0.01;

	// buildneibs at every iteration
	simparams()->buildneibsfreq = 1;

	// *** Other parameters and settings
	add_writer(VTKWRITER, 1e-2f);
	m_name = "CompleteSaExample";

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

	// To simulate with no floating cube: comment the special boundary in the ini file, rerun Crixus, use the
	// appropriate file radix (e.g. complete_sa_example_nobox and do not add the "cube" geometry

	// fluid
	addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/CompleteSaExample/0.complete_sa_example.fluid.h5sph", NULL);

	// main container
	GeometryID container =
		addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/CompleteSaExample/0.complete_sa_example.boundary.kent0.h5sph", NULL);
	disableCollisions(container);

	// Inflow square. Load it as GT_FIXED_BOUNDARY to disable it.
	GeometryID inlet =
		addHDF5File(GT_OPENBOUNDARY, Point(0,0,0), "./data_files/CompleteSaExample/0.complete_sa_example.boundary.kent1.h5sph", NULL);
	disableCollisions(inlet);

	// set velocity or pressure driven (see define in header)
	// TODO call this function setInflowType with enum VELOCITY_DRIVEN, PRESSURE_DRIVEN
	setVelocityDriven(inlet, VELOCITY_DRIVEN);

	// Floating box, with STL mesh for collision detection
	// GT_FLOATING_BODY for floating, GT_MOVING_BODY for force measurement only
	GeometryID cube =
		addHDF5File(GT_FLOATING_BODY, Point(0,0,0), "./data_files/CompleteSaExample/0.complete_sa_example.boundary.kent2.h5sph",
			"./data_files/CompleteSaExample/CompleteSaExample_cube_coarse.obj");

	enableFeedback(cube);

	// NOTE: physparams()->rho0[0] is not available yet if set_density() was not explicitly called,
	// so we use an absolute value instead (half water density)
	setMassByDensity(cube, 500);
}

/*
void CompleteSaExample::init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash)
{
	const float k0 = 1.0f/sqrtf(0.09f);

	for (uint i = 0; i < numpart; i++) {
		k[i] = k0;
		e[i] = 2.874944542f*k0*0.01f;
	}
} // */

/* TODO this routine is never called
void CompleteSaExample::imposeForcedMovingObjects(
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

uint CompleteSaExample::max_parts(uint numpart)
{
	return (uint)((float)numpart*2.0f);
}

void CompleteSaExample::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}

namespace cuCompleteSaExample
{
using namespace cuforces;
using namespace cubounds;

__device__
void
CompleteSaExample_imposeBoundaryCondition(
	const	particleinfo	info,
	const	float3			absPos,
			float			waterdepth,
	const	float			t,
			float4&			vel,
			float4&			eulerVel,
			float&			tke,
			float&			eps)
{
	// Default value for eulerVel
	// Note that this default value needs to be physically feasible, as it is used in case of boundary elements
	// without fluid particles in their support. It is also possible to use this default value to impose tangential
	// velocities for pressure outlets.
	eulerVel = make_float4(0.0f, 0.0f, 0.0f, d_rho0[fluid_num(info)]);
	vel = make_float4(0.0f);
	tke = 0.0f;
	eps = 0.0f;

	// open boundary conditions
	if (IO_BOUNDARY(info)) {

		if (!VEL_IO(info)) {
			// impose pressure

			/*
			if (t < 1.0)
				// inlet pressure grows to target in 1s settling time
				waterdepth = 0.5 + t * (INLET_WATER_LEVEL - 0.5F);
			else
			*/
				// set inflow waterdepth
				waterdepth = INLET_WATER_LEVEL;
			const float localdepth = fmaxf(waterdepth - absPos.z, 0.0f);
			const float pressure = 9.81e3f*localdepth;
			eulerVel.w = RHO(pressure, fluid_num(info));
		} else {
			// impose velocity
			if (t < INLET_VELOCITY_FADE)
				eulerVel.x = INLET_VELOCITY * t / INLET_VELOCITY_FADE;
			else
				eulerVel.x = INLET_VELOCITY;
		}
	}
}

__global__ void
CompleteSaExample_imposeBoundaryConditionDevice(
	pos_info_wrapper	params,
			float4*		__restrict__ newVel,
			float4*		__restrict__ newEulerVel,
			float*		__restrict__ newTke,
			float*		__restrict__ newEpsilon,
	const	uint*		IOwaterdepth,
	const	float		t,
	const	uint		numParticles,
	const	hashKey*	particleHash)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	float4 vel = make_float4(0.0f);			// imposed velocity for moving objects
	float4 eulerVel = make_float4(0.0f);	// imposed velocity/pressure for open boundaries
	float tke = 0.0f;						// imposed turbulent kinetic energy for open boundaries
	float eps = 0.0f;						// imposed turb. diffusivity for open boundaries

	if(index < numParticles) {
		const particleinfo info = params.fetchInfo(index);
		// open boundaries and forced moving objects
		// the case of a corner needs to be treated as follows:
		// - for a velocity inlet nothing is imposed (in case of k-eps newEulerVel already contains the info
		//   from the viscosity
		// - for a pressure inlet the pressure is imposed on the corners. If we are in the k-epsilon case then
		//   we need to get the viscosity info from newEulerVel (x,y,z) and add the imposed density in .w
		if ((VERTEX(info) || BOUNDARY(info)) && IO_BOUNDARY(info) && (!CORNER(info) || !VEL_IO(info))) {
			// For corners we need to get eulerVel in case of k-eps and pressure outlet
			if (CORNER(info) && newTke && !VEL_IO(info))
				eulerVel = newEulerVel[index];
			const float3 absPos = d_worldOrigin + pos_mass(params.fetchPos(index)).pos
									+ calcGridPosFromParticleHash(particleHash[index])*d_cellSize
									+ 0.5f*d_cellSize;
			float waterdepth = 0.0f;
			if (!VEL_IO(info) && IOwaterdepth) {
				waterdepth = ((float)IOwaterdepth[object(info)])/((float)UINT_MAX); // now between 0 and 1
				waterdepth *= d_cellSize.z*d_gridSize.z; // now between 0 and world size
				waterdepth += d_worldOrigin.z; // now absolute z position
			}
			// this now calls the virtual function that is problem specific
			CompleteSaExample_imposeBoundaryCondition(info, absPos, waterdepth, t, vel, eulerVel, tke, eps);
			// copy values to arrays
			newVel[index] = vel;
			newEulerVel[index] = eulerVel;
			if(newTke)
				newTke[index] = tke;
			if(newEpsilon)
				newEpsilon[index] = eps;
		}
	}
}

} // end of cuCompleteSaExample namespace

void
CompleteSaExample::imposeBoundaryConditionHost(
			BufferList&		bufwrite,
			BufferList const&	bufread,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numOpenBoundaries,
			const	uint			particleRangeEnd)
{
	float4	*newVel = bufwrite.getData<BUFFER_VEL>();
	float4	*newEulerVel = bufwrite.getData<BUFFER_EULERVEL>();
	float	*newTke = bufwrite.getData<BUFFER_TKE>();
	float	*newEpsilon = bufwrite.getData<BUFFER_EPSILON>();

	const hashKey *particleHash = bufread.getData<BUFFER_HASH>();

	const uint numThreads = min(BLOCK_SIZE_IOBOUND, particleRangeEnd);
	const uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	cuCompleteSaExample::CompleteSaExample_imposeBoundaryConditionDevice<<< numBlocks, numThreads, dummy_shared >>>
		(pos_info_wrapper(bufread), newVel, newEulerVel, newTke, newEpsilon, IOwaterdepth, t, numParticles, particleHash);

	// reset waterdepth calculation
	if (IOwaterdepth)
		SAFE_CALL(cudaMemset(IOwaterdepth, 0, numOpenBoundaries*sizeof(int)));

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}
