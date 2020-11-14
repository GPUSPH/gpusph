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

#include "ChannelIO.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

ChannelIO::ChannelIO(GlobalData *_gdata) : Problem(_gdata)
{
	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		formulation<SPH_F1>,
		viscosity<DYNAMICVISC>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_NONE>,
		densitydiffusion<BREZZI>,
		add_flags<ENABLE_DTADAPT | ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM | ENABLE_WATER_DEPTH>
	);

	// *** Initialization of minimal physical parameters
	set_deltap(0.07f);
	set_gravity(0.0, 0.0,-9.81);

	// *** Initialization of minimal simulation parameters
	simparams()->tend = 20.f;
	simparams()->dtadaptfactor = 0.3f;
	resize_neiblist(128+128, 64);
	// ferrari correction
	//simparams()->ferrariLengthScale = 0.25f;
	simparams()->densityDiffCoeff = 0.00;

	// buildneibs at every iteration
	simparams()->buildneibsfreq = 1;

	// *** Other parameters and settings
	add_writer(VTKWRITER, 1.0e-2f);
	m_name = "ChannelIO";

	m_origin = make_double3(0, 0, 0);
	m_size = make_double3(4, 1, 1.5);

	add_fluid(1000.0);
	set_equation_of_state(0, 7.0f, 30.0f);
	set_kinematic_visc(0, 0.1f);

	// fluid
	addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/ChannelIO/0.ChannelIO.fluid.h5sph", NULL);

	// main container
	GeometryID container =
		addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/ChannelIO/0.ChannelIO.kent0.h5sph", NULL);
	disableCollisions(container);

	// inflow
	GeometryID inlet =
		addHDF5File(GT_OPENBOUNDARY, Point(0,0,0), "./data_files/ChannelIO/0.ChannelIO.kent1.h5sph", NULL);
	disableCollisions(inlet);
	setVelocityDriven(inlet, 1);

	// outflow
	GeometryID outlet =
		addHDF5File(GT_OPENBOUNDARY, Point(0,0,0), "./data_files/ChannelIO/0.ChannelIO.kent2.h5sph", NULL);
	disableCollisions(outlet);
	setVelocityDriven(outlet, 0);
}

uint ChannelIO::max_parts(uint numpart)
{
	return (uint)((float)numpart*1.2f);
}

namespace cuChannelIO
{
using namespace cuforces;
using namespace cubounds;

__device__
void
ChannelIO_imposeBoundaryCondition(
		const	particleinfo	info,
		const	float3	absPos,
		float	waterdepth,
		const	float	t,
		float4&	vel,
		float4&	eulerVel,
		float&	tke,
		float&	eps)
{
	// Default value for eulerVel
	// Note that this default value needs to be physically feasible, as it is used in case of boundary elements
	// without fluid particles in their support. It is also possible to use this default value to impose tangential
	// velocities for pressure outlets.
	eulerVel = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
	vel = make_float4(0.0f);
	tke = 0.0f;
	eps = 0.0f;

	// open boundary conditions
	if (IO_BOUNDARY(info)) {

		if (!VEL_IO(info)) {
			const float localdepth = fmaxf(waterdepth - absPos.z, 0.0f);
			const float pressure = 9.81f*localdepth*d_rho0[fluid_num(info)];
			eulerVel.w = RHO(pressure, fluid_num(info));
		} else {
			const float U = 0.05;
			// impose velocity
			eulerVel.x = U;// * (1-(powf(absPos.y,2.)+powf(absPos.z,2.))/powf(R,2.))*fmaxf(t/3.,1.);
		}
	}
}

__global__ void
ChannelIO_imposeBoundaryConditionDevice(
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
//		if ((VERTEX(info) || BOUNDARY(info)) && IO_BOUNDARY(info) && (!CORNER(info) || !VEL_IO(info))) {
		if (IO_BOUNDARY(info)) {
			// For corners we need to get eulerVel in case of k-eps and pressure outlet
			if (CORNER(info) && newTke && !VEL_IO(info))
				eulerVel = newEulerVel[index];
			const float3 absPos = d_worldOrigin + as_float3(params.fetchPos(index))
									+ calcGridPosFromParticleHash(particleHash[index])*d_cellSize
									+ 0.5f*d_cellSize;

      float waterdepth = 1.0f;
			if (VEL_IO(info)) {
				waterdepth = ((float)IOwaterdepth[object(info)])/((float)UINT_MAX); // now between 0 and 1
				waterdepth *= d_cellSize.z*d_gridSize.z;
				waterdepth += d_worldOrigin.z;
				if (waterdepth <= 0.0) {
					waterdepth = 1.0;
				}
			}
			// this now calls the virtual function that is problem specific
			ChannelIO_imposeBoundaryCondition(info, absPos, waterdepth, t, vel, eulerVel, tke, eps);
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

} // end of cuChannelIO namespace

void
ChannelIO::imposeBoundaryConditionHost(
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

	cuChannelIO::ChannelIO_imposeBoundaryConditionDevice<<< numBlocks, numThreads, dummy_shared >>>
		(pos_info_wrapper(bufread), newVel, newEulerVel, newTke, newEpsilon, IOwaterdepth, t, numParticles, particleHash);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}
