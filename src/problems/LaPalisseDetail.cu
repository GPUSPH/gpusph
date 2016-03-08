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

#include "LaPalisseDetail.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "textures.cuh"
#include "utils.h"

LaPalisseDetail::LaPalisseDetail(GlobalData *_gdata) : XProblem(_gdata)
{
	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		formulation<SPH_F1>,
		viscosity<KEPSVISC>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_NONE>,
		add_flags<ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM | ENABLE_WATER_DEPTH | ENABLE_FERRARI>
	);

	// *** Initialization of minimal physical parameters
	set_deltap(0.005f);
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);

	// *** Initialization of minimal simulation parameters
	simparams()->maxneibsnum = 256 + 64 + 32; // 352
	// ferrari correction
	simparams()->ferrari = 1.0f;

	// buildneibs at every iteration
	simparams()->buildneibsfreq = 1;

	// *** Other parameters and settings
	add_writer(VTKWRITER, 1e-1f);
	m_name = "LaPalisseDetail";

	// Size and origin of the simulation domain
	H = 1.35;
	l = 3.6; w = 6.1; h = 1.7;
	m_size = make_double3(l, w ,h);
	m_origin = make_double3(-0.38, -5.48, -0.65);

	setWaterLevel(H);
	setMaxParticleSpeed(5.0);

	size_t water = add_fluid(1000.0);
	set_kinematic_visc(water, 1.0e-6f);
	set_equation_of_state(water, 7.0f, 50.0f);

	// fluid
	addHDF5File(GT_FLUID, Point(0,0,0), "./meshes/LaPalisseDetail/0.la_palisse_detail.fluid.h5sph", NULL);

	// main container
	GeometryID container =
		addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./meshes/LaPalisseDetail/0.la_palisse_detail.boundary.kent0.h5sph", NULL);
	disableCollisions(container);

	// Inflow area. Load it as GT_FIXED_BOUNDARY to disable it.
	GeometryID inlet =
		addHDF5File(GT_OPENBOUNDARY, Point(0,0,0), "./meshes/LaPalisseDetail/0.la_palisse_detail.boundary.kent1.h5sph", NULL);
	disableCollisions(inlet);
	setVelocityDriven(inlet, PRESSURE_DRIVEN);

	// Outflow area. Load it as GT_FIXED_BOUNDARY to disable it.
	GeometryID outlet =
		addHDF5File(GT_OPENBOUNDARY, Point(0,0,0), "./meshes/LaPalisseDetail/0.la_palisse_detail.boundary.kent2.h5sph", NULL);
	disableCollisions(outlet);
	setVelocityDriven(outlet, PRESSURE_DRIVEN);
}

void LaPalisseDetail::initializeParticles(BufferList &buffers, const uint numParticles)
{
	printf("Initializing particle properties...\n");

	// grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	float4 *pos = buffers.getData<BUFFER_POS>();
	float4 *eulerVel = buffers.getData<BUFFER_EULERVEL>();
	float *k = buffers.getData<BUFFER_TKE>();
	float *e = buffers.getData<BUFFER_EPSILON>();
	const particleinfo *info = buffers.getData<BUFFER_INFO>();
	const hashKey *hash = buffers.getData<BUFFER_HASH>();

	const float Htilde = H + 0.1f*m_deltap;

	// iterate on the particles
	for (uint i = 0; i < numParticles; i++) {

		// get absolute z position
		const unsigned int cellHash = cellHashFromParticleHash(hash[i]);
		const float gridPosZ = float((cellHash % (m_gridsize.COORD2*m_gridsize.COORD1)) / m_gridsize.COORD1);
		const float z = pos[i].z + m_origin.z + (gridPosZ + 0.5f)*m_cellsize.z;

		const float rho = density(H - z, 0);
		const float lvel = 0.0f;

		if (FLUID(info[i])) {
			vel[i].x = lvel;
			if (z < Htilde) {
				// turbulent intensity
				const float Ti = 0.01f;
				// length scale of the flow (water depth)
				const float L = H;

				k[i] = fmax(3.0f/2.0f*lvel*Ti*lvel*Ti, 1e-6f);
				// constant is C_\mu^(3/4)/0.07*sqrt(3/2)
				// formula is epsilon = C_\mu^(3/4) k^(3/2)/(0.07 L)
				e[i] = fmax(2.874944542f*k[i]*lvel*Ti/L, 1e-6f);
			}
			else {
				k[i] = 1e-6f;
				e[i] = 1e-6f;
			}
		}
		else if (eulerVel) {
			if (!MOVING(info[i]))
				eulerVel[i].x = lvel;
		}
	}
}

void LaPalisseDetail::init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash)
{
	/* do nothing, init of keps is in general init routine */
}

uint LaPalisseDetail::max_parts(uint numpart)
{
	return (uint)((float)numpart*2.0f);
}

void LaPalisseDetail::fillDeviceMap()
{
	fillDeviceMapByAxisBalanced(Y_AXIS);
}

namespace cuLaPalisseDetail
{
using namespace cubounds;
using namespace cuforces;

__device__
void
LaPalisseDetail_imposeBoundaryCondition(
	const	particleinfo	info,
	const	float3			absPos,
			float			waterdepth,
	const	float			t,
			float4&			vel,
			float4&			eulerVel,
			float&			tke,
			float&			eps)
{
	vel = make_float4(0.0f);
	tke = 1e-6f;
	eps = 1e-6f;
	eulerVel = make_float4(0.0f, 0.0f, 0.0f, d_rho0[fluid_num(info)]);

	if (IO_BOUNDARY(info)) {
		if (object(info)==0) // inlet has prescribed water depth
			waterdepth = INITIAL_WATER_LEVEL + (INLET_WATER_LEVEL - INITIAL_WATER_LEVEL)*fmax(t,RISE_TIME)/RISE_TIME;
		const float localdepth = fmax(waterdepth - absPos.z, 0.0f);
		const float pressure = 9.807e3f*localdepth;
		eulerVel.w = RHO(pressure, fluid_num(info));
	}

}

__global__ void
LaPalisseDetail_imposeBoundaryConditionDevice(
			float4*		newVel,
			float4*		newEulerVel,
			float*		newTke,
			float*		newEpsilon,
	const	float4*		oldPos,
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
		const particleinfo info = tex1Dfetch(infoTex, index);
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
			const float3 absPos = d_worldOrigin + as_float3(oldPos[index])
									+ calcGridPosFromParticleHash(particleHash[index])*d_cellSize
									+ 0.5f*d_cellSize;
			// when pressure outlets require the water depth compute it from the IOwaterdepth integer
			float waterdepth = 0.0f;
			if (!VEL_IO(info) && IOwaterdepth) {
				waterdepth = ((float)IOwaterdepth[object(info)])/((float)UINT_MAX); // now between 0 and 1
				waterdepth *= d_cellSize.z*d_gridSize.z; // now between 0 and world size
				waterdepth += d_worldOrigin.z; // now absolute z position
			}
			// this now calls the virtual function that is problem specific
			LaPalisseDetail_imposeBoundaryCondition(info, absPos, waterdepth, t, vel, eulerVel, tke, eps);
			// copy values to arrays
			newVel[index] = vel;
			newEulerVel[index] = eulerVel;
			if(newTke)
				newTke[index] = tke;
			if(newEpsilon)
				newEpsilon[index] = eps;
		}
		// all other vertex particles had their eulerVel set in euler already
	}
}

} // end of cuLaPalisseDetail namespace

void
LaPalisseDetail::imposeBoundaryConditionHost(
			MultiBufferList::iterator		bufwrite,
			MultiBufferList::const_iterator	bufread,
					uint*			IOwaterdepth,
			const	float			t,
			const	uint			numParticles,
			const	uint			numOpenBoundaries,
			const	uint			particleRangeEnd)
{
	float4	*newVel = bufwrite->getData<BUFFER_VEL>();
	float4	*newEulerVel = bufwrite->getData<BUFFER_EULERVEL>();
	float	*newTke = bufwrite->getData<BUFFER_TKE>();
	float	*newEpsilon = bufwrite->getData<BUFFER_EPSILON>();

	const particleinfo *info = bufread->getData<BUFFER_INFO>();
	const float4 *oldPos = bufread->getData<BUFFER_POS>();
	const hashKey *particleHash = bufread->getData<BUFFER_HASH>();

	const uint numThreads = min(BLOCK_SIZE_IOBOUND, particleRangeEnd);
	const uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	cuLaPalisseDetail::LaPalisseDetail_imposeBoundaryConditionDevice<<< numBlocks, numThreads, dummy_shared >>>
		(newVel, newEulerVel, newTke, newEpsilon, oldPos, IOwaterdepth, t, numParticles, particleHash);

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// reset waterdepth calculation
	if (IOwaterdepth) {
		uint h_IOwaterdepth[numOpenBoundaries];
		for (uint i=0; i<numOpenBoundaries; i++)
			h_IOwaterdepth[i] = 0;
		CUDA_SAFE_CALL(cudaMemcpy(IOwaterdepth, h_IOwaterdepth, numOpenBoundaries*sizeof(int), cudaMemcpyHostToDevice));
	}

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}
