#include <math.h>
#include <string>
#include <iostream>

#include "LaPalisse.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "textures.cuh"
#include "utils.h"

#define USE_PLANES 0

LaPalisse::LaPalisse(GlobalData *_gdata) : Problem(_gdata)
{
	h5File.setFilename("meshes/0.LaPalisse.h5sph");

	SETUP_FRAMEWORK(
		viscosity<KEPSVISC>,
		boundary<SA_BOUNDARY>,
		formulation<SPH_F2>,
		add_flags<
			ENABLE_INLET_OUTLET |
			ENABLE_FERRARI |
			ENABLE_WATER_DEPTH |
			ENABLE_DENSITY_SUM>
	);

	simparams()->sfactor=1.3f;
	set_deltap(0.015f);

	add_fluid(1000.0f);
	set_equation_of_state(0,  7.0f, 50.0f);
	set_kinematic_visc(0, 1.0e-6f);
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);

	simparams()->neiblistsize = 240;

	simparams()->tend = 10.0;

	// SPH parameters
	simparams()->dt = 0.00001f;
	simparams()->dtadaptfactor = 0.1;
	simparams()->buildneibsfreq = 1;
	simparams()->ferrari= 1.0f;
	simparams()->nlexpansionfactor = 1.1;

	// Size and origin of the simulation domain
	m_size = make_double3(5.8f, 7.6f, 2.4f);
	m_origin = make_double3(-2.35f, -3.5f, -1.3f);

	// Drawing and saving times
	add_writer(VTKWRITER, 1e-2f);

	// Name of problem used for directory creation
	m_name = "LaPalisse";
}

LaPalisse::~LaPalisse()
{
}

int LaPalisse::fill_parts()
{
	return h5File.getNParts();
}

void LaPalisse::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	vertexinfo *vertices = buffers.getData<BUFFER_VERTICES>();
	float4 *boundelm = buffers.getData<BUFFER_BOUNDELEMENTS>();
	float4 *eulerVel = buffers.getData<BUFFER_EULERVEL>();

	h5File.read();

	uint n_parts = 0;
	uint n_vparts = 0;
	uint n_bparts = 0;

	for (uint i = 0; i<h5File.getNParts(); i++) {
		switch(h5File.buf[i].ParticleType) {
			case CRIXUS_FLUID:
				n_parts++;
				break;
			case CRIXUS_VERTEX:
				n_vparts++;
				break;
			case CRIXUS_BOUNDARY:
				n_bparts++;
				break;
		}
	}

	std::cout << "Fluid parts: " << n_parts << "\n";
	for (uint i = 0; i < n_parts; i++) {
		float rho = density(INLET_WATER_LEVEL - h5File.buf[i].Coords_2, 0);
		//float rho = physparams()->rho0[0];
		vel[i] = make_float4(0, 0, 0, rho);
		if (eulerVel)
			eulerVel[i] = make_float4(0);
		info[i] = make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
			physparams()->rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_parts;
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	if(n_vparts) {
		std::cout << "Vertex parts: " << n_vparts << "\n";
		for (uint i = j; i < j + n_vparts; i++) {
			float rho = density(INLET_WATER_LEVEL - h5File.buf[i].Coords_2, 0);
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
			if (eulerVel)
				eulerVel[i] = vel[i];
			int specialBoundType = h5File.buf[i].KENT;
			// count the number of different objects
			// note that we assume all objects to be sorted from 1 to n. Not really a problem if this
			// is not true it simply means that the IOwaterdepth object is bigger than it needs to be
			// in cases of ODE objects this array is allocated as well, even though it is not needed.
			simparams()->numOpenBoundaries = max(specialBoundType, simparams()->numOpenBoundaries);
			// TODO FIXME MERGE the object id should be sequential from 0, no shifting
			info[i] = make_particleinfo(PT_VERTEX, specialBoundType, i);
			// Define the type of boundaries
			if (specialBoundType != 0) {
				// this vertex is part of an open boundary
				// TODO FIXME MERGE inlet or outlet?
				SET_FLAG(info[i], FG_INLET | FG_OUTLET);
			}
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
				physparams()->rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
		}
		j += n_vparts;
		std::cout << "Vertex part mass: " << pos[j-1].w << "\n";
	}

	if(n_bparts) {
		std::cout << "Boundary parts: " << n_bparts << "\n";
		for (uint i = j; i < j + n_bparts; i++) {
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
			if (eulerVel)
				eulerVel[i] = vel[i];
			int specialBoundType = h5File.buf[i].KENT;
			// TODO FIXME MERGE the object id should be sequential from 0, no shifting
			info[i] = make_particleinfo(PT_BOUNDARY, specialBoundType, i);
			// Define the type of boundaries
			if (specialBoundType != 0) {
				// this vertex is part of an open boundary
				// TODO FIXME MERGE inlet or outlet?
				SET_FLAG(info[i], FG_INLET | FG_OUTLET);
			}
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2, 0.0), info[i], pos[i], hash[i]);
			vertices[i].x = h5File.buf[i].VertexParticle1;
			vertices[i].y = h5File.buf[i].VertexParticle2;
			vertices[i].z = h5File.buf[i].VertexParticle3;
			boundelm[i].x = h5File.buf[i].Normal_0;
			boundelm[i].y = h5File.buf[i].Normal_1;
			boundelm[i].z = h5File.buf[i].Normal_2;
			boundelm[i].w = h5File.buf[i].Surface;
		}
		j += n_bparts;
		std::cout << "Boundary part mass: " << pos[j-1].w << "\n";
	}
	// Make sure that fluid + vertex + boundaries are done in that order
	// before adding any other items like testpoints, etc.

	//Testpoints
	if (test_points.size()) {
		std::cout << "\nTest points: " << test_points.size() << "\n";
		for (uint i = j; i < j+test_points.size(); i++) {
			vel[i] = make_float4(0, 0, 0, 0.0);
			info[i]= make_particleinfo(PT_TESTPOINT, 0, i);
			calc_localpos_and_hash(test_points[i-j], info[i], pos[i], hash[i]);
		}
		j += test_points.size();
		std::cout << "Test point mass:" << pos[j-1].w << "\n";
	}

	std::flush(std::cout);

	h5File.empty();
}

void
LaPalisse::init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash)
{
	for (uint i = 0; i < numpart; i++) {
		k[i] = 0.0f;
		e[i] = 1e-5f;
	}
}

uint
LaPalisse::max_parts(uint numpart)
{
	return (uint)((float)numpart*2.0f);
}

void LaPalisse::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}

void LaPalisse::imposeForcedMovingObjects(
			float3	&centerOfGravity,
			float3	&translation,
			float*	rotationMatrix,
	const	uint	ob,
	const	double	t,
	const	float	dt)
{
	switch (ob) {
		default:
			break;
	}
}

namespace cuLaPalisse
{
using namespace cubounds;
using namespace cuforces;

__device__
void
LaPalisse_imposeBoundaryCondition(
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
	tke = 0.0f;
	eps = 0.0f;

	// open boundary conditions
	if (IO_BOUNDARY(info)) {
		// impose pressure
		if (!VEL_IO(info)) {
			if (object(info)==1) {
				// rise slowly over 6 seconds
				waterdepth = (INLET_WATER_LEVEL - 1.08f - INITIAL_WATER_LEVEL)*fmin(t/RISE_TIME, 1.0f) + INITIAL_WATER_LEVEL;
			}
			const float localdepth = fmax(waterdepth - absPos.z, 0.0f);
			const float pressure = 9.81e3f*localdepth;
			eulerVel.w = RHO(pressure, fluid_num(info));
		}
	}
}

__global__ void
LaPalisse_imposeBoundaryConditionDevice(
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
		if (VERTEX(info) && IO_BOUNDARY(info) && (!CORNER(info) || !VEL_IO(info))) {
			// For corners we need to get eulerVel in case of k-eps and pressure outlet
			if (CORNER(info) && newTke && !VEL_IO(info))
				eulerVel = newEulerVel[index];
			const float3 absPos = d_worldOrigin + as_float3(oldPos[index])
									+ calcGridPosFromParticleHash(particleHash[index])*d_cellSize
									+ 0.5f*d_cellSize;
			float waterdepth = 0.0f;
			if (!VEL_IO(info) && IOwaterdepth) {
				waterdepth = ((float)IOwaterdepth[object(info)-1])/((float)UINT_MAX); // now between 0 and 1
				waterdepth *= d_cellSize.z*d_gridSize.z; // now between 0 and world size
				waterdepth += d_worldOrigin.z; // now absolute z position
			}
			// this now calls the virtual function that is problem specific
			LaPalisse_imposeBoundaryCondition(info, absPos, waterdepth, t, vel, eulerVel, tke, eps);
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

} // end of cuLaPalisse namespace

void
LaPalisse::imposeBoundaryConditionHost(
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

	cuLaPalisse::LaPalisse_imposeBoundaryConditionDevice<<< numBlocks, numThreads, dummy_shared >>>
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
