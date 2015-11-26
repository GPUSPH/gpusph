#include <math.h>
#include <string>
#include <iostream>

#include "LaPalisse.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "textures.cuh"
#include "utils.h"
#include "Problem.h"

#define USE_PLANES 0

LaPalisse::LaPalisse(GlobalData *_gdata) : Problem(_gdata)
{
	h5File.setFilename("meshes/0.LaPalisse.h5sph");

	SETUP_FRAMEWORK(
		viscosity<KEPSVISC>,
		boundary<SA_BOUNDARY>,
		kernel<WENDLAND>,
		add_flags<ENABLE_FERRARI | ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM | ENABLE_WATER_DEPTH>
	);

	addPostProcess(FLUX_COMPUTATION);

	set_deltap(0.015f);
	simparams()->maxneibsnum = 240;
	simparams()->tend = 40.0;
	simparams()->ferrari= 1.0f;

	size_t water = add_fluid(1000.0f);
	set_equation_of_state(water, 7.0f, 50.0f);
	set_kinematic_visc(water, 1.0e-6f);
	simparams()->numOpenBoundaries=2;

	m_size = make_double3(5.8f, 7.6f, 2.4f);
	m_origin = make_double3(-2.35f, -3.5f, -1.3f);
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);

	// SPH parameters
	simparams()->dt = 0.00004f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 1;
	simparams()->nlexpansionfactor = 1.1;

	// Physical parameters
	float g = length(physparams()->gravity);

	physparams()->dcoeff = 5.0f*g*H;

	physparams()->r0 = m_deltap;

	physparams()->artvisccoeff = 0.3f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	physparams()->epsxsph = 0.5f;

	// Drawing and saving times
	add_writer(VTKWRITER, 1e-7f);

	// Name of problem used for directory creation
	m_name = "LaPalisse";
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
		vel[i] = make_float4(0, 0, 0, rho);
		// Fluid particles don't have a eulerian velocity
		if (eulerVel)
			eulerVel[i] = make_float4(0.0f);
		info[i] = make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2, physparams()->rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_parts;
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	if(n_vparts) {
		std::cout << "Vertex parts: " << n_vparts << "\n";
		const float referenceVolume = m_deltap*m_deltap*m_deltap;
		for (uint i = j; i < j + n_vparts; i++) {
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
			if (eulerVel)
				eulerVel[i] = vel[i];
			int openBoundType = h5File.buf[i].KENT;
			// count the number of different objects
			// note that we assume all objects to be sorted from 1 to n. Not really a problem if this
			// is not true it simply means that the IOwaterdepth object is bigger than it needs to be
			// in cases of ODE objects this array is allocated as well, even though it is not needed.
			simparams()->numOpenBoundaries = max(openBoundType, simparams()->numOpenBoundaries);
			info[i] = make_particleinfo_by_ids(PT_VERTEX, 0, max(openBoundType-1,0), i);
			// Define the type of open boundaries
			// two pressure boundaries
			if (openBoundType != 0)
				SET_FLAG(info[i], FG_INLET | FG_OUTLET);
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2, physparams()->rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
			// boundelm.w contains the reference mass of a vertex particle, actually only needed for IO_BOUNDARY
			boundelm[i].w = h5File.buf[i].Volume/referenceVolume;
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
			// two pressure boundaries
			int openBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo_by_ids(PT_BOUNDARY, 0, max(openBoundType-1,0), i);
			if (openBoundType != 0)
				SET_FLAG(info[i], FG_INLET | FG_OUTLET);
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
	const float k0 = 1.0f/sqrtf(0.09f);

	for (uint i = 0; i < numpart; i++) {
		const unsigned int cellHash = cellHashFromParticleHash(hash[i]);
		const float gridPosZ = float((cellHash % (m_gridsize.COORD2*m_gridsize.COORD1)) / m_gridsize.COORD1);
		const float z = pos[i].z + m_origin.z + (gridPosZ + 0.5f)*m_cellsize.z;
		const float Ti = 0.01f;
		const float u = 0.0f; // TODO set according to initial velocity
		const float L = 1.0f; // TODO set according to geometry
		k[i] = fmax(1e-5f, 3.0f/2.0f*(u*Ti)*(u*Ti));
		e[i] = fmax(1e-5f, 2.874944542f*k[i]*u*Ti/L);
		//k[i] = k0;
		//e[i] = 1.0f/0.41f/fmax(1.0f-fabs(z),0.5f*(float)m_deltap);
	}
}

uint
LaPalisse::max_parts(uint numpart)
{
	// gives an estimate for the maximum number of particles
	return (uint)((float)numpart*2.0f);
}

void LaPalisse::fillDeviceMap()
{
	fillDeviceMapByAxis(X_AXIS);
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

	if (IO_BOUNDARY(info)) {
		if (object(info)==0)
			//waterdepth = 0.255; // set inflow waterdepth to 0.21 (with respect to world_origin)
			//waterdepth = -0.1 + 0.355*fmin(t,20.0f)/20.0f; // set inflow waterdepth to 0.21 (with respect to world_origin)
			//waterdepth = -0.1 + 0.355*fmin(t,5.0f)/5.0f; // set inflow waterdepth to 0.21 (with respect to world_origin)
			waterdepth = (INLET_WATER_LEVEL - 1.08f - INITIAL_WATER_LEVEL)*fmin(t/RISE_TIME, 1.0f) + INITIAL_WATER_LEVEL;
		const float localdepth = fmax(waterdepth - absPos.z, 0.0f);
		const float pressure = 9.81e3f*localdepth;
		eulerVel.w = RHO(pressure, fluid_num(info));
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
			LaPalisse_imposeBoundaryCondition(info, absPos, waterdepth, t, vel, eulerVel, tke, eps);
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
