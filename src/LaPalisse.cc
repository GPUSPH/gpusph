#include <math.h>
#include <string>
#include <iostream>

#include "LaPalisse.h"
#include "GlobalData.h"

#define USE_PLANES 0

LaPalisse::LaPalisse(const GlobalData *_gdata) : Problem(_gdata)
{
	h5File.setFilename("meshes/0.LaPalisse.h5sph");

	m_simparams.sfactor=1.3f;
	set_deltap(0.015f);

	m_physparams.kinematicvisc = 1.0e-6f;
	m_simparams.visctype = KEPSVISC;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	m_physparams.set_density(0, 1000.0, 7.0f, 70.0f);

	m_simparams.maxneibsnum = 240;

	m_simparams.tend = 10.0;
	m_simparams.testpoints = false;
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;
	initial_water_level = 1.23f;
	expected_final_water_level = INLET_WATER_LEVEL;
	m_simparams.calcPrivate = false;
	m_simparams.inoutBoundaries = true;
	m_simparams.movingBoundaries = false;
	m_simparams.floatingObjects = false;

	// SPH parameters
	m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 1;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.ferrari= 1.0f;
	m_simparams.mbcallback = false;
	m_simparams.boundarytype = SA_BOUNDARY;
	m_simparams.nlexpansionfactor = 1.1;

	// Size and origin of the simulation domain
	m_size = make_double3(5.8f, 7.6f, 2.4f);
	m_origin = make_double3(-2.35f, -3.5f, -1.3f);

	// Drawing and saving times
	add_writer(VTKWRITER, 1e-6f);

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
			case 1: // AM-TODO call this CRIXUS_FLUID
				n_parts++;
				break;
			case 2: // AM-TODO call this CRIXUS_VERTEX
				n_vparts++;
				break;
			case 3: // AM-TODO call this CRIXUS_BOUNDARY
				n_bparts++;
				break;
		}
	}

	std::cout << "Fluid parts: " << n_parts << "\n";
	for (uint i = 0; i < n_parts; i++) {
		float rho = density(initial_water_level - 1.08f - h5File.buf[i].Coords_2, 0);
		//float rho = m_physparams.rho0[0];
		vel[i] = make_float4(0, 0, 0, rho);
		if (eulerVel)
			eulerVel[i] = make_float4(0);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
			m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_parts;
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	if(n_vparts) {
		std::cout << "Vertex parts: " << n_vparts << "\n";
		for (uint i = j; i < j + n_vparts; i++) {
			float rho = density(initial_water_level - 1.08f - h5File.buf[i].Coords_2, 0);
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			if (eulerVel)
				eulerVel[i] = make_float4(0);
			int specialBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo(VERTEXPART, specialBoundType, i);
			// Define the type of boundaries
			if (specialBoundType != 0) {
				// this vertex is part of an open boundary
				SET_FLAG(info[i], IO_PARTICLE_FLAG);
			}
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
				m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
		}
		j += n_vparts;
		std::cout << "Vertex part mass: " << pos[j-1].w << "\n";
	}

	uint numOdeObjParts = 0;

	if(n_bparts) {
		std::cout << "Boundary parts: " << n_bparts << "\n";
		for (uint i = j; i < j + n_bparts; i++) {
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			if (eulerVel)
				eulerVel[i] = make_float4(0);
			int specialBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo(BOUNDPART, specialBoundType, i);
			// Define the type of boundaries
			if (specialBoundType != 0) {
				// this vertex is part of an open boundary
				SET_FLAG(info[i], IO_PARTICLE_FLAG);
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
			info[i]= make_particleinfo(TESTPOINTSPART, 0, i);
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
		k[i] = k0;
		e[i] = 2.874944542f*k0*0.01f;
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
