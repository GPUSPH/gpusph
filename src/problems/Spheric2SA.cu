#include <math.h>
#include <string>
#include <iostream>

#include "Spheric2SA.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "textures.cuh"
#include "utils.h"
#include "Problem.h"

#define USE_PLANES 0

Spheric2SA::Spheric2SA(GlobalData *_gdata) : Problem(_gdata)
{
	h5File.setFilename("meshes/0.spheric2.h5sph");

	SETUP_FRAMEWORK(
		viscosity<DYNAMICVISC>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_NONE>,
		kernel<WENDLAND>,
		add_flags<ENABLE_FERRARI | ENABLE_GAMMA_QUADRATURE>
	);

	set_deltap(0.01833f);

	size_t water = add_fluid(1000.0);
	set_equation_of_state(water,  7.0f, 130.f);
	set_kinematic_visc(water, 1.0e-2f);
	physparams()->gravity = make_float3(0.0, 0.0, -9.81f);

	simparams()->tend = 5.0;
	addPostProcess(SURFACE_DETECTION);
	addPostProcess(TESTPOINTS);
	H = 0.55;
	l = 3.5+0.02; w = 1.0+0.02; h = 2.0;
	m_origin = make_double3(-0.01, -0.01, -0.01);
	simparams()->ferrariLengthScale = 0.161f;
	simparams()->maxneibsnum = 240;

	// SPH parameters
	simparams()->dt = 0.00004f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 1;
	simparams()->nlexpansionfactor = 1.1;

	// Size and origin of the simulation domain
	m_size = make_double3(l, w ,h);

	// Physical parameters
	float g = length(physparams()->gravity);

	physparams()->dcoeff = 5.0f*g*H;

	physparams()->r0 = m_deltap;

	physparams()->artvisccoeff = 0.3f;
	physparams()->epsartvisc = 0.01*simparams()->slength*simparams()->slength;
	physparams()->epsxsph = 0.5f;

	// Drawing and saving times
	add_writer(VTKWRITER, 1e-2f);

	// Name of problem used for directory creation
	m_name = "Spheric2SA";
}


int Spheric2SA::fill_parts()
{
	// Setting probe for Box test case
	//*******************************************************************
	// Wave gages
	add_gage(m_origin + make_double3(2.724, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	add_gage(m_origin + make_double3(2.228, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	add_gage(m_origin + make_double3(1.732, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	add_gage(m_origin + make_double3(0.582, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	// Pressure probes
	if (m_simframework->hasPostProcessEngine(TESTPOINTS)) {
		test_points.push_back(m_origin + make_double3(2.3955, 0.5, 0.021) + make_double3(0.01, 0.01, 0.01)); // the (0.01,0.01,0.01) vector accounts for the slightly shifted origin
		test_points.push_back(m_origin + make_double3(2.3955, 0.5, 0.061) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.3955, 0.5, 0.101) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.3955, 0.5, 0.141) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.4165, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.4565, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.4965, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.5365, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
	}

	return h5File.getNParts() + test_points.size();
}

void Spheric2SA::copy_to_array(BufferList &buffers)
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
		//float rho = density(H - h5File.buf[i].Coords_2, 0);
		float rho = physparams()->rho0[0];
		vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
		// Fluid particles don't have a eulerian velocity
		if (eulerVel)
			eulerVel[i] = make_float4(0.0f);
		info[i] = make_particleinfo(PT_FLUID, 0, i);
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2, rho*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_parts;
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	if(n_vparts) {
		std::cout << "Vertex parts: " << n_vparts << "\n";
		const float referenceVolume = m_deltap*m_deltap*m_deltap;
		for (uint i = j; i < j + n_vparts; i++) {
			float rho = density(H - h5File.buf[i].Coords_2, 0);
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
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2, rho*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
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

			int openBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo_by_ids(PT_BOUNDARY, 0, max(openBoundType-1, 0), i);
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
Spheric2SA::init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash)
{
	const float k0 = 1.0f/sqrtf(0.09f);

	for (uint i = 0; i < numpart; i++) {
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
Spheric2SA::max_parts(uint numpart)
{
	// gives an estimate for the maximum number of particles
	return numpart;
}

void Spheric2SA::fillDeviceMap()
{
	fillDeviceMapByAxis(X_AXIS);
}
