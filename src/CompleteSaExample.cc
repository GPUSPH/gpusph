#include <math.h>
#include <string>
#include <iostream>

#include "CompleteSaExample.h"
#include "GlobalData.h"

#define USE_PLANES 0

CompleteSaExample::CompleteSaExample(const GlobalData *_gdata) : Problem(_gdata)
{
	h5File.setFilename("meshes/0.complete_sa_example.h5sph");

	container = STLMesh::load_stl("./meshes/CompleteSaExample_container_coarse.stl");
	cube = STLMesh::load_stl("./meshes/CompleteSaExample_cube_coarse.stl");

	m_simparams.numObjects = 2;

	m_simparams.sfactor=1.3f;
	set_deltap(0.02f);

	m_physparams.kinematicvisc = 1.0e-2f;
	m_simparams.visctype = DYNAMICVISC;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	m_physparams.set_density(0, 1000.0, 7.0f, 70.0f);

	// ugh (cit.)
	m_simparams.maxneibsnum = 384;

	m_simparams.tend = 10.0;
	m_simparams.testpoints = false;
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;
	H = 1.0;
	// extra margin around the domain size
	const double MARGIN = 0.1;
	const double INLET_BOX_LENGTH = 0.25;
	l = 1.0 + INLET_BOX_LENGTH + 2 * MARGIN; // length is 1 (box) + 0.2 (inlet box length)
	w = 1.0 + 2 * MARGIN;
	h = 1.0 + 2 * MARGIN;
	m_origin = make_double3(- INLET_BOX_LENGTH - MARGIN, - MARGIN, - MARGIN);
	m_simparams.calcPrivate = false;
	m_simparams.inoutBoundaries = true;
	m_simparams.movingBoundaries = true;
	m_simparams.floatingObjects = true;

	// SPH parameters
	m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 1;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.ferrari = 0.1;
	m_simparams.mbcallback = false;
	m_simparams.boundarytype = SA_BOUNDARY;
	m_simparams.nlexpansionfactor = 1.1;

	// Size and origin of the simulation domain
	m_size = make_double3(l, w ,h);

	// Physical parameters
	float g = length(m_physparams.gravity);

	m_physparams.dcoeff = 5.0f*g*H;

	m_physparams.r0 = m_deltap;

	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	m_physparams.epsxsph = 0.5f;

	// Drawing and saving times
	set_timer_tick(1.0e-2);
	add_writer(VTKWRITER, 1);

	/*
	// will use only 1 ODE body: the floating/moving cube (container is fixed)
	allocate_ODE_bodies(1);
	dInitODE();
	// world setup
	m_ODEWorld = dWorldCreate();
	m_ODESpace = dHashSpaceCreate(0);
	//m_ODEJointGroup = dJointGroupCreate(0);
	// Set gravity（x, y, z)
	dWorldSetGravity(m_ODEWorld,
		m_physparams.gravity.x, m_physparams.gravity.y, m_physparams.gravity.z);

	// Name of problem used for directory creation
	m_name = "CompleteSaExample";
	*/
}


int CompleteSaExample::fill_parts()
{
	/*
	container->ODEGeomCreate(m_ODESpace, m_deltap);
	cube->ODEGeomCreate(m_ODESpace, m_deltap);

	// cube density half water density
	const double cube_size = 0.2;
	cube->SetMass( (cube_size * cube_size * cube_size) * 0.5 );

	// no need to unfill, done by Crixus
	// cube->Unfill(fluid_parts, m_deltap);

	//cube->ODEBodyCreate(m_ODEWorld, m_deltap, ); // only dynamics
	//cube->ODEGeomCreate(m_ODESpace, m_deltap); // only collisions
	cube->ODEBodyCreate(m_ODEWorld, m_deltap, m_ODESpace); // dynamics + collisions
	add_ODE_body(cube);

	*/

	return h5File.getNParts();
}

void CompleteSaExample::copy_to_array(BufferList &buffers)
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
			case 1:
				n_parts++;
				break;
			case 2:
				n_vparts++;
				break;
			case 3:
				n_bparts++;
				break;
		}
	}

	std::cout << "Fluid parts: " << n_parts << "\n";
	for (uint i = 0; i < n_parts; i++) {
		float rho = density(H - h5File.buf[i].Coords_2, 0);
		//float rho = m_physparams.rho0[0];
		vel[i] = make_float4(0, 0, 0, rho);
		//vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		if (eulerVel)
			eulerVel[i] = vel[i];
		info[i] = make_particleinfo(FLUIDPART, 0, i);
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
			m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_parts;
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	if(n_vparts) {
		std::cout << "Vertex parts: " << n_vparts << "\n";
		for (uint i = j; i < j + n_vparts; i++) {
			float rho = density(H - h5File.buf[i].Coords_2, 0);
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			if (eulerVel)
				eulerVel[i] = vel[i];
			int specialBoundType = h5File.buf[i].KENT;
			// count the number of different objects
			// note that we assume all objects to be sorted from 1 to n. Not really a problem if this
			// is not true it simply means that the IOwaterdepth object is bigger than it needs to be
			// in cases of ODE objects this array is allocated as well, even though it is not needed.
			m_simparams.numObjects = max(specialBoundType, m_simparams.numObjects);
			info[i] = make_particleinfo(VERTEXPART, specialBoundType, i);
			// Define the type of boundaries
			if (specialBoundType == 1) {
				// this vertex is part of an open boundary
				SET_FLAG(info[i], IO_PARTICLE_FLAG);
				// if you need to impose the velocity uncomment the following
				//// open boundary imposes velocity
				//SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
				// open boundary is an inflow
				SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
			} else if (specialBoundType == 2) {
				// this vertex is part of a moving object
				SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
				// this moving object is also floating
				SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
			}
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2,
				m_physparams.rho0[0]*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
		}
		j += n_vparts;
		std::cout << "Vertex part mass: " << pos[j-1].w << "\n";
	}

	if(n_bparts) {
		std::cout << "Boundary parts: " << n_bparts << "\n";
		for (uint i = j; i < j + n_bparts; i++) {
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			if (eulerVel)
				eulerVel[i] = vel[i];
			int specialBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo(BOUNDPART, specialBoundType, i);
			// Define the type of boundaries
			if (specialBoundType == 1) {
				// this vertex is part of an open boundary
				SET_FLAG(info[i], IO_PARTICLE_FLAG);
				// if you need to impose the velocity uncomment the following
				//// open boundary imposes velocity
				//SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
				// open boundary is an inflow
				SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
			} else if (specialBoundType == 2) {
				// this vertex is part of a moving object
				SET_FLAG(info[i], MOVING_PARTICLE_FLAG);
				// this moving object is also floating
				SET_FLAG(info[i], FLOATING_PARTICLE_FLAG);
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
CompleteSaExample::init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash)
{
	const float k0 = 1.0f/sqrtf(0.09f);

	for (uint i = 0; i < numpart; i++) {
		k[i] = k0;
		e[i] = 2.874944542f*k0*0.01f;
	}
}

uint
CompleteSaExample::max_parts(uint numpart)
{
	return (uint)((float)numpart*2.0f);
}

void CompleteSaExample::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}
