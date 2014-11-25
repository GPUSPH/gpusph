#include <math.h>
#include <string>
#include <iostream>

#include "InputProblem.h"
#include "GlobalData.h"

#define USE_PLANES 0

InputProblem::InputProblem(const GlobalData *_gdata) : Problem(_gdata)
{
	// Error catcher for SPECIFIC_PROBLEM definition
	// If the value is not defined properly this will throw a compile error
	int i = SPECIFIC_PROBLEM;
	//StillWater periodic (symmetric)
	//*************************************************************************************
#if SPECIFIC_PROBLEM == StillWater
		h5File.setFilename("/home/vorobyev/Crixus/geometries/plane_periodicity/0.plane_0.1_sym.h5sph");

		set_deltap(0.1f);

		m_simparams.testpoints = false;
		H = 2.0;
		l = 2.0; w = 2.0; h = 2.2;

		m_physparams.kinematicvisc = 3.0e-2f;
		m_simparams.visctype = DYNAMICVISC;
		m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);
		m_simparams.tend = 5.0;

		//periodic boundaries
		m_simparams.periodicbound = PERIODIC_X;
		m_physparams.dispvect = make_float3(l, l, 0.0);
		m_physparams.minlimit = make_float3(0.0f, 0.0f, 0.0f);
		m_physparams.maxlimit = make_float3(l, l, 0.0f);
		m_origin = make_double3(0.0, 0.0, 0.0);
		m_physparams.set_density(0, 1000.0, 7.0f, 20.0f);
	//*************************************************************************************

	//Spheric2 (DamBreak)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == Spheric2
		h5File.setFilename("meshes/0.spheric2.h5sph");

		set_deltap(0.01833f);

		m_physparams.kinematicvisc = 1.0e-2f;
		m_simparams.visctype = DYNAMICVISC;
		m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);

		m_simparams.tend = 5.0;
		m_simparams.testpoints = true;
		m_simparams.csvtestpoints = true;
		m_simparams.surfaceparticle = true;
		H = 0.55;
		l = 3.5+0.02; w = 1.0+0.02; h = 2.0;
		m_origin = make_double3(-0.01, -0.01, -0.01);
		m_physparams.set_density(0, 1000.0, 7.0f, 130.0f);
	//*************************************************************************************

	//Box (Dambreak)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == BoxCorner || SPECIFIC_PROBLEM == Box
#if SPECIFIC_PROBLEM == BoxCorner
			h5File.setFilename("meshes/0.box_corner.h5sph");
#else
			h5File.setFilename("meshes/0.box_blend_16.h5sph");
#endif

		set_deltap(0.125f);

		m_physparams.kinematicvisc = 1.0e-2f;
		m_simparams.visctype = DYNAMICVISC;
		m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);

		m_simparams.tend = 5.0;
		m_simparams.testpoints = true;
		m_simparams.csvtestpoints = true;
		m_simparams.surfaceparticle = true;
		H = 1.0;
		l = 2.2; w = 2.2; h = 2.2;
		m_origin = make_double3(-1.1, -1.1, -1.1);
		m_physparams.set_density(0, 1000.0, 7.0f, 45.0f);
		m_simparams.calcPrivate = true;
	//*************************************************************************************

	//SmallChannelFlow (a small channel flow for debugging viscosity)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlow
		h5File.setFilename("meshes/0.small_channel.h5sph");

		set_deltap(0.0625f);

		m_physparams.kinematicvisc = 1.0e-2f;
		m_simparams.visctype = DYNAMICVISC;
		m_physparams.gravity = make_float3(8.0*m_physparams.kinematicvisc, 0.0, 0.0);
		m_physparams.set_density(0, 1000.0, 7.0f, 10.0f);

		m_simparams.tend = 100.0;
		m_simparams.periodicbound = PERIODIC_XY;
		m_simparams.testpoints = false;
		m_simparams.surfaceparticle = false;
		m_simparams.savenormals = false;
		H = 1.0;
		l = 1.0; w = 1.0; h = 1.02;
		m_origin = make_double3(-0.5, -0.5, -0.51);
		m_simparams.calcPrivate = true;
	//*************************************************************************************

	//SmallChannelFlowKEPS (a small channel flow for debugging the k-epsilon model)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowKEPS
		h5File.setFilename("meshes/0.small_channel_keps.h5sph");

		m_simparams.sfactor=2.0f;
		set_deltap(0.05f);

		// turbulent (as in agnes' paper)
		m_physparams.kinematicvisc = 1.5625e-3f;
		m_simparams.visctype = KEPSVISC;
		m_physparams.gravity = make_float3(1.0, 0.0, 0.0);
		m_physparams.set_density(0, 1000.0, 7.0f, 200.0f);

		m_simparams.tend = 100.0;
		m_simparams.periodicbound = PERIODIC_XY;
		m_simparams.testpoints = true;
		m_simparams.csvtestpoints = true;
		m_simparams.surfaceparticle = false;
		m_simparams.savenormals = false;
		H = 2.0;
		l = 0.8; w = 0.8; h = 2.02;
		m_origin = make_double3(-0.4, -0.4, -1.01);
		m_simparams.calcPrivate = false;
	//*************************************************************************************

	//SmallChannelFlowIO (a small channel flow for debugging in/outflow)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowIO
		h5File.setFilename("meshes/0.small_channel_io_walls.h5sph");

		set_deltap(0.2f);

		m_physparams.kinematicvisc = 1.0e-2f;
		m_simparams.visctype = DYNAMICVISC;
		m_physparams.gravity = make_float3(0.0, 0.0, 0.0);
		m_physparams.set_density(0, 1000.0, 7.0f, 10.0f);

		m_simparams.tend = 100.0;
		m_simparams.testpoints = false;
		m_simparams.surfaceparticle = false;
		m_simparams.savenormals = false;
		H = 2.0;
		l = 2.1; w = 2.1; h = 2.1;
		m_origin = make_double3(-1.05, -1.05, -1.05);
		m_simparams.calcPrivate = false;
		m_simparams.inoutBoundaries = true;
	//*************************************************************************************

	//SmallChannelFlowIOPer (a small channel flow for debugging in/outflow with periodicity)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPer
		h5File.setFilename("meshes/0.small_channel_io_2d_per.h5sph");

		m_simparams.sfactor=1.3f;
		set_deltap(0.05f);

		m_physparams.kinematicvisc = 1.0e-1f;
		m_simparams.visctype = DYNAMICVISC;
		m_physparams.gravity = make_float3(0.0, 0.0, 0.0);
		m_physparams.set_density(0, 1000.0, 7.0f, 10.0f);

		m_simparams.tend = 10.0;
		m_simparams.testpoints = false;
		m_simparams.surfaceparticle = false;
		m_simparams.savenormals = false;
		m_simparams.periodicbound = PERIODIC_Y;
		H = 2.0;
		l = 1.1; w = 1.0; h = 2.1;
		m_origin = make_double3(-0.55, -0.5, -1.05);
		m_simparams.calcPrivate = false;
		m_simparams.inoutBoundaries = true;
	//*************************************************************************************

	//SmallChannelFlowIOKeps (a small channel flow for debugging in/outflow with keps)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
		h5File.setFilename("meshes/0.small_channel_io_2d_per.h5sph");

		m_simparams.sfactor=1.3f;
		set_deltap(0.05f);

		m_physparams.kinematicvisc = 1.0e-1f;
		m_simparams.visctype = KEPSVISC;
		m_physparams.gravity = make_float3(0.0, 0.0, 0.0);
		m_physparams.set_density(0, 1000.0, 7.0f, 200.0f);

		m_simparams.tend = 10.0;
		m_simparams.testpoints = false;
		m_simparams.surfaceparticle = false;
		m_simparams.savenormals = false;
		m_simparams.periodicbound = PERIODIC_Y;
		H = 2.0;
		l = 1.1; w = 1.0; h = 2.1;
		m_origin = make_double3(-0.55, -0.5, -1.05);
		m_simparams.calcPrivate = false;
		m_simparams.inoutBoundaries = true;
	//*************************************************************************************

	//IOWithoutWalls (i/o between two plates without walls)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == IOWithoutWalls
		h5File.setFilename("meshes/0.io_without_walls.h5sph");

		set_deltap(0.2f);

		m_physparams.kinematicvisc = 1.0e-2f;
		m_simparams.visctype = DYNAMICVISC;
		m_physparams.gravity = make_float3(-0.1, 0.0, 0.0);
		m_physparams.set_density(0, 1000.0, 7.0f, 10.0f);

		m_simparams.tend = 100.0;
		m_simparams.periodicbound = PERIODIC_YZ;
		m_simparams.testpoints = false;
		m_simparams.surfaceparticle = false;
		m_simparams.savenormals = false;
		H = 2.0;
		l = 2.2; w = 2.0; h = 2.0;
		m_origin = make_double3(-1.1, -1.0, -1.0);
		m_simparams.calcPrivate = false;
		m_simparams.inoutBoundaries = true;
	//*************************************************************************************

	//IOWithoutWalls (i/o between two plates without walls)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == LaPalisseSmallTest
		h5File.setFilename("meshes/0.la_palisse_small_test.h5sph");

		set_deltap(0.1f);

		m_physparams.kinematicvisc = 1.0e-2f;
		m_simparams.visctype = DYNAMICVISC;
		m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
		m_physparams.set_density(0, 1000.0, 7.0f, 110.0f);

		m_simparams.tend = 10.0;
		m_simparams.testpoints = false;
		m_simparams.surfaceparticle = false;
		m_simparams.savenormals = false;
		H = 4.0;
		l = 10.8; w = 2.2; h = 4.2;
		m_origin = make_double3(-5.4, -1.1, -2.1);
		m_simparams.calcPrivate = false;
		m_simparams.inoutBoundaries = true;
		m_simparams.ioWaterdepthComputation = true;
	//*************************************************************************************

#endif

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
	set_timer_tick(1.0e-6);
	add_writer(VTKWRITER, 1);

	// Name of problem used for directory creation
	m_name = "InputProblem";
}


int InputProblem::fill_parts()
{
	// Setting probe for Box test case
	//*******************************************************************
#if SPECIFIC_PROBLEM == Box
	add_gage(m_origin + make_double3(1.0, 1.8, 0.0) + make_double3(0.1, 0.1, 0.1));
	if (m_simparams.testpoints) {
		test_points.push_back(m_origin + make_double3(1.0, 2.0, 0.0) + make_double3(0.1, 0.1, 0.1));
	}
	//*******************************************************************
	// Setting probes for Spheric2 test case
	//*******************************************************************
#elif SPECIFIC_PROBLEM == Spheric2
	// Wave gages
	add_gage(m_origin + make_double3(2.724, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	add_gage(m_origin + make_double3(2.228, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	add_gage(m_origin + make_double3(1.732, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	add_gage(m_origin + make_double3(0.582, 0.5, 0.0) + make_double3(0.01, 0.01, 0.01));
	// Pressure probes
	if (m_simparams.testpoints) {
		test_points.push_back(m_origin + make_double3(2.3955, 0.5, 0.021) + make_double3(0.01, 0.01, 0.01)); // the (0.01,0.01,0.01) vector accounts for the slightly shifted origin
		test_points.push_back(m_origin + make_double3(2.3955, 0.5, 0.061) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.3955, 0.5, 0.101) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.3955, 0.5, 0.141) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.4165, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.4565, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.4965, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
		test_points.push_back(m_origin + make_double3(2.5365, 0.5, 0.161) + make_double3(0.01, 0.01, 0.01));
	}
	//*******************************************************************
	// Setting probes for KEPS test case
	//*******************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowKEPS || SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
	if (m_simparams.testpoints) {
		// create test points at (0,0,.) with dp spacing from bottom to top
		for(uint i=0; i<=40; i++)
			test_points.push_back(m_origin + make_double3(0.4, 0.4, 0.05*(float)i) + make_double3(0.0, 0.0, 0.01));
	}
	//*******************************************************************
#endif

	return h5File.getNParts() + test_points.size();
}

void InputProblem::copy_to_array(BufferList &buffers)
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
		//float rho = density(H - h5File.buf[i].Coords_2, 0);
		float rho = m_physparams.rho0[0];
#if SPECIFIC_PROBLEM == SmallChannelFlowKEPS || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
			const float lvel = log(fmax(1.0f-fabs(h5File.buf[i].Coords_2), 0.5*m_deltap)/0.0015625f)/0.41f+5.2f;
			vel[i] = make_float4(lvel, 0, 0, m_physparams.rho0[0]);
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPer
			const float lvel = 1.0f-h5File.buf[i].Coords_2*h5File.buf[i].Coords_2;
			vel[i] = make_float4(lvel, 0.0f, 0.0f, m_physparams.rho0[0]);
#elif SPECIFIC_PROBLEM == SmallChannelFlowIO
			const float y2 = h5File.buf[i].Coords_1*h5File.buf[i].Coords_1;
			const float z2 = h5File.buf[i].Coords_2*h5File.buf[i].Coords_2;
			const float y4 = y2*y2;
			const float z4 = z2*z2;
			const float y6 = y2*y4;
			const float z6 = z2*z4;
			const float y8 = y4*y4;
			const float z8 = z4*z4;
			const float lvel = (461.0f+y8-392.0f*z2-28.0f*y6*z2-70.0f*z4+z8+70.0f*y4*(z4-1.0f)-28.0f*y2*(14.0f-15.0f*z2+z6))/461.0f;
			vel[i] = make_float4(lvel, 0, 0, m_physparams.rho0[0]);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
			vel[i] = make_float4(1.0f, 0.0f, 0.0f, m_physparams.rho0[0]);
#else
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
#endif
		// Fluid particles don't have a eulerian velocity
		if (eulerVel)
			eulerVel[i] = make_float4(0.0f);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
		calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2, rho*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_parts;
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	if(n_vparts) {
		std::cout << "Vertex parts: " << n_vparts << "\n";
		for (uint i = j; i < j + n_vparts; i++) {
			float rho = density(H - h5File.buf[i].Coords_2, 0);
#if SPECIFIC_PROBLEM == SmallChannelFlowKEPS || \
	SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
				const float lvel = log(fmax(1.0f-fabs(h5File.buf[i].Coords_2), 0.5*m_deltap)/0.0015625f)/0.41f+5.2f;
				vel[i] = make_float4(0.0f, 0.0f, 0.0f, m_physparams.rho0[0]);
				eulerVel[i] = make_float4(lvel, 0.0f, 0.0f, m_physparams.rho0[0]);
#else
				vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
				if (eulerVel)
					eulerVel[i] = vel[i];
#endif
			int openBoundType = h5File.buf[i].KENT;
			// count the number of different objects
			// note that we assume all objects to be sorted from 1 to n. Not really a problem if this
			// is not true it simply means that the IOwaterdepth object is bigger than it needs to be
			// in cases of ODE objects this array is allocated as well, even though it is not needed.
			m_simparams.numObjects = max(openBoundType, m_simparams.numObjects);
			info[i] = make_particleinfo(VERTEXPART, openBoundType, i);
			// Define the type of open boundaries
#if SPECIFIC_PROBLEM == SmallChannelFlowIO || \
    SPECIFIC_PROBLEM == IOWithoutWalls || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOPer || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
				if (openBoundType == 1) {
					// this vertex is part of an open boundary
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					// open boundary imposes velocity
					SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
					// open boundary is an inflow
					SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
				} else if (openBoundType == 2) {
					// this vertex is part of an open boundary
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					// open boundary imposes pressure => VEL_IO_PARTICLE_FLAG not set
					// open boundary is an outflow => INFLOW_PARTICLE_FLAG not set
				}
#elif SPECIFIC_PROBLEM == LaPalisseSmallTest
				// two pressure boundaries
				if (openBoundType != 0)
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
				if (openBoundType == 1)
					SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
#endif
			calc_localpos_and_hash(Point(h5File.buf[i].Coords_0, h5File.buf[i].Coords_1, h5File.buf[i].Coords_2, rho*h5File.buf[i].Volume), info[i], pos[i], hash[i]);
		}
		j += n_vparts;
		std::cout << "Vertex part mass: " << pos[j-1].w << "\n";
	}

	if(n_bparts) {
		std::cout << "Boundary parts: " << n_bparts << "\n";
		for (uint i = j; i < j + n_bparts; i++) {
#if SPECIFIC_PROBLEM == SmallChannelFlowKEPS || \
	SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
				const float lvel = log(fmax(1.0f-fabs(h5File.buf[i].Coords_2), 0.5*m_deltap)/0.0015625f)/0.41f+5.2f;
				vel[i] = make_float4(0.0f, 0.0f, 0.0f, m_physparams.rho0[0]);
				eulerVel[i] = make_float4(lvel, 0.0f, 0.0f, m_physparams.rho0[0]);
#else
				vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
				if (eulerVel)
					eulerVel[i] = vel[i];
#endif
			int openBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo(BOUNDPART, openBoundType, i);
			// Define the type of open boundaries
#if SPECIFIC_PROBLEM == SmallChannelFlowIO || \
    SPECIFIC_PROBLEM == IOWithoutWalls || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOPer || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
				if (openBoundType == 1) {
					// this vertex is part of an open boundary
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					// open boundary imposes velocity
					SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
					// open boundary is an inflow
					SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
				} else if (openBoundType == 2) {
					// this vertex is part of an open boundary
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					// open boundary imposes pressure => VEL_IO_PARTICLE_FLAG not set
					// open boundary is an outflow => INFLOW_PARTICLE_FLAG not set
				}
#elif SPECIFIC_PROBLEM == LaPalisseSmallTest
				// two pressure boundaries
				if (openBoundType != 0)
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
				if (openBoundType == 1)
					SET_FLAG(info[i], INFLOW_PARTICLE_FLAG);
#endif
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
InputProblem::init_keps(float* k, float* e, uint numpart, particleinfo* info, float4* pos, hashKey* hash)
{
	const float k0 = 1.0f/sqrtf(0.09f);

	for (uint i = 0; i < numpart; i++) {
		const unsigned int cellHash = cellHashFromParticleHash(hash[i]);
		const float gridPosZ = float((cellHash % (m_gridsize.COORD2*m_gridsize.COORD1)) / m_gridsize.COORD1);
		const float z = pos[i].z + m_origin.z + (gridPosZ + 0.5f)*m_cellsize.z;
		k[i] = k0;
		e[i] = 1.0f/0.41f/fmax(1.0f-fabs(z),0.5f*(float)m_deltap);
	}
}

uint
InputProblem::max_parts(uint numpart)
{
	// gives an estimate for the maximum number of particles
#if SPECIFIC_PROBLEM == SmallChannelFlowIO || \
    SPECIFIC_PROBLEM == IOWithoutWalls || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOPer || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
		return (uint)((float)numpart*1.2f);
#elif SPECIFIC_PROBLEM == LaPalisseSmallTest
		return (uint)((float)numpart*2.0f);
#else
		return numpart;
#endif
}

void InputProblem::fillDeviceMap()
{
	fillDeviceMapByAxis(X_AXIS);
}
