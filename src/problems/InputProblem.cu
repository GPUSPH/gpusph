#include <math.h>
#include <string>
#include <iostream>

#include "InputProblem.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "textures.cuh"
#include "utils.h"
#include "Problem.h"

#define USE_PLANES 0

InputProblem::InputProblem(GlobalData *_gdata) : Problem(_gdata)
{
	// Error catcher for SPECIFIC_PROBLEM definition
	// If the value is not defined properly this will throw a compile error
	int i = SPECIFIC_PROBLEM;

	//Spheric2 (DamBreak)
	//*************************************************************************************
#if SPECIFIC_PROBLEM == Spheric2
		h5File.setFilename("meshes/0.spheric2.h5sph");

		SETUP_FRAMEWORK(
			viscosity<DYNAMICVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_NONE>,
			kernel<WENDLAND>,
			flags<ENABLE_DTADAPT | ENABLE_FERRARI>
		);

		set_deltap(0.01833f);

		m_physparams->kinematicvisc = 1.0e-2f;
		m_physparams->gravity = make_float3(0.0, 0.0, -9.81f);

		m_simparams->tend = 5.0;
		m_simparams->testpoints = true;
		m_simparams->csvtestpoints = true;
		m_simparams->surfaceparticle = true;
		H = 0.55;
		l = 3.5+0.02; w = 1.0+0.02; h = 2.0;
		m_origin = make_double3(-0.01, -0.01, -0.01);
		m_simparams->ferrariLengthScale = 0.161f;
		m_physparams->set_density(0, 1000.0, 7.0f, 130.0f);
		m_simparams->maxneibsnum = 240;
	//*************************************************************************************

	//Box (Dambreak)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == BoxCorner || SPECIFIC_PROBLEM == Box
#if SPECIFIC_PROBLEM == BoxCorner
			h5File.setFilename("meshes/0.box_corner.h5sph");
#else
			h5File.setFilename("meshes/0.box_blend_16.h5sph");
#endif

		SETUP_FRAMEWORK(
			viscosity<DYNAMICVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_NONE>,
			kernel<WENDLAND>,
			flags<ENABLE_DTADAPT | ENABLE_FERRARI | ENABLE_DENSITY_SUM>
		);

		set_deltap(0.125f);

		m_physparams->gravity = make_float3(0.0, 0.0, -9.81f);

		m_simparams->tend = 5.0;
		addPostProcess(SURFACE_DETECTION);
		addPostProcess(TESTPOINTS);
		m_simparams->csvtestpoints = true;
		H = 1.0;
		l = 2.2; w = 2.2; h = 2.2;
		m_origin = make_double3(-1.1, -1.1, -1.1);
		//m_simparams->ferrariLengthScale = 1.0f;
		m_simparams->ferrari = 1.0f;
		size_t water = add_fluid(1000.0);
		set_equation_of_state(water,  7.0f, 60.f);
		set_kinematic_visc(0, 1.0e-2f);
		addPostProcess(CALC_PRIVATE);
	//*************************************************************************************

	//SmallChannelFlow (a small channel flow for debugging viscosity)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlow
		h5File.setFilename("meshes/0.small_channel.h5sph");

		set_deltap(0.0625f);

		m_physparams->kinematicvisc = 1.0e-2f;
		m_simparams->visctype = DYNAMICVISC;
		m_physparams->gravity = make_float3(8.0*m_physparams->kinematicvisc, 0.0, 0.0);
		m_physparams->set_density(0, 1000.0, 7.0f, 10.0f);

		m_simparams->tend = 100.0;
		m_simparams->periodicbound = PERIODIC_XY;
		m_simparams->testpoints = false;
		m_simparams->surfaceparticle = false;
		m_simparams->savenormals = false;
		H = 1.0;
		l = 1.0; w = 1.0; h = 1.02;
		m_simparams->ferrariLengthScale = 0.5f;
		m_origin = make_double3(-0.5, -0.5, -0.51);
		m_simparams->calcPrivate = true;
	//*************************************************************************************

	//SmallChannelFlowKEPS (a small channel flow for debugging the k-epsilon model)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowKEPS
		h5File.setFilename("meshes/0.small_channel_keps.h5sph");

		m_simparams->sfactor=2.0f;
		set_deltap(0.05f);

		// turbulent (as in agnes' paper)
		m_physparams->kinematicvisc = 1.5625e-3f;
		m_simparams->visctype = KEPSVISC;
		m_physparams->gravity = make_float3(1.0, 0.0, 0.0);
		m_physparams->set_density(0, 1000.0, 7.0f, 200.0f);

		m_simparams->tend = 100.0;
		m_simparams->periodicbound = PERIODIC_XY;
		m_simparams->testpoints = true;
		m_simparams->csvtestpoints = true;
		m_simparams->surfaceparticle = false;
		m_simparams->savenormals = false;
		H = 2.0;
		l = 0.8; w = 0.8; h = 2.02;
		m_simparams->ferrariLengthScale = 1.0f;
		m_origin = make_double3(-0.4, -0.4, -1.01);
		m_simparams->calcPrivate = false;
	//*************************************************************************************

	//SmallChannelFlowIO (a small channel flow for debugging in/outflow)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowIO
		h5File.setFilename("meshes/0.small_channel_io_walls.h5sph");

		set_deltap(0.2f);

		m_physparams->kinematicvisc = 1.0e-2f;
		m_simparams->visctype = DYNAMICVISC;
		m_physparams->gravity = make_float3(0.0, 0.0, 0.0);
		m_physparams->set_density(0, 1000.0, 7.0f, 10.0f);

		m_simparams->tend = 100.0;
		m_simparams->testpoints = false;
		m_simparams->surfaceparticle = false;
		m_simparams->savenormals = false;
		H = 2.0;
		l = 2.1; w = 2.1; h = 2.1;
		m_origin = make_double3(-1.05, -1.05, -1.05);
		m_simparams->ferrariLengthScale = 1.0f;
		m_simparams->calcPrivate = false;
		m_simparams->inoutBoundaries = true;
		m_simparams->maxneibsnum = 220;
	//*************************************************************************************

	//SmallChannelFlowIOPer (a small channel flow for debugging in/outflow with periodicity)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPer
		h5File.setFilename("meshes/0.small_channel_io_2d_per.h5sph");

		m_simparams->sfactor=1.3f;
		set_deltap(0.05f);

		m_physparams->kinematicvisc = 1.0e-1f;
		m_simparams->visctype = DYNAMICVISC;
		m_physparams->gravity = make_float3(0.0, 0.0, 0.0);
		m_physparams->set_density(0, 1000.0, 7.0f, 10.0f);

		m_simparams->tend = 10.0;
		m_simparams->testpoints = false;
		m_simparams->surfaceparticle = false;
		m_simparams->savenormals = false;
		m_simparams->periodicbound = PERIODIC_Y;
		H = 2.0;
		l = 1.1; w = 1.0; h = 2.1;
		m_simparams->ferrariLengthScale = 1.0f;
		m_origin = make_double3(-0.55, -0.5, -1.05);
		m_simparams->calcPrivate = false;
		m_simparams->inoutBoundaries = true;
	//*************************************************************************************

	//SmallChannelFlowIOKeps (a small channel flow for debugging in/outflow with keps)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
		h5File.setFilename("meshes/0.small_channel_io_2d_per.h5sph");

		m_simparams->sfactor=1.3f;
		set_deltap(0.05f);

		m_physparams->kinematicvisc = 1.5625e-3f;
		m_simparams->visctype = KEPSVISC;
		m_physparams->gravity = make_float3(0.0, 0.0, 0.0);
		m_physparams->set_density(0, 1000.0, 7.0f, 200.0f);

		m_simparams->tend = 10.0;
		m_simparams->testpoints = false;
		m_simparams->surfaceparticle = false;
		m_simparams->savenormals = false;
		m_simparams->periodicbound = PERIODIC_Y;
		H = 2.0;
		l = 1.1; w = 1.0; h = 2.1;
		m_simparams->ferrariLengthScale = 1.0f;
		m_origin = make_double3(-0.55, -0.5, -1.05);
		m_simparams->calcPrivate = false;
		m_simparams->inoutBoundaries = true;
	//*************************************************************************************

	//IOWithoutWalls (i/o between two plates without walls)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == IOWithoutWalls
		h5File.setFilename("meshes/0.io_without_walls.h5sph");

		set_deltap(0.2f);

		m_physparams->kinematicvisc = 1.0e-2f;
		m_simparams->visctype = DYNAMICVISC;
		m_physparams->gravity = make_float3(0.0, 0.0, 0.0);
		m_physparams->set_density(0, 1000.0, 7.0f, 10.0f);

		m_simparams->tend = 100.0;
		m_simparams->periodicbound = PERIODIC_YZ;
		m_simparams->testpoints = false;
		m_simparams->surfaceparticle = false;
		m_simparams->savenormals = false;
		H = 2.0;
		l = 2.2; w = 2.0; h = 2.0;
		m_origin = make_double3(-1.1, -1.0, -1.0);
		m_simparams->ferrariLengthScale = 1.0f;
		m_simparams->calcPrivate = false;
		m_simparams->inoutBoundaries = true;
	//*************************************************************************************

	//Small test case with similar features to La Palisse
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == LaPalisseSmallTest
		h5File.setFilename("meshes/0.la_palisse_small_test.h5sph");

		set_deltap(0.1f);

		m_physparams->kinematicvisc = 1.0e-2f;
		m_simparams->visctype = DYNAMICVISC;
		m_physparams->gravity = make_float3(0.0, 0.0, -9.81);
		m_physparams->set_density(0, 1000.0, 7.0f, 110.0f);

		m_simparams->tend = 40.0;
		m_simparams->testpoints = false;
		m_simparams->surfaceparticle = false;
		m_simparams->savenormals = false;
		H = 4.0;
		l = 10.8; w = 2.2; h = 4.2;
		m_origin = make_double3(-5.4, -1.1, -2.1);
		//m_simparams->ferrariLengthScale = 0.2f;
		m_simparams->ferrari= 1.0f;
		m_simparams->calcPrivate = false;
		m_simparams->inoutBoundaries = true;
		m_simparams->ioWaterdepthComputation = true;
		m_simparams->maxneibsnum = 240;
	//*************************************************************************************

	//Periodic wave with IO
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == PeriodicWave
		h5File.setFilename("meshes/0.periodic_wave_0.02.h5sph");

		set_deltap(0.02f);

		m_physparams->kinematicvisc = 1.0e-6f;
		m_simparams->visctype = DYNAMICVISC;
		m_physparams->gravity = make_float3(0.0, 0.0, -9.81);
		m_physparams->set_density(0, 1000.0, 7.0f, 25.0f);

		m_simparams->tend = 10.0;
		//m_simparams->tend = 0.2;
		m_simparams->testpoints = true;
		m_simparams->periodicbound = PERIODIC_Y;
		m_simparams->surfaceparticle = true;
		m_simparams->savenormals = true;
		H = 0.5;
		l = 2.7; w = 0.5; h = 1.2;
		//m_simparams->sfactor=1.3f;
		m_simparams->sfactor=2.0f;
		m_origin = make_double3(-1.35, -0.25, -0.1);
		m_simparams->ferrari = 0.1f;
		m_simparams->calcPrivate = false;
		m_simparams->inoutBoundaries = true;
		m_simparams->ioWaterdepthComputation = false;
		m_simparams->maxneibsnum = 240;
	//*************************************************************************************

#endif

	// SPH parameters
	m_simparams->dt = 0.00004f;
	m_simparams->dtadaptfactor = 0.3;
	m_simparams->buildneibsfreq = 1;
	m_simparams->nlexpansionfactor = 1.1;
	//m_simparams->densitySum = true;

	// Size and origin of the simulation domain
	m_size = make_double3(l, w ,h);

	// Physical parameters
	float g = length(m_physparams->gravity);

	m_physparams->dcoeff = 5.0f*g*H;

	m_physparams->r0 = m_deltap;

	m_physparams->artvisccoeff = 0.3f;
	m_physparams->epsartvisc = 0.01*m_simparams->slength*m_simparams->slength;
	m_physparams->epsxsph = 0.5f;

	// Drawing and saving times
	add_writer(VTKWRITER, 1e-2f);

	// Name of problem used for directory creation
	m_name = "InputProblem";
}


int InputProblem::fill_parts()
{
	// Setting probe for Box test case
	//*******************************************************************
#if SPECIFIC_PROBLEM == Box
	add_gage(m_origin + make_double3(1.0, 1.8, 0.0) + make_double3(0.1, 0.1, 0.1));
	if (m_simframework->hasPostProcessEngine(TESTPOINTS)) {
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
	//*******************************************************************
	// Setting probes for channel flow keps test cases (with and without io)
	//*******************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowKEPS || SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
	if (m_simframework->hasPostProcessEngine(TESTPOINTS)) {
		// create test points at (0,0,.) with dp spacing from bottom to top
		for(uint i=0; i<=40; i++)
			test_points.push_back(m_origin + make_double3(0.4, 0.4, 0.05*(float)i) + make_double3(0.0, 0.0, 0.01));
	}
	//*******************************************************************
	// Setting probes for PeriodicWave test case
	//*******************************************************************
#elif SPECIFIC_PROBLEM == PeriodicWave
	add_gage(make_double3(0.0, 0.0, 0.2));
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

#if SPECIFIC_PROBLEM == PeriodicWave
	// define some constants
	const float L = 2.5f;
	const float k = 2.0f*M_PI/L;
	const float D = 0.5f;
	const float A = 0.05f;
	const float phi = M_PI/2.0f;
	const float omega = sqrt(9.807f*k*tanh(k*D));
	printf("Periodic Wave: Wave period: %e\n", 2.0f*M_PI/omega);
#endif

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
		float rho = m_physparams->rho0[0];
#if SPECIFIC_PROBLEM == SmallChannelFlowKEPS || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
			const float lvel = log(fmax(1.0f-fabs(h5File.buf[i].Coords_2), 0.5*m_deltap)/0.0015625f)/0.41f+5.2f;
			vel[i] = make_float4(lvel, 0, 0, m_physparams->rho0[0]);
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPer
			const float lvel = 1.0f-h5File.buf[i].Coords_2*h5File.buf[i].Coords_2;
			vel[i] = make_float4(lvel, 0.0f, 0.0f, m_physparams->rho0[0]);
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
			vel[i] = make_float4(lvel, 0, 0, m_physparams->rho0[0]);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
			vel[i] = make_float4(1.0f, 0.0f, 0.0f, (m_physparams->rho0[0]+2.0f));//+1.0f-1.0f*h5File.buf[i].Coords_0));
#elif SPECIFIC_PROBLEM == PeriodicWave
			const float x = h5File.buf[i].Coords_0;
			const float z = h5File.buf[i].Coords_2;
			const float eta = A*cos(k*x+phi);
			const float h = D + eta;
			//const float p = rho*9.807f*(z-h) + cosh(k*z)/cosh(k*D)*rho*9.807f*eta;
			const float p = rho*-9.807f*(z-h) + cosh(k*z)/cosh(k*D)*rho*9.807f*eta;
			const float _rho = powf(p/(625.0f*rho/7.0f) + 1.0f, 1.0f/7.0f)*rho;
			const float u = A*omega*cosh(k*z)/sinh(k*D)*cos(k*x+phi);
			const float w = A*omega*sinh(k*z)/sinh(k*D)*sin(k*x+phi);
			vel[i] = make_float4(u, 0, w, _rho);
#else
			vel[i] = make_float4(0, 0, 0, m_physparams->rho0[0]);
#endif
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
		for (uint i = j; i < j + n_vparts; i++) {
			float rho = density(H - h5File.buf[i].Coords_2, 0);
#if SPECIFIC_PROBLEM == SmallChannelFlowKEPS || \
	SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
				const float lvel = log(fmax(1.0f-fabs(h5File.buf[i].Coords_2), 0.5*m_deltap)/0.0015625f)/0.41f+5.2f;
				vel[i] = make_float4(0.0f, 0.0f, 0.0f, m_physparams->rho0[0]);
				eulerVel[i] = make_float4(lvel, 0.0f, 0.0f, m_physparams->rho0[0]);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
				vel[i] = make_float4(0, 0, 0, m_physparams->rho0[0]+2.0f);
#else
				vel[i] = make_float4(0, 0, 0, m_physparams->rho0[0]);
				if (eulerVel)
					eulerVel[i] = vel[i];
#endif
			// CAM-TODO use different indices here
			int openBoundType = h5File.buf[i].KENT;
			// count the number of different objects
			// note that we assume all objects to be sorted from 1 to n. Not really a problem if this
			// is not true it simply means that the IOwaterdepth object is bigger than it needs to be
			// in cases of ODE objects this array is allocated as well, even though it is not needed.
			info[i] = make_particleinfo(PT_VERTEX, openBoundType, i);
			// Define the type of open boundaries
#if SPECIFIC_PROBLEM == SmallChannelFlowIO || \
    SPECIFIC_PROBLEM == IOWithoutWalls || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOPer || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
				if (openBoundType == 1) {
					// this vertex is part of an open boundary
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					// open boundary imposes velocity
#if SPECIFIC_PROBLEM != IOWithoutWalls
					SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
#endif
				} else if (openBoundType == 2) {
					// this vertex is part of an open boundary
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					// open boundary imposes pressure => VEL_IO_PARTICLE_FLAG not set
				}
#elif SPECIFIC_PROBLEM == PeriodicWave
				// two pressure boundaries
				if (openBoundType != 0) {
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
				}
#elif SPECIFIC_PROBLEM == LaPalisseSmallTest
				// two pressure boundaries
				if (openBoundType != 0)
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
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
				vel[i] = make_float4(0.0f, 0.0f, 0.0f, m_physparams->rho0[0]);
				eulerVel[i] = make_float4(lvel, 0.0f, 0.0f, m_physparams->rho0[0]);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
				vel[i] = make_float4(0, 0, 0, m_physparams->rho0[0]+2.0f);
#else
				vel[i] = make_float4(0, 0, 0, m_physparams->rho0[0]);
				if (eulerVel)
					eulerVel[i] = vel[i];
#endif
			int openBoundType = h5File.buf[i].KENT;
			info[i] = make_particleinfo(PT_BOUNDARY, openBoundType, i);
			// Define the type of open boundaries
#if SPECIFIC_PROBLEM == SmallChannelFlowIO || \
    SPECIFIC_PROBLEM == IOWithoutWalls || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOPer || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
				if (openBoundType == 1) {
					// this vertex is part of an open boundary
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					// open boundary imposes velocity
#if SPECIFIC_PROBLEM != IOWithoutWalls
					SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
#endif
				} else if (openBoundType == 2) {
					// this vertex is part of an open boundary
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					// open boundary imposes pressure => VEL_IO_PARTICLE_FLAG not set
				}
#elif SPECIFIC_PROBLEM == PeriodicWave
				// two pressure boundaries
				if (openBoundType != 0) {
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
					SET_FLAG(info[i], VEL_IO_PARTICLE_FLAG);
				}
#elif SPECIFIC_PROBLEM == LaPalisseSmallTest
				// two pressure boundaries
				if (openBoundType != 0)
					SET_FLAG(info[i], IO_PARTICLE_FLAG);
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
    SPECIFIC_PROBLEM == SmallChannelFlowIOKeps || \
    SPECIFIC_PROBLEM == PeriodicWave
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

namespace cuInputProblem
{
#include "cuda/cellgrid.cuh"
// Core SPH functions
#include "cuda/sph_core_utils.cuh"

__device__
void
InputProblem_imposeBoundaryCondition(
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
		if (VEL_IO(info)) {
#if SPECIFIC_PROBLEM == SmallChannelFlowIO
			// third order approximation to the flow in a rectangular duct
			const float y2 = absPos.y*absPos.y;
			const float z2 = absPos.z*absPos.z;
			const float y4 = y2*y2;
			const float z4 = z2*z2;
			const float y6 = y2*y4;
			const float z6 = z2*z4;
			const float y8 = y4*y4;
			const float z8 = z4*z4;
			eulerVel.x = (461.0f+y8-392.0f*z2-28.0f*y6*z2-70.0f*z4+z8+70.0f*y4*(z4-1.0f)-28.0f*y2*(14.0f-15.0f*z2+z6))/461.0f;
			eulerVel.x = fmax(eulerVel.x, 0.0f);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
			eulerVel.x = 1.0f;
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPer
			eulerVel.x = 1.0f-absPos.z*absPos.z;
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
			// the 0.025 is deltap*0.5 = 0.05*0.5
			eulerVel.x = log(fmax(1.0f-fabs(absPos.z), 0.025f)/0.0015625f)/0.41f+5.2f;
#elif SPECIFIC_PROBLEM == PeriodicWave
			// define some constants
			const float L = 2.5f;
			const float k = 2.0f*M_PI/L;
			const float D = 0.5f;
			const float A = 0.05f;
			const float phi = M_PI/2.0f;
			const float omega = sqrt(9.807f*k*tanh(k*D));
			const float x = absPos.x;
			const float z = absPos.z;
			const float eta = A*cos(k*x-omega*t+phi);
			const float h = D + eta;
			if (z-0.005 <= h) {
				const float u = A*omega*cosh(k*z)/sinh(k*D)*cos(k*x-omega*t+phi);
				const float w = A*omega*sinh(k*z)/sinh(k*D)*sin(k*x-omega*t+phi);
				eulerVel.x = u;
				eulerVel.z = w;
			}
#else
			eulerVel.x = 0.0f;
#endif
		}
		else {
#if SPECIFIC_PROBLEM == LaPalisseSmallTest
			if (object(info)==1)
				waterdepth = 0.255; // set inflow waterdepth to 0.21 (with respect to world_origin)
				//waterdepth = -0.1 + 0.355*t/20.0f; // set inflow waterdepth to 0.21 (with respect to world_origin)
			const float localdepth = fmax(waterdepth - absPos.z, 0.0f);
			const float pressure = 9.81e3f*localdepth;
			eulerVel.w = RHO(pressure, PART_FLUID_NUM(info));
#elif SPECIFIC_PROBLEM == IOWithoutWalls
			if (object(info)==1)
				eulerVel.w = 1002.0f;
			else
				eulerVel.w = 1002.0f;
				//eulerVel.w = 1000.0f;
#else
			eulerVel.w = 1000.0f;
#endif
		}

		// impose tangential velocity
		if (VEL_IO(info)) {
			eulerVel.y = 0.0f;
#if SPECIFIC_PROBLEM != PeriodicWave
			eulerVel.z = 0.0f;
#endif
#if SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
			// k and eps based on Versteeg & Malalasekera (2001)
			// turbulent intensity (between 1% and 6%)
			const float Ti = 0.01f;
			// in case of a pressure inlet eulerVel.x = 0 so we set u to 1 to multiply it later once
			// we know the correct velocity
			const float u = eulerVel.x > 1e-6f ? eulerVel.x : 1.0f;
			tke = 3.0f/2.0f*(u*Ti)*(u*Ti);
			tke = 3.33333f;
			// length scale of the flow
			const float L = 1.0f;
			// constant is C_\mu^(3/4)/0.07*sqrt(3/2)
			// formula is epsilon = C_\mu^(3/4) k^(3/2)/(0.07 L)
			eps = 2.874944542f*tke*u*Ti/L;
			eps = 1.0f/0.41f/fmax(1.0f-fabs(absPos.z),0.025f);
#endif
		}
	}

}

__global__ void
InputProblem_imposeBoundaryConditionDevice(
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
			// when pressure outlets require the water depth compute it from the IOwaterdepth integer
			float waterdepth = 0.0f;
			if (!VEL_IO(info) && IOwaterdepth) {
				waterdepth = ((float)IOwaterdepth[object(info)-1])/((float)UINT_MAX); // now between 0 and 1
				waterdepth *= d_cellSize.z*d_gridSize.z; // now between 0 and world size
				waterdepth += d_worldOrigin.z; // now absolute z position
			}
			// this now calls the virtual function that is problem specific
			InputProblem_imposeBoundaryCondition(info, absPos, waterdepth, t, vel, eulerVel, tke, eps);
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

} // end of cuInputProblem namespace

extern "C"
{

void
InputProblem::setboundconstants(
	const	PhysParams	*physparams,
	float3	const&		worldOrigin,
	uint3	const&		gridSize,
	float3	const&		cellSize)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_worldOrigin, &worldOrigin, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_cellSize, &cellSize, sizeof(float3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_gridSize, &gridSize, sizeof(uint3)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_rho0, &physparams->rho0, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_bcoeff, &physparams->bcoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_gammacoeff, &physparams->gammacoeff, MAX_FLUID_TYPES*sizeof(float)));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuInputProblem::d_sscoeff, &physparams->sscoeff, MAX_FLUID_TYPES*sizeof(float)));

}

}

void
InputProblem::imposeBoundaryConditionHost(
			float4*			newVel,
			float4*			newEulerVel,
			float*			newTke,
			float*			newEpsilon,
	const	particleinfo*	info,
	const	float4*			oldPos,
			uint			*IOwaterdepth,
	const	float			t,
	const	uint			numParticles,
	const	uint			numOpenBoundaries,
	const	uint			particleRangeEnd,
	const	hashKey*		particleHash)
{
	uint numThreads = min(BLOCK_SIZE_IOBOUND, particleRangeEnd);
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	cuInputProblem::InputProblem_imposeBoundaryConditionDevice<<< numBlocks, numThreads, dummy_shared >>>
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
	CUT_CHECK_ERROR("imposeBoundaryCondition kernel execution failed");
}
