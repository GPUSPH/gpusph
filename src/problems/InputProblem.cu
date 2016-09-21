#include <string>
#include <iostream>

#include "InputProblem.h"
#include "GlobalData.h"
#include "cudasimframework.cu"
#include "textures.cuh"
#include "utils.h"

#define USE_PLANES 0

InputProblem::InputProblem(GlobalData *_gdata) : XProblem(_gdata)
{
	// Error catcher for SPECIFIC_PROBLEM definition
	// If the value is not defined properly this will throw a compile error
	int i = SPECIFIC_PROBLEM;

	//Box (Dambreak)
	//*************************************************************************************
#if SPECIFIC_PROBLEM == BoxCorner || SPECIFIC_PROBLEM == Box

		SETUP_FRAMEWORK(
			viscosity<DYNAMICVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_NONE>,
			kernel<WENDLAND>,
			add_flags<ENABLE_FERRARI | ENABLE_DENSITY_SUM>
		);

		set_deltap(0.125f);

		physparams()->gravity = make_float3(0.0, 0.0, -9.81f);

		simparams()->tend = 5.0;
		addPostProcess(SURFACE_DETECTION);
		addPostProcess(TESTPOINTS);
		H = 1.0;
		l = 2.2; w = 2.2; h = 2.2;
		m_origin = make_double3(-1.1, -1.1, -1.1);
		//simparams()->ferrariLengthScale = 1.0f;
		simparams()->ferrari = 1.0f;
		size_t water = add_fluid(1000.0);
		set_equation_of_state(water,  7.0f, 60.f);
		set_kinematic_visc(water, 1.0e-2f);
		addPostProcess(CALC_PRIVATE);
		// Building the geometry
#if SPECIFIC_PROBLEM == BoxCorner
			addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.box_corner.fluid.h5sph", NULL);
			addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.box_corner.boundary.kent0.h5sph", NULL);
#else
			addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.box_blend_16.fluid.h5sph", NULL);
			addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.box_blend_16.boundary.kent0.h5sph", NULL);
			// Add gage
			add_gage(m_origin.x + 1.0 + 0.1, m_origin.y + 1.8 + 0.1, m_origin.z + 0.1);
			// add testpoints and dump them separately
			addPostProcess(TESTPOINTS);
			addTestPoint(m_origin + make_double3(1.0, 2.0, 0.0) + make_double3(0.1, 0.1, 0.1));
#endif
	//**********************************************************************

	//SmallChannelFlow (a small channel flow for debugging viscosity)
	//**********************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlow
		addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.small_channel.fluid.h5sph", NULL);
		addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel.boundary.kent0.h5sph", NULL);

		SETUP_FRAMEWORK(
			viscosity<DYNAMICVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_XY>,
			kernel<WENDLAND>,
			add_flags<ENABLE_FERRARI>
		);

		set_deltap(0.0625f);
		simparams()->tend = 100.0;
		simparams()->ferrari = 1.0f;

		size_t water = add_fluid(1000.0f);
		set_equation_of_state(water, 7.0f, 10.0f);
		set_kinematic_visc(water, 1.0e-2f);

		H = 1.0;
		l = 1.0; w = 1.0; h = 1.02;
		m_origin = make_double3(-0.5, -0.5, -0.51);
		physparams()->gravity = make_float3(8.0*physparams()->kinematicvisc[water], 0.0, 0.0);
	//*************************************************************************************

	//SmallChannelFlowKEPS (a small channel flow for debugging the k-epsilon model)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowKEPS
		addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.small_channel_keps.fluid.h5sph", NULL);
		addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_keps.boundary.kent0.h5sph", NULL);

		SETUP_FRAMEWORK(
			viscosity<KEPSVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_XY>,
			kernel<WENDLAND>,
			add_flags<ENABLE_FERRARI>
		);

		simparams()->sfactor=2.0f;
		set_deltap(0.05f);
		simparams()->tend = 100.0;
		simparams()->ferrariLengthScale = 1.0f;

		// turbulent (as in Leroy[2014], JCP)
		size_t water = add_fluid(1000.0f);
		set_equation_of_state(water, 7.0f, 200.0f);
		set_kinematic_visc(water, 1.5625e-3f);

		addPostProcess(SURFACE_DETECTION);
		addPostProcess(TESTPOINTS);

		H = 2.0;
		l = 0.8; w = 0.8; h = 2.02;
		m_origin = make_double3(-0.4, -0.4, -1.01);
		physparams()->gravity = make_float3(1.0, 0.0, 0.0);

		// Setting test points for channel flow keps test case
		addPostProcess(TESTPOINTS);
		// create test points at (0,0,.) with dp spacing from bottom to top
		for(uint i=0; i<=40; i++)
			addTestPoint(m_origin + make_double3(0.4, 0.4, 0.05*(float)i) + make_double3(0.0, 0.0, 0.01));

	//**********************************************************************

	//SmallChannelFlowIO (a small channel flow for debugging in/outflow)
	//**********************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowIO
		GeometryID fluid = addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_walls.fluid.h5sph", NULL);
		GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_walls.boundary.kent0.h5sph", NULL);
		GeometryID inlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_walls.boundary.kent1.h5sph", NULL);
		setVelocityDriven(inlet, VELOCITY_DRIVEN);
		GeometryID outlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_walls.boundary.kent2.h5sph", NULL);
		setVelocityDriven(outlet, PRESSURE_DRIVEN);

		SETUP_FRAMEWORK(
			viscosity<DYNAMICVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_Y>,
			kernel<WENDLAND>,
			add_flags<ENABLE_FERRARI | ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM>
		);

		set_deltap(0.2f);
		simparams()->maxneibsnum = 220;
		simparams()->tend = 100.0;
		simparams()->ferrariLengthScale = 1.0f;
		simparams()->numOpenBoundaries = 2;

		size_t water = add_fluid(1000.0f);
		set_equation_of_state(water, 7.0f, 10.0f);
		set_kinematic_visc(water, 1.0e-2f);

		H = 2.0;
		l = 2.1; w = 2.1; h = 2.1;
		m_origin = make_double3(-1.05, -1.05, -1.05);
		physparams()->gravity = make_float3(0.0, 0.0, 0.0);
	//*************************************************************************************

	//SmallChannelFlowIOPer (a small channel flow for debugging in/outflow with periodicity)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPer || \
      SPECIFIC_PROBLEM == SmallChannelFlowIOPerOpen
		GeometryID fluid = addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_2d_per.fluid.h5sph", NULL);
		GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_2d_per.boundary.kent0.h5sph", NULL);
		GeometryID inlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_2d_per.boundary.kent1.h5sph", NULL);
#if SPECIFIC_PROBLEM == SmallChannelFlowIOPerOpen
		setVelocityDriven(inlet, PRESSURE_DRIVEN);
#else
		setVelocityDriven(inlet, VELOCITY_DRIVEN);
#endif
		GeometryID outlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_2d_per.boundary.kent2.h5sph", NULL);
		setVelocityDriven(outlet, PRESSURE_DRIVEN);

		SETUP_FRAMEWORK(
			viscosity<DYNAMICVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_Y>,
			kernel<WENDLAND>,
#if SPECIFIC_PROBLEM == SmallChannelFlowIOPerOpen
			add_flags<ENABLE_FERRARI | ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM | ENABLE_WATER_DEPTH>
#else
			add_flags<ENABLE_FERRARI | ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM>
#endif
		);

		simparams()->sfactor=1.3f;
		set_deltap(0.05f);
		simparams()->tend = 10.0;
		simparams()->ferrari = 1.0f;
		simparams()->numOpenBoundaries = 2;

		size_t water = add_fluid(1000.0f);
		set_equation_of_state(water, 7.0f, 10.0f);
#if SPECIFIC_PROBLEM == SmallChannelFlowIOPerOpen
		set_equation_of_state(water, 7.0f, 50.0f);
#endif
		set_kinematic_visc(water, 1.0e-1f);

		H = 2.0;
		l = 1.1; w = 1.0; h = 2.1;
		m_origin = make_double3(-0.55, -0.5, -1.05);
		physparams()->gravity = make_float3(0.0, 0.0, 0.0f);
#if SPECIFIC_PROBLEM == SmallChannelFlowIOPerOpen
		physparams()->gravity = make_float3(1.0, 0.0, -9.759f);
#endif
	//*************************************************************************************

	//SmallChannelFlowIOKeps (a small channel flow for debugging in/outflow with keps)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
		GeometryID fluid = addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_2d_per.fluid.h5sph", NULL);
		GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_2d_per.boundary.kent0.h5sph", NULL);
		GeometryID inlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_2d_per.boundary.kent1.h5sph", NULL);
		setVelocityDriven(inlet, VELOCITY_DRIVEN);
		GeometryID outlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.small_channel_io_2d_per.boundary.kent2.h5sph", NULL);
		setVelocityDriven(outlet, PRESSURE_DRIVEN);

		SETUP_FRAMEWORK(
			viscosity<KEPSVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_Y>,
			kernel<WENDLAND>,
			add_flags<ENABLE_FERRARI | ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM>
		);

		simparams()->sfactor=1.3f;
		set_deltap(0.05f);
		simparams()->tend = 10.0;
		simparams()->ferrariLengthScale = 1.0f;
		simparams()->numOpenBoundaries = 2;

		size_t water = add_fluid(1000.0f);
		set_equation_of_state(water, 7.0f, 200.0f);
		set_kinematic_visc(water, 1.5625e-3f);

		H = 2.0;
		l = 1.1; w = 1.0; h = 2.1;
		m_origin = make_double3(-0.55, -0.5, -1.05);
		physparams()->gravity = make_float3(0.0, 0.0, 0.0);

		// Setting test points for channel flow keps test case
		addPostProcess(TESTPOINTS);
		// create test points at (0,0,.) with dp spacing from bottom to top
		for(uint i=0; i<=40; i++)
			addTestPoint(m_origin + make_double3(0.4, 0.4, 0.05*(float)i) + make_double3(0.0, 0.0, 0.01));

	//*************************************************************************************

	//IOWithoutWalls (i/o between two plates without walls)
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == IOWithoutWalls
		GeometryID fluid = addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.io_without_walls.fluid.h5sph", NULL);
		GeometryID inlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.io_without_walls.boundary.kent0.h5sph", NULL);
		setVelocityDriven(inlet, PRESSURE_DRIVEN);
		GeometryID outlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.io_without_walls.boundary.kent1.h5sph", NULL);
		setVelocityDriven(outlet, PRESSURE_DRIVEN);

		SETUP_FRAMEWORK(
			viscosity<DYNAMICVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_YZ>,
			kernel<WENDLAND>,
			add_flags<ENABLE_FERRARI | ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM>
		);

		set_deltap(0.2f);
		simparams()->tend = 100.0;
		simparams()->ferrari = 1.0f;
		simparams()->numOpenBoundaries = 2;

		size_t water = add_fluid(1000.0f);
		set_equation_of_state(water, 7.0f, 10.0f);
		set_kinematic_visc(water, 1.0e-2f);

		H = 2.0;
		l = 2.2; w = 2.0; h = 2.0;
		m_origin = make_double3(-1.1, -1.0, -1.0);
		physparams()->gravity = make_float3(0.0, 0.0, 0.0);
	//*************************************************************************************
	// Solitary Wave with IO
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == SolitaryWave
		GeometryID fluid = addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.solitaryWave.fluid.h5sph", NULL);
		GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.solitaryWave.boundary.kent0.h5sph", NULL);
		GeometryID outlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.solitaryWave.boundary.kent1.h5sph", NULL);
		setVelocityDriven(outlet, PRESSURE_DRIVEN);
		GeometryID inlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.solitaryWave.boundary.kent2.h5sph", NULL);
		setVelocityDriven(inlet, VELOCITY_DRIVEN);

		SETUP_FRAMEWORK(
			viscosity<DYNAMICVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_NONE>,
			kernel<WENDLAND>,
			add_flags<ENABLE_FERRARI | ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM | ENABLE_WATER_DEPTH>
		);

		//set_deltap(0.026460);
		set_deltap(0.017637f);
		simparams()->maxneibsnum = 512;
		simparams()->tend = 7.0;
		simparams()->ferrari = 1.0f;
		simparams()->numOpenBoundaries = 2;

		size_t water = add_fluid(1000.0f);
		set_equation_of_state(water, 7.0f, 20.0f);
		//set_equation_of_state(water, 7.0f, 45.0f);
		set_kinematic_visc(water, 1.0e-6f);

		addPostProcess(SURFACE_DETECTION);

		H = 0.6;
		//l = 3.2; w=1.7; h=1.4;
		//m_origin = make_double3(-3.85, -0.1, -0.1);
		l = 7.7; w=3.4; h=1.4;
		m_origin = make_double3(-3.85, -1.7, -0.1);
		physparams()->gravity = make_float3(0.0, 0.0, -9.81);
	//*************************************************************************************

	//Periodic wave with IO
	//*************************************************************************************
#elif SPECIFIC_PROBLEM == PeriodicWave
		GeometryID fluid = addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/InputProblem/0.periodic_wave_0.02.fluid.h5sph", NULL);
		GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.pariodic_wave_0.02.boundary.kent0.h5sph", NULL);
		GeometryID inlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.periodic_wave_0.02.boundary.kent1.h5sph", NULL);
		setVelocityDriven(inlet, VELOCITY_DRIVEN);
		GeometryID outlet = addHDF5File(GT_OPEN_BOUNDARY, Point(0,0,0), "./data_files/InputProblem/0.periodic_wave_0.02.boundary.kent2.h5sph", NULL);
		setVelocityDriven(outlet, VELOCITY_DRIVEN);

		SETUP_FRAMEWORK(
			viscosity<DYNAMICVISC>,
			boundary<SA_BOUNDARY>,
			periodicity<PERIODIC_Y>,
			kernel<WENDLAND>,
			add_flags<ENABLE_FERRARI | ENABLE_INLET_OUTLET | ENABLE_DENSITY_SUM>
		);

		simparams()->sfactor=2.0f;
		set_deltap(0.02f);
		simparams()->maxneibsnum = 440;
		simparams()->tend = 10.0;
		simparams()->ferrari = 0.1f;
		simparams()->numOpenBoundaries = 2;

		addPostProcess(SURFACE_DETECTION);
		addPostProcess(CALC_PRIVATE);

		size_t water = add_fluid(1000.0f);
		set_equation_of_state(water, 7.0f, 25.0f);
		set_kinematic_visc(water, 1.0e-6f);

		H = 0.5;
		l = 2.7; w = 0.5; h = 1.2;
		m_origin = make_double3(-1.35, -0.25, -0.1);
		physparams()->gravity = make_float3(0.0, 0.0, -9.81);

		// Setting probe
		add_gage(0.0, 0.0, 0.2);

	//*************************************************************************************

#endif

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
	m_name = "InputProblem";
}

// Custom initialization
	void
InputProblem::initializeParticles(BufferList &buffers, const uint numParticles)
{
	printf("Custom initialization...\n");

	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos = buffers.getData<BUFFER_POS_GLOBAL>();
	float4 *eulerVel = buffers.getData<BUFFER_EULERVEL>();
	float *k = buffers.getData<BUFFER_TKE>();
	float *epsilon = buffers.getData<BUFFER_EPSILON>();

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

	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		// Initialize turbulent quantities
		const float k0 = 1.0f/sqrtf(0.09f);
		if (k && epsilon) {
		k[i] = k0;
		epsilon[i] = 1.0f/0.41f/fmaxf(1.0f-fabsf(pos[i].z),0.5f*(float)m_deltap);
		}
		// Initialize fluid particles
		if (FLUID(info[i])) {
			float rho = physparams()->rho0[0];
#if SPECIFIC_PROBLEM == SmallChannelFlowKEPS || \
			SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
			const float lvel = log(fmaxf(1.0f-fabsf(pos[i].z), 0.5*m_deltap)/0.0015625f)/0.41f+5.2f;
			vel[i] = make_float4(lvel, 0, 0, physparams()->rho0[0]);
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPer
			const float lvel = 1.0f-pos[i].z*pos[i].z;
			vel[i] = make_float4(lvel, 0.0f, 0.0f, physparams()->rho0[0]);
#elif SPECIFIC_PROBLEM == SmallChannelFlowIO
			const float y2 = pos[i].y*pos[i].y;
			const float z2 = pos[i].z*pos[i].z;
			const float y4 = y2*y2;
			const float z4 = z2*z2;
			const float y6 = y2*y4;
			const float z6 = z2*z4;
			const float y8 = y4*y4;
			const float z8 = z4*z4;
			const float lvel = (461.0f+y8-392.0f*z2-28.0f*y6*z2-70.0f*z4+z8+70.0f*y4*(z4-1.0f)-28.0f*y2*(14.0f-15.0f*z2+z6))/461.0f;
			vel[i] = make_float4(lvel, 0, 0, physparams()->rho0[0]);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
			vel[i] = make_float4(1.0f, 0.0f, 0.0f, (physparams()->rho0[0]+2.0f));//+1.0f-1.0f*pos[i].x));
#elif SPECIFIC_PROBLEM == PeriodicWave
			const float x = pos[i].x;
			const float z = pos[i].z;
			const float eta = A*cos(k*x+phi);
			const float h = D + eta;
			//const float p = rho*9.807f*(z-h) + cosh(k*z)/cosh(k*D)*rho*9.807f*eta;
			const float p = rho*-9.807f*(z-h) + cosh(k*z)/cosh(k*D)*rho*9.807f*eta;
			const float _rho = powf(p/(625.0f*rho/7.0f) + 1.0f, 1.0f/7.0f)*rho;
			const float u = A*omega*cosh(k*z)/sinh(k*D)*cos(k*x+phi);
			const float w = A*omega*sinh(k*z)/sinh(k*D)*sin(k*x+phi);
			vel[i] = make_float4(u, 0, w, _rho);
#elif SPECIFIC_PROBLEM == SolitaryWave
			vel[i] = make_float4(0, 0, 0, powf(((0.58212-pos[i].z)*9.807f*physparams()->rho0[0])*7.0f/physparams()->sscoeff[0]/physparams()->sscoeff[0]/physparams()->rho0[0] + 1.0f,1.0f/7.0f)*physparams()->rho0[0]);
#else
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
#endif
			// Fluid particles don't have a eulerian velocity
			if (eulerVel)
				eulerVel[i] = make_float4(0.0f);
			// Initialize vertex particles
		} else if (VERTEX(info[i])) {
			float rho = density(H - pos[i].z, 0);
#if SPECIFIC_PROBLEM == SmallChannelFlowKEPS || \
			SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
			const float lvel = log(fmaxf(1.0f-fabsf(pos[i].z), 0.5*m_deltap)/0.0015625f)/0.41f+5.2f;
			vel[i] = make_float4(0.0f, 0.0f, 0.0f, physparams()->rho0[0]);
			eulerVel[i] = make_float4(lvel, 0.0f, 0.0f, physparams()->rho0[0]);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]+2.0f);
#else
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
			if (eulerVel)
				eulerVel[i] = vel[i];
#endif
			// Initialize boundary particles
		} else if (BOUNDARY(info[i])) {
#if SPECIFIC_PROBLEM == SmallChannelFlowKEPS || \
			SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
			const float lvel = log(fmaxf(1.0f-fabsf(pos[i].z), 0.5*m_deltap)/0.0015625f)/0.41f+5.2f;
			vel[i] = make_float4(0.0f, 0.0f, 0.0f, physparams()->rho0[0]);
			eulerVel[i] = make_float4(lvel, 0.0f, 0.0f, physparams()->rho0[0]);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]+2.0f);
#else
			vel[i] = make_float4(0, 0, 0, physparams()->rho0[0]);
			if (eulerVel)
				eulerVel[i] = vel[i];
#endif
		}
	}
}

uint
InputProblem::max_parts(uint numpart)
{
	// gives an estimate for the maximum number of particles
#if SPECIFIC_PROBLEM == SmallChannelFlowIO || \
    SPECIFIC_PROBLEM == IOWithoutWalls || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOPer || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOPerOpen || \
    SPECIFIC_PROBLEM == SmallChannelFlowIOKeps || \
    SPECIFIC_PROBLEM == PeriodicWave
		return (uint)((float)numpart*1.2f);
#elif SPECIFIC_PROBLEM == SolitaryWave
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
using namespace cubounds;
using namespace cuforces;

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
			eulerVel.x = fmaxf(eulerVel.x, 0.0f);
#elif SPECIFIC_PROBLEM == IOWithoutWalls
			eulerVel.x = 1.0f;
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPer
			eulerVel.x = 1.0f-absPos.z*absPos.z;
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOKeps
			// the 0.025 is deltap*0.5 = 0.05*0.5
			eulerVel.x = log(fmaxf(1.0f-fabsf(absPos.z), 0.025f)/0.0015625f)/0.41f+5.2f;
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
#elif SPECIFIC_PROBLEM == SolitaryWave
			const float d = 0.6;
			const float a = 0.5*d;
			const float g = 9.81;
			const float theta = atan(1.);
			const float3 normalWave = make_float3(cos(theta), sin(theta), 0);
			const float c = sqrt(g*(d + a));
			const float k = sqrt(0.75*a/d)/d;
			const float xMin = -2.5*1.5;
			const float x0 = xMin - 4./k;

			//const float kxct = k*(dot(normalWave, absPos)-c*(t-2.0f)-x0);
			const float kxct = k*(dot(normalWave, absPos)-c*t-x0);
			const float eta = a/(cosh(kxct)*cosh(kxct));
			const float detadt = 2*a*k*c*tanh(kxct)/(cosh(kxct)*cosh(kxct));

			const float h = d + eta;
			if (absPos.z < h) {
				const float u = (c*eta/h)*normalWave.x;
				const float v = (c*eta/h)*normalWave.y;
				const float w = absPos.z*detadt/h;
				eulerVel.x = u;
				eulerVel.y = v;
				eulerVel.z = w;
			}
			/*
			if (t < 2.0f) {
				eulerVel.x = 0.0f;
				eulerVel.y = 0.0f;
				eulerVel.z = 0.0f;
			}
			*/
#else
			eulerVel.x = 0.0f;
#endif
		}
		else {
#if SPECIFIC_PROBLEM == IOWithoutWalls
			if (object(info)==0)
				eulerVel.w = 1002.0f;
			else
				eulerVel.w = 1002.0f;
				//eulerVel.w = 1000.0f;
#elif SPECIFIC_PROBLEM == SmallChannelFlowIOPerOpen
			if (object(info)==0)
				waterdepth = 0.0f;
			const float localdepth = fmaxf(waterdepth - absPos.z, 0.0f);
			const float pressure = 9.81e3f*localdepth;
			eulerVel.w = RHO(pressure, fluid_num(info));
#elif SPECIFIC_PROBLEM == SolitaryWave
			waterdepth = 0.6 - 0.5*0.0195;
			const float localdepth = fmaxf(waterdepth - absPos.z, 0.0f);
			const float pressure = 9.81e3f*localdepth;
			eulerVel.w = RHO(pressure, fluid_num(info));
#else
			eulerVel.w = 1000.0f;
#endif
		}
		// impose tangential velocity
		if (VEL_IO(info)) {
#if SPECIFIC_PROBLEM != SolitaryWave
			eulerVel.y = 0.0f;
#endif
#if SPECIFIC_PROBLEM != PeriodicWave && \
    SPECIFIC_PROBLEM != SolitaryWave
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
			eps = 1.0f/0.41f/fmaxf(1.0f-fabsf(absPos.z),0.025f);
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

	// Default value for eulerVel
	// Note that this default value needs to be physically feasible, as it is used in case of boundary elements
	// without fluid particles in their support. It is also possible to use this default value to impose tangential
	// velocities for pressure outlets.
	float4 eulerVel = make_float4(0.0f);	// imposed velocity/pressure for open boundaries
	float4 vel = make_float4(0.0f);			// imposed velocity for moving objects
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

void
InputProblem::imposeBoundaryConditionHost(
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
	KERNEL_CHECK_ERROR;
}
