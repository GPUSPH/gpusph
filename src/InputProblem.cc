#include <math.h>
#include <string>
#include <iostream>

#include "InputProblem.h"
#include "HDF5SphReader.h"
#include "GlobalData.h"

static const std::string SPECIFIC_PROBLEM("SmallChannelFlow");

/* Implemented problems:
 *
 *	Keyword			Description
 ***********************************************
 *	StillWater			Periodic stillwater (lacking file)
 *	Spheric2			Spheric2 dambreak with obstacle
 *	Box					Small dambreak in a box
 *	BoxCorner			Small dambreak in a box with a corner
 *	SmallChannelFlow	Small channel flow for debugging
 *
 */

#define USE_PLANES 0

InputProblem::InputProblem(const GlobalData *_gdata) : Problem(_gdata)
{
	numparticles = 0;

	//StillWater periodic (symmetric)
	//*************************************************************************************
	if (SPECIFIC_PROBLEM == "StillWater") {
		inputfile = "/home/vorobyev/Crixus/geometries/plane_periodicity/0.plane_0.1_sym.h5sph";

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
		m_origin = make_double3(0.0, 0.0, 0.0);
		m_physparams.set_density(0, 1000.0, 7.0f, 20.0f);
	}
	//*************************************************************************************

	//Spheric2 (DamBreak)
	//*************************************************************************************
	else if (SPECIFIC_PROBLEM == "Spheric2") {
		inputfile = "/home/arnom/work/post-doc-2013/crixus/crixus-build/geometries/140311-spheric2/0.spheric2.h5sph";

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
	}
	//*************************************************************************************

	//Box (Dambreak)
	//*************************************************************************************
	else if (SPECIFIC_PROBLEM.substr(0,3) == "Box") {
		if (SPECIFIC_PROBLEM == "BoxCorner")
			inputfile = "/home/arnom/work/post-doc-2013/crixus/crixus-build/geometries/111116-box/box-corner/0.box_corner.h5sph";
		else
			inputfile = "/home/arnom/work/post-doc-2013/crixus/crixus-build/geometries/111116-box/0.box_blend_16.h5sph";

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
	}
	//*************************************************************************************

	//SmallChannelFlow (a small channel flow for debugging viscosity and k-epsilon)
	//*************************************************************************************
	else if (SPECIFIC_PROBLEM == "SmallChannelFlow") {
		inputfile = "/home/arnom/work/post-doc-2013/crixus/crixus-build/geometries/140109-small-channel/0.small_channel.h5sph";

		set_deltap(0.0625f);

		// laminar
		//m_physparams.kinematicvisc = 1.0e-2f;
		//m_simparams.visctype = DYNAMICVISC;
		//m_physparams.gravity = make_float3(8.0*m_physparams.kinematicvisc, 0.0, 0.0);
		//m_physparams.set_density(0, 1000.0, 7.0f, 10.0f);

		// turbulent (as in agnes' paper)
		m_physparams.kinematicvisc = 1.5625e-3f;
		m_simparams.visctype = KEPSVISC;
		m_physparams.gravity = make_float3(1.0, 0.0, 0.0);
		m_physparams.set_density(0, 1000.0, 7.0f, 200.0f);

		m_simparams.tend = 100.0;
		m_simparams.periodicbound = PERIODIC_XY;
		m_simparams.testpoints = false;
		m_simparams.surfaceparticle = false;
		m_simparams.savenormals = false;
		H = 1.0;
		l = 1.0; w = 1.0; h = 1.02;
		m_origin = make_double3(-0.5, -0.5, -0.51);
		m_simparams.calcPrivate = true;
	}
	//*************************************************************************************
	// Fishpass
	//*************************************************************************************
	// Poitier geometry
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	inputfile = "/home/vorobyev/Crixus/geometries/fishpass3D/wrong.fishpass_covered_0.0075_sl10.h5sph";

//	set_deltap(0.0075f);

//	m_simparams.testpoints = false;
//	H = 0.2;
//	l = 0.75; w = 0.675; h = 0.4;

//	float slope = 0.1;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	// BAW geometry
//	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	inputfile = "/home/vorobyev/Crixus/geometries/fishpass3D/0.BAW.fishpass.0.01.h5sph";

//	set_deltap(0.01f);

//	m_simparams.testpoints = false;
//	H = 0.25;
//	l = 1.019; w = 0.785; h = 0.4;

//	float slope = 0.027;
//	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//	m_physparams.kinematicvisc = 1.0e-6f;
//	m_physparams.gravity = make_float3(9.81f*sin(atan(slope)), 0.0, -9.81f*cos(atan(slope)));

//	//periodic boundaries
//	m_simparams.periodicbound = PERIODIC_X;
//	//*************************************************************************************

//	// Poiseuille flow
//	//*************************************************************************************
//	inputfile = "/home/vorobyev/Crixus/geometries/2planes_periodicity/0.2planes_0.02.h5sph";
//
//	set_deltap(0.02f);
//
//	m_simparams.testpoints = false;
//	H = 1.0;
//	l = 0.26; w = 0.26; h = 1.0;
//
//	m_physparams.kinematicvisc = 0.1f;
//	m_physparams.gravity = make_float3(0.8, 0.0, 0.0);		// laminar
//
//	//m_physparams.kinematicvisc = 0.00078125f;
//	//m_physparams.gravity = make_float3(2.0, 0.0, 0.0);	// turbulent
//
//	//periodic boundaries
//	m_simparams.periodicbound = PERIODIC_XY;
//	//*************************************************************************************

	// SPH parameters
	m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
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
	add_writer(VTKWRITER, 0.1);

	// Name of problem used for directory creation
	m_name = "InputProblem";
}


int InputProblem::fill_parts()
{
	std::cout << std::endl << "Reading particle data from the input:" << std::endl << inputfile << std::endl;
	const char *ch_inputfile = inputfile.c_str();

	// Setting probes for Spheric2 test case
	//*******************************************************************
	if (SPECIFIC_PROBLEM == "Box") {
		add_gage(m_origin + make_double3(1.0, 1.8, 0.0) + make_double3(0.1, 0.1, 0.1));
		if (m_simparams.testpoints) {
			test_points.push_back(m_origin + make_double3(1.0, 2.0, 0.0) + make_double3(0.1, 0.1, 0.1));
		}
	}
	// Setting probes for Spheric2 test case
	//*******************************************************************
	if (SPECIFIC_PROBLEM == "Spheric2") {
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
	}
	//*******************************************************************

	int npart = HDF5SphReader::getNParts(ch_inputfile) + test_points.size();

	return npart;
}

void InputProblem::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	vertexinfo *vertices = buffers.getData<BUFFER_VERTICES>();
	float4 *boundelm = buffers.getData<BUFFER_BOUNDELEMENTS>();

	const char *ch_inputfile = inputfile.c_str();
	uint npart = HDF5SphReader::getNParts(ch_inputfile);

	HDF5SphReader::ReadParticles *buf = new HDF5SphReader::ReadParticles[npart];
	HDF5SphReader::readParticles(buf, ch_inputfile, npart);

	uint n_parts = 0;
	uint n_vparts = 0;
	uint n_bparts = 0;

	for (uint i = 0; i<npart; i++) {
		switch(buf[i].ParticleType) {
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
		//float rho = density(H - buf[i].Coords_2, 0);
		float rho = m_physparams.rho0[0];
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
		calc_localpos_and_hash(Point(buf[i].Coords_0, buf[i].Coords_1, buf[i].Coords_2, rho*buf[i].Volume), info[i], pos[i], hash[i]);
	}
	uint j = n_parts;
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	if(n_vparts) {
		std::cout << "Vertex parts: " << n_vparts << "\n";
		for (uint i = j; i < j + n_vparts; i++) {
			float rho = density(H - buf[i].Coords_2, 0);
			vel[i] = make_float4(0, 0, 0, rho);
			info[i] = make_particleinfo(VERTEXPART, 0, i);
			calc_localpos_and_hash(Point(buf[i].Coords_0, buf[i].Coords_1, buf[i].Coords_2, rho*buf[i].Volume), info[i], pos[i], hash[i]);
		}
		j += n_vparts;
		std::cout << "Vertex part mass: " << pos[j-1].w << "\n";
	}

	if(n_bparts) {
		std::cout << "Boundary parts: " << n_bparts << "\n";
		for (uint i = j; i < j + n_bparts; i++) {
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			info[i] = make_particleinfo(BOUNDPART, 0, i);
			calc_localpos_and_hash(Point(buf[i].Coords_0, buf[i].Coords_1, buf[i].Coords_2, 0.0), info[i], pos[i], hash[i]);
			vertices[i].x = buf[i].VertexParticle1;
			vertices[i].y = buf[i].VertexParticle2;
			vertices[i].z = buf[i].VertexParticle3;
			boundelm[i].x = buf[i].Normal_0;
			boundelm[i].y = buf[i].Normal_1;
			boundelm[i].z = buf[i].Normal_2;
			boundelm[i].w = buf[i].Surface;
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

	delete [] buf;
}
