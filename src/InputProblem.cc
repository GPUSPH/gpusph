
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif
#include <math.h>
#include <iostream>

#include "InputProblem.h"
#include "HDF5SphReader.h"


#define USE_PLANES 0

InputProblem::InputProblem(const Options &options) : Problem(options)
{
	numparticles = 0;

	//StillWater periodic (symmetric)
	//*************************************************************************************
//	inputfile = "/home/vorobyev/Crixus/geometries/plane_periodicity/0.plane_0.1_sym.h5sph";

//	set_deltap(0.1f);

//	n_probeparts = 0;
//	H = 2.0;
//	l = 2.0; w = 2.0; h = 2.2;

//	m_physparams.kinematicvisc = 3.0e-2f;
//	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);

//	//periodic boundaries
//	m_simparams.periodicbound = true;
//	m_physparams.dispvect = make_float3(l, l, 0.0);
//	m_physparams.minlimit = make_float3(0.0f, 0.0f, 0.0f);
//	m_physparams.maxlimit = make_float3(l, l, 0.0f);
	//*************************************************************************************

	//Spheric 2 (DamBreak)
	//*************************************************************************************
//	inputfile = "/home/vorobyev/Crixus/geometries/spheric2/0.spheric2-dr-0.01833-dp-0.02.h5sph";

//	set_deltap(0.02f);

//	m_physparams.kinematicvisc = 1.0e-6f;
//	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);

//	n_probeparts = 208;
//	H = 0.55;
//	l = 3.5; w = 1.0; h = 1.0;
	//*************************************************************************************

	// Fishpass
	//*************************************************************************************
	// Poitier geometry
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	inputfile = "/home/vorobyev/Crixus/geometries/fishpass3D/wrong.fishpass_covered_0.0075_sl10.h5sph";

//	set_deltap(0.0075f);

//	n_probeparts = 0;
//	H = 0.2;
//	l = 0.75; w = 0.675; h = 0.4;

//	float slope = 0.1;
	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	// BAW geometry
//	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//	inputfile = "/home/vorobyev/Crixus/geometries/fishpass3D/0.BAW.fishpass.0.01.h5sph";

//	set_deltap(0.01f);

//	n_probeparts = 0;
//	H = 0.25;
//	l = 1.019; w = 0.785; h = 0.4;

//	float slope = 0.027;
//	//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

//	m_physparams.kinematicvisc = 1.0e-6f;
//	m_physparams.gravity = make_float3(9.81f*sin(atan(slope)), 0.0, -9.81f*cos(atan(slope)));

//	//periodic boundaries
//	m_simparams.periodicbound = true;
//	m_physparams.dispvect = make_float3(l, 0.0f, 0.0f);
//	m_physparams.minlimit = make_float3(0.0f, 0.0f, 0.0f);
//	m_physparams.maxlimit = make_float3(l, 0.0f, 0.0f);
//	//*************************************************************************************

	// Poiseuille flow
	//*************************************************************************************
	inputfile = "/home/vorobyev/Crixus/geometries/2planes_periodicity/0.2planes_0.02.h5sph";

	set_deltap(0.02f);

	n_probeparts = 0;
	H = 1.0;
	l = 0.26; w = 0.26; h = 1.0;

	m_physparams.kinematicvisc = 0.1f;
	m_physparams.gravity = make_float3(0.8, 0.0, 0.0);		// laminar

	//m_physparams.kinematicvisc = 0.00078125f;
	//m_physparams.gravity = make_float3(2.0, 0.0, 0.0);	// turbulent

	//periodic boundaries
	m_simparams.periodicbound = true;
	m_physparams.dispvect = make_float3(l, w, 0.0f);
	m_physparams.minlimit = make_float3(0.0f, 0.0f, 0.0f);
	m_physparams.maxlimit = make_float3(l, w, 0.0f);
	//*************************************************************************************

	// SPH parameters
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 1;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.ferrari = 0.1;
	m_simparams.visctype = DYNAMICVISC;
	m_simparams.mbcallback = false;
	m_simparams.boundarytype = MF_BOUNDARY;

	// Size and origin of the simulation domain
	m_size = make_float3(l, w ,h);
	m_origin = make_float3(0.0f, 0.0f, 0.0f);

	m_writerType = VTKWRITER;

	// Physical parameters
	//m_physparams.gravity = make_float3(0.8, 0.0, 0.0); //body forse for plane Poiseuille flow
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0, 1000.0, 7.0f, 40.0f);

	m_physparams.dcoeff = 5.0f*g*H;

	m_physparams.r0 = m_deltap;

	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;
	m_physparams.epsxsph = 0.5f;

	// Scales for drawing
	m_maxrho = density(H, 0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	m_maxvel = 1.0f;

	// Drawing and saving times
	m_displayinterval = 1.0e-4;
	m_writefreq = 1000;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "InputProblem";
	create_problem_dir();
}


int InputProblem::fill_parts()
{
	std::cout << std::endl << "Reading particle data from the input:" << std::endl << inputfile << std::endl;
	const char *ch_inputfile = inputfile.c_str();
	int npart = HDF5SphReader::getNParts(ch_inputfile) + n_probeparts;

	return npart;
}

void InputProblem::copy_to_array(float4 *pos, float4 *vel, particleinfo *info, vertexinfo *vertices, float4 *boundelm, uint *hash)
{
	const char *ch_inputfile = inputfile.c_str();
	int npart = HDF5SphReader::getNParts(ch_inputfile);
	float4 localpos;
	uint hashvalue;

	HDF5SphReader::ReadParticles *buf = new HDF5SphReader::ReadParticles[npart];
	HDF5SphReader::readParticles(buf, ch_inputfile, npart);
	
	int n_parts = 0;
	int n_vparts = 0;
	int n_bparts = 0;
	
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
		calc_localpos_and_hash(Point(buf[i].Coords_0, buf[i].Coords_1, buf[i].Coords_2, rho*buf[i].Volume), localpos, hashvalue);
		pos[i] = localpos;
		vel[i] = make_float4(0, 0, 0, rho);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
		hash[i] = hashvalue;
	}
	int j = n_parts;
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";

	std::cout << "Vertex parts: " << n_vparts << "\n";
	for (uint i = j; i < j + n_vparts; i++) {
		float rho = density(H - buf[i].Coords_2, 0);
		calc_localpos_and_hash(Point(buf[i].Coords_0, buf[i].Coords_1, buf[i].Coords_2, rho*buf[i].Volume), localpos, hashvalue);
		pos[i] = localpos;
		vel[i] = make_float4(0, 0, 0, rho);
		info[i] = make_particleinfo(VERTEXPART, 0, i);
		hash[i] = hashvalue;
	}
	j += n_vparts;
	std::cout << "Vertex part mass: " << pos[j-1].w << "\n";

	std::cout << "Boundary parts: " << n_bparts << "\n";
	for (uint i = j; i < j + n_bparts; i++) {
		// Crixus sets zero volume for boundary particles resulting in zero mass. It doesn't affect calculations, but it
		// contradicts with new feature of indication particles, which left the outlet, by zeroing their mass.
		// To avoid problems masses of boundary particles are set to 99
		calc_localpos_and_hash(Point(buf[i].Coords_0, buf[i].Coords_1, buf[i].Coords_2, 99.0), localpos, hashvalue);
		pos[i] = localpos;
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(BOUNDPART, 0, i);
		hash[i] = hashvalue;
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

	// Setting probes for Spheric2 test case
	//*******************************************************************
	if(n_probeparts) {
		std::cout << "Probe parts: " << n_probeparts << "\n";
		float4 probe_coord[n_probeparts];

		// Probe H1
		for (uint i = 0; i < 50; i++) {
			probe_coord[i] = make_float4(2.724, 0.5, 0.02*i, 0);
		}
		// Probe H2
		for (uint i = 50; i < 100; i++) {
			probe_coord[i] = make_float4(2.228, 0.5, 0.02*(i-50), 0);
		}
		// Probe H3
		for (uint i = 100; i < 150; i++) {
			probe_coord[i] = make_float4(1.732, 0.5, 0.02*(i-100), 0);
		}
		// Probe H4
		for (uint i = 150; i < 200; i++) {
			probe_coord[i] = make_float4(0.582, 0.5, 0.02*(i-150), 0);
		}
		// Pressure probes
		probe_coord[200] = make_float4(2.3955, 0.529, 0.021, 0); // Probe P1
		probe_coord[201] = make_float4(2.3955, 0.529, 0.061, 0); // Probe P2
		probe_coord[202] = make_float4(2.3955, 0.529, 0.101, 0); // Probe P3
		probe_coord[203] = make_float4(2.3955, 0.529, 0.141, 0); // Probe P4
		probe_coord[204] = make_float4(2.4165, 0.471, 0.161, 0); // Probe P5
		probe_coord[205] = make_float4(2.4565, 0.471, 0.161, 0); // Probe P6
		probe_coord[206] = make_float4(2.4965, 0.471, 0.161, 0); // Probe P7
		probe_coord[207] = make_float4(2.5365, 0.471, 0.161, 0); // Probe P8

		for (uint i = j; i < j + n_probeparts; i++) {
			calc_localpos_and_hash(probe_coord[i-j], localpos, hashvalue);
			pos[i] = localpos;
			vel[i] = make_float4(0, 0, 0, 1000);
			info[i] = make_particleinfo(PROBEPART, 0, i);
			hash[i] = hashvalue;
		}
	}
	//*******************************************************************

	delete [] buf;
}
