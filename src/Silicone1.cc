#include <cmath>
#include <iostream>
#include <stdexcept>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "Silicone1.h"
#include "particledefine.h"

#define MK_par 2

Silicone1::Silicone1(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	m_size = make_double3(0.2f, 0.2f, 0.2f);
	m_origin = make_double3(0.0f, 0.0f,0.0f);

	m_writerType = VTKWRITER;


	// use a plane for the bottom
	i_use_bottom_plane = 1; // 1 for real plane instead of boundary parts

	// SPH parameters
	set_deltap(0.002f);
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 3.0e-5f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 30;
	m_simparams.mlsfreq = 0;
	m_simparams.visctype = KINEMATICVISC;
	m_simparams.usedem = false;
	m_simparams.tend = 20.0;

	m_simparams.boundarytype = LJ_BOUNDARY;  //LJ_BOUNDARY or MK_BOUNDARY

    // Physical parameters
	H = 0.45f;
	m_physparams.gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(m_physparams.gravity);

	m_physparams.set_density(0, 974.0f, 7.0f, 5);
	m_physparams.numFluids = 1;
	m_physparams.r0 = m_deltap;

	m_physparams.kinematicvisc =  0.01f;
	m_physparams.artvisccoeff =  0.3f;
	m_physparams.smagfactor = 0.12*0.12*m_deltap*m_deltap;
	m_physparams.kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;

	// BC when using LJ
	m_physparams.dcoeff = 5.0f*g*H;
    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// BC when using MK
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;

	// Scales for drawing
	m_maxrho = density(H,0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	//m_maxvel = sqrt(m_physparams.gravity*H);
	m_maxvel = 0.4f;

	// Drawing and saving times
	m_displayinterval = 0.01f;
	m_writefreq = 0;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "Silicone1";
	create_problem_dir();
}


Silicone1::~Silicone1(void)
{
	release_memory();
}


void Silicone1::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}


int Silicone1::fill_parts()
{
	const float r0 = m_physparams.r0;
	const float width = m_size.y;
	const float br = (m_simparams.boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);

    experiment_box = Cube(Point(0, 0, 0), Vector(m_size.x, 0, 0),
						Vector(0, m_size.y, 0), Vector(0, 0, m_size.z));

	boundary_parts.reserve(100);
	parts.reserve(34000);

	if (i_use_bottom_plane == 0) {
	   experiment_box.SetPartMass(m_deltap, m_physparams.rho0[0]);
	   bool edges_to_fill[4] = {true, true, true, true};
	   experiment_box.FillBorder(boundary_parts, br, 4, edges_to_fill);
	   std::cout << "bottom rectangle filled" <<"\n";
	   }

	float radius = 0.004;
	cyl = Cylinder(Point(0.1,0.1,r0),Vector(2.0*radius, 0, 0),Vector(0,0,50.0*r0));
	cyl.SetPartMass(m_deltap, m_physparams.rho0[0]);
	cyl.Fill(parts, m_deltap);

//	Cube cube = Cube(Point(0.1-5.0*radius,0.1-5.0*radius,r0),Vector(5.0*radius, 2*radius, 0),Vector(-2*radius, 5.0*radius, 0),Vector(0,0,100.0*r0));
//	cube.SetPartMass(m_deltap, m_physparams.rho0[0]);
//	cube.Fill(parts, m_deltap, true);
    return parts.size() + boundary_parts.size();

	}


uint Silicone1::fill_planes()
{

    if (i_use_bottom_plane == 0) {
		return 0;
		}
	else {
		return 1;
		} //corresponds to number of planes
}


void Silicone1::copy_planes(float4 *planes, float *planediv)
{
	//  plane is defined as a x + by +c z + d= 0
	planes[0] = make_float4(0, 0, 1.0, 0);   //bottom, where the first three numbers are the normal, and the last is d.
	planediv[0] = 1.0;
}


void Silicone1::draw_boundary(float t)
{
	glColor3f(0.0, 1.0, 0.0);
	experiment_box.GLDraw();
 	if (i_use_bottom_plane == 1)
		experiment_box.GLDraw();

	//cyl.GLDraw();
}


void Silicone1::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
		std::cout << "      "<< 0  <<"--"<< boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART, 0, i);  // first is type, object, 3rd id
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";


	float g = length(m_physparams.gravity);
	std::cout << "\nFluid parts: " << parts.size() << "\n";
	std::cout << "      "<< j  <<"--"<< j+ parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		pos[i] = make_float4(parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
	    info[i]= make_particleinfo(FLUIDPART,0,i);

	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";

	std::cout << " Everything uploaded" <<"\n";
}

#undef MK_par
