/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
*/
//This is to make fluid 1 the dense fluid
#include <math.h>
#include <iostream>
#include <stdexcept>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "PaddleTest1.h"
#include "particledefine.h"

#define MK_par 2

PaddleTest1::PaddleTest1(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	m_size = make_float3(9.0f, 0.4f, 1.5f);
	m_origin = make_float3(0.0f, 0.0f,0.0f);

	m_writerType = TEXTWRITER;

	// Data for problem setup
	slope_length = 7.5f;
	h_length = 5.5f;
	height = .63f;
 	beta = 10*M_PI/180.0;

	// We have at least 1 moving boundary, the paddle
	m_mbnumber = 1;
	m_simparams.mbcallback = true;


	// use a plane for the bottom
	i_use_bottom_plane = 1; // 1 for real plane instead of boundary parts

	// SPH parameters
	set_deltap(0.033f);  //0.005f;
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.5e-4;
	m_simparams.xsph = false;
	m_simparams.dtadapt = false;
	m_simparams.dtadaptfactor = 0.2;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 10;
	m_simparams.mlsfreq = 0;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = KINEMATICVISC;
	//m_simparams.visctype = SPSVISC;
	m_simparams.usedem = false;
	m_simparams.tend = 10.0;

	m_simparams.vorticity = false;
	m_simparams.boundarytype = LJ_BOUNDARY;  //LJ_BOUNDARY or MK_BOUNDARY
    m_simparams.sph_formulation = SPH_F2;
    // Physical parameters
	H = 0.45f;
	m_physparams.gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(m_physparams.gravity);
	m_physparams.numFluids = 2;
	m_physparams.set_density(0, 1000.0f, 7.0f, 20.f);  //water
	m_physparams.set_density(1, 1200.0f, 7.0f, 20.f);  //mud is heavier

	float r0 = m_deltap;
	m_physparams.r0 = r0;
	m_physparams.kinematicvisc = 1.6e-05;                      //1.0e-6f
	m_physparams.artvisccoeff =  0.3;                          //0.3f
	m_physparams.smagfactor = 0.12*m_deltap*m_deltap;
	m_physparams.kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	m_physparams.epsartvisc =  0.01*m_simparams.slength*m_simparams.slength;                          // 0.01*m_simparams.slength*m_simparams.slength;

	// BC when using LJ
	m_physparams.dcoeff = 5.0f*g*H;
    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// BC when using MK
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;

	//Wave paddle definition:  location, start & stop times, stroke and frequency (2 \pi/period)
	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
	paddle_length = 1.0f;
	paddle_width = m_size.y;
	mbpaddledata.type = PADDLEPART;
	mbpaddledata.origin = make_float3(0.13f, r0, -0.1344);
	mbpaddledata.tstart = 0.5f;                 // m_simparams.tend;  //still water test
	mbpaddledata.tend = m_simparams.tend;
	// The stroke value is given at free surface level H
	float stroke = 0.2;
	// m_mbamplitude is the maximal angular value par paddle angle
	// Paddle angle is in [-m_mbamplitude, m_mbamplitude]
	mbpaddledata.amplitude = atan(stroke/(2.0*(H - mbpaddledata.origin.z)));
	mbpaddledata.omega = 2.0*M_PI;		// period T = 1.0 s
	// Call mb_callback for paddle a first time to initialise
	// values set by the call back function
	mb_callback(0.0, 0.0, 0);


	
	// Scales for drawing
	m_maxrho = density(H,0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	//m_maxvel = sqrt(m_physparams.gravity*H);
	m_maxvel = 0.4f;

	// Drawing and saving times
	m_displayinterval = 0.001f;
	m_writefreq =  0;
	m_screenshotfreq = 10;
	
	// Name of problem used for directory creation
	m_name = "PaddleTest";
	create_problem_dir();
}


PaddleTest1::~PaddleTest1(void)
{
	release_memory();
}


void PaddleTest1::release_memory(void)
{
	parts.clear();
	paddle_parts.clear();
	gate_parts.clear();
	boundary_parts.clear();
}


MbCallBack& PaddleTest1::mb_callback(const float t, const float dt, const int i)
{

	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
    float theta = mbpaddledata.amplitude;
	if (t >= mbpaddledata.tstart && t < mbpaddledata.tend) {
		theta = mbpaddledata.amplitude*cos(mbpaddledata.omega*(t - mbpaddledata.tstart));
		}
	mbpaddledata.sintheta = sin(theta);
	mbpaddledata.costheta = cos(theta);

	return m_mbcallbackdata[i];
}


int PaddleTest1::fill_parts()
{
	const float r0 = m_physparams.r0;
	const float width = m_size.y;
	const float br = (m_simparams.boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);

    experiment_box = Cube(Point(0, 0, 0), Vector(h_length + slope_length, 0, 0),
						Vector(0, width, 0), Vector(0, 0, height));

	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
	Rect paddle = Rect(Point(mbpaddledata.origin), Vector(0, paddle_width, 0),
				Vector(paddle_length*mbpaddledata.sintheta, 0,
						paddle_length*mbpaddledata.costheta));

	boundary_parts.reserve(100);
	paddle_parts.reserve(500);
	parts.reserve(34000);
   
	paddle.SetPartMass(m_deltap, m_physparams.rho0[1]);// might use 1 if we know it is always multifluid
	paddle.Fill(paddle_parts, br, true);

	if( i_use_bottom_plane  == 0){
	   Rect bottom = Rect(Point(h_length,0,0  ), Vector(0, width, 0),
			Vector(slope_length/cos(beta), 0.0, slope_length*tan(beta)));
	   bottom.SetPartMass(m_deltap, m_physparams.rho0[1]);
	   bottom.Fill(boundary_parts,br,true);
	   std::cout << "bottom rectangle defined" <<"\n";
	 }   
    

 
	Rect fluid;
	float z = 0;
	int n = 0;
	const float amplitude = mbpaddledata.amplitude;
	while (z < H/2) {              // lower layer
		z = n*m_deltap + .8*r0;    //old: z = n*m_deltap + 1.5*r0;
		std::cout << "z = " << z <<"\n";
		float x = mbpaddledata.origin.x + (z - mbpaddledata.origin.z)*tan(amplitude) + 1.0*r0/cos(amplitude);
 
		float l = h_length + z/tan(beta) - 1.5*r0/sin(beta) - x;
 
		fluid = Rect(Point(x,  2.0*r0, z), Vector(l, 0, 0), Vector(0, width-4.0*r0, 0));
		fluid.SetPartMass(m_deltap, m_physparams.rho0[1]);  //should be 1, the fluid with fastest speed of sound
		fluid.Fill(parts, m_deltap, true);
		n++;
	 }
	num_parts[1] = parts.size();  // number of rho0[1] fluid particles
	std::cout << "num_parts[1] = " << num_parts[1] <<"\n";
    while (z < H) {
		z = n*m_deltap + 1.5*r0;    //z = n*m_deltap + 2*r0;
		std::cout << "z = " << z <<"\n";
	 	float x = mbpaddledata.origin.x + (z - mbpaddledata.origin.z)*tan(amplitude) + 1.0*r0/cos(amplitude);	 
	 	float l = h_length + z/tan(beta) - 1.5*r0/sin(beta) - x;
		fluid = Rect(Point(x,  2.*r0, z), Vector(l, 0, 0), Vector(0, width-4.0*r0, 0));
		fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid.Fill(parts, m_deltap, true);
		n++;
	 }
	num_parts[0] = parts.size()-num_parts[1];
	std::cout <<"num_parts[0] = " <<num_parts[0] <<"\n";
	std::cout <<"parts.size() = " <<parts.size() <<"\n";
 
/*
	Cube fluid;
	fluid = Cube(Point(mbpaddledata.origin.x+r0, r0, r0), Vector(h_length + slope_length-mbpaddledata.origin.x-2*r0, 0, 0),
						Vector(0, width-2*r0, 0), Vector(0, 0, height));
	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	fluid.Fill(parts, m_deltap, true);
	num_parts[0]=parts.size()/2;
	num_parts[1]=parts.size()-num_parts[0];
	*/
    return parts.size() + boundary_parts.size() + paddle_parts.size() ;

	}

 
uint PaddleTest1::fill_planes()
{
 
    if (i_use_bottom_plane == 0) {
		return 5;
		}
	else {
		return 6;
		} //corresponds to number of planes
 
}


void PaddleTest1::copy_planes(float4 *planes, float *planediv)
{
	const float w = m_size.y;
	const float l = h_length + slope_length;

	//  plane is defined as a x + by +c z + d= 0
	planes[0] = make_float4(0, 0, 1.0, 0);   //bottom, where the first three numbers are the normal, and the last is d.
	planediv[0] = 1.0;
	planes[1] = make_float4(0, 1.0, 0, 0);   //wall
	planediv[1] = 1.0;
	planes[2] = make_float4(0, -1.0, 0, w); //far wall
	planediv[2] = 1.0;
 	planes[3] = make_float4(1.0, 0, 0, 0);  //end
 	planediv[3] = 1.0;
 	planes[4] = make_float4(-1.0, 0, 0, l);  //one end
 	planediv[4] = 1.0;
 	if (i_use_bottom_plane == 1)  {
 		planes[5] = make_float4(-sin(beta),0,cos(beta), h_length*sin(beta));  //sloping bottom starting at x=h_length
 		planediv[5] = 1.0;
 	}
}


void PaddleTest1::draw_boundary(float t)
{
	glColor3f(0.0, 1.0, 0.0);
	experiment_box.GLDraw();
 	if (i_use_bottom_plane == 1)
		experiment_box1.GLDraw();

	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
	glColor3f(1.0, 0.0, 0.0);
	Rect actual_paddle = Rect(Point(mbpaddledata.origin), Vector(0, paddle_width, 0),
				Vector(paddle_length*mbpaddledata.sintheta, 0,
						paddle_length*mbpaddledata.costheta));

	actual_paddle.GLDraw();

	glColor3f(0.5, 0.5, 1.0);
	const float displace = m_mbcallbackdata[1].disp.z;
	const float width = m_size.y;
}


void PaddleTest1::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	/*  No boundary particles if using planes
	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
		std::cout << "      "<< 0  <<"--"<< boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[11]);
		info[i]= make_particleinfo(BOUNDPART, 0, i);  // first is type, object, 3rd id
	}
	int j = boundary_parts.size();
	std::cout <<" j = " << j <<", Boundary part mass:" << pos[j-1].w << "\n";
    */
	int j = 0;
	
	// The object id of moving boundaries parts must be coherent with mb_callback function and follow
	// those rules:
	//		1. object id must be unique (you cannot have a PADDLE with object id 0 and a GATEPART with same id)
	//		2. particle of the same type having the object id move in the same way
	// In this exemple we have 2 type of moving boudaries PADDLE, and GATE for 11 moving boundaries and 2
	// different movements. There is one PADDLE moving boundary with a rotational movement and 10 GATES (actually
	// the cylinders) sharing the same translational movement. So in this case PADDLEPARTS have objectid = 0 and
	// GATEPARTS have object id = 1.
	std::cout << "\nPaddle parts: " << paddle_parts.size() << "\n";
		std::cout << "      "<< j  <<"--"<< j+ paddle_parts.size() << "\n";
	for (uint i = j; i < j + paddle_parts.size(); i++) {
		pos[i] = make_float4(paddle_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(PADDLEPART, 0, i);
	}
	j += paddle_parts.size();
	std::cout << "j = "<< j <<", Paddle part mass:" << pos[j-1].w << "\n";



	std::cout << "\nFluid parts: " << parts.size() << " =(" << num_parts[0] <<" + "<< num_parts[1] << ")" <<"\n";
	std::cout << "      "<< j  <<"--"<< j+ parts.size() << "\n";
	
	// lower fluid; fluid 1
	for (uint i = j; i < j + num_parts[1]; i++) {
		pos[i] = make_float4(parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[1]);// should be 1
	    info[i]= make_particleinfo(FLUIDPART + 1, 0, i);// should be one
	}
	j += num_parts[1];
	std::cout << "j = " << j << ", Fluid [1] particle mass:" << pos[j-1].w << "\n";


	// upper fluid; fluid 0
	for (uint i = j; i < j + num_parts[0]; i++) {
		pos[i] = make_float4(parts[i-j+num_parts[1]]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
	    info[i]= make_particleinfo(FLUIDPART, 0, i);

	}
	j += num_parts[0];
	std::cout << "j = " << j << ", Fluid [0] particle mass:" << pos[j-1].w << "\n";

	std::cout << " Everything uploaded" <<"\n";
}

#undef MK_par