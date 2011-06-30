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

#include <math.h>
#include <iostream>
#include <stdexcept>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "WaveTank.h"
#include "particledefine.h"

#define MK_par 2

WaveTank::WaveTank(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	m_size = make_float3(9.0f, 0.4f, 1.0f);
	m_origin = make_float3(0.0f, 0.0f,0.0f);

	m_writerType = TEXTWRITER;

	// Data for problem setup
	slope_length = 8.5f;
	h_length = 0.5f;
	height = .63f;
	beta = 4.2364*M_PI/180.0;

	// We have at least 1 moving boundary, the paddle
	m_mbnumber = 1;
	m_simparams.mbcallback = true;

	// Add objects to the tank
    icyl = 0;	// icyl = 0 means no cylinders
	icone = 0;	// icone = 0 means no cone
	// If presents, cylinders and cone are moving alltogether with
	// the same velocity
	if (icyl || icone)
		m_mbnumber++;

	// use a plane for the bottom
	i_use_bottom_plane = 0; // 1 for real plane instead of boundary parts

	// SPH parameters
	set_deltap(0.04f);  //0.005f;
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.00013f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.2;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 20;
	//m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = KINEMATICVISC;
	m_simparams.visctype = SPSVISC;
	m_simparams.usedem = false;
	m_simparams.tend = 10.0;

	m_simparams.vorticity = true;
	//Testpoints
	m_simparams.testpoints = false;
	if (m_simparams.testpoints)
	numTestpoints = 3;

	m_simparams.boundarytype = LJ_BOUNDARY;  //LJ_BOUNDARY or MK_BOUNDARY

    // Physical parameters
	H = 0.45f;
	m_physparams.gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(m_physparams.gravity);

	m_physparams.set_density(0, 1000.0f, 7.0f, 300*H);
	m_physparams.numFluids = 1;
	float r0 = m_deltap;
	m_physparams.r0 = r0;

	m_physparams.kinematicvisc =  1.0e-6f;
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

	//Wave paddle definition:  location, start & stop times, stroke and frequency (2 \pi/period)
	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
	paddle_length = 1.0f;
	paddle_width = m_size.y - 2*r0;
	mbpaddledata.type = PADDLEPART;
	mbpaddledata.origin = make_float3(0.13f, r0, -0.1344);
	mbpaddledata.tstart = 0.2f;
	mbpaddledata.tend = m_simparams.tend;
	// The stroke value is given at free surface level H
	float stroke = 0.1;
	// m_mbamplitude is the maximal angular value par paddle angle
	// Paddle angle is in [-m_mbamplitude, m_mbamplitude]
	mbpaddledata.amplitude = atan(stroke/(2.0*(H - mbpaddledata.origin.z)));
	mbpaddledata.omega = 2.0*M_PI;		// period T = 1.0 s
	// Call mb_callback for paddle a first time to initialise
	// values set by the call back function
	mb_callback(0.0, 0.0, 0);

	// Moving boundary initialisation data for cylinders and cone
	// used only if needed(cyl = 1 or cone = 1)
	MbCallBack& mbcyldata = m_mbcallbackdata[1];
	mbcyldata.type = GATEPART;
	mbcyldata.tstart = 0.0f;
	mbcyldata.tend =  1.0f;
	// Call mb_callback  for cylindres and cone a first time to initialise
	// values set by the call back function
	mb_callback(0.0, 0.0, 0);
	
	// Scales for drawing
	m_maxrho = density(H,0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	//m_maxvel = sqrt(m_physparams.gravity*H);
	m_maxvel = 0.4f;

	// Drawing and saving times
	m_displayinterval = 0.01f;
	m_writefreq = 10;
	m_screenshotfreq = 0;
	
	// Name of problem used for directory creation
	m_name = "WaveTank";
	create_problem_dir();
}


WaveTank::~WaveTank(void)
{
	release_memory();
}


void WaveTank::release_memory(void)
{
	parts.clear();
	paddle_parts.clear();
	gate_parts.clear();
	boundary_parts.clear();
}


MbCallBack& WaveTank::mb_callback(const float t, const float dt, const int i)
{
	switch (i) {
		// Paddle
		case 0:
			{
			MbCallBack& mbpaddledata = m_mbcallbackdata[0];
			float theta = mbpaddledata.amplitude;
			if (t >= mbpaddledata.tstart && t < mbpaddledata.tend) {
				theta = mbpaddledata.amplitude*cos(mbpaddledata.omega*(t - mbpaddledata.tstart));
				}
			mbpaddledata.sintheta = sin(theta);
			mbpaddledata.costheta = cos(theta);
			}
			break;

		// Cylinders and cone
		case 1:
			{
			MbCallBack& mbcyldata = m_mbcallbackdata[1];
			if (t >= mbcyldata.tstart && t < mbcyldata.tend) {
				mbcyldata.vel = make_float3(0.0f, 0.0f, 0.5f);
				mbcyldata.disp += mbcyldata.vel*dt;
				}
			else
				mbcyldata.vel = make_float3(0.0f, 0.0f, 0.0f);
			}
			break;

		default:
			throw runtime_error("Incorrect moving boundary object number");
			break;
		}
        
	return m_mbcallbackdata[i];
}


int WaveTank::fill_parts()
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
   
	paddle.SetPartMass(m_deltap, m_physparams.rho0[0]);
	paddle.Fill(paddle_parts, br, true);

	if (i_use_bottom_plane == 0) {
	   experiment_box1 = Rect(Point(h_length,0,0  ), Vector(0, width, 0),
			Vector(slope_length/cos(beta), 0.0, slope_length*tan(beta)));
	   experiment_box1.SetPartMass(m_deltap, m_physparams.rho0[0]);
	   experiment_box1.Fill(boundary_parts,br,true);
	   std::cout << "bottom rectangle defined" <<"\n";
	   }

	if (icyl == 1) {
		Point p1 = Point(h_length + slope_length/(cos(beta)*10), width/2, -height);
		Point p2 = Point(h_length + slope_length/(cos(beta)*10), width/6,  -height);
		Point p3 = Point(h_length + slope_length/(cos(beta)*10), 5*width/6, -height);
		Point p4 = Point(h_length + slope_length/(cos(beta)*5), 0, -height);
		Point p5 = Point(h_length + slope_length/(cos(beta)*5),  width/3, -height);
		Point p6 = Point(h_length + slope_length/(cos(beta)*5), 2*width/3, -height);
		Point p7 = Point(h_length + slope_length/(cos(beta)*5),  width, -height);
		Point p8 = Point(h_length + 3*slope_length/(cos(beta)*10),  width/6, -height);
		Point p9 = Point(h_length + 3*slope_length/(cos(beta)*10),  width/2, -height);
		Point p10 = Point(h_length+ 3*slope_length/(cos(beta)*10), 5*width/6, -height);
		Point p11 = Point(h_length+ 4*slope_length/(cos(beta)*10), width/2, -height*.75);

	    cyl1 = Cylinder(p1,Vector(.025, 0, 0),Vector(0,0,height));
	    cyl1.SetPartMass(m_deltap, m_physparams.rho0[0]);
	    cyl1.FillBorder(gate_parts, br, true, true);
		cyl2 = Cylinder(p2,Vector(.025, 0, 0),Vector(0,0,height));
		cyl2.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl2.FillBorder(gate_parts, br, false, false);
		cyl3 = Cylinder(p3,Vector(.025, 0, 0),Vector(0,0,height));
		cyl3.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl3.FillBorder(gate_parts, br, false, false);
		cyl4 = Cylinder(p4,Vector(.025, 0, 0),Vector(0,0,height));
		cyl4.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl4.FillBorder(gate_parts, br, false, false);
		cyl5  = Cylinder(p5,Vector(.025, 0, 0),Vector(0,0,height));
		cyl5.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl5.FillBorder(gate_parts, br, false, false);
		cyl6 = Cylinder(p6,Vector(.025, 0, 0),Vector(0,0,height));
		cyl6.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl6.FillBorder(gate_parts, br, false, false);
		cyl7 = Cylinder(p7,Vector(.025, 0, 0),Vector(0,0,height));
		cyl7.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl7.FillBorder(gate_parts, br, false, false);
		cyl8 = Cylinder(p8,Vector(.025,0,0),Vector(0,0,height));
		cyl8.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl8.FillBorder(gate_parts, br, false, false);
		cyl9 = Cylinder(p9,Vector(.025, 0, 0),Vector(0,0,height));
		cyl9.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl9.FillBorder(gate_parts, br, false, false);
		cyl10 = Cylinder(p10,Vector(.025, 0, 0),Vector(0,0,height));
		cyl10.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cyl10.FillBorder(gate_parts, br, false, false);
	}
	if (icone == 1) {
		Point p1 = Point(h_length + slope_length/(cos(beta)*10), width/2, -height);
		cone = Cone(p1,Vector(width/4,0.0,0.0), Vector(width/10,0.,0.), Vector(0,0,height));
		cone.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cone.FillBorder(gate_parts, br, false, true);
    }


	Rect fluid;
	float z = 0;
	int n = 0;
	const float amplitude = mbpaddledata.amplitude;
	while (z < H) {
		z = n*m_deltap + 1.5*r0;    //z = n*m_deltap + 1.5*r0;
		float x = mbpaddledata.origin.x + (z - mbpaddledata.origin.z)*tan(amplitude) + 1.0*r0/cos(amplitude);
		float l = h_length + z/tan(beta) - 1.5*r0/sin(beta) - x;
		fluid = Rect(Point(x,  r0, z),
				Vector(0, width-2.0*r0, 0), Vector(l, 0, 0));
		fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid.Fill(parts, m_deltap, true);
		n++;
	 }

	//Testpoints
	if (m_simparams.testpoints)
		return parts.size() + boundary_parts.size() + paddle_parts.size() + gate_parts.size()+numTestpoints;
	else
    return parts.size() + boundary_parts.size() + paddle_parts.size() + gate_parts.size();
	
    

	}


uint WaveTank::fill_planes()
{

    if (i_use_bottom_plane == 0) {
		return 5;
		}
	else {
		return 6;
		} //corresponds to number of planes
}


void WaveTank::copy_planes(float4 *planes, float *planediv)
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


void WaveTank::draw_boundary(float t)
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
	if (icyl ==1) {
		Point p1 = Point(h_length + slope_length/(cos(beta)*10), width/2,    -height + displace);
	    Point p2 = Point(h_length + slope_length/(cos(beta)*10), width/6,    -height + displace);
	    Point p3 = Point(h_length + slope_length/(cos(beta)*10), 5*width/6,  -height + displace);
	    Point p4 = Point(h_length + slope_length/(cos(beta)*5), 0,           -height + displace);
	    Point p5 = Point(h_length + slope_length/(cos(beta)*5),  width/3,    -height + displace);
	    Point p6 = Point(h_length + slope_length/(cos(beta)*5), 2*width/3,   -height + displace);
	    Point p7 = Point(h_length + slope_length/(cos(beta)*5),  width,      -height + displace);
	    Point p8 = Point(h_length + 3*slope_length/(cos(beta)*10),  width/6, -height + displace);
        Point p9 = Point(h_length + 3*slope_length/(cos(beta)*10),  width/2, -height + displace);
	    Point p10 = Point(h_length+ 3*slope_length/(cos(beta)*10), 5*width/6,-height + displace);
        Point p11 = Point(h_length+ 4*slope_length/(cos(beta)*10), width/2,  -height + displace);

	    cyl1 = Cylinder(p1,Vector(.05,0,0),Vector(0,0,height));
	    cyl1.GLDraw();
		cyl2 = Cylinder(p2,Vector(.025,0,0),Vector(0,0,height));
		cyl2.GLDraw();
		cyl3= Cylinder(p3,Vector(.025,0,0),Vector(0,0,height));
		cyl3.GLDraw();
		cyl4= Cylinder(p4,Vector(.025,0,0),Vector(0,0,height));
		cyl4.GLDraw();
		cyl5= Cylinder(p5,Vector(.025,0,0),Vector(0,0,height));
		cyl5.GLDraw();
		cyl6= Cylinder(p6,Vector(.025,0,0),Vector(0,0,height));
		cyl6.GLDraw();
		cyl7= Cylinder(p7,Vector(.025,0,0),Vector(0,0,height));
		cyl7.GLDraw();
		cyl8= Cylinder(p8,Vector(.025,0,0),Vector(0,0,height));
		cyl8.GLDraw();
		cyl9= Cylinder(p9,Vector(.025,0,0),Vector(0,0,height));
		cyl9.GLDraw();
		cyl10= Cylinder(p10,Vector(.025,0,0),Vector(0,0,height));
		cyl10.GLDraw();
		}

	if (icone == 1) {
	 	Point p1 = Point(h_length + slope_length/(cos(beta)*10), width/2, -height + displace);
	  	cone = Cone(p1,Vector(width/4,0.0,0.0), Vector(width/10,0.,0.), Vector(0,0,height));
		cone.GLDraw();
		}
}


void WaveTank::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{

	
	//Testpoints
	int j;
	if (m_simparams.testpoints ) {
	    std::cout << "\nTestpoints parts: " << numTestpoints << "\n";
		std::cout << "      "<< 0  <<"--"<< numTestpoints << "\n";

		pos[0] = make_float4(0.364,0.16,0.04,0.0);
		pos[1] = make_float4(0.37,0.17,0.04,0.0);
        pos[2] = make_float4(1.5748,0.2799,0.2564,0.0);


		for (uint i = 0; i < numTestpoints; i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(TESTPOINTSPART, 0, i);  // first is type, object, 3rd id
		}

    j =numTestpoints;
	std::cout << "Testpoints part mass:" << pos[j-1].w << "\n";
	}
	
	else
		 j=0;



	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
		std::cout << "      "<< 0  <<"--"<< boundary_parts.size() << "\n";
	for (uint i = j; i < j+boundary_parts.size(); i++) {
		pos[i] = make_float4(boundary_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART, 0, i);  // first is type, object, 3rd id
	}
    j += boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";
	//

//	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
//		std::cout << "      "<< 0  <<"--"<< boundary_parts.size() << "\n";
//	for (uint i = 0; i < boundary_parts.size(); i++) {
//		pos[i] = make_float4(boundary_parts[i]);
//		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
//		info[i]= make_particleinfo(BOUNDPART, 0, i);  // first is type, object, 3rd id
//	}
//	int j = boundary_parts.size();
//	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

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
	std::cout << "Paddle part mass:" << pos[j-1].w << "\n";

	std::cout << "\nCylinders and/or cone parts: " << gate_parts.size() << "\n";
	std::cout << "       " << j << "--" << j+gate_parts.size() <<"\n";
	for (uint i = j; i < j + gate_parts.size(); i++) {
		pos[i] = make_float4(gate_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(GATEPART, 1, i);
	}
	j += gate_parts.size();
	std::cout << "Cylinders and/or cone part mass:" << pos[j-1].w << "\n";

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