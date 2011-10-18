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

#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif
#include <cmath>
#include <iostream>
#include <stdexcept>

#include "EnergyGenerator.h"


#define MK_par 2

EnergyGenerator::EnergyGenerator(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	lx = 9.0;
	ly = 2.0;
	lz = 1.5;
	
	m_size = make_float3(lx, ly, lz);
	m_origin = make_float3(0.0, 0.0, 0.0);

	m_writerType = VTKWRITER;

	// Data for problem setup
	slope_length = 8.5;
	h_length = 0.5;
	height = .63;
	beta = 4.2364*M_PI/180.0;

	// We have at least 1 moving boundary, the paddle
	m_mbnumber = 1;
	m_simparams.mbcallback = true;

	// use a plane for the bottom
	use_bottom_plane = true;
	
	// SPH parameters
	set_deltap(0.06f);  //0.005f;
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.00013;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.2;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 17;
	//m_simparams.mlsfreq = 20;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = KINEMATICVISC;
	//m_simparams.visctype = SPSVISC;
	m_simparams.usedem = false;
	m_simparams.tend = 10.0;

	m_simparams.vorticity = true;

	m_simparams.boundarytype = LJ_BOUNDARY;  //LJ_BOUNDARY or MK_BOUNDARY

    // Physical parameters
	H = 0.65f;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(m_physparams.gravity);

	m_physparams.set_density(0, 1000.0, 7.0, 20);
	m_physparams.numFluids = 1;
	float r0 = m_deltap;
	m_physparams.r0 = r0;

	m_physparams.kinematicvisc =  1.0e-6;
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

	// Allocate data for floating bodies
	allocate_bodies(1);
	
	//Wave paddle definition:  location, start & stop times, stroke and frequency (2 \pi/period)
	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
	paddle_length = 1.4f;
	paddle_width = m_size.y - 2*r0;
	mbpaddledata.type = PADDLEPART;
	mbpaddledata.origin = make_float3(0.13f, r0, -0.1344);
	mbpaddledata.tstart = 0.1f;
	mbpaddledata.tend = m_simparams.tend;
	// The stroke value is given at free surface level H
	float stroke = 0.6;
	// m_mbamplitude is the maximal angular value par paddle angle
	// Paddle angle is in [-m_mbamplitude, m_mbamplitude]
	mbpaddledata.amplitude = atan(stroke/(2.0*(H - mbpaddledata.origin.z)));
	mbpaddledata.omega = 0.5*M_PI;		// period T = 1.0 s
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
	m_maxvel = 1.4f;

	// Drawing and saving times
	m_displayinterval = 0.01f;
	m_writefreq = 0;
	m_screenshotfreq = 0;
	
	// Name of problem used for directory creation
	m_name = "EnergyGenerator";
	create_problem_dir();
}


EnergyGenerator::~EnergyGenerator(void)
{
	release_memory();
}


void EnergyGenerator::release_memory(void)
{
	parts.clear();
	paddle_parts.clear();
	boundary_parts.clear();
}


MbCallBack& EnergyGenerator::mb_callback(const float t, const float dt, const int i)
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

		default:
			throw runtime_error("Incorrect moving boundary object number");
			break;
		}
        
	return m_mbcallbackdata[i];
}


int EnergyGenerator::fill_parts()
{
	const float r0 = m_physparams.r0;
	const float br = (m_simparams.boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);

    experiment_box = Cube(Point(0, 0, 0), Vector(h_length + slope_length, 0, 0),
						Vector(0, ly, 0), Vector(0, 0, height));

	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
	Rect paddle = Rect(Point(mbpaddledata.origin), Vector(0, paddle_width, 0),
				Vector(paddle_length*mbpaddledata.sintheta, 0, paddle_length*mbpaddledata.costheta));

	boundary_parts.reserve(100);
	paddle_parts.reserve(500);
	parts.reserve(34000);
   
	paddle.SetPartMass(m_deltap, m_physparams.rho0[0]);
	paddle.Fill(paddle_parts, br, true);

	bottom_rect = Rect(Point(h_length, 0, 0), Vector(0, ly, 0),
			Vector(slope_length/cos(beta), 0.0, slope_length*tan(beta)));
	if (!use_bottom_plane) {
	   bottom_rect.SetPartMass(m_deltap, m_physparams.rho0[0]);
	   bottom_rect.Fill(boundary_parts, br, true);
	   std::cout << "bottom rectangle defined" <<"\n";
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
				Vector(0, ly-2.0*r0, 0), Vector(l, 0, 0));
		fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid.Fill(parts, m_deltap, true);
		n++;
	 }
	
	Point p = Point(2.0, 1.0, 0.0);
	cyl = Cylinder(p, 0.12, lz, EulerParameters());
	cyl.SetPartMass(m_deltap, m_physparams.rho0[0]);
	cyl.FillBorder(boundary_parts, br);
	cyl.Unfill(parts, br);

	torus = Torus(p + Vector(0, 0, H + 0.2 + 2*br), 0.4, 0.2, EulerParameters());
	torus.SetPartMass(br, m_physparams.rho0[0]*0.2);
	torus.SetInertia(br);
	torus.Unfill(parts, 2*br);
	
	RigidBody* rigid_body = get_body(0);
	rigid_body->AttachObject(&torus);
	torus.FillBorder(rigid_body->GetParts(), br);
	rigid_body->SetInitialValues(Vector(0.0, 0.0, 0.0), Vector(0.0, 0.0, 0.0));
	
	return parts.size() + boundary_parts.size() + paddle_parts.size() + get_bodies_numparts();
}


uint EnergyGenerator::fill_planes()
{

    if (!use_bottom_plane) {
		return 5;
		}
	else {
		return 6;
		} //corresponds to number of planes
}


void EnergyGenerator::copy_planes(float4 *planes, float *planediv)
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
 	if (use_bottom_plane == 1)  {
		planes[5] = make_float4(-sin(beta),0,cos(beta), h_length*sin(beta));  //sloping bottom starting at x=h_length
		planediv[5] = 1.0;
	}
}


void EnergyGenerator::draw_boundary(float t)
{
	glColor3f(0.0, 1.0, 0.0);
	experiment_box.GLDraw();
	experiment_box.GLDraw();

	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
	glColor3f(1.0, 0.0, 0.0);
	Rect actual_paddle = Rect(Point(mbpaddledata.origin), Vector(0, paddle_width, 0),
				Vector(paddle_length*mbpaddledata.sintheta, 0,
						paddle_length*mbpaddledata.costheta));

	actual_paddle.GLDraw();

	glColor3f(0.5, 0.5, 1.0);
	cyl.GLDraw();
	glColor3f(1.0, 0, 0.0);
	for (int i = 0; i < m_simparams.numbodies; i++)
		get_body(i)->GLDraw();
}


void EnergyGenerator::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
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

	for (int k = 0; k < m_simparams.numbodies; k++) {
		PointVect & rbparts = get_body(k)->GetParts();
		std::cout << "Rigid body " << k << ": " << rbparts.size() << " particles ";
		for (uint i = j; i < j + rbparts.size(); i++) {
			pos[i] = make_float4(rbparts[i - j]);
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			info[i]= make_particleinfo(OBJECTPART, k, i - j);
		}
		j += rbparts.size();
		std::cout << ", part mass: " << pos[j-1].w << "\n";
	}
	
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

