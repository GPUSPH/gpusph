/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

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

#include <cmath>
#include <iostream>

#include "WaveTank.h"


#define MK_par 2

WaveTank::WaveTank(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	lx = 9.0;
	ly = 0.6;
	lz = 1.0;

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0.0, 0.0, 0.0);

	m_writerType = VTKWRITER;

	// Data for problem setup
	slope_length = 8.5;
	h_length = 0.5;
	height = .63;
	beta = 4.2364*M_PI/180.0;

	// We have at least 1 moving boundary, the paddle
	m_mbnumber = 1;
	m_simparams.mbcallback = true;

	// Add objects to the tank
	use_cyl = false;
	use_cone = false;

	// use a plane for the bottom
	use_bottom_plane = 1;

	// SPH parameters
	set_deltap(0.04);  //0.005f;
	m_simparams.dt = 0.00013;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.2;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 20;
	m_simparams.mlsfreq = 0;
	//m_simparams.visctype = ARTVISC;
	m_simparams.visctype = KINEMATICVISC;
	//m_simparams.visctype = SPSVISC;
	m_simparams.usedem = false;
	m_simparams.tend = 10.0;

	m_simparams.vorticity = false;
	//Testpoints
	m_simparams.testpoints = false;

	// Free surface detection
	m_simparams.surfaceparticle = true;
	m_simparams.savenormals = false;

	//WaveGage
	add_gage(1, 0.3);
	add_gage(0.5, 0.3);

	m_simparams.boundarytype = LJ_BOUNDARY;  //LJ_BOUNDARY or MK_BOUNDARY

	// Physical parameters
	H = 0.45;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(m_physparams.gravity);

	m_physparams.set_density(0, 1000.0, 7.0, 50);
	m_physparams.numFluids = 1;
	float r0 = m_deltap;
	m_physparams.r0 = r0;

	m_physparams.kinematicvisc =  1.0e-6;
	m_physparams.artvisccoeff =  0.3;
	m_physparams.smagfactor = 0.12*0.12*m_deltap*m_deltap;
	m_physparams.kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;

	// BC when using LJ
	m_physparams.dcoeff = 5.0*g*H;
	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5

	// BC when using MK
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;

	//Wave paddle definition:  location, start & stop times, stroke and frequency (2 \pi/period)
	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
	paddle_length = 1.0;
	paddle_width = m_size.y - 2*r0;
	mbpaddledata.type = PADDLEPART;
	mbpaddledata.origin = make_float3(0.13f, r0, -0.1344);
	mbpaddledata.tstart = 0.2;
	mbpaddledata.tend = m_simparams.tend;
	// The stroke value is given at free surface level H
	float stroke = 0.18;
	// m_mbamplitude is the maximal angular value par paddle angle
	// Paddle angle is in [-m_mbamplitude, m_mbamplitude]
	mbpaddledata.amplitude = atan(stroke/(2.0*(H - mbpaddledata.origin.z)));
	mbpaddledata.omega = 2.0*M_PI/0.7;		// period T = 0.8 s
	// Call mb_callback for paddle a first time to initialise
	// values set by the call back function
	mb_callback(0.0, 0.0, 0);

	// Drawing and saving times
	m_displayinterval = 0.01;
	m_writefreq = 20;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "WaveTank";
}


WaveTank::~WaveTank(void)
{
	release_memory();
}


void WaveTank::release_memory(void)
{
	parts.clear();
	paddle_parts.clear();
	boundary_parts.clear();
	test_points.clear();
}


MbCallBack& WaveTank::mb_callback(const float t, const float dt, const int i)
{

	MbCallBack& mbpaddledata = m_mbcallbackdata[0];
	float theta = mbpaddledata.amplitude;
	float dthetadt = 0;
	if (t >= mbpaddledata.tstart && t < mbpaddledata.tend) {
		const float arg = mbpaddledata.omega*(t - mbpaddledata.tstart);
		theta = mbpaddledata.amplitude*cos(arg);
		dthetadt = - mbpaddledata.amplitude*mbpaddledata.omega*sin(arg);
		}
	mbpaddledata.sintheta = sin(theta);
	mbpaddledata.costheta = cos(theta);
	mbpaddledata.dthetadt = dthetadt;
	return m_mbcallbackdata[0];
}


int WaveTank::fill_parts()
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
	   bottom_rect.Fill(boundary_parts,br,true);
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

	if (m_simparams.testpoints) {
		Point pos = Point(0.5748, 0.1799, 0.2564, 0.0);
		test_points.push_back(pos);
		pos = Point(0.5748, 0.2799, 0.2564, 0.0);
		test_points.push_back(pos);
		pos = Point(1.5748, 0.2799, 0.2564, 0.0);
		test_points.push_back(pos);
	}

	if (use_cyl) {
		Point p[10];
		p[0] = Point(h_length + slope_length/(cos(beta)*10), ly/2., 0);
		p[1] = Point(h_length + slope_length/(cos(beta)*10), ly/6.,  0);
		p[2] = Point(h_length + slope_length/(cos(beta)*10), 5*ly/6, 0);
		p[3] = Point(h_length + slope_length/(cos(beta)*5), 0, 0);
		p[4] = Point(h_length + slope_length/(cos(beta)*5), ly/3, 0);
		p[5] = Point(h_length + slope_length/(cos(beta)*5), 2*ly/3, 0);
		p[6] = Point(h_length + slope_length/(cos(beta)*5), ly, 0);
		p[7] = Point(h_length + 3*slope_length/(cos(beta)*10), ly/6, 0);
		p[8] = Point(h_length + 3*slope_length/(cos(beta)*10), ly/2, 0);
		p[9] = Point(h_length+ 3*slope_length/(cos(beta)*10), 5*ly/6, 0);
		p[10] = Point(h_length+ 4*slope_length/(cos(beta)*10), ly/2, 0);

		for (int i = 0; i < 11; i++) {
			cyl[i] = Cylinder(p[i], Vector(.025, 0, 0), Vector(0, 0, height));
			cyl[i].SetPartMass(m_deltap, m_physparams.rho0[0]);
			cyl[i].FillBorder(boundary_parts, br, false, false);
			cyl[i].Unfill(parts, br);
		}
	}
	if (use_cone) {
		Point p1 = Point(h_length + slope_length/(cos(beta)*10), ly/2, 0);
		cone = Cone(p1,Vector(ly/4, 0.0, 0.0), Vector(ly/10, 0., 0.), Vector(0, 0, height));
		cone.SetPartMass(m_deltap, m_physparams.rho0[0]);
		cone.FillBorder(boundary_parts, br, false, true);
		cone.Unfill(parts, br);
    }

	return parts.size() + boundary_parts.size() + paddle_parts.size() + test_points.size();
}


uint WaveTank::fill_planes()
{
    if (!use_bottom_plane) {
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
 	if (use_bottom_plane)  {
		planes[5] = make_float4(-sin(beta),0,cos(beta), h_length*sin(beta));  //sloping bottom starting at x=h_length
		planediv[5] = 1.0;
	}
}

void WaveTank::copy_to_array(float4 *pos, float4 *vel, particleinfo *info, hashKey *hash)
{
	int j = 0;
	if (test_points.size()) {
		//Testpoints
		std::cout << "\nTest points: " << test_points.size() << "\n";
		std::cout << "      " << j << "--" << test_points.size() << "\n";
		for (uint i = 0; i < test_points.size(); i++) {
			calc_localpos_and_hash(test_points[i], pos[i], hash[i]);
			vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
			info[i]= make_particleinfo(TESTPOINTSPART, 0, i);  // first is type, object, 3rd id
		}
		j += test_points.size();
		std::cout << "Test point mass:" << pos[j-1].w << "\n";
	}

	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
	std::cout << "      " << j  << "--" << boundary_parts.size() << "\n";
	for (uint i = j; i < j + boundary_parts.size(); i++) {
		calc_localpos_and_hash(boundary_parts[i-j], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART, 0, i);  // first is type, object, 3rd id
	}
	j += boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	// The object id of moving boundaries parts must be coherent with mb_callback function and follow
	// those rules:
	//		1. object id must be unique (you cannot have a PADDLE with object id 0 and a GATEPART with same id)
	//		2. particle of the same type having the object id move in the same way
	std::cout << "\nPaddle parts: " << paddle_parts.size() << "\n";
	std::cout << "      " << j  << "--" << j + paddle_parts.size() << "\n";
	for (uint i = j; i < j + paddle_parts.size(); i++) {
		calc_localpos_and_hash(paddle_parts[i-j], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(PADDLEPART, 0, i);
	}
	j += paddle_parts.size();
	std::cout << "Paddle part mass:" << pos[j-1].w << "\n";

	std::cout << "\nFluid parts: " << parts.size() << "\n";
	std::cout << "      "<< j  << "--" << j + parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		calc_localpos_and_hash(parts[i-j], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART, 0, i);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";

	std::cout << "Everything uploaded" <<"\n";
}

#undef MK_par
