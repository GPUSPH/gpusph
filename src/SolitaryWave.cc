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
#include <stdexcept>

#include "SolitaryWave.h"
#include "particledefine.h"
#include "GlobalData.h"

#define MK_par 2

SolitaryWave::SolitaryWave(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 9.0;
	ly = 0.4;
	lz = 3.0;
	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0.0, 0.0, 0.0);

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

	i_use_bottom_plane = 1; // 1 for real plane instead of boundary parts

	// SPH parameters
	set_deltap(0.04f);  //0.005f;
	m_simparams.dt = 0.00013f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 20;
	m_simparams.mlsfreq = 0;
	//m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = KINEMATICVISC;
	m_simparams.visctype = SPSVISC;
	m_simparams.usedem = false;
	m_simparams.tend = 10.0;

	m_simparams.vorticity = true;
	m_simparams.boundarytype = LJ_BOUNDARY;  //LJ_BOUNDARY or MK_BOUNDARY

	// Physical parameters
	H = 0.45f;
	m_physparams.gravity = make_float3(0.0f, 0.0f, -9.81f);
	float g = length(m_physparams.gravity);

	m_physparams.set_density(0, 1000.0f, 7.0f, 20.f);
	m_physparams.numFluids = 1;
	float r0 = m_deltap;
	m_physparams.r0 = r0;

	m_physparams.artvisccoeff = 0.3f;
	m_physparams.kinematicvisc =  1.0e-6f;
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

	//Wave paddle definition:  location, start & stop times
	MbCallBack& mbpistondata = m_mbcallbackdata[0];
	mbpistondata.type = PISTONPART;
	mbpistondata.origin = make_float3(r0, 0.0, 0.0);
	float amplitude = 0.2f;
	m_Hoh = amplitude/H;
	float kappa = sqrt((3*m_Hoh)/(4.0*H*H));
	float cel = sqrt(g*(H + amplitude));
	m_S = sqrt(16.0*amplitude*H/3.0);
	m_tau = 2.0*(3.8 + m_Hoh)/(kappa*cel);
//	std::cout << "m_tau: " << m_tau << "\n";
	mbpistondata.tstart = 0.2f;
	mbpistondata.tend = m_tau;
	// Call mb_callback for piston a first time to initialise
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

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);

	// Name of problem used for directory creation
	m_name = "SolitaryWave";
}


SolitaryWave::~SolitaryWave(void)
{
	release_memory();
}


void SolitaryWave::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
	gate_parts.clear();
	piston_parts.clear();
}


MbCallBack& SolitaryWave::mb_callback(const double t, const float dt, const int i)
{
	switch (i) {
		// Piston
		case 0:
			{
			MbCallBack& mbpistondata = m_mbcallbackdata[0];
			mbpistondata.type = PISTONPART;
			const float posx = mbpistondata.origin.x;
			if (t >= mbpistondata.tstart && t < mbpistondata.tend) {
				float arg = 2.0*((3.8 + m_Hoh)*((t - mbpistondata.tstart)/m_tau - 0.5)
							- 2.0*m_Hoh*((posx/m_S) - 0.5));
				mbpistondata.disp.x = m_S*(1.0 + tanh(arg))/2.0;
				mbpistondata.vel.x = (3.8 + m_Hoh)*m_S/(m_tau*cosh(arg)*cosh(arg));
			}
			else {
				mbpistondata.vel.x = 0;
				}
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
			break;
			}

		default:
			throw runtime_error("Incorrect moving boundary object number");
			break;
		}

	return m_mbcallbackdata[i];
}


int SolitaryWave::fill_parts()
{
	const float r0 = m_physparams.r0;
	const float width = ly;

	const float br = (m_simparams.boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);

	experiment_box = Cube(Point(0, 0, 0), h_length + slope_length, width, height);

	boundary_parts.reserve(100);
	parts.reserve(34000);
	gate_parts.reserve(2000);
	piston_parts.reserve(500);

	MbCallBack& mbpistondata = m_mbcallbackdata[0];
	Rect piston = Rect(Point(mbpistondata.origin),
						Vector(0, width, 0), Vector(0, 0, height));
	piston.SetPartMass(m_deltap, m_physparams.rho0[0]);
	piston.Fill(piston_parts, br, true);

	if (i_use_bottom_plane == 0) {
	   experiment_box1 = Rect(Point(h_length, 0, 0), Vector(0, width, 0),
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

	    cyl1 = Cylinder(p1,Vector(.05, 0, 0),Vector(0,0,height));
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
	while (z < H) {
		z = n*m_deltap + 1.5*r0;    //z = n*m_deltap + 1.5*r0;
		float x = mbpistondata.origin.x + r0;
		float l = h_length + z/tan(beta) - 1.5*r0/sin(beta) - x;
		fluid = Rect(Point(x,  r0, z),
				Vector(0, width-2.0*r0, 0), Vector(l, 0, 0));
		fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
		fluid.Fill(parts, m_deltap, true);
		n++;
	 }

    return parts.size() + boundary_parts.size() + gate_parts.size() + piston_parts.size();
}


uint SolitaryWave::fill_planes()
{

    if (i_use_bottom_plane == 0) {
		return 5;
		}
	else {
		return 6;
		} //corresponds to number of planes
}


void SolitaryWave::copy_planes(float4 *planes, float *planediv)
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


void SolitaryWave::copy_to_array(BufferList &buffers)
{
	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
		std::cout << "      "<< 0  <<"--"<< boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART, 0, i);  // first is type, object, 3rd id
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass:" << pos[j-1].w << "\n";

	std::cout << "\nPiston parts: " << piston_parts.size() << "\n";
	std::cout << "     " << j << "--" << j + piston_parts.size() << "\n";
	for (uint i = j; i < j + piston_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(PISTONPART, 0, i);
		calc_localpos_and_hash(piston_parts[i - j], info[i], pos[i], hash[i]);
	}
	j += piston_parts.size();
	std::cout << "Piston part mass:" << pos[j-1].w << "\n";

	std::cout << "\nGate parts: " << gate_parts.size() << "\n";
	std::cout << "       " << j << "--" << j+gate_parts.size() <<"\n";
	for (uint i = j; i < j + gate_parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(GATEPART, 1, i);
		calc_localpos_and_hash(gate_parts[i - j], info[i], pos[i], hash[i]);
	}
	j += gate_parts.size();
	std::cout << "Gate part mass:" << pos[j-1].w << "\n";


	std::cout << "\nFluid parts: " << parts.size() << "\n";
	std::cout << "      "<< j  <<"--"<< j+ parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
	    info[i]= make_particleinfo(FLUIDPART,0,i);
		calc_localpos_and_hash(parts[i - j], info[i], pos[i], hash[i]);
		// initializing density
		//       float rho = m_physparams.rho0*pow(1.+g*(H-pos[i].z)/m_physparams.bcoeff,1/m_physparams.gammacoeff);
		//        vel[i] = make_float4(0, 0, 0, rho);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";

	std::cout << " Everything uploaded" <<"\n";
}
#undef MK_par
