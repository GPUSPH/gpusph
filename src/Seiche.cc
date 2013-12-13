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
#include "Seiche.h"
#include "particledefine.h"


Seiche::Seiche(const Options &options) : Problem(options)
{
	set_deltap(0.015f);
	H = .5f;
	l = sqrt(2)*H; w = l/2; h = 1.5*H;
	std::cout << "length= " << l<<"\n";
	std::cout << "width= " << w <<"\n";
	std::cout << "h = " << h <<"\n";

	// Size and origin of the simulation domain
	m_size = make_double3(l, w ,h);
	m_origin = make_double3(0.0, 0.0, 0.0);

	m_writerType = VTKWRITER;

	// SPH parameters
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.00004f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.2;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 20;
	m_simparams.visctype = SPSVISC;
	m_simparams.mbcallback = false;
	m_simparams.gcallback = true;
	m_simparams.usedem=false;
	m_simparams.tend=10.0f;
	m_simparams.vorticity = false;
	//m_simparams.boundarytype=LJ_BOUNDARY;

	// Physical parameters
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f); //must be set first
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0,1000.0, 7.0f, 20.f);
	m_physparams.numFluids = 1;

    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.dcoeff = 5.0f*g*H;
	m_physparams.r0 = m_deltap;

	// BC when using MK boundary condition: Coupled with m_simsparams.boundarytype=MK_BOUNDARY
	#define MK_par 2
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;
	#undef MK_par

	m_physparams.kinematicvisc = 5.0e-6f;
	m_physparams.artvisccoeff = 0.3f;
	m_physparams.smagfactor = 0.12*0.12*m_deltap*m_deltap;
	m_physparams.kspsfactor = (2.0/3.0)*0.0066*m_deltap*m_deltap;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;

	m_simparams.periodicbound = false;

	// Variable gravity terms:  starting with m_physparams.gravity as defined above
	m_gtstart=0.3f;
	m_gtend=3.0f;

	// Drawing and saving times
	m_displayinterval = 0.01f;
	m_writefreq = 10;
	m_screenshotfreq = 5;

	// Name of problem used for directory creation
	m_name = "Seiche";
	create_problem_dir();
}


Seiche::~Seiche(void)
{
	release_memory();
}


void Seiche::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}

float3 Seiche::g_callback(const float t)
{
	if(t > m_gtstart && t < m_gtend)
		m_physparams.gravity=make_float3(2.*sin(9.8*(t-m_gtstart)), 0.0, -9.81f);
	else
		m_physparams.gravity=make_float3(0.,0.,-9.81f);
	return m_physparams.gravity;
}


int Seiche::fill_parts()
{
	// distance between fluid box and wall
	float wd = m_deltap; //Used to be divided by 2


	parts.reserve(14000);

	experiment_box = Cube(Point(0, 0, 0), Vector(l, 0, 0), Vector(0, w, 0), Vector(0, 0, h));
	Cube fluid = Cube(Point(wd, wd, wd), Vector(l-2*wd, 0, 0), Vector(0, w-2*wd, 0), Vector(0, 0, H-2*wd));
	fluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	// InnerFill puts particle in the center of boxes of step m_deltap, hence at
	// m_deltap/2 from the sides, so the total distance between particles and walls
	// is m_deltap = r0
//	fluid.InnerFill(parts, m_deltap);
	fluid.Fill(parts,m_deltap,true);// it used to be InnerFill


	return parts.size() + boundary_parts.size();
}

uint Seiche::fill_planes()
{
	return 5;
}

void Seiche::copy_planes(float4 *planes, float *planediv)
{
	planes[0] = make_float4(0, 0, 1.0, 0);
	planediv[0] = 1.0;
	planes[1] = make_float4(0, 1.0, 0, 0);
	planediv[1] = 1.0;
	planes[2] = make_float4(0, -1.0, 0, w);
	planediv[2] = 1.0;
	planes[3] = make_float4(1.0, 0, 0, 0);
	planediv[3] = 1.0;
	planes[4] = make_float4(-1.0, 0, 0, l);
	planediv[4] = 1.0;
}


void Seiche::copy_to_array(float4 *pos, float4 *vel, particleinfo *info, uint *hash)
{
	std::cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		calc_localpos_and_hash(boundary_parts[i], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(BOUNDPART, 0, i);
	}
	int j = boundary_parts.size();
	std::cout << "Boundary part mass: " << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		calc_localpos_and_hash(parts[i-j], pos[i], hash[i]);
	//	float rho = density(H - pos[i].z,0);
	//	vel[i] = make_float4(0, 0, 0, rho);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i] = make_particleinfo(FLUIDPART, 0, i);
	}
	j += parts.size();
	std::cout << "Fluid part mass: " << pos[j-1].w << "\n";
}

