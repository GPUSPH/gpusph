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
//This is to make fluid 1 the dense fluid
#include <math.h>
#include <iostream>
#include <stdexcept>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "FluidMix.h"
#include "particledefine.h"

#define MK_par 2

FluidMix::FluidMix(const Options &options) : Problem(options)
{
	lx = 2.0;
	ly = 1.0;
	lz = 1.0;
	H = lz;

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0.0, 0.0, 0.0);

	m_writerType = VTKWRITER;

	// SPH parameters
	set_deltap(0.05f);  //0.005f;
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 1.e-4;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
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
	H -= m_deltap;
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


	// Scales for drawing
	m_maxrho = density(H, 0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;
	//m_maxvel = sqrt(m_physparams.gravity*H);
	m_maxvel = 0.4f;

	// Drawing and saving times
	m_displayinterval = 0.01f;
	m_writefreq =  5;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "FluidMix";
	create_problem_dir();
}


FluidMix::~FluidMix(void)
{
	release_memory();
}


void FluidMix::release_memory(void)
{
	parts0.clear();
	parts1.clear();
	boundary_parts.clear();
}


int FluidMix::fill_parts()
{
	const float r0 = m_physparams.r0;
	const float width = m_size.y;
	const float br = (m_simparams.boundarytype == MK_BOUNDARY ? m_deltap/MK_par : r0);

	experiment_box = Cube(Point(0, 0, 0), Vector(lx, 0, 0),
						Vector(0, ly, 0), Vector(0, 0, lz));

	Cube fluid0 = Cube(Point(r0, r0, r0), Vector(lx/2.0 - r0 - r0/2, 0, 0),
						Vector(0, ly - 2*r0, 0), Vector(0, 0, H - r0));
	Cube fluid1 = Cube(Point(lx/2.0 + r0/2, r0, r0), Vector(lx/2.0 - r0/2 - r0, 0, 0),
						Vector(0, ly - 2*r0, 0), Vector(0, 0, H - r0));

	boundary_parts.reserve(1000);
	parts0.reserve(10000);
	parts1.reserve(10000);

	experiment_box.SetPartMass(r0, m_physparams.rho0[0]);
	experiment_box.FillBorder(boundary_parts, r0, true);

	double dx3 = m_deltap*m_deltap*m_deltap;
	fluid0.SetPartMass(dx3*m_physparams.rho0[0]);
	fluid1.SetPartMass(dx3*m_physparams.rho0[1]);

	fluid0.Fill(parts0, r0);
	fluid1.Fill(parts1, r0);

	return parts0.size() + parts1.size() + boundary_parts.size();
}



void FluidMix::draw_boundary(float t)
{
	glColor3f(0.0, 1.0, 0.0);
	experiment_box.GLDraw();
}


void FluidMix::copy_to_array(float4 *pos, float4 *vel, particleinfo *info, uint *hash)
{
	int j = 0;
	std::cout << "\nBoundary parts: " << boundary_parts.size() << "\n";
	std::cout << "\t\t" << j  << "--" << j + boundary_parts.size() << "\n";
	for (uint i = j; i < j + boundary_parts.size(); i++) {
		calc_localpos_and_hash(boundary_parts[i-j], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART, 0, i);
	}
	j += boundary_parts.size();
	std::cout << "j = " << j <<", Boundary  part mass:" << pos[j-1].w << "\n";

	std::cout << "\nFluid [0] parts: " << parts0.size() <<"\n";
	std::cout << "\t\t" << j  << "--" << j + parts0.size() << "\n";
	for (uint i = j; i < j + parts0.size(); i++) {
		calc_localpos_and_hash(parts0[i-j], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART, 0, i);
	}
	j += parts0.size();
	std::cout << "j = " << j << ", Fluid [1] particle mass:" << pos[j-1].w << "\n";

	std::cout << "\nFluid [1] parts: " << parts1.size() <<"\n";
	std::cout << "\t\t" << j  << "--" << j + parts1.size() << "\n";
	for (uint i = j; i < j + parts1.size(); i++) {
		calc_localpos_and_hash(parts1[i-j], pos[i], hash[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[1]);
	}
	j += parts1.size();
	std::cout << "j = " << j << ", Fluid [1] particle mass:" << pos[j-1].w << "\n";

	std::cout << " Everything uploaded" <<"\n";
}

#undef MK_par
