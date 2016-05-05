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

#include <cmath>
#include <iostream>

#include "AccuracyTest.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

AccuracyTest::AccuracyTest(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	lx = 4.0;
	ly = 0.7;
	lz = 1.0;
	H = 0.7;

	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		boundary<DYN_BOUNDARY>,
		add_flags<ENABLE_INTERNAL_ENERGY>
	);

	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0.0, 0.0, 0.0);

	// SPH parameters
	set_deltap(0.02); //0.008

	simparams()->dt = 1e-5f;
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 1.5f; //0.00036f

	// Physical parameters
	H = 0.6f;
	physparams()->gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(physparams()->gravity);
	add_fluid(1000.0);
	set_equation_of_state(0, 7.0, 50);

    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	physparams()->dcoeff = 5.0f*g*H;
	physparams()->r0 = m_deltap;

	// BC when using MK boundary condition: Coupled with m_simsparams.boundarytype=MK_BOUNDARY
	#define MK_par 2
	physparams()->MK_K = g*H;
	physparams()->MK_d = 1.1*m_deltap/MK_par;
	physparams()->MK_beta = MK_par;
	#undef MK_par

	set_kinematic_visc(0, 1.0e-6f);
	physparams()->artvisccoeff = 0.3*0.005/m_deltap;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);

	// Name of problem used for directory creation
	m_name = "AccuracyTest";
}


AccuracyTest::~AccuracyTest(void)
{
	release_memory();
}


void AccuracyTest::release_memory(void)
{
	parts.clear();
	boundary_parts.clear();
}


int AccuracyTest::fill_parts()
{
	float r0 = physparams()->r0;
	const float dp = m_deltap;

	Cube fluid, fluid1;

	experiment_box = Cube(Point(0, 0, 0), lx, ly, lz);

	Cube side0 = Cube(Point(0, 0, 0), lx, ly, 3*dp);

	Cube side1 = Cube(Point(0, 0, 4.0*dp), 3*dp, ly, lz - 4*dp);

	Cube side2 = Cube(Point(lx - 3.0*dp, 0, 4.0*dp),
		3*dp, ly, lz - 4*dp);

	Cube side3 = Cube(Point(4.0*dp, 0, 4.0*dp),
		lx - 8*dp, 3*dp, lz - 4*dp);

	Cube side4 = Cube(Point(4.0*dp, ly, 4.0*dp),
		lx - 8*dp, 3*dp, lz - 4*dp);

	fluid = Cube(Point(4.0*dp, 4.0*dp, 4.0*dp), 0.4, ly - 8*dp, H);

	boundary_parts.reserve(2000);
	parts.reserve(14000);

	side0.SetPartMass(dp, physparams()->rho0[0]);
	side0.Fill(boundary_parts, dp, true);
	side1.SetPartMass(dp, physparams()->rho0[0]);
	side1.Fill(boundary_parts, dp, true);
	side2.SetPartMass(dp, physparams()->rho0[0]);
	side2.Fill(boundary_parts, dp, true);
	side3.SetPartMass(dp, physparams()->rho0[0]);
	side3.Fill(boundary_parts, dp, true);
	side4.SetPartMass(dp, physparams()->rho0[0]);
	side4.Fill(boundary_parts, dp, true);

	fluid.SetPartMass(m_deltap, physparams()->rho0[0]);
	fluid.Fill(parts, m_deltap, true);

	return parts.size() + boundary_parts.size();
}

void AccuracyTest::copy_to_array(BufferList &buffers)
{
	const float rho0 = physparams()->rho0[0];

	float4 *pos = buffers.getData<BUFFER_POS>();
	hashKey *hash = buffers.getData<BUFFER_HASH>();
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();

	cout << "Boundary parts: " << boundary_parts.size() << "\n";
	for (uint i = 0; i < boundary_parts.size(); i++) {
		calc_localpos_and_hash(boundary_parts[i], info[i], pos[i], hash[i]);

		vel[i] = make_float4(0, 0, 0, rho0);
		info[i] = make_particleinfo(PT_BOUNDARY, 0, i);
	}
	int j = boundary_parts.size();
	cout << "Boundary part mass:" << pos[j-1].w << "\n";

	cout << "Fluid parts: " << parts.size() << "\n";
	for (uint i = j; i < j + parts.size(); i++) {
		calc_localpos_and_hash(parts[i-j], info[i], pos[i], hash[i]);

		vel[i] = make_float4(0, 0, 0, rho0);
		info[i] = make_particleinfo(PT_FLUID, 0, i);
	}
	j += parts.size();
	cout << "Fluid part mass:" << pos[j-1].w << "\n";
	flush(cout);
}
