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

#include "SharedNeibsTest.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"


SharedNeibsTest::SharedNeibsTest(const Options &options) : Problem(options)
{
	// Size and origin of the simulation domain
	set_deltap(0.1f);
	lx = 90*m_deltap;
	ly = 90*m_deltap;
	lz = 30*m_deltap;

	m_size = make_float3(lx, ly, lz);
	m_origin = make_float3(0.0, 0.0, 0.0);

	m_writerType = VTKWRITER;

	// SPH parameters
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.01f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = false;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = DYNAMICVISC;
    m_simparams.boundarytype= LJ_BOUNDARY;
	m_simparams.tend = 0.01;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	m_physparams.gravity = make_float3(0.0, 0.0, 0.0);
	float g = length(m_physparams.gravity);
	m_physparams.set_density(0,1000.0, 7.0f, 5.0);

    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.dcoeff = 50.0f;
	m_physparams.r0 = m_deltap;

	m_physparams.kinematicvisc = 1.0e-6f;
	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;

	// Drawing and saving times
	m_displayinterval = 0.01f;
	m_writefreq = 1;
	m_screenshotfreq = 0;

	// Name of problem used for directory creation
	m_name = "SharedNeibsTest";
	create_problem_dir();
}


SharedNeibsTest::~SharedNeibsTest(void)
{
	release_memory();
}


void SharedNeibsTest::release_memory(void)
{
	parts.clear();
}


int SharedNeibsTest::fill_parts()
{
	float r0 = m_physparams.r0;

	Cube fluid, fluid1;

	experiment_box = Cube(Point(0, 0, 0), Vector(lx, 0, 0),
						Vector(0, ly, 0), Vector(0, 0, lz));
	parts.reserve(14000);

	experiment_box.SetPartMass(r0, m_physparams.rho0[0]);
	experiment_box.Fill(parts, r0);

	return parts.size();
}


void SharedNeibsTest::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Fluid parts: " << parts.size() << "\n";
	int j = 0;
	for (uint i = 0; i < parts.size(); i++) {
		pos[i] = make_float4(parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
	}
	j += parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}
