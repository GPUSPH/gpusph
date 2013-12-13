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

#include "Waterfall.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"


Waterfall::Waterfall(const Options &options) : Problem(options)
{
	WORLD_LENGTH = 2.0F;
	WORLD_WIDTH = 1.0F;
	STEP_HEIGHT = 0.5F;
	SIDES_HEIGHT = 2 * STEP_HEIGHT;
	WATER_LEVEL = STEP_HEIGHT / 3.0F;

	float r0 = m_physparams.r0;

	// Size and origin of the simulation domain
	m_size = make_double3(WORLD_WIDTH, WORLD_LENGTH, SIDES_HEIGHT);
	m_origin = make_double3(0.0, 0.0, 0.0);

	m_writerType = VTKWRITER;

	// Y periodicity
	m_simparams.periodicbound = true;
	m_physparams.dispvect = make_float3(0.0F, WORLD_LENGTH, 0.0F);
	m_physparams.minlimit = make_float3(0.0F, 0.0F, 0.0F);
	m_physparams.maxlimit = make_float3(0.0F, WORLD_LENGTH, 0.0F);

	// extra Z offset on Y periodicity (lifting)
	m_physparams.dispOffset = make_float3(0.0F, 0.0F, - STEP_HEIGHT);

	// SPH parameters
	set_deltap(0.02f); // remember: deltap needs to be set at the beginning of the constructor if it is used for setting geomestry
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.0001f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 0;
	m_simparams.visctype = ARTVISC;
	//m_simparams.visctype = DYNAMICVISC;
	m_simparams.boundarytype= LJ_BOUNDARY;
	m_simparams.tend = 5.0f;

	// Free surface detection
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	H = WATER_LEVEL;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(m_physparams.gravity);
	//m_physparams.set_density(0,1000.0, 7.0f, 20.f);

	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.dcoeff = 5.0f*g*H;
	m_physparams.r0 = m_deltap;

	// BC when using MK boundary condition: Coupled with m_simsparams.boundarytype=MK_BOUNDARY
	#define MK_par 2
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;
	#undef MK_par

	m_physparams.kinematicvisc = 1.0e-6f;
	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;

	// Drawing and saving times
	m_displayinterval = 0.001f;
	m_writefreq = 20;
	m_screenshotfreq = 20;

	// Name of problem used for directory creation
	m_name = "Waterfall";
	create_problem_dir();
}


Waterfall::~Waterfall(void)
{
	release_memory();
}


void Waterfall::release_memory(void)
{
	fluid_parts.clear();
	floor_parts.clear();
	walls_parts.clear();
}

int Waterfall::fill_parts()
{
	float r0 = m_physparams.r0;
	int totParts = 0;

	upperFloor = Rect(
		Point(0, - r0, STEP_HEIGHT),
		Vector(WORLD_WIDTH, 0, 0),
		Vector(0, WORLD_LENGTH/2.0F + 2*r0, 0)
	);
	upperFloor.SetPartMass(r0, m_physparams.rho0[0]);
	upperFloor.Fill(floor_parts, r0, false, true); // no borders

	loweFloor = Rect(
		Point(0, WORLD_LENGTH/2.0F - r0, 0),
		Vector(WORLD_WIDTH, 0, 0),
		Vector(0, WORLD_LENGTH/2.0F + r0, 0)
	);
	loweFloor.SetPartMass(r0, m_physparams.rho0[0]);
	loweFloor.Fill(floor_parts, r0, false, true);

	step = Rect(
		Point(0, WORLD_LENGTH/2.0F, 0),
		Vector(WORLD_WIDTH, 0, 0),
		Vector(0, 0, STEP_HEIGHT)
	);
	step.SetPartMass(r0, m_physparams.rho0[0]);
	step.Fill(floor_parts, r0, false, true); // no borders

	right_side = Rect(
		Point(0, 0, 0),
		Vector(0, WORLD_LENGTH, 0),
		Vector(0, 0, SIDES_HEIGHT)
	);
	right_side.SetPartMass(r0, m_physparams.rho0[0]);
	right_side.Fill(walls_parts, r0, true);

	left_side = Rect(
		Point(WORLD_WIDTH, 0, 0),
		Vector(0, WORLD_LENGTH, 0),
		Vector(0, 0, SIDES_HEIGHT)
	);
	left_side.SetPartMass(r0, m_physparams.rho0[0]);
	left_side.Fill(walls_parts, r0, true);

	lowerFluid = Cube(
		Point(r0, WORLD_LENGTH/2.0F + r0, r0),
		Vector(WORLD_WIDTH - 2*r0, 0, 0),
		Vector(0, WORLD_LENGTH/2.0F - r0 - m_deltap, 0),
		Vector(0, 0, WATER_LEVEL - r0) );
	lowerFluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	lowerFluid.Fill(fluid_parts, m_deltap, true);

	upperFluid = Cube(
		Point(r0, 0, STEP_HEIGHT + r0),
		Vector(WORLD_WIDTH - 2*r0, 0, 0),
		Vector(0, WORLD_LENGTH/2.0F, 0),
		Vector(0, 0, WATER_LEVEL - r0) );
	upperFluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	upperFluid.Fill(fluid_parts, m_deltap, true);

	return floor_parts.size() + fluid_parts.size() + walls_parts.size();
}


void Waterfall::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Floor parts: " << floor_parts.size() << "\n";
	for (uint i = 0; i < floor_parts.size(); i++) {
		pos[i] = make_float4(floor_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,0,i);
	}
	int j = floor_parts.size();
	std::cout << "Floor part mass:" << pos[j-1].w << "\n";

	std::cout << "Walls parts: " << walls_parts.size() << "\n";
	for (uint i = j; i < j + walls_parts.size(); i++) {
		pos[i] = make_float4(walls_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,1,i);
	}
	j += walls_parts.size();
	std::cout << "Walls part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << fluid_parts.size() << "\n";
	for (uint i = j; i < j + fluid_parts.size(); i++) {
		pos[i] = make_float4(fluid_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,2,i);
	}
	j += fluid_parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}

void Waterfall::fillDeviceMap(GlobalData* gdata)
{
	// TODO: test which split performs better, if Y (not many particles passing) or X (smaller section)
	// fillDeviceMapByAxisSplit(gdata, 2, 6, );
	//fillDeviceMapByEquation(gdata);
	// fillDeviceMapByAxis(gdata, LONGEST_AXIS);
	// partition by performing the specified number of cuts along the three cartesian axes
	fillDeviceMapByRegularGrid(gdata);
}
