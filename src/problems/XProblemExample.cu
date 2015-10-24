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

#include "XProblemExample.h"
/*#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"*/
#include "cudasimframework.cu"

XProblemExample::XProblemExample(GlobalData *_gdata) : XProblem(_gdata)
{
	SETUP_FRAMEWORK(
		// viscosities: ARTVISC, KINEMATICVISC, DYNAMICVISC, SPSVISC, KEPSVISC
		viscosity<ARTVISC>,
		// boundary types: LJ_BOUNDARY, MK_BOUNDARY, SA_BOUNDARY, DYN_BOUNDARY
		boundary<LJ_BOUNDARY>);

	// *** Initialization of minimal physical parameters
	set_deltap(0.02f);
	physparams()->r0 = m_deltap;
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);
	float g = length(physparams()->gravity);
	double H = 3;
	physparams()->dcoeff = 5.0f*g*H;
	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, 20.0f);
	//set_kinematic_visc(0, 1.0e-2f);

	// *** Initialization of minimal simulation parameters
	simparams()->maxneibsnum = 256 + 32;

	// *** Other parameters and settings
	add_writer(VTKWRITER, 1e-1f);
	m_name = "XProblemExample";

	// domain size
	const double dimX = 10;
	const double dimY = 10;
	const double dimZ = 3;

	// world size
	m_origin = make_double3(0, 0, 0);
	// NOTE: NAN value means that will be computed automatically
	m_size = make_double3(dimX, dimY, dimZ);

	// size and height of grid of cubes
	const double cube_size = 0.4;
	const double cube_Z = 1;

	// size and height of spheres of water
	const double sphere_radius = 0.5;
	const double sphere_Z = 2;

	// will create a grid of cubes and spheres
	const double grid_size = dimX / 5;
	const uint cubes_grid_size = 4;
	const uint spheres_grid_size = 3;

	// every geometry will be centered in the given coordinate
	setPositioning(PP_CENTER);

	// create infinite floor
	addPlane(0, 0, 1, 0);

	// origin of the grid of cubes and spheres
	const double cornerXY = (dimX / 2) - (grid_size / 2);

	// grid of cubes
	for (uint i=0; i < cubes_grid_size; i++)
		for (uint j=0; j < cubes_grid_size; j++) {
			// create cube
			GeometryID current = addCube(GT_FIXED_BOUNDARY, FT_BORDER,
				Point( cornerXY + i*grid_size/(cubes_grid_size-1),
				cornerXY + j*grid_size/(cubes_grid_size-1), cube_Z), cube_size);
			// rotate it
			rotate(current, i * (M_PI/2) / cubes_grid_size, j * (M_PI/2) / cubes_grid_size, 0);
		}

	// grid of spheres
	for (uint i=0; i < spheres_grid_size; i++)
		for (uint j=0; j < spheres_grid_size; j++)
			addSphere(GT_FLUID, FT_SOLID,
				Point( cornerXY + i*grid_size/(spheres_grid_size-1),
				cornerXY + j*grid_size/(spheres_grid_size-1), sphere_Z), sphere_radius);

	// setMassByDensity(floating_obj, physparams()->rho0[0] / 2);
}

