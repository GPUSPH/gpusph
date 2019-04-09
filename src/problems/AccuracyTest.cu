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

#include <iostream>

#include "AccuracyTest.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

AccuracyTest::AccuracyTest(GlobalData *_gdata) : XProblem(_gdata)
{

	SETUP_FRAMEWORK(
		viscosity<ARTVISC>,
		boundary<DYN_BOUNDARY>,
		add_flags<ENABLE_INTERNAL_ENERGY>
	);

	// Size and origin of the simulation domain
	lx = 4.0;
	ly = 0.7;
	lz = 1.0;
	H = 0.7;
	m_size = make_double3(lx, ly, lz);
	m_origin = make_double3(0.0, 0.0, 0.0);

	// SPH parameters
	set_deltap(0.02); //0.008

	set_timestep(1e-5f);
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 1.5f; //0.00036f

	// Physical parameters
	H = 0.6f;
	set_gravity(-9.81f);
	float g = get_gravity_magnitude();
	add_fluid(1000.0);
	set_equation_of_state(0, 7.0, 50);

    //set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	physparams()->dcoeff = 5.0f*g*H;
	physparams()->r0 = m_deltap;

	// BC when using MK boundary condition:
	// Coupled with m_simsparams.boundarytype=MK_BOUNDARY
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

	// set positioning policy to PP_CORNER:
	// the given point will be the corner of the geometry
	setPositioning(PP_CORNER);

	// Building the geometry
	GeometryID side0 = addBox(GT_FIXED_BOUNDARY, FT_BORDER, Point(0, 0, 0),
		lx, ly, 3*m_deltap);
	disableCollisions(side0);

	GeometryID side1 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(0, 0, 4.0*m_deltap),
		3*m_deltap, ly, lz - 4*m_deltap);
	disableCollisions(side1);

	GeometryID side2 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(lx - 3.0*m_deltap, 0, 4.0*m_deltap),
		3*m_deltap, ly, lz - 4*m_deltap);
	disableCollisions(side2);

	GeometryID side3 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(4.0*m_deltap, 0, 4.0*m_deltap),
		lx - 8*m_deltap, 3*m_deltap, lz - 4*m_deltap);
	disableCollisions(side3);

	GeometryID side4 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(4.0*m_deltap, ly, 4.0*m_deltap),
		lx - 8*m_deltap, 3*m_deltap, lz - 4*m_deltap);
	disableCollisions(side4);

	GeometryID fluid = addBox(GT_FLUID, FT_SOLID,
		Point(4.0*m_deltap, 4.0*m_deltap, 4.0*m_deltap),
		0.4, ly - 8*m_deltap, H);
}
