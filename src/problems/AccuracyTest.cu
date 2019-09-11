/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

AccuracyTest::AccuracyTest(GlobalData *_gdata) : Problem(_gdata)
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
	setMaxFall(H);
	add_fluid(1000.0);
	set_equation_of_state(0, 7.0, 50);

	set_kinematic_visc(0, 1.0e-6f);
	set_artificial_visc(0.3*0.005/m_deltap);

	// Drawing and saving times
	add_writer(VTKWRITER, 0.1);

	// Name of problem used for directory creation
	m_name = "AccuracyTest";

	// set positioning policy to PP_CORNER:
	// the given point will be the corner of the geometry
	setPositioning(PP_CORNER);

	const int num_layers = (simparams()->boundarytype > SA_BOUNDARY) ?
		simparams()->get_influence_layers() : 1;
	const double wall_size = num_layers*m_deltap;
	const double box_thickness = wall_size - m_deltap;

	// Building the geometry
	GeometryID side0 = addBox(GT_FIXED_BOUNDARY, FT_BORDER, Point(0, 0, 0),
		lx, ly, box_thickness);
	disableCollisions(side0);

	GeometryID side1 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(0, 0, wall_size),
		box_thickness, ly, lz - wall_size);
	disableCollisions(side1);

	GeometryID side2 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(lx - box_thickness, 0, wall_size),
		box_thickness, ly, lz - wall_size);
	disableCollisions(side2);

	GeometryID side3 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(wall_size, 0, wall_size),
		lx - 2*wall_size, box_thickness, lz - wall_size);
	disableCollisions(side3);

	GeometryID side4 = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(wall_size, ly - box_thickness, wall_size),
		lx - 2*wall_size, box_thickness, lz - wall_size);
	disableCollisions(side4);

	GeometryID fluid = addBox(GT_FLUID, FT_SOLID,
		Point(wall_size, wall_size, wall_size),
		0.4, ly - 2*wall_size, H);
}
