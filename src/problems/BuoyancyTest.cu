/*  Copyright (c) 2014-2019 INGV, EDF, UniCT, JHU

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
#include "BuoyancyTest.h"
#include <iostream>

#include "GlobalData.h"
#include "cudasimframework.cu"
#include "Cube.h"
#include "Sphere.h"
#include "Point.h"
#include "Vector.h"


BuoyancyTest::BuoyancyTest(GlobalData *_gdata) : Problem(_gdata)
{
	// Size and origin of the simulation domain
	const double lx = 1.0;
	const double ly = 1.0;
	const double lz = 1.0;
	const double H = 0.6;

	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		viscosity<ARTVISC>,
		//viscosity<SPSVISC>,
		//viscosity<KINEMATICVISC>,
		boundary<DYN_BOUNDARY>
	);

	const int ppH = get_option("ppH", 16);

	// SPH parameters
	set_deltap(H/ppH);
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 10;
	simparams()->tend = 8;

	// Physical parameters
	set_gravity(-9.81f);
	setMaxFall(H);

	auto water = add_fluid(1000.0);
	set_equation_of_state(water,  7.0f, NAN /* autocompute speed of sound */);

	set_kinematic_visc(0, 1.0e-6f);

	add_writer(VTKWRITER, 0.1);

	setPositioning(PP_BOTTOM_CENTER);
	setFillingMethod(Object::BORDER_TANGENT);

	const Point bottom_center(0, 0, 0);

	GeometryID domain_box = addBox(GT_FIXED_BOUNDARY, FT_OUTER_BORDER,
		bottom_center, lx, ly, lz);
	disableCollisions(domain_box);
	// cut off the top of the domain box
	addPlane(0, 0, -1, lz, FT_UNFILL);

	addBox(GT_FLUID, FT_SOLID, bottom_center, lx, ly, H);

	setPositioning(PP_CENTER);
	const double floater_side = lx *0.4;
	const Point floater_center = bottom_center + Vector(0, 0, H/2);
	const std::string floater_type = get_option("floater", "cube");
	GeometryID floater;

	if (floater_type == "cube") {
		floater = addCube(GT_FLOATING_BODY, FT_INNER_BORDER, floater_center, floater_side);
	} else if (floater_type == "sphere") {
		floater = addSphere(GT_FLOATING_BODY, FT_INNER_BORDER, floater_center, floater_side/2);
	} else if (floater_type == "torus") {
		floater = addTorus(GT_FLOATING_BODY, FT_INNER_BORDER, floater_center, floater_side/2, floater_side/4);
	}

	setMassByDensity(floater, physparams()->rho0[0]*0.5);

}
