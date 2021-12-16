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

#include "StillWater.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

StillWater::StillWater(GlobalData *_gdata) : Problem(_gdata)
{
	const bool use_planes = get_option("use-planes", false); // --use-planes true to enable use of planes for boundaries
	const int mlsIters = get_option("mls", 0); // --mls N to enable MLS filter every N iterations
	const int ppH = get_option("ppH", 16); // --ppH N to change deltap to H/N

	// density diffusion terms, see DensityDiffusionType
	const DensityDiffusionType rhodiff = get_option("density-diffusion", FERRARI);

	SETUP_FRAMEWORK(
		//viscosity<KINEMATICVISC>,
		viscosity<DYNAMICVISC>,
		//viscosity<ARTVISC>,
		boundary<DYN_BOUNDARY>
		//boundary<LJ_BOUNDARY>
	).select_options(
		rhodiff,
		use_planes, add_flags<ENABLE_PLANES>()
	);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// water height
	const double H = 1;

	set_deltap(H/ppH);

	setMaxFall(H);

	// box size
	const double l = sqrt(2)*H;
	const double w = l;
	const double box_height = 1.1*H;

	// SPH parameters
	set_timestep(0.00004f);
	simparams()->dtadaptfactor = 0.3;
	simparams()->buildneibsfreq = 20;
	simparams()->ferrariLengthScale = H;

	simparams()->tend = 100.0;
	if (simparams()->boundarytype == SA_BOUNDARY) {
		resize_neiblist(128, 128);
	};

	// Physical parameters
	set_gravity(-9.81f);
	const float g = get_gravity_magnitude();
	const float maxvel = sqrt(2*g*H);
	// purely for cosmetic reason, let's round the soundspeed to the next
	// integer
	const float c0 = ceil(10*maxvel);
	auto water = add_fluid(1000.0);
	set_equation_of_state(water, 7.0f, c0);

	//physparams()->visccoeff = 0.05f;
	set_kinematic_visc(0, 3.0e-2f);
	//set_kinematic_visc(0, 1.0e-6f);

	// Drawing and saving times
	add_writer(VTKWRITER, 1.0);

	// BORDER_TANGENT filling method means that particle filling begins
	// half a deltap “inside” the geometry
	setFillingMethod(Object::BORDER_TANGENT);

	// place geometries by their corner
	setPositioning(PP_CORNER);

	const Point box_corner(0, 0, 0);

	// outer walls: if using particles, build a box and remove the top,
	// otherwise defines the 5 planes
	if (use_planes) {
		const Point other_corner = box_corner + Vector(l, w, 0);
		// geometric planes are FT_NOFILL
		addPlane(box_corner, Vector(1, 0, 0), FT_NOFILL);
		addPlane(box_corner, Vector(0, 1, 0), FT_NOFILL);
		addPlane(box_corner, Vector(0, 0, 1), FT_NOFILL);
		addPlane(other_corner, Vector(-1, 0, 0), FT_NOFILL);
		addPlane(other_corner, Vector(0, -1, 0), FT_NOFILL);
		// planes will end up half a dp from the fluid, so set r0 for the LJ force correctly:
		physparams()->r0 = m_deltap/2;
	} else {
		addBox(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, box_corner, l, w, box_height);
		// cutting plane: the normal points downwards so that it can unfill upwards
		// NOTE: fill type is FT_UNFILL because this is NOT a geometric plane involved in the simulation,
		// it's only used for cutting out unnecessary particles
		addPlane(box_corner + Vector(0, 0, box_height), Vector(0, 0, -1), FT_UNFILL);
	}

	// water
	addBox(GT_FLUID, FT_SOLID, box_corner, l, w, H);
}
