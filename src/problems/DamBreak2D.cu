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

#include "DamBreak2D.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

// Geometry taken from the SPHERIC test case 2
DamBreak2D::DamBreak2D(GlobalData *_gdata) : Problem(_gdata)
{
	// *** user parameters from command line
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", DENSITY_DIFFUSION_NONE);
	// artificial viscosity: if set to a positive value, the problem witll use artificial viscosity
	// instead of kinematic viscosity (with the specified viscosity value)
	const float artvisc = get_option("artvisc", 0.0f);
	// particles in the initial water height
	const uint ppH = get_option("ppH", 30);
	// add obstacle; default is true per SPHERIC test case 2
	const bool has_obstacle = get_option("obstacle", true);

	// *** Geometrical parameters, starting from the size of the domain

	constexpr double domain_height = 1;
	//const double domain_width = 1; // unused in 2D test case
	constexpr double water_height = 0.55;
	constexpr double water_length = 1.228; // initial length of the water box
	constexpr double front_to_obstacle_center = 1.248; // distance from the initial water front to the center of the obstacle
	constexpr double obstacle_center_to_back_wall = 0.744; // distance from the obstacle center to the back wall
	constexpr double domain_length = water_length + front_to_obstacle_center + obstacle_center_to_back_wall;
	//const double obstacle_width = 0.403; // unused in 2D test case
	constexpr double obstacle_length = 0.161;
	constexpr double obstacle_height = 0.161;

	// *** Framework setup
	SETUP_FRAMEWORK(
		space_dimensions<R2>,
		viscosity<KINEMATICVISC>,
		boundary<DUMMY_BOUNDARY>
	).select_options(
		RHODIFF,
		artvisc > 0, viscosity<ARTVISC>()
	);

	// will dump testpoints separately
	addPostProcess(TESTPOINTS);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		(simparams()->densitydiffusiontype != DENSITY_DIFFUSION_NONE) ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// *** Initialization of minimal physical parameters
	set_deltap(water_height/ppH);
	set_gravity(-9.81);

	add_fluid(1000.0);
	set_equation_of_state(0,  7.0f, NAN); // sound speed NAN = autocompute
	set_kinematic_visc(0, 1.0e-6f);
	set_artificial_visc(artvisc);

	// 6s of runtime by default
	simparams()->tend=6;

	// The maximum fall height is normally taken as the initial max filling height.
	// However, we know that the splash in this case can cover the whole domain height, so:
	setMaxFall(domain_height);

	// Save every 100th of simulated second
	add_writer(VTKWRITER, 0.01f);

	// *** Setup geometries

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);

	// When using DUMMY_BOUNDARY, the walls should be m_deltap/2 “outside” the wall. We achieve this
	// by defining the domain box with an extra m_deltap in size, and shifting it by half m_deltap
	const double half_dp = m_deltap/2;
	GeometryID domain_box = addRect(GT_FIXED_BOUNDARY, FT_OUTER_BORDER,
		Point(-half_dp, -half_dp, 0), domain_length+m_deltap, domain_height+m_deltap);

	// Conversely, the water box must be shifted half_dp _inside_
	GeometryID water_box = addRect(GT_FLUID, FT_SOLID, Point(half_dp, half_dp, 0), water_length, water_height);

	if (has_obstacle) {
		setPositioning(PP_CENTER);
		GeometryID obstacle = addRect(GT_FIXED_BOUNDARY, FT_INNER_BORDER,
			Point(water_length + front_to_obstacle_center, obstacle_height/2, 0),
			// we remove one deltap from length and height because we're using dummy boundaries
			// and the particles should start half a dp _inside_ the obstacle
			obstacle_length - m_deltap, obstacle_height - m_deltap);
		// in some resolutions (e.g. 32 PPH), the obstacle deletes one row of floor boundary particles,
		// avoid that
		setEraseOperation(obstacle, ET_ERASE_NOTHING);
	}

	// 4 water gages every 0.496 from the back
	// For each gage we add a nearest-neighbor one and a smoothing one
	constexpr double gage_step = 0.496;
	constexpr int ngages = 4;
	const double gage_smoothing = simparams()->slength;
	for (int g = 0; g < ngages; ++g) {
		// gage 0 will be the one closest to the front
		const double x = domain_length - gage_step*(ngages - g);
		add_gage(x); // nearest-neighbor gage
		add_gage(x, gage_smoothing); // Wendland smoothing gage
	}

	// 8 (4x2) testpoints every 0.04 along the front and top of the obstacle, starting at 0.021
	if (has_obstacle) {
		constexpr int ntps = 4;
		constexpr double tp_x0 = water_length + front_to_obstacle_center - obstacle_length/2;
		constexpr double tp_offset = 0.021;
		constexpr double tp_gap = 0.04;
		for (int t = 0; t < ntps; ++t) {
			addTestPoint(tp_x0 , tp_offset + t*tp_gap, 0);
			addTestPoint(tp_x0 + tp_offset + t*tp_gap, obstacle_height, 0);
		}
	}

}
