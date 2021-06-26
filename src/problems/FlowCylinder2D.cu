/*  Copyright (c) 2021 INGV, EDF, UniCT, JHU

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

#include "FlowCylinder2D.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

// Flow around a periodic lattice of cylinders
// The geometry is taken from Morris et al. (1997) JCP
// with the option to add side walls

FlowCylinder2D::FlowCylinder2D(GlobalData *_gdata) : Problem(_gdata)
{
	// *** user parameters from command line
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", COLAGROSSI);
	// particles in the domain length
	const uint ppH = get_option("ppH", 64);
	// Periodicity along Y (the flow is always periodic in the X direction
	const bool periodic_y = get_option("periodic-y", true);

	// *** Geometrical parameters, starting from the size of the domain

	constexpr double domain_size = 0.1;
	constexpr double cylinder_radius = 0.02;
	constexpr double F = 1.5e-7;

	// *** Framework setup
	SETUP_FRAMEWORK(
		space_dimensions<R2>,
		viscosity<KINEMATICVISC>,
		boundary<DUMMY_BOUNDARY>,
		periodicity<PERIODIC_X>
	).select_options(
		RHODIFF,
		periodic_y, periodicity<PERIODIC_XY>()
	);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		(simparams()->densitydiffusiontype != DENSITY_DIFFUSION_NONE) ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// *** Initialization of minimal physical parameters
	set_deltap(domain_size/ppH);
	// external force in the X direction
	set_gravity(F, 0.0, 0.0);

	auto fluid = add_fluid(1.0);
	// Morris uses a speed of sound of 5.77e-4, we round up
	set_equation_of_state(fluid,  7.0f, 6.0e-4);
	set_kinematic_visc(0, 1.0e-6);

	simparams()->tend = 6000;

	// Save every 10 simulated seconds
	add_writer(VTKWRITER, 60.0f);

	// *** Setup geometries

	// set positioning policy to PP_CENTER: given point will be the center of the geometry.
	setPositioning(PP_CENTER);

	// The water box is centered at the origin. Its sizes are dictated by the domain_size
	// in both direction, minus one deltap to account for half deltap gap between particles
	// and their periodic neighbors
	addRect(GT_FLUID, FT_SOLID, Point(0, 0, 0), domain_size-m_deltap, domain_size-m_deltap);

	// The cylinder is also centered at the origin. The radius is half dp less than the
	// analytical to account for the DUMMY gap
	GeometryID disk = addDisk(GT_FIXED_BOUNDARY, FT_INNER_BORDER, Point(0, 0, 0), cylinder_radius - m_deltap/2);
	// the disk is a bit too generous when unfilling, try to keep more fluid particles around it
	setUnfillRadius(disk, 0.9*m_deltap);

	// Finally, if not periodic along the y axis, we need to add the top and bottom floor.
	if (!periodic_y) {
		const double wall_thickness = getDynamicBoundariesLayers()*m_deltap;
		const double offset = (domain_size + m_deltap + wall_thickness)/2;
		GeometryID bottom = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
			Point(0, -offset, 0),
			domain_size - m_deltap, wall_thickness);
		GeometryID top = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
			Point(0, offset, 0),
			domain_size - m_deltap, wall_thickness);
		// reduce the clearing radius, since it's a bit too generous in removing things
		setUnfillRadius(bottom, 0.9*m_deltap);
		setUnfillRadius(top, 0.9*m_deltap);
	}
}
