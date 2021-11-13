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

#include "LidDrivenCavity2D.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

// Geometry taken from the SPHERIC test case 2
LidDrivenCavity2D::LidDrivenCavity2D(GlobalData *_gdata) : Problem(_gdata),
	// initialize the lid velocity from the command-line, defaulting to 1m/s
	lid_vel(get_option("lid-velocity", 1.0)),
	// allow users to override the lead-in time, defaulting to 1s
	lead_in_time(get_option("lead-in-time", 1.0))
{
	// *** user parameters from command line
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", COLAGROSSI);
	// particles in the domain length
	const uint ppH = get_option("ppH", 64);
	// Reynolds number of the simulation
	const double Re = get_option("Re", 100.0);
	// Incompressibility: ratio of c0 to the lid velocity.
	// This problem benefits from “true” incompressibility, so using a higher ratio
	// than the default can help. This option allows running multiple simulations
	// with different values without recompiling (e.g. try --c0-ratio 200 and
	// compare with the default 20).
	const double c0_ratio = get_option("c0-ratio", 20);

	// *** Geometrical parameters, starting from the size of the domain

	constexpr double domain_length = 1;
	constexpr double domain_height = domain_length;

	// *** Framework setup
	SETUP_FRAMEWORK(
		space_dimensions<R2>,
		viscosity<KINEMATICVISC>,
		boundary<DUMMY_BOUNDARY>
	).select_options(
		RHODIFF
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
	set_deltap(domain_length/ppH);
	// no external forces
	set_gravity(0);

	auto fluid = add_fluid(1.0);
	// The lid driven cavity should really be incompressible, so for WCSPH we want a _very_
	// weak compressibility. Rather than the usual factor of 10 between the maximum speed 
	// (i.e. the lid velocity) and the speed of sound, we'll go with a factor of 200
	set_equation_of_state(fluid,  7.0f, c0_ratio*lid_vel);
	// Since we have length 1, Re = lid_vel/nu => nu = lid_vel/Re
	set_kinematic_visc(0, lid_vel/Re);

	// We should run until we reach a steady state. I can't find a formula for
	// the time to reach it, but it seems to be no larger than Re/5, so:
	simparams()->tend=lead_in_time + Re/5;

	// Save every 10th of simulated second
	add_writer(VTKWRITER, 0.1f);

	// *** Setup geometries

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);
	// set filling method to BORDER_TANGENT: particle layers will start half a m_deltap inside
	// (or outside) the geometry.
	setFillingMethod(Object::BORDER_TANGENT);

	const Point corner = Point(0, 0, 0);
	GeometryID domain_box = addRect(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, corner, domain_length, domain_height);

	GeometryID water_box = addRect(GT_FLUID, FT_SOLID, corner, domain_length, domain_height);

	// We need to cut off the top of the box to replace it with the moving lid.
	// We use a plane to cut the existing particles off, and then use a Segment to recreate it.

	// cutting plane at y = domain_height (i.e. y - domain_height = 0), with downwards normal
	// FT_UNFILL means: use this plane for unfilling (cutting out particles), but do not
	// actually add the plane to the geometries of the simulation
	GeometryID cutting_plane = addPlane(0, -1, 0, domain_height, FT_UNFILL);
	// the plane should only cut _behind_. Normaly it cuts up to dp forward too
	setUnfillRadius(cutting_plane, 0);

	// place the lid, taking into account that its length should be enough to cover
	// the extra boundary layers
	const double outer_padding = getNumBoundaryLayers()*m_deltap;
	GeometryID lid = addSegment(GT_MOVING_BODY, FT_OUTER_BORDER,
		corner + Vector(-outer_padding, domain_height, 0),
		domain_length + 2*outer_padding);
	// note that by default the outer normal is downwards,
	// so we need to rotate the segment. We do the rotation around the x axis
	// so that it actually remains in-place, just flipping the normal
	rotate(lid, M_PI, 0, 0);
}

void
LidDrivenCavity2D::moving_bodies_callback(const uint index,
	Object *object, const double t0, const double t1,
	const float3& force, const float3& torque,
	const KinematicData& initial_kdata, KinematicData& kdata,
	double3& dx, EulerParameters& dr)
{
	// no rotation
	dr.Identity();
	// no angular velocity
	kdata.avel = make_double3(0.0);

	// linear velocity, growing from 0 to lid_vel in lead_in_time,
	// if lead_in_time is > 0
	const double scale = lead_in_time > 0 ? fmin(t1/lead_in_time, 1) : 1.0;
	kdata.lvel = make_double3(scale*lid_vel, 0, 0);
	// don't actually move the particles though
	dx = make_double3(0.0);
}
