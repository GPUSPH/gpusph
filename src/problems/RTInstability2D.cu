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

#include "RTInstability2D.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

RTInstability2D::RTInstability2D(GlobalData *_gdata) : Problem(_gdata),
	domain_height(1.0),
	domain_width(domain_height/2)
{
	// *** user parameters from command line
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", COLAGROSSI);
	// periodicity along the X direction
	const bool periodic_domain = get_option("periodic", false);
	// particles in the initial water height
	const uint ppH = get_option("ppH", 128);
	// density ratio between right and heavy fluid
	const double rho_ratio = get_option("rho-ratio", 0.5);
	// kinematic viscosity of the two fluids (assumed the same)
	const float kinvisc = get_option("kinvisc", 1.0e-6f);

	SETUP_FRAMEWORK(
		space_dimensions<R2>,
		boundary<DUMMY_BOUNDARY>,
		rheology<NEWTONIAN>,
		computational_visc<KINEMATIC>,
		visc_model<MORRIS>,
		visc_average<HARMONIC>,
		formulation<SPH_F2>,
		add_flags<ENABLE_MULTIFLUID>
	).select_options(
		RHODIFF,
		periodic_domain, periodicity<PERIODIC_X>()
	);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled, 10 otherwise
	const int mlsIters = get_option("mls", 0);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// *** Initialization of minimal physical parameters
	set_deltap(domain_height/ppH);
	set_gravity(-9.81);
	// epsilon for pseudo-interface-tension in Grenier's formulation
	set_interface_epsilon(0.08);

	// max fall height is autocomputed from the position of the highest particle,
	// but we specify it anyway
	setMaxFall(domain_height);

	float rho0 = 1000;
	float rho1 = rho0*rho_ratio;

	heavy = add_fluid(rho0);
	light = add_fluid(rho1);

	// set the speed of sound for both fluids automatically
	set_equation_of_state(heavy,  7.0f, NAN);
	set_equation_of_state(light,  7.0f, NAN);

	set_kinematic_visc(heavy, kinvisc);
	set_kinematic_visc(light, kinvisc);

	simparams()->tend=1.5;

	// Save every 100th of simulated second
	add_writer(VTKWRITER, 0.01f);

	// *** Setup geometries

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);

	// main container: the top and bottom segments are always present,
	// the sides are only present if not periodic.
	// we would use segments to define the boundaries if we had the geometry (TODO),
	// but since we're using dummy boundaries we can just use appropriately-sized
	// Rectangles instead
	const double wall_thickness = getDynamicBoundariesLayers()*m_deltap;

	// When using DUMMY_BOUNDARY, the walls should be m_deltap/2 “outside” the wall. We achieve this
	// by defining the domain box components with an extra m_deltap in size, and shifting it by half m_deltap>
	// Of course, the top/bottom slabs start aligned with the fluid in the horizontal direction,
	// due to the possible presence of the side walls
	const double half_dp = m_deltap/2;

	GeometryID bottom = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
		Point( half_dp, -half_dp-wall_thickness, 0),
		domain_width, wall_thickness);

	GeometryID top = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
		Point( half_dp, domain_height + half_dp, 0),
		domain_width, wall_thickness);

	// TODO multifluid. Currently multi-fluid is not supported natively in Problem API 1,
	// so we set all particle masses assuming density 1 and then we will multiply by the actual density
	setParticleMassByDensity(top, 1);
	setParticleMassByDensity(bottom, 1);

	if (!periodic_domain) {
		const double wall_shift = wall_thickness + half_dp;
		const double wall_height = domain_height + 2*wall_thickness + m_deltap;
		GeometryID left_wall = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
			Point( -wall_shift, -wall_shift, 0),
			wall_thickness, wall_height);
		GeometryID right_wall = addRect(GT_FIXED_BOUNDARY, FT_SOLID,
			Point( domain_width + half_dp, -wall_shift, 0),
			wall_thickness, wall_height);
		setParticleMassByDensity(left_wall, 1);
		setParticleMassByDensity(right_wall, 1);
	}

	// Fluid box: we fill the entire domain with a single box, and then mark particles
	// as belonging to one fluid or the other based on the position
	GeometryID fluid_box = addRect(GT_FLUID, FT_SOLID,
		Point(half_dp, half_dp, 0), domain_width-m_deltap, domain_height-m_deltap);

	setParticleMassByDensity(fluid_box, 1);
}

double
RTInstability2D::interface_height(double x) const
{
	return domain_height/2 + 0.05*sin(2*M_PI/domain_width*x);
}

// Mass and density initialization
void
RTInstability2D::initializeParticles(BufferList &buffers, const uint numParticles)
{
	// Example usage

	// 1. warn the user if this is expected to take much time
	printf("Initializing particles density and mass...\n");

	// 2. grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	const double4 *pos_global = buffers.getData<BUFFER_POS_GLOBAL>();
	float4 *pos = buffers.getData<BUFFER_POS>();

	// we will need the gravity magnitude for the hydrostatic initialization
	const float g = get_gravity_magnitude();

	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		float rho;
		double4 const& gpos = pos_global[i];
		const double depth = domain_height - gpos.y;

		// for boundary particles, we use the heavy density,
		int fluid_idx = heavy;

		if (FLUID(info[i])) {
			const double intf = interface_height(gpos.x);
			// fluid index depends on the position wrt the interface
			fluid_idx = gpos.y < intf ? light : heavy;

			rho = hydrostatic_density(depth, fluid_idx);
			if (fluid_idx == light) {
				// pressure at interface, from heavy fluid
				float P = physparams()->rho0[heavy]*(domain_height - intf)*g;
				// plus hydrostatic pressure from _our_ fluid
				P += physparams()->rho0[light]*(intf - gpos.y)*g;
				rho = density_for_pressure(P, light);
			} else {
				rho = hydrostatic_density(depth, fluid_idx);
			}
			info[i]= make_particleinfo(PT_FLUID, fluid_idx, i);
		} else if (BOUNDARY(info[i])) {
			rho = hydrostatic_density(depth, fluid_idx);
			info[i] = make_particleinfo(PT_BOUNDARY, fluid_idx, i);
		}
		// fix up the particle mass according to the actual density
		pos[i].w *= physical_density(rho,fluid_idx);
		vel[i].w = rho;
	}
}





