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

#include "LockExchange.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

LockExchange::LockExchange(GlobalData *_gdata) : Problem(_gdata)
{
	const bool USE_PLANES = get_option("use_planes", false);
	const int ppH = get_option("ppH", 32);
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", COLAGROSSI);

	SETUP_FRAMEWORK(
		formulation<SPH_GRENIER>,
		//formulation<SPH_F2>,
		viscosity<DYNAMICVISC>,
		boundary<DYN_BOUNDARY>,
		add_flags<ENABLE_MULTIFLUID>
	).select_options(
		RHODIFF,
		USE_PLANES, add_flags<ENABLE_PLANES>()
	);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		(simparams()->densitydiffusiontype != DENSITY_DIFFUSION_NONE) ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	dimX = 1.82;
	dimY = 0.23;
	dimZ = 0.2;

	H = dimZ;

	// *** Initialization of minimal physical parameters
	set_deltap(H/ppH);
	set_gravity(-9.81);
	set_interface_epsilon(0.08);

	float rho0 = 1000;
	float rho1 = 2350;

	light = add_fluid(rho0);
	heavy = add_fluid(rho1);

	// autocompute speed of sound
	set_equation_of_state(light,  7.0f, NAN);
	set_equation_of_state(heavy,  7.0f, NAN);

	set_kinematic_visc(light, 1.0e-2f);
	set_kinematic_visc(heavy, 1.0e-2f);

	simparams()->tend=20.0f;
	simparams()->densityDiffCoeff = 0.1f;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.005f);

	m_name = "LockExchange";

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);

	// use BORDER_TANGNET filling so that geometries are filled starting half a dp inside
	// rather than on the border
	setFillingMethod(Object::BORDER_TANGENT);

	const Point corner(0, 0, 0);

	// main container
	if (USE_PLANES) {
		// limit domain with 6 planes. Due to our filling method, using:
		//   makeUniverseBox(corner, corner + Vector(dimX, dimY, dimZ));
		// would place the planes half a dp from the fluid,
		// which would be correct if we used ghost particles,
		// but with the LJ planes we have currently, we should have a full dp.
		const double half_dp = m_deltap/2;
		const Vector half_dp_vec = Vector(half_dp, half_dp, half_dp);
		const Vector dim_vec = Vector(dimX, dimY, dimZ);
		makeUniverseBox(corner - half_dp_vec, corner + dim_vec + half_dp_vec);
	} else {
		GeometryID domain_box = addBox(GT_FIXED_BOUNDARY, FT_OUTER_BORDER, corner, dimX, dimY, dimZ);
		// actual masses and densities will be initialized in initializeParticles()
		setParticleMassByDensity(domain_box, 1);
	}

	// Add the main water part
	GeometryID light_box = addBox(GT_FLUID, FT_SOLID, corner, dimX/2, dimY, dimZ);
	GeometryID heavy_box = addBox(GT_FLUID, FT_SOLID, corner + Vector(dimX/2, 0, 0), dimX/2, dimY, dimZ);
	// there is no interference in the filling, so avoid any spurious unfill due to rounding issues
	setEraseOperation(heavy_box, ET_ERASE_NOTHING);

	// actual masses and densities will be initialized in initializeParticles()
	setParticleMassByDensity(light_box, 1);
	setParticleMassByDensity(heavy_box, 1);
}

// since the fluid topology is roughly symmetric along Y through the whole simulation, prefer Y split
void LockExchange::fillDeviceMap()
{
	fillDeviceMapByAxis(Y_AXIS);
}

bool is_light(float R, double4 const& pt)
{
	return pt.x < R/2;
}

// Mass and density initialization
	void
LockExchange::initializeParticles(BufferList &buffers, const uint numParticles)
{
	// Example usage

	// 1. warn the user if this is expected to take much time
	printf("Initializing particles density and mass...\n");

	// 2. grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos_global = buffers.getData<BUFFER_POS_GLOBAL>();
	float4 *pos = buffers.getData<BUFFER_POS>();

	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		float rho;
		double depth = H - pos_global[i].z;
		// for boundary particles, we use the heavy density,
		int fluid_idx = heavy;
		if (FLUID(info[i])) {
			fluid_idx = is_light(dimX, pos_global[i]) ? light : heavy;
			info[i]= make_particleinfo(PT_FLUID, fluid_idx, i);
		} else if (BOUNDARY(info[i])) {
			info[i]= make_particleinfo(PT_BOUNDARY, fluid_idx, i);
		}
		// fix up the particle mass according to the actual density
		rho = hydrostatic_density(depth, fluid_idx);
		pos[i].w *= physical_density(rho,fluid_idx);
		vel[i].w = rho;
	}
}





