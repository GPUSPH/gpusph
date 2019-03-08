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

#include <iostream>

#include "LockExchange.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

LockExchange::LockExchange(GlobalData *_gdata) : XProblem(_gdata)
{
	//const double dimX = 1.82;
	dimX = 0.6;
	dimY = 0.23;
	dimZ = 0.2;

	H = dimZ;

	const bool USE_PLANES = get_option("use_planes", false);
	const uint NUM_TESTPOINTS = get_option("num_testpoints", 0);
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

	// will dump testpoints separately
	addPostProcess(TESTPOINTS);

	// Allow user to set the MLS frequency at runtime. Default to 0 if density
	// diffusion is enabled, 10 otherwise
	const int mlsIters = get_option("mls",
		(simparams()->densitydiffusiontype != DENSITY_DIFFUSION_NONE) ? 0 : 10);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// Explicitly set number of layers. Also, prevent having undefined number of layers before the constructor ends.
	setDynamicBoundariesLayers(3);

	resize_neiblist(128);

	// *** Initialization of minimal physical parameters
	set_deltap(H/32);
	//simparams()->dt = 0.00005;
	physparams()->r0 = m_deltap;
	physparams()->gravity = make_float3(0.0, 0.0, -9.81);
	const float g = length(physparams()->gravity);
	physparams()->epsinterface = 0.08;

	// If we used only makeUniverseBox(), origin and size would be computed automatically
	m_origin = make_double3(0, 0, 0);
	m_size = make_double3(dimX, dimY, dimZ);

	physparams()->dcoeff = 5.0f * g * H;

	float rho0 = 1000;
	float rho1 = 2350;

	light = add_fluid(rho0);
	heavy = add_fluid(rho1);

	set_equation_of_state(light,  7.0f, 20.0f);
	set_equation_of_state(heavy,  7.0f, 20.0f);

	set_kinematic_visc(light, 1.0e-2f);
	set_kinematic_visc(heavy, 1.0e-2f);

	// default tend 1.5s
	simparams()->tend=20.0f;
	//simparams()->ferrariLengthScale = H;
	simparams()->densityDiffCoeff = 0.1f;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.005f);

	m_name = "LockExchange";

	// set positioning policy to PP_CORNER: given point will be the corner of the geometry
	setPositioning(PP_CORNER);

	// main container
	if (USE_PLANES) {
		// limit domain with 6 planes
		makeUniverseBox(m_origin, m_origin + m_size);
	} else {
		GeometryID box =
			addBox(GT_FIXED_BOUNDARY, FT_BORDER, m_origin, dimX, dimY, dimZ);
		// we simulate inside the box, so do not erase anything
		setEraseOperation(box, ET_ERASE_NOTHING);
		setParticleMassByDensity(box, 1);
	}

	// Planes unfill automatically but the box won't, to void deleting all the water. Thus,
	// we define the water at already the right distance from the walls.
	double BOUNDARY_DISTANCE = m_deltap;

	if (simparams()->boundarytype == DYN_BOUNDARY && !USE_PLANES)
			BOUNDARY_DISTANCE *= getDynamicBoundariesLayers();

	// Add the main water part
	GeometryID light_box = addBox(GT_FLUID, FT_SOLID, Point(BOUNDARY_DISTANCE, BOUNDARY_DISTANCE, BOUNDARY_DISTANCE),
		dimX/2 - BOUNDARY_DISTANCE + m_deltap, dimY - 2 * BOUNDARY_DISTANCE , dimZ - 2 *BOUNDARY_DISTANCE);

	GeometryID heavy_box = addBox(GT_FLUID, FT_SOLID, Point(m_deltap + dimX/2, BOUNDARY_DISTANCE, BOUNDARY_DISTANCE),
		dimX/2 - BOUNDARY_DISTANCE - m_deltap, dimY - 2 * BOUNDARY_DISTANCE , dimZ - 2 *BOUNDARY_DISTANCE);

	disableCollisions(light_box);
	setParticleMassByDensity(light_box, 1);

	disableCollisions(heavy_box);
	setParticleMassByDensity(heavy_box, 1);

	// set positioning policy to PP_BOTTOM_CENTER: given point will be the center of the base
	//setPositioning(PP_BOTTOM_CENTER);

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





