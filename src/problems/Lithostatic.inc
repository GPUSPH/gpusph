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

#ifndef LITHOSTATIC_PROBLEM
	#error "Please define LITHOSTATIC_PROBLEM"
#endif

#include "utils.h"
#include "Lithostatic.h.inc"
#include "GlobalData.h"
#include "cudasimframework.cu"

LITHOSTATIC_PROBLEM::LITHOSTATIC_PROBLEM(GlobalData *_gdata) :
	Problem(_gdata),
	hs ((floor(1.0f/m_deltap)-0.5f)*m_deltap), // sediment height
	hw (floor(1.0f/m_deltap)*m_deltap), // water height
	H (2.5f), // reservoir height
	l (1.0f), // reservoir length and width
	// For LJ/DYN_boundary, choose if Salome geometry is used
	salomeGeom (true)
{
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const int RHODIFF = get_option("density-diffusion", 0);

	SETUP_FRAMEWORK(
		formulation<SPH_HA>,
		viscosity<GRANULARVISC>,
		boundary<LITHOSTATIC_BOUNDARY>,
		add_flags<LITHOSTATIC_FLAGS> // Enable for SA_BOUNDARY
		).select_options(
		RHODIFF == FERRARI, densitydiffusion<FERRARI>(),
		RHODIFF == BREZZI, densitydiffusion<BREZZI>(),
		RHODIFF == COLAGROSSI, densitydiffusion<COLAGROSSI>()
	);


	// Detection of free-surface and interface is mandatory for GRANULARVISC
	addPostProcess(INTERFACE_DETECTION);

	// SPH parameters
	simparams()->sfactor = 2.0;
	set_deltap(0.025);
	simparams()->dtadaptfactor = 0.3;
	resize_neiblist(512, 128);
	simparams()->buildneibsfreq = 10;
	simparams()->densityDiffCoeff = 0.5f;

	// Effective pressure Jacobi solver parameters
	simparams()->jacobi_maxiter = 10000;
	simparams()->jacobi_backerr = 3e-5;
	simparams()->jacobi_residual = 3e-5;

	/* Geometrical parameters 
	 * 	-the sediment and water heights hs and hw are determined to
	 *	 be the closest value to 1 that can be divided by deltap.
	 *	-the sediment height is calculated taking the bottom vertex
	 *	 particle height: deltap/2
	*/
	hs = (floor(1.0f/m_deltap)-0.5f)*m_deltap ; // sediment height
	hw =  floor(1.0f/m_deltap)*m_deltap; // water height
	H = 2.5f; // reservoir height
	l = 1.0f; // reservoir length and width

	// Origin and size of the domain
	m_size = make_double3(l + 20.f*m_deltap, l + 20.f*m_deltap, H + 20.f*m_deltap);
	m_origin = make_double3(-10.f*m_deltap, -10.f*m_deltap, -10.f*m_deltap);

	// Gravity
	const float g = 9.81f;
	physparams()->gravity = make_float3(0.0, 0.0, -g);

	// Fluid 0 (water)
	const float rho0 = 1000.f;
	const float nu0 = 1.0e-6;
	const float mu0 = rho0*nu0;

	// Fluid 1 (sediment)
	const float rho1 = 1892.0f;

	// Speed of sound (same for the two phases)
	const float c0 = 10.f*sqrtf(g*H);

	// Characteristic time
	const float tref = sqrtf(H/g);

	add_fluid(rho0);
	set_dynamic_visc(0, mu0);
	add_fluid(rho1);

	set_sinpsi(1, 0.5);
	set_cohesion(1, 0);
	// lower bound of kinematic effective viscosity
	// is set to the interstitial fluid viscosity
	set_kinematic_visc(1, nu0);
	// upper bound of kinematic effective viscosity
	// (here we just want to validate effective pressure)
	set_limiting_kinvisc(1e-6);


	set_equation_of_state(0,  7.0f, c0);
	set_equation_of_state(1,  7.0f, c0);

	simparams()->tend = 100*tref;

	// Drawing and saving times
	add_writer(VTKWRITER, tref/1000.f);

	// Name of problem used for directory creation
	m_name = "LithostaticEquilibriumSA";
	GeometryID container(0);

	// Add the geometry: depends on boundarytype. For boundarytype != SA_BOUNDARY, the geometry can be
	// either imported from a salome h5sph file, or created with addBox.

	// SA case: geometry is imported from Salome file
	if (simparams()->boundarytype == SA_BOUNDARY) {
		// Add the fluid from Salome geometry
		addHDF5File(GT_FLUID, Point(0,0,0),
		"./data_files/LithostaticEquilibrium/LithostaticEquilibriumSA_presph/dr0dot025/0.LithostaticEquilibrium_dr0dot025.fluid.h5sph", NULL);
		// Add the main container from Salome geometry
		container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0),
		"./data_files/LithostaticEquilibrium/LithostaticEquilibriumSA_presph/dr0dot025/0.LithostaticEquilibrium_dr0dot025.boundary.kent0.h5sph", NULL);

	// LJ with Salome geometry
	} else if (simparams()->boundarytype == LJ_BOUNDARY && salomeGeom) {
		// Add the fluid from Salome geometry
		addHDF5File(GT_FLUID, Point(0,0,0),
		"./data_files/LithostaticEquilibrium/LithostaticEquilibriumLJ_presph/dr0dot025/dr0dot025.fluid.h5sph", NULL);
		// Add the main container from Salome geometry
		container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0),
		"./data_files/LithostaticEquilibrium/LithostaticEquilibriumLJ_presph/dr0dot025/dr0dot025.Mesh_tank.h5sph", NULL);

	// DYN with Salome geometry
	} else if (simparams()->boundarytype == DYN_BOUNDARY && salomeGeom) {
		// Add the fluid from Salome geometry
		addHDF5File(GT_FLUID, Point(0,0,0),
		"./data_files/LithostaticEquilibrium/LithostaticEquilibriumDB_presph/dr0dot025/dr0dot025.fluid.h5sph", NULL);
		// Add the main container from Salome geometry
		container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0),
		"./data_files/LithostaticEquilibrium/LithostaticEquilibriumDB_presph/dr0dot025/dr0dot025.tank.h5sph", NULL);

	// LJ or DYN without Salome geometry
	} else {

		// Add the fluid with addBox 
		addBox(GT_FLUID, FT_SOLID, Point(m_origin + make_double3(l/2.,l/2., (hw+hs)/2.)), l, l, hw+hs);
		// Add the main container with addBox 
		double boundary_thickness = simparams()->boundarytype == DYN_BOUNDARY ? 3*m_deltap : 0.0;
		container = addBox(GT_FIXED_BOUNDARY, FT_BORDER,
			Point(m_origin + make_double3(l/2.,l/2.,H/2.) - make_double3(0.0f)),
				l+2*boundary_thickness, l+2*boundary_thickness, H+2*boundary_thickness);
		setEraseOperation(container, ET_ERASE_NOTHING);
	}
	disableCollisions(container);
}

// fluid is sediment if pos.z <= hs
bool is_sediment(double4 const& pt, float hs)
{
	return pt.z <= hs;
}

// Mass and density initialization
	void
LITHOSTATIC_PROBLEM::initializeParticles(BufferList &buffers, const uint numParticles)
{
#if 1
	// Warn the user if this is expected to take much time
	printf("Initializing particles density and mass...\n");

	// Choose to initialize effective pressure with
	// analytical lithostatic profile.
	bool lithostaticInitialization = false;

	// Grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos_global = buffers.getData<BUFFER_POS_GLOBAL>();
	float4 *pos = buffers.getData<BUFFER_POS>();
	float *effvisc = buffers.getData<BUFFER_EFFVISC>();
	float *effpres = buffers.getData<BUFFER_EFFPRES>();
	float g = length(physparams()->gravity);

	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		int fluid_idx = is_sediment(pos_global[i], hs) ? 1 : 0;
		float tol = 1e-6;
		if (FLUID(info[i])) {
			info[i]= make_particleinfo(PT_FLUID, fluid_idx, i);
			if (pos_global[i].z <= hs + tol &&
					pos_global[i].z >= hs - m_deltap) {
				SET_FLAG(info[i], FG_INTERFACE);
			}

			if (is_sediment(pos_global[i], hs)) {
				SET_FLAG(info[i], FG_SEDIMENT);
			}
			// initialize pressure for all particles
			float z_max = is_sediment(pos_global[i], hs) ? 1. : 2.;
			float P_min = is_sediment(pos_global[i], hs) ? physparams()->rho0[0]*g : 0.;
			float P = max(physparams()->rho0[fluid_idx]*g*(z_max - pos_global[i].z) + P_min, 0.);
			float rho = density_for_pressure(P, fluid_idx);

			pos[i].w *= physical_density(rho,fluid_idx)/physparams()->rho0[0];
			vel[i].w = rho;
		}
		// initialize effective pressure for all particles
		if (is_sediment(pos_global[i], hs)) { // if sediment
			const float delta_rho = physparams()->rho0[1]-physparams()->rho0[0];
			if (lithostaticInitialization) {
				// initialize with analytical profile
				effpres[i] = fmax(delta_rho*g*(m_deltap+hs-pos_global[i].z), 0.f);
			} else {
				// initialize to zero
				effpres[i] = 0.;
			}
			effvisc[i] = 1e-6;
		} else {
			effpres[i] = 0.f;
			effvisc[i] = 1e-6;
		}
	}
#endif
}


// since the fluid topology is roughly symmetric along Z through the whole simulation, prefer Z split
void LITHOSTATIC_PROBLEM::fillDeviceMap()
{
	fillDeviceMapByAxis(Z_AXIS);
}

