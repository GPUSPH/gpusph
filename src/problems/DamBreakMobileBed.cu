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

#include "DamBreakMobileBed.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

DamBreakMobileBed::DamBreakMobileBed(GlobalData *_gdata) : XProblem(_gdata)
{
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const int RHODIFF = get_option("density-diffusion", 3);

	SETUP_FRAMEWORK(
		formulation<SPH_HA>,
		viscosity<GRANULARVISC>,
//		boundary<SA_BOUNDARY>,
//		add_flags<ENABLE_MULTIFLUID | ENABLE_DTADAPT | ENABLE_DENSITY_SUM> // Enable for SA_BOUNDARY
		periodicity<PERIODIC_Y>,
		boundary<DYN_BOUNDARY>,
		add_flags<ENABLE_MULTIFLUID> // Enable for LJ/DYN_BOUNDARY

		).select_options(
		RHODIFF == FERRARI, densitydiffusion<FERRARI>(),
		RHODIFF == BREZZI, densitydiffusion<BREZZI>(),
		RHODIFF == COLAGROSSI, densitydiffusion<COLAGROSSI>()
	);

/*	SETUP_FRAMEWORK(
		formulation<SPH_HA>,
		viscosity<RHEOLOGY>,
		boundary<SA_BOUNDARY>,
		periodicity<PERIODIC_Y>,
		densitydiffusion<FERRARI>,
		flags<ENABLE_DTADAPT | ENABLE_DENSITY_SUM>
	);
*/
	addPostProcess(INTERFACE_DETECTION);

	// SPH parameters
	simparams()->sfactor = 1.3;
	set_deltap(0.01);
	simparams()->dtadaptfactor = 0.3;
	resize_neiblist(256, 64);
	simparams()->buildneibsfreq = 10;
	simparams()->densityDiffCoeff = 0.05f;

	// Rheological parameters
	effvisc_max = 0.0960952;

	// Geometrical parameters
	hs = 0.1f; // sediment height
	hw = 0.35f; // water height
	lx = 8.2f; // reservoir length
	ly = 31.*m_deltap; // reservoir width
	lz = 2.2f; // reservoir height
	zi = 0.f; // interface horizontal position

	// Origin and size of the domain
	m_origin = make_double3(-3.1f, 0.f, -1.8f);
	m_size = make_double3(lx, ly, lz);

	// Gravity
	const float g = 9.81f;
	physparams()->gravity = make_float3(0.0, 0.0, -g);

	// Fluid 0 (water)
	const float rho0 = 1000.0f;
	const float nu0 = 1.0e-6;
	const float mu0 = rho0*nu0;

	// Fluid 1 (sediment)
	const float phi = 0.47; // porosity of the sediment
	const float rhog = 2683.0f; // density of the grains
	const float rho1 = phi*rho0 + (1.f - phi)*rhog; // density of the saturated bed

	// Speed of sound (same for the two phases)
	const float c0 = 10.f*sqrtf(g*hw);

	add_fluid(rho0);
	set_dynamic_visc(0, mu0);
	add_fluid(rho1);

	set_sinpsi(1, 0.5);
	set_cohesion(1, 0);
	// lower bound of kinematic effective viscosity
	// is set to the interstitial fluid viscosity
	set_kinematic_visc(1, nu0);
	// upper bound of kinematic effective viscosity
	set_limiting_kinvisc(effvisc_max);

	set_equation_of_state(0,  7.0f, c0);
	set_equation_of_state(1,  7.0f, c0);

	// Final time
	simparams()->tend = 20;

	// Drawing and saving times
	add_writer(VTKWRITER, 0.0625);

	// Name of problem used for directory creation
	m_name = "DamBreakMobileBed";

	// Building the geometry
	//*** Add the Fluid
	addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/DamBreakMobileBed/DYN_dr0dot01/fluid.h5sph", NULL);

	//*** Add the Main Container
	GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/DamBreakMobileBed/DYN_dr0dot01/tank.h5sph", NULL);
	disableCollisions(container);
}

// fluid is sediment if pos.z <= zi
bool is_sediment(double4 const& pt, float zi)
{
	return pt.z <= zi;
}

// Mass and density initialization
	void
DamBreakMobileBed::initializeParticles(BufferList &buffers, const uint numParticles)
{
	// 1. warn the user if this is expected to take much time
	printf("Initializing particles density and mass...\n");

	// 2. grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos_global = buffers.getData<BUFFER_POS_GLOBAL>();
	float4 *pos = buffers.getData<BUFFER_POS>();
	float *effvisc = buffers.getData<BUFFER_EFFVISC>();
	float *effpres = buffers.getData<BUFFER_EFFPRES>();
	const float g = length(physparams()->gravity);

	float rho = 1;
	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		float P(0.);
		int fluid_idx = is_sediment(pos_global[i], zi - m_deltap/2.f) ? 1 : 0;
		if (FLUID(info[i])) {
			info[i]= make_particleinfo(PT_FLUID, fluid_idx, i);

			if (pos_global[i].z <= zi && 
					pos_global[i].z >= zi - 2.*m_deltap) {
				SET_FLAG(info[i], FG_INTERFACE);
			}

			if (is_sediment(pos_global[i], zi)) {
				SET_FLAG(info[i], FG_SEDIMENT);
				P = hw*g*physparams()->rho0[0]
					+ (zi - pos_global[i].z)*g*physparams()->rho0[1];
			} else {
				P = (hw-pos_global[i].z)*g*physparams()->rho0[0];
			}
			rho = density_for_pressure(P, fluid_idx);
			pos[i].w *= physparams()->rho0[fluid_idx]/physparams()->rho0[0];
			vel[i].w = rho;
		}

		// initialize effective pressure for all particles
		if (is_sediment(pos_global[i], zi)) { // if sediment
			const float delta_rho = physparams()->rho0[1]-physparams()->rho0[0];
			effpres[i] = fmax(delta_rho*g*(m_deltap+zi-pos_global[i].z), 0.f);
			effvisc[i] = effvisc_max;
		} else {
			effpres[i] = 0.f;
			effvisc[i] = 1e-6;
		}
	}
}
