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

#ifndef BIFLUIDPOISEUILLE_PROBLEM
	#error "Please define BIFLUIDPOISEUILLE_PROBLEM"
#endif

#include "utils.h"
#include "BiFluidPoiseuille.h.inc"
#include "GlobalData.h"
#include "cudasimframework.cu"

BIFLUIDPOISEUILLE_PROBLEM::BIFLUIDPOISEUILLE_PROBLEM(GlobalData *_gdata) :
	Problem(_gdata)
{
	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const int RHODIFF = get_option("density-diffusion", 3);

	SETUP_FRAMEWORK(
		formulation<SPH_HA>,
		rheology<NEWTONIAN>,
		turbulence_model<LAMINAR_FLOW>,
		computational_visc<DYNAMIC>,
		visc_model<MORRIS>,
		visc_average<HARMONIC>,
		boundary<BIFLUIDPOISEUILLE_BOUNDARY>,
		periodicity<PERIODIC_XY>,
		add_flags<BIFLUIDPOISEUILLE_FLAGS>
		).select_options(
		RHODIFF == FERRARI, densitydiffusion<FERRARI>(),
		RHODIFF == BREZZI, densitydiffusion<BREZZI>(),
		RHODIFF == COLAGROSSI, densitydiffusion<COLAGROSSI>()
	);

	// SPH parameters
	simparams()->sfactor = 1.3;
	set_deltap(0.05);
	simparams()->dtadaptfactor = 0.3;
	resize_neiblist(210, 70);
	simparams()->buildneibsfreq = 10;

	H = 2.0; // channel height
	simparams()->densityDiffCoeff = 0.05;

	if (simparams()->boundarytype == SA_BOUNDARY) {
		margin = make_double3(0.0);
	} else {
		margin = make_double3(0., 0, 0.2); // no margin in the periodic directions X & Y
	}

	if (simparams()->boundarytype == SA_BOUNDARY) {
		// Size and origin of the simulation domain
		l = H/4.; // channel length (X-perdiodic direction)
		a = H/4.; // channel width (Y-periodic direction)
	} else {
		// Size and origin of the simulation domain
		l = round_up(H/4., m_deltap);
		a = round_up(H/4., m_deltap);
		h = round_up(H, m_deltap);
	}

	m_size = make_double3(l, a, H) + 2.f*make_double3(margin.x, margin.y, margin.z);
	m_origin = make_double3(-a/2., -l/2., -H/2.f) - make_double3(margin.x, margin.y, margin.z);

	// Physical parameters
	const float g = 0.1;
	physparams()->gravity = make_float3(g, 0.0, 0.0);

	// Interface position
	alpha = 0.5f;
	// Fluid 0
	const float rho0 = 4000.0f;
	const float nu0 = 0.4;
	float lambda = 1; // lambda = rho0/rho1
	float omega = 1; // omega = nu0/nu1
	// Fluid 1
	int config = 1;
	/* Configurations description
	config = 0 -- standard one fluid Poiseuille flow
	config = 1 -- reference case omega = 4 / lambda = 4
	config = 2 -- high density ratio case with parabolic profile (same theoretical
	solution as the standard one fluid Poiseuille flow)
	*/
	if (config == 0) {
		lambda = 1;
		omega = 1;
	} else if (config == 1) {
		lambda = 4;
		omega = 4;
	} else if (config == 2) {
		lambda = 0.01;
		omega = 1;
	}

	const float rho1 = rho0/lambda;
	const float nu1 = nu0/omega;

	const float uref = g*H*H/(2*nu0);

	/* Compute the max velocity in fluid 0 and fluid 1 */
	const float umax0 = uref*(
		(omega+2.*alpha*(lambda-1.)*omega+alpha*alpha*(1.+omega-2.*lambda*omega))*
		(omega+2.*alpha*(lambda-1.)*omega+alpha*alpha*(1.+omega-2.*lambda*omega))/
		(4*
		(alpha+lambda*omega-alpha*lambda*omega)*
		(alpha+lambda*omega-alpha*lambda*omega))
		);

	const float umax1 = uref*(omega*
		(alpha*(2.+alpha*(lambda-2.))+(alpha-1.)*(alpha-1.)*lambda*omega)*
		(alpha*(2.+alpha*(lambda-2.))+(alpha-1.)*(alpha-1.)*lambda*omega)/
		(4.*
		(alpha+lambda*omega-alpha*lambda*omega)*
		(alpha+lambda*omega-alpha*lambda*omega))
		);

	/* Compute the max velocity in the domaim */
	const float umax = fmaxf(umax0, umax1);

	/* Compute the viscous effect charateristic propagation time */
	const float tvisc=fmaxf(H*H/nu0, H*H/nu1);

	bottom =add_fluid(rho0);
	top =	add_fluid(rho1);

	set_equation_of_state(top,	7.0f, 10.f*umax);
	set_equation_of_state(bottom,	7.0f, 10.f*umax);

	set_kinematic_visc(bottom, nu0);
	set_kinematic_visc(top, nu1);

	simparams()->tend = 250.f*tvisc;

	// Drawing and saving times
	add_writer(VTKWRITER, 1);

	// Name of problem used for directory creation
	m_name = "BiFluidPoiseuille";

	if (simparams()->boundarytype == SA_BOUNDARY) {
		// Building the geometry
		//*** Add the Fluid
		GeometryID fluid = addHDF5File(GT_FLUID, Point(0,0,0), "./data_files/PoiseuilleFlowSA/0.PoiseuilleFlowSA.fluid.h5sph", NULL);

		//*** Add the Main Container
		GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0,0,0), "./data_files/PoiseuilleFlowSA/0.PoiseuilleFlowSA.boundary.kent0.h5sph", NULL);
		disableCollisions(container);
	} else {
		// Building the geometry
		dyn_layers = ceil(simparams()->influenceRadius/m_deltap) + 1;
		// no extra offset in the X direction, since we have periodicity there
		// no extra offset in the Y direction either
		dyn_offset = dyn_layers*make_double3(0, 0, m_deltap);
		// gap due to periodicity
		const double3 periodicity_gap = make_double3(m_deltap/2, m_deltap/2, 0);

		/* Walls */
		GeometryID walls = addBox(GT_FIXED_BOUNDARY, FT_SOLID,
				periodicity_gap + make_double3(0, 0, 0), l - m_deltap, a - m_deltap, H - m_deltap + 2*dyn_offset.z) ;
		disableCollisions(walls);
		GeometryID unfill_walls = addBox(GT_FIXED_BOUNDARY, FT_NOFILL,
				periodicity_gap + make_double3(0, 0, 0), l - m_deltap, a - m_deltap, H - 2*m_deltap) ;
		disableCollisions(unfill_walls);
		setEraseOperation(unfill_walls, ET_ERASE_BOUNDARY);
		/* Fluid */
		addBox(GT_FLUID, FT_SOLID,
				periodicity_gap + make_double3(0, 0, 0), l - m_deltap, a - m_deltap, H - m_deltap) ;

	}
}

// fluid 0 is in z < H*(alpha-0.5f) 
bool is_fluid0(double4 const& pt, float H, float alpha)
{
	return pt.z < H*(alpha-0.5f);
}

// Mass and density initialization
	void
BIFLUIDPOISEUILLE_PROBLEM::initializeParticles(BufferList &buffers, const uint numParticles)
{
	// 1. warn the user if this is expected to take much time
	printf("Initializing particles density and mass...\n");

	// 2. grab the particle arrays from the buffer list
	float4 *vel = buffers.getData<BUFFER_VEL>();
	particleinfo *info = buffers.getData<BUFFER_INFO>();
	double4 *pos_global = buffers.getData<BUFFER_POS_GLOBAL>();
	float4 *pos = buffers.getData<BUFFER_POS>();
	// 3. iterate on the particles
	for (uint i = 0; i < numParticles; i++) {
		int fluid_idx = is_fluid0(pos_global[i], H, alpha) ? bottom : top;
		if (FLUID(info[i])) {
			info[i]= make_particleinfo(PT_FLUID, fluid_idx, i);
		// with SPH_HA, the density of boundaries shouldn't have any effect.
		// use the following condition to do some test.
		} else if (VERTEX(info[i])) {
			info[i]= make_particleinfo(PT_VERTEX, fluid_idx, i);
		} else if (BOUNDARY(info[i])) {
			info[i]= make_particleinfo(PT_BOUNDARY, fluid_idx, i);
		}
		// masses were initialized to 1 when using the addBox etc in the DYN case,
		// but were initialized assuming rho0[0] in the SA case, so to get the proper
		// mass for each particle type we have to multiply by a scale factor that depends
		// on the boundary type we're using:
		float mass_scale = physparams()->rho0[fluid_idx];
		if (simparams()->boundarytype == SA_BOUNDARY)
			mass_scale /= physparams()->rho0[0];
		pos[i].w *= mass_scale;
		vel[i].w = atrest_density(fluid_idx);
	}
}
