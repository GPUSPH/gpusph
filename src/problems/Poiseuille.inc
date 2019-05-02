/* This is the 'core' implementation of the POISEUILLE_PROBLEM problem,
 * used to set up the problem in the same way for all rheological models.
 * This allows us to define multiple problems by just including this file with appropriate
 * rheological models.
 * For usage, create a problem file that defines the following macros:
 * POISEUILLE_PROBLEM to the actual name of the problem
 * POISEUILLE_RHEOLOGY to the name of the rheological model
 * and includes Poiseuille.h and Poiseuille.inc
 * The default for POISEUILLE_PROBLEM is POISEUILLE_PROBLEM,
 * and the default for POISEUILLE_RHEOLOGY is NEWTONIAN.
 * If POISEUILLE_RHEOLOGY is defined, it is assumed that we are using a non-Newtonian rheology
 */

#include "utils.h"

#ifndef POISEUILLE_PROBLEM
	#define POISEUILLE_PROBLEM Poiseuille
#endif

#include "Poiseuille.h"

#include <iostream>

#include "cudasimframework.cu"

// Some options are not compatible with non-Newtonian rheologies, so
// make sure to skip them if it's enabled
#ifndef POISEUILLE_RHEOLOGY
#	define POISEUILLE_RHEOLOGY NEWTONIAN
#	define HAS_NONNEWTONIAN 0
#else
#	define HAS_NONNEWTONIAN 1
#endif


POISEUILLE_PROBLEM::POISEUILLE_PROBLEM(GlobalData *_gdata) :
	Problem(_gdata),

	lz(1.0),
	ly(lz),
	lx(lz),

	rho(get_option("density", 1.0f)),
	kinvisc(get_option("kinvisc", 0.1f)),
	driving_force(get_option("driving-force", 0.05f)),

	ys( YIELDING_RHEOLOGY(POISEUILLE_RHEOLOGY) ? driving_force*rho*lz/4 : 0 ),
	max_vel(compute_poiseuille_vel(0)),
	Re(lz*max_vel/kinvisc)
{
	// *** user parameters from command line

	// particles per depth: allow specification of the particle size in terms of
	// the number of particles in z direction
	const int ppH = get_option("ppH", 32);

	// density diffusion terms: 0 none, 1 Ferrari, 2 Molteni & Colagrossi, 3 Brezzi
	const DensityDiffusionType RHODIFF = get_option("density-diffusion", DENSITY_DIFFUSION_NONE);

	// Allow user to set the MLS frequency at runtime. Default to 0 (no MLS).
	const int mlsIters = get_option("mls", 0);

	// Allow user to set the computational viscosity; accepted values: dyn, kin
	const ComputationalViscosityType compvisc = get_option("compvisc", KINEMATIC);

	const AverageOperator viscavg = get_option("viscavg", HARMONIC);

	// Allow use to set the viscous operator model: morris or monaghan
	const ViscousModel viscmodel = get_option("viscmodel", MORRIS);

	if (HAS_NONNEWTONIAN && viscmodel == ESPANOL_REVENGA)
		throw std::invalid_argument("cannot use " +
			string(ViscousModelName[viscmodel]) +
			" with non-Newtonian rheologies");

	SETUP_FRAMEWORK(
		kernel<WENDLAND>,
		rheology<POISEUILLE_RHEOLOGY>,
		turbulence_model<LAMINAR_FLOW>,
		computational_visc<KINEMATIC>,
		visc_model<MORRIS>,
		visc_average<ARITHMETIC>,
		periodicity<PERIODIC_XY>,
		boundary<DYN_BOUNDARY>
	).select_options
		( RHODIFF  // switch to the user-selected density diffusion
		, compvisc // switch to the user-selected computational viscosity
		, viscavg  // switch to the user-selected viscous averaging operator
#if !HAS_NONNEWTONIAN
		, viscmodel // switch to the user-selected viscous model
#endif
		);

	if (mlsIters > 0)
		addFilter(MLS_FILTER, mlsIters);

	// SPH parameters

	set_deltap(lz/ppH);
	simparams()->tend = 100;

	// Physical parameters
	set_gravity(driving_force, 0.0, 0.0);

	auto fluid_idx = add_fluid(rho);
	set_kinematic_visc(fluid_idx, kinvisc);

	// Set yield strength (if needed) to get a plug which is about 1/2 of the channel height
	if (YIELDING_RHEOLOGY(POISEUILLE_RHEOLOGY))
		set_yield_strength(fluid_idx, ys);

	printf("Reynolds number : %g\n", Re);
	if (YIELDING_RHEOLOGY(POISEUILLE_RHEOLOGY))
		printf("Yield strength : %g Pa.s\n", ys);
	printf("Max flow velocity : %g m/s\n", max_vel);

	const double hydrostatic_vel = sqrt(2*driving_force*lz);
	printf("Hydrostatic velocity : %g m/s\n", hydrostatic_vel);

	const double c0 = 20*fmax(hydrostatic_vel, max_vel);

	set_equation_of_state(0,  7.0f, c0);

	// Geometry--------------------------------------------------
	setPositioning(PP_CENTER);
	GeometryID bottom = addRect(GT_FIXED_BOUNDARY, FT_BORDER,
		Point(0,0,-(lz/2)), lx - m_deltap, ly - m_deltap);
	GeometryID fluid = addBox(GT_FLUID, FT_SOLID, Point(0,0,0),
		lx - m_deltap, ly - m_deltap, lz- 2*m_deltap );

	setPositioning(PP_CORNER);
	GeometryID top = addRect(GT_FIXED_BOUNDARY, FT_BORDER,
		Point((lx - m_deltap)/2, -(ly - m_deltap)/2, (lz/2)), lx - m_deltap, ly - m_deltap);

	setEraseOperation(top, ET_ERASE_NOTHING);
	rotate(top, 0, M_PI, 0);

	add_writer(VTKWRITER, 1.0);

	// Name of problem used for directory creation
	m_name = STR(POISEUILLE_PROBLEM);
}

void POISEUILLE_PROBLEM::initializeParticles(BufferList &buffer, const uint numParticle)
{
	const bool init_vel = get_option("steady-init", false);

	if (!init_vel)
		return;

	double4 *pos = buffer.getData<BUFFER_POS_GLOBAL>();
	float4 *vel = buffer.getData<BUFFER_VEL>();
	const ushort4 *pinfo = buffer.getData<BUFFER_INFO>();

	for (uint i = 0 ; i < numParticle ; i++) {
		if (FLUID(pinfo[i]))
			vel[i].x = compute_poiseuille_vel(pos[i].z);
	}
}

float POISEUILLE_PROBLEM::compute_poiseuille_vel(float pos_z) const
{
	if (EXPONENTIAL_RHEOLOGY(POISEUILLE_RHEOLOGY))
		throw invalid_argument("Exponential rheology profile computation not yet supported");

	const float n = NONLINEAR_RHEOLOGY(POISEUILLE_RHEOLOGY) ?
		physparams()->visc_nonlinear_param[0] :
		1.0f;
	const float n1n = 1.0 + 1.0/n;

	const float plug_limit = ys/(rho*driving_force);

	// The formula is A ( B - C(z) ) where:
	//
	// A = n / (n+1) * (driving_force/kinvisc)^(1/n)
	// B = (h/2 - plug_limit)^n1n
	// C(z) = (abs(z) - plug_limit)^n1n
	// TODO A and B should be computed only once

	// Compute A
	float A = driving_force/kinvisc;
	if (n != 1)
		A = powf(A, 1.0/n);
	A = (n*A)/(n+1);

	// Compute B
	float B = lz/2 - plug_limit;
	if (n == 1)
		B *= B;
	else
		B = powf(B, n1n);

	if (fabs(pos_z) < plug_limit)
		return A*B;

	// Compute C
	float C = pos_z - plug_limit;
	if (n == 1)
		C *= C;
	else
		C = powf(C, n1n);

	return A*(B - C);
}

/* vim: set ft=cuda */
