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

#include "GenericProblem.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

// custom function to print warnings during the problem generation
void warn( const char* theWarn )
{
	std::cout << std::endl << "--------------------" << std::endl;
	std::cout << "WARNING: " << theWarn << std::endl << std::endl;
}

// Generic problem class
GenericProblem::GenericProblem(GlobalData *_gdata)
		: Problem(_gdata)
{
  // Set the problem name. PSTR is defined in GenericProblem.h,
  // it reads string type parameters from the parameters file
  // (header file created by Salome). Each parameter
  // is defined through a line:
  // GPUSPH_section_name_parameter_name__ value
  // for example:
  // GPUSPH_general_name__ ProblemName
	m_name = PSTR(general, name);

	// Setup the simulation framework
	SETUP_FRAMEWORK(kernel<KERNEL_TYPE>,
					formulation<SPH_FORMULATION>,
					densitydiffusion<DENSITY_DIFFUSION_TYPE>,
					rheology<RHEOLOGY_TYPE>,
					turbulence_model<TURBULENCE_MODEL>,
					computational_visc<VISCOSITY_TYPE>,
					visc_model<VISCOUS_MODEL>,
					visc_average<VISCOSITY_AVERAGING>,
					boundary<BOUNDARY_TYPE>,
					periodicity<PERIODICITY>,
					add_flags<FLAGS_LIST>);

	// Initialization of the discretisation parameters
#if ISDEF(discretisation,sfactor)
	set_smoothing( PVAL( discretisation, sfactor ) );
#endif
	set_deltap ( PVAL( discretisation, m_deltap ));

	// Initialization of the neighbours parameters
#if ISENUM_EQ(boundaries,bnd_type,SA_BOUNDARY)
	resize_neiblist(PVAL(neighbours, neiblistsize), PVAL(neighbours, neibboundpos));
#else
	resize_neiblist(PVAL(neighbours, neiblistsize));
#endif
	simparams()->buildneibsfreq = PVAL(neighbours, buildneibsfreq);

	// Time parameters
#if ISDEF(time,dt)
	set_timestep(PVAL( time, dt ));
#endif
#if ISDEF(time,dtadaptfactor)
	simparams()->dtadaptfactor = PVAL( time, dtadaptfactor );
#endif
	simparams()->tend = PVAL(time, tend);

	// Density diffusion
#if ISDEF(density_calculation, ferrariDiffCoeff)
	simparams()->densityDiffCoeff= PVAL( density_calculation, ferrariDiffCoeff );
#endif
#if ISDEF(density_calculation, brezziDiffCoeff)
	simparams()->densityDiffCoeff= PVAL( density_calculation, brezziDiffCoeff );
#endif
#if ISDEF(density_calculation, densityDiffCoeff)
	simparams()->densityDiffCoeff= PVAL( density_calculation, densityDiffCoeff );
#endif

  // Gravity
	set_gravity(PVAL(physics, gravity_1), PVAL(physics, gravity_2), PVAL(physics, gravity_3));


	// Writer settings
	add_writer(VTKWRITER, PVAL(output, vtk_frequency));
#if ISDEF(output,commonwriter)
		add_writer(COMMONWRITER, PVAL(output, commonwriter));
#endif

	// fluids settings
	size_t fluid_0 = add_fluid(PVAL(fluid_0, rho0));
	set_kinematic_visc(fluid_0, PVAL(fluid_0, kinematicvisc));

#if ISDEF(fluid_0,m_waterLevel)
	setWaterLevel( PVAL( fluid_0, m_waterLevel ) );
#endif
#if ISDEF(fluid_0,m_maxParticleSpeed)
	setMaxParticleSpeed( PVAL( fluid_0, m_maxParticleSpeed ) );
#endif
#if ISDEF(fluid_0,sscoeff)
	set_equation_of_state( fluid_0, PVAL( fluid_0, gammacoeff), PVAL(fluid_0, sscoeff));
#else
	set_equation_of_state(fluid_0, PVAL(fluid_0, gammacoeff), NAN);
#endif

	//	Add more fluids
#if NB_SECTIONS( fluid ) > 1
	// density
  const double fluidDensity[] =
	{	PVALS( fluid, rho0 )};
  // kinematic viscosity
	const double fluidViscosity[] =
	{	PVALS( fluid, kinematicvisc )};
  // speed of sound
	const double fluidSSCoeff[] =
	{	PVALS( fluid, sscoeff )};
  // equation of state exponent
	const double fluidEOS[] =
	{	PVALS( fluid, gammacoeff )};
  // speed of sound input method
  const char c0InputMethod[] =
  { PSTRVALS( fluid, input_method)};
	for ( uint i = 1; i < NB_SECTIONS( fluid ); i++ )
	{
		size_t fluid_id = add_fluid( i, fluidDensity[i] );
		set_kinematic_visc( fluid_id, fluidViscosity[i] );
    if (c0InputMethod[i] == "direct_input")
  		set_equation_of_state( fluid_id, fluidEOS[i], fluidSSCoeff[i] );
    else
  		set_equation_of_state( fluid_id, fluidEOS[i], NAN );
	}
#endif

	// Repacking settings
#if ISDEF( initialisation, repack_maxiter )
	simparams()->repack_maxiter = PVAL(initialisation, repack_maxiter);
#endif
#if ISDEF( initialisation, repack_a )
	simparams()->repack_a = PVAL(initialisation, repack_a);
#endif
#if ISDEF( initialisation, repack_alpha )
	simparams()->repack_alpha = PVAL(initialisation, repack_alpha);
#endif

	// Geometry settings
	m_origin = make_double3( NAN, NAN, NAN);
	if (strcmp("0", PSTR(geometry, m_origin_1)) != 0)
	{
		m_origin.x = PVAL(geometry, m_origin_1);
		warn(WARN_ORIGIN_DEFINED("X"));
	}
	if (strcmp("0", PSTR(geometry, m_origin_2)) != 0)
	{
		m_origin.y = PVAL(geometry, m_origin_2);
		warn(WARN_ORIGIN_DEFINED("Y"));
	}
	if (strcmp("0", PSTR(geometry, m_origin_3)) != 0)
	{
		m_origin.z = PVAL(geometry, m_origin_3);
		warn(WARN_ORIGIN_DEFINED("Z"));
	}

	m_size = make_double3( NAN, NAN, NAN);
	if (strcmp("1e+09", PSTR(geometry, m_size_1)) != 0)
	{
		m_size.x = PVAL(geometry, m_size_1);
		warn(WARN_SIZE_DEFINED("X"));
	}
	if (strcmp("1e+09", PSTR(geometry, m_size_2)) != 0)
	{
		m_size.y = PVAL(geometry, m_size_2);
		warn(WARN_SIZE_DEFINED("Y"));
	}
	if (strcmp("1e+09", PSTR(geometry, m_size_3)) != 0)
	{
		m_size.z = PVAL(geometry, m_size_3);
		warn(WARN_SIZE_DEFINED("Z"));
	}

	// Fluid particles definition
	GeometryID fluid = addHDF5File(GT_FLUID, Point(0, 0, 0),
		PSTR(geometry, fluid_file), NULL);

	// Main container definition
#if ISDEF(boundaries,main_container_collision_file)
	const char* collisionsFileString = PSTR(boundaries, main_container_collision_file);
#else
	const char* collisionsFileString = NULL;
#endif
	GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0, 0, 0),
		PSTR(geometry, walls_file), collisionsFileString);

#if !ISDEF(boundaries,main_container_collision_file)
			disableCollisions( container );
#endif
	// Special boundaries definition
#ifdef GPUSPH_special_boundary_SECTIONS
#define enable true
#define disable false
#if NB_SECTIONS(special_boundary) > 0
	// Lists of special boundaries parameters values.
#if ISDEF(special_boundary,collisions_VALS)
	const bool enableCollisionsArray[] =
	{	PVALS( special_boundary, collisions )};
#endif

#if ISDEF(special_boundary,feedback_VALS)
	const bool enableFeedbackArray[] =
	{	PVALS( special_boundary, feedback )};
#endif

#if ISDEF(special_boundary,collisions_file_VALS)
	const char* collisionsFiles[] =
	{	PSTRVALS( special_boundary, collisions_file )};
#endif

#if ISDEF(special_boundary,object_geometry_file_VALS)
	const char* objectFiles[] =
	{	PSTRVALS( special_boundary, object_geometry_file )};
#endif

#if ISDEF(special_boundary,open_bnd_type_VALS)
	const int openBndType[] =
	{	PVALS( special_boundary, open_bnd_type )};
#endif

#if ISDEF(special_boundary,object_density_VALS)
	const double objectDensity[] =
	{	PVALS( special_boundary, object_density )};
#endif

#if ISDEF(special_boundary,object_mass_VALS)
	const double objectMass[] =
	{	PVALS( special_boundary, object_mass )};
#endif

#if ISDEF(special_boundary,object_cg_x_VALS)
	const double3 objectCG[] =
	{	make_double3 (PVALS (special_boundary, object_cg_x),
                  PVALS (special_boundary, object_cg_y),
                  PVALS (special_boundary, object_cg_z))};
#endif
#if ISDEF(special_boundary,object_inertia_x_VALS)
	const double objectInertiaX[] =
	{ PVALS (special_boundary, object_inertia_x)};
#endif
#if ISDEF(special_boundary,object_inertia_y_VALS)
	const double objectInertiaY[] =
	{ PVALS (special_boundary, object_inertia_y)};
#endif
#if ISDEF(special_boundary,object_inertia_z_VALS)
	const double objectInertiaZ[] =
	{ PVALS (special_boundary, object_inertia_z)};
#endif

	const int boundaryType[] =
	{	PVALS( special_boundary, type )};
	const char* boundaryFile[] =
	{	PSTRVALS( special_boundary, file )};
	// Set parameters for each special boundary
	for (uint i = 0; i < NB_SECTIONS(special_boundary); i++)
	{
		GeometryType specialBoundaryType;
		if ( boundaryType[i] == moving_body )
		{
			specialBoundaryType = GT_MOVING_BODY;
		}
		else if ( boundaryType[i] == floating_body )
		{
			specialBoundaryType = GT_FLOATING_BODY;
		}
		else if ( boundaryType[i] == open_boundary )
		{
			specialBoundaryType = GT_OPENBOUNDARY;
		}
		else if ( boundaryType[i] == free_surface )
		{
			specialBoundaryType = GT_FREE_SURFACE;
		}
#if ISDEF(special_boundary,collisions_file_VALS)
    const char* collisionsFile = (collisionsFiles[i]) ? collisionsFiles[i] : NULL;
#elif ISDEF(special_boundary,object_geometry_file_VALS)
		const char*	collisionsFile = (objectFiles[i]) ? objectFiles[i] : NULL;
#else
    const char* collisionsFile = NULL;
#endif
#if ISDEF(special_boundary,start_time_VALS)
		m_bndtstart [ i ] = PVALS (special_boundary, start_time)[i];
#endif

#if ISDEF(special_boundary,end_time_VALS)
		m_bndtend [ i ] = PVALS (special_boundary, end_time)[i];
#endif

    // Define the special boundary
		GeometryID specialBoundary = addHDF5File( specialBoundaryType,
				Point(0,0,0), boundaryFile[i], collisionsFile );

#if ISDEF(special_boundary,collisions_VALS)
		if ( !enableCollisionsArray[i] )
		{
			disableCollisions( specialBoundary );
		}
#endif

#if ISDEF(special_boundary,feedback_VALS)
		if (enableFeedbackArray[i])
		{
			enableFeedback( specialBoundary );
		}
#endif

#if ISDEF(special_boundary,open_bnd_type_VALS)
		if ( specialBoundaryType == GT_OPENBOUNDARY
				&& openBndType[ i ] == velocity_driven )
		{
			setVelocityDriven( specialBoundary, true );
		}
#endif

#if ISDEF(special_boundary,object_density_VALS)
		if ( objectDensity[ i ] > 0 && ::isfinite( objectDensity[ i ] ) )
		{
			setMassByDensity( specialBoundary, objectDensity[ i ] );
		}
#endif

#if ISDEF(special_boundary,object_mass_VALS)
		if ( objectMass[ i ] > 0 && ::isfinite( objectMass[ i ] ) )
		{
			setMass( specialBoundary, objectMass[i] );
		}
#endif

#if ISDEF(special_boundary,object_cg_x_VALS)
    setCenterOfGravity( specialBoundary, objectCG[i] );
#endif

#if ISDEF(special_boundary,object_inertia_x_VALS)
    setInertia( specialBoundary, objectInertiaX[i], objectInertiaY[i], objectInertiaZ[i] );
#endif

	}
#endif
#endif

}

GPUSPH_USER_FUNCTIONS
