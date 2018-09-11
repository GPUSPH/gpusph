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

#include "GenericProblem.h"
#include "GlobalData.h"
#include "cudasimframework.cu"

void warn( const char* theWarn )
{
	std::cout << std::endl << "--------------------------------------------------------------" << std::endl;
	std::cout << "WARNING: " << theWarn << std::endl << std::endl;
}

GenericProblem::GenericProblem(GlobalData *_gdata)
		: XProblem(_gdata)
{
	// Private variables
	m_name = PSTR(general, name);
#ifdef GPUSPH_special_boundary_SECTIONS
#if ISDEF(special_boundary,start_time_VALS)
	int aStart[] =
	{	PVALS( special_boundary, start_time )};
#endif

#if ISDEF(special_boundary,end_time_VALS)
	int anEnd[] =
	{	PVALS( special_boundary, end_time )};
#endif

	for (uint i = 0; i < NB_SECTIONS( special_boundary ); i++)
	{
#if ISDEF(special_boundary,start_time_VALS)
		m_bndtstart [ i ] = aStart[ i ];
#endif

#if ISDEF(special_boundary,end_time_VALS)
		m_bndtend [ i ] = anEnd[ i ];
#endif

	}
#endif

	// Setup the simulation framework
	SETUP_FRAMEWORK(kernel<KERNEL_TYPE>,
					formulation<SPH_FORMULATION>,
					viscosity<VISCOSITY_TYPE>,
					boundary<BOUNDARY_TYPE>,
					periodicity<PERIODICITY>,
					flags<FLAGS_LIST>);

	// Initialization of the physical parameters
	set_deltap ( PVAL( sph, m_deltap ));
	physparams()->r0 = m_deltap;

	// Gravity
	physparams()->gravity = make_float3(PVAL(physics, gravity_1), PVAL(physics, gravity_2), PVAL(physics, gravity_3));
	// Gravity
	// Assume that if gravity is enabled then both start and end time are
	// mandatory to be defined.
#if ISDEF(physics,variable_gravity_begin)
	m_gtstart = PVAL( physics, variable_gravity_begin );
	m_gtend = PVAL( physics, variable_gravity_end );
	simparams()->gcallback = true;
#endif

	// Initialization of the neighbours parameters
	simparams()->maxneibsnum = PVAL(neighbours, maxneibsnum);
	simparams()->buildneibsfreq = PVAL(neighbours, buildneibsfreq);

	// Time parameters
#if ISDEF(time,dt)
	simparams()->dt = PVAL( time, dt );
#endif
#if ISDEF(time,dtadaptfactor)
	simparams()->dtadaptfactor = PVAL( time, dtadaptfactor );
#endif
	simparams()->tend = PVAL(time, tend);

	// Ferrari correction
#if ISDEF(sph,ferrari)
	simparams()->ferrari= PVAL( sph, ferrari );
#endif
#if ISDEF(sph,ferrariLengthScale)
	simparams()->ferrariLengthScale = PVAL( sph, ferrariLengthScale );
#endif

	// Writer settings
	add_writer(VTKWRITER, PVAL(output, vtk_frequency));
	if ( PVAL( output, commonwriter ) > 0) {
		add_writer(COMMONWRITER, PVAL(output, commonwriter));
	}

	size_t fluid_0 = add_fluid(PVAL(fluid_0, rho0));
	set_kinematic_visc(fluid_0, PVAL(fluid_0, kinematicvisc));

#if ISDEF(fluid_0,m_waterLevel)
	setWaterLevel( PVAL( fluid_0, m_waterLevel ) );
	setMaxParticleSpeed( PVAL( fluid_0, m_maxParticleSpeed ) );
	set_equation_of_state( fluid_0, PVAL( fluid_0, kinematicvisc ), NAN);
#else
	set_equation_of_state(fluid_0, PVAL(fluid_0, kinematicvisc), PVAL(fluid_0, sscoeff));
#endif

	//	ADD_MORE_FLUIDS
#if NB_SECTIONS( fluid ) > 1
	double dens[] =
	{	PVALS( fluid, rho0 )};
	double visc[] =
	{	PVALS( fluid, kinematicvisc )};
	double ss[] =
	{	PVALS( fluid, sscoeff )};
	double eos[] =
	{	PVALS( fluid, gammacoeff )};
	for ( uint i = 1; i < NB_SECTIONS( fluid ); i++ )
	{
		size_t fluid_id = add_fluid( i, dens[i] );
		set_kinematic_visc( fluid_id, visc[i] );
		set_equation_of_state( fluid_id, eos[i], ss[i] );
	}
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
		m_origin.x = PVAL(geometry, m_size_1);
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

	// Fluid definition
	GeometryID fluid = addHDF5File(GT_FLUID, Point(0, 0, 0),
		PSTR(geometry, fluid_file), NULL);

	// Main container
	const char* collisionsFileString = NULL;
#if ISDEF( geometry, collision_file )
	collisionsFileString = PSTR( geometry, collision_file );
#endif
	GeometryID container = addHDF5File(GT_FIXED_BOUNDARY, Point(0, 0, 0),
		PSTR(geometry, walls_file), collisionsFileString);
	disableCollisions(container);

	// Special boundaries
#ifdef GPUSPH_special_boundary_SECTIONS
#if NB_SECTIONS(special_boundary) > 0
#define enable true
#define disable false
	// Lists of special boundaries parameters values.
#if ISDEF(special_boundary,collisions_VALS)
	const bool aCollisionsFlags[] =
	{	PBOOLVALS( special_boundary, collisions )};
#endif

#if ISDEF(special_boundary,feedback_VALS)
	bool aFeedbackFlags[] =
	{	PBOOLVALS( special_boundary, feedback )};
#endif

#if ISDEF(special_boundary,collisions_file_VALS)
	const char* aCollisionsFiles[] =
	{	PSTRVALS( special_boundary, collisions_file )};
#endif

#if ISDEF(special_boundary,floating_body_geometry_VALS)
	const char* aFloatingGeomFiles[] =
	{	PSTRVALS( special_boundary, floating_body_geometry )};
#endif

#if ISDEF(special_boundary,open_bnd_type_VALS)
	int anOpenBndType[] =
	{	PINTVALS( special_boundary, open_bnd_type )};
#endif

#if ISDEF(special_boundary,density_VALS)
	double aDensity[] =
	{	PVALS( special_boundary, density )};
#endif

	int aBndType[] =
	{	PINTVALS( special_boundary, type )};
	const char* aBndFile[] =
	{	PSTRVALS( special_boundary, file )};
	// Set parameters for each special boundary
	for (uint i = 0; i < NB_SECTIONS(special_boundary); i++)
	{
		GeometryType specialBoundaryType;
		if ( aBndType[i] == moving_body )    // moving_body
		{
			specialBoundaryType = GT_MOVING_BODY;
			// TODO: Synchronize with common flags.
//      if (SetUp.MovingBodies == "disable") {
//        std::cout << std::endl << "--------------------------------------------------------------" << std::endl << std::endl;
//        std::cout << "WARNING !!!" << std::endl
//          << "You have a moving body but the moving_bodies are disabled in the sph section!" << std::endl;
//      }
		}
		else if ( aBndType[i] == floating_body )    // floating_body
		{
			specialBoundaryType = GT_FLOATING_BODY;
			// TODO: Synchronize with common flags.
//      if (SetUp.FloatingBodies == "disable") {
//        std::cout << std::endl << "WARNING !!!" << std::endl
//          << "You have a floating body but the floating_bodies are disabled in the sph section!" << std::endl;
//      }
		}
		else if ( aBndType[i] == open_boundary )    // open_boundary
		{
			specialBoundaryType = GT_OPENBOUNDARY;
			// TODO: Synchronize with common flags.
//      if (SetUp.OpenBoundaries == "disable") {
//        std::cout << std::endl << "--------------------------------------------------------------" << std::endl << std::endl;
//        std::cout << "WARNING !!!" << std::endl
//          << "You have an open boundary but the open_boundaries are disabled in the sph section!" << std::endl;
//      }
		}
		else if ( aBndType[i] == free_surface )    // free_surface
		{
			specialBoundaryType = GT_FREE_SURFACE;
		}
		const char* aCollisionsFile = NULL;
#if ISDEF(special_boundary,collisions_file_VALS)
		if( aCollisionsFiles[ i ] )
		{
			aCollisionsFile = aCollisionsFiles[ i ];
		}
#endif
#if ISDEF(special_boundary,floating_body_geometry_VALS)
		if ( aCollisionsFile == NULL && aBndType[ i ] == floating_body )
		{
			aCollisionsFile = aFloatingGeomFiles[ i ];
		}
#endif

		// Define a special boundary
		GeometryID aSpecialBnd = addHDF5File( specialBoundaryType,
				Point(0,0,0), aBndFile[i], aCollisionsFile );

#if ISDEF(special_boundary,collisions_VALS)
		if ( !aCollisionsFlags[i] )    // != "true"
		{
			// If collisions are disabled.
			disableCollisions( aSpecialBnd );
		}
#endif

#if ISDEF(special_boundary,feedback_VALS)
		if ( aFeedbackFlags[ i ] )    // enable
		{
			// If feedback is enabled.
			enableFeedback( aSpecialBnd );
		}
#endif

#if ISDEF(special_boundary,open_bnd_type_VALS)
		if ( specialBoundaryType == GT_OPENBOUNDARY
				&& anOpenBndType[ i ] == velocity_driven )
		{
			setVelocityDriven( aSpecialBnd, true );
		}
#endif

#if ISDEF(special_boundary,density_VALS)
		if ( aDensity[ i ] > 0 && ::isfinite( aDensity[ i ] ) )
		{
			setMassByDensity( aSpecialBnd, aDensity[ i ] );
		}
#endif

	}
#endif
#endif
}

GPUSPH_USER_FUNCTIONS
