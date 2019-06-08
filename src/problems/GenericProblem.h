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

#ifndef _GENERICPROBLEM_H
#define	_GENERICPROBLEM_H

#define PROBLEM_API 1
#include "Problem.h"
#define GPUSPH_INCLUDE_PARAMS
#define GPUSPH_USER_FUNCTIONS

// Include  parameters macro definitions auto-generated from ini file.
GPUSPH_INCLUDE_PARAMS

#define WARN_ORIGIN_DEFINED(x) "The domain origin " x " is defined even "\
	"though external *.h5sph geometrical elements are read."
#define WARN_SIZE_DEFINED(x) "The domain size " x " is defined even "\
	"though external *.h5sph geometrical elements are read."

/*! \def PVAL(s,p)
	Value of the parameter p in the section s.
	\param s the section name
	\param p the parameter name
 */
#define PVAL(s,p) GPUSPH_##s##_##p##__
/*! \def ISNAN(s,p)
	Check if the parameter p in the section s is NAN.
	\param s the section name
	\param p the parameter name
 */
#define ISNAN(s,p) GPUSPH_##s##_##p##__NAN != 0
/*! Check if the parameter p in the section s is defined.
	\param s the section name
	\param p the parameter name
 */
#define ISDEF(s,p) defined (GPUSPH_##s##_##p##__)
/*! Number of indexed sections with the base section name s.
	\param s the base section name
 */
#define NB_SECTIONS(s) GPUSPH_NB_##s##__
/*!
 * The comma separated list of the parameter p values in indexed sections with
 * the base section name s. If in some sections the parameter is absent
 * then NAN is inserted into the list of values.
 * \param s the base section name
 * \param p the parameter name
 */
#define PVALS(s,p) GPUSPH_##s##_##p##_VALS__
/*!
 * The comma separated list of the parameter p quoted values in indexed sections
 * with the base section name s. If in some sections the parameter is absent
 * then NULL is inserted into the list of values.
 * \param s the base section name
 * \param p the parameter name
 */
#define PSTRVALS(s,p) GPUSPH_##s##_##p##_STRVALS__
/*!
 * The comma separated list of the parameter p values in indexed sections
 * with the base section name s. If in some sections the parameter is absent
 * then -1 is inserted into the list of values.
 * \param s the base section name
 * \param p the parameter name
 */
#define PINTVALS(s,p) GPUSPH_##s##_##p##_INTVALS__
/*!
 * The comma separated list of the parameter p values in indexed sections
 * with the base section name s. If in some sections the parameter is absent
 * then false is inserted into the list of values.
 * \param s the base section name
 * \param p the parameter name
 */
#define PBOOLVALS(s,p) GPUSPH_##s##_##p##_BOOLVALS__
/*! Enclose the value with double quotes to use it as a string literal.*/
#define PQUOTE(x) #x
/*! Auxiliary macro to expand macros in the argument if any.*/
#define _PSTR(theValue) PQUOTE(theValue)
/*! The string value of the parameter p in the section s.
	\param s the section name
	\param p the parameter name
 */
#define PSTR(s,p) _PSTR(PVAL(s,p))
/*! Check if the parameter p in the section s has value v.
	\param s the section name
	\param p the parameter name
	\param v the parameter's value
 */
#define ISENUM_EQ(s,p,v) defined( GPUSPH_##s##_##p##_##v )

/*! Define problem name according to general.name parameter value.*/
#if ISDEF(general, name)
#define GenericProblem PVAL(general, name)
#endif

/*! Wave gages */
#ifdef GPUSPH_wave_gage_SECTIONS
#define WAVE_GAGES \
{ \
	double wave_gage_x[] = { PVALS( wave_gage, x ) }; \
	double wave_gage_y[] = { PVALS( wave_gage, y ) }; \
	double wave_gage_z[] = { PVALS( wave_gage, z ) }; \
	for ( uint i = 0; i < NB_SECTIONS(wave_gage) ) \
	{ \
		add_gage(make_double3( wave_gage_x[i], wave_gage_y[i], wave_gage_z[i] )); \
	} \
}
#else
#define WAVE_GAGES
#endif

/*! Test points */
#ifdef GPUSPH_probe_SECTIONS
#define TEST_POINTS \
{ \
  addPostProcess(TESTPOINTS);\
	double probe_x[] = { PVALS( probe, x ) }; \
	double probe_y[] = { PVALS( probe, y ) }; \
	double probe_z[] = { PVALS( probe, z ) }; \
	for ( uint i = 0; i < NB_SECTIONS(probe) ) \
	{ \
		addTestPoint(make_double3( probe_x[i], probe_y[i], probe_z[i] )); \
	} \
}
#else
#define TEST_POINTS
#endif

/*! Split axis definitions */
#define x_AXIS X_AXIS
#define y_AXIS Y_AXIS
#define z_AXIS Z_AXIS
#if ISDEF(domain_splitting,split_axis)
#define __SPLIT_AXIS(a) a##_AXIS
#define _SPLIT_AXIS(a) __SPLIT_AXIS(a)
#define SPLIT_AXIS _SPLIT_AXIS( PVAL( domain_splitting, split_axis ) )
#else
#define SPLIT_AXIS LONGEST_AXIS
#endif

/*! Periodicity definition */
#if ISDEF(boundaries,periodicity_x)
#define _periodicity_x_true PERIODIC_X
#define _periodicity_x_false PERIODIC_NONE
#define _periodicity_y_true PERIODIC_Y
#define _periodicity_y_false PERIODIC_NONE
#define _periodicity_z_true PERIODIC_Z
#define _periodicity_z_false PERIODIC_NONE
#define __PERIODIC(x,v) _periodicity##_##x##_##v
#define _PERIODIC(x,v) __PERIODIC(x,v)
#define PERIODIC(x) _PERIODIC(x, PVAL(boundaries, periodicity_##x ))
#define PERIODICITY (Periodicity)(PERIODIC(x) | PERIODIC(y) | PERIODIC(z))
#else
#define PERIODICITY PERIODIC_NONE
#endif

/*! Boundary conditions type definition */
#if ISDEF(boundaries,bnd_type)
#define BOUNDARY_TYPE PVAL(boundaries,bnd_type)
#else
#define BOUNDARY_TYPE SA_BOUNDARY
#endif

/*! Rheology definition */
#if ISENUM_EQ(viscous_options,rheology,inviscid)
#define RHEOLOGY_TYPE INVISCID
#elif ISENUM_EQ(viscous_options,rheology,Newtonian)
#define RHEOLOGY_TYPE NEWTONIAN
#else
#define RHEOLOGY_TYPE NEWTONIAN
#endif

/*! Turbulence model definition */
#if ISENUM_EQ(viscous_options,turbulence,disable)
#define TURBULENCE_MODEL LAMINAR_FLOW
#elif ISENUM_EQ(viscous_options,turbulence,artificial_viscosity)
#define TURBULENCE_MODEL ARTIFICIAL
#elif ISENUM_EQ(viscous_options,turbulence,k_epsilon)
#define TURBULENCE_MODEL KEPSILON
#elif ISENUM_EQ(viscous_options,turbulence,SPS_model)
#define TURBULENCE_MODEL SPS
#else
#define TURBULENCE_MODEL LAMINAR_FLOW
#endif

/*! Viscosity averaging definition */
#if ISENUM_EQ(viscous_options,viscosityAveraging,Arithmetic)
#define VISCOSITY_AVERAGING ARITHMETIC
#elif ISENUM_EQ(viscous_options,viscosityAveraging,Harmonic)
#define VISCOSITY_AVERAGING HARMONIC
#elif ISENUM_EQ(viscous_options,viscosityAveraging,Geometric)
#define VISCOSITY_AVERAGING GEOMETRIC
#else
#define VISCOSITY_AVERAGING ARITHMETIC
#endif

/*! Viscous model definition */
#if ISENUM_EQ(viscous_options,viscousModel,Morris)
#define VISCOUS_MODEL MORRIS
#else
#define VISCOUS_MODEL MORRIS
#endif

/*! Viscosity type definition */
#if ISENUM_EQ(viscous_options,viscosityType,kinematic)
#define VISCOSITY_TYPE KINEMATIC
#elif ISENUM_EQ(viscous_options,viscosityType,dynamic)
#define VISCOSITY_TYPE DYNAMIC
#else
#define VISCOSITY_TYPE KINEMATIC
#endif

/*! Kernel type definition */
#if ISENUM_EQ(discretisation,kernel_type,Cubic_spline)
#define KERNEL_TYPE CUBICSPLINE
#elif ISENUM_EQ(discretisation,kerne_type,Quadratic)
#define KERNEL_TYPE QUADRATIC
#elif ISENUM_EQ(discretisation,kerne_type,Wendland)
#define KERNEL_TYPE WENDLAND
#elif ISENUM_EQ(discretisation,kerne_type,Gaussian)
#define KERNEL_TYPE GAUSSIAN
#else
#define KERNEL_TYPE WENDLAND
#endif

/*! SPH formulation definition */
#if ISENUM_EQ(discretisation,sph_formulation, Single_fluid_WCSPH)
#define SPH_FORMULATION SPH_F1
#elif ISENUM_EQ(discretisation,sph_formulation, Multi_fluid)
#define SPH_FORMULATION SPH_F2
#elif ISENUM_EQ(discretisation,sph_formulation, Multi_fluid_Grenier)
#define SPH_FORMULATION SPH_GRENIER
#else
#define SPH_FORMULATION SPH_F1
#endif

/*! Density diffusion type definition */
#if ISENUM_EQ(density_calculation,density_diff_type, none)
#define DENSITY_DIFFUSION_TYPE DENSITY_DIFFUSION_NONE
#elif ISENUM_EQ(density_calculation,density_diff_type, Colagrossi)
#define DENSITY_DIFFUSION_TYPE COLAGROSSI
#elif ISENUM_EQ(density_calculation,density_diff_type, Brezzi)
#define DENSITY_DIFFUSION_TYPE BREZZI
#elif ISENUM_EQ(density_calculation,density_diff_type, Ferrari)
#define DENSITY_DIFFUSION_TYPE FERRARI
#else
#define DENSITY_DIFFUSION_TYPE DENSITY_DIFFUSION_NONE
#endif

/*! Flags definitions */
#if ISDEF(special_boundary,open_bnd_type_VALS)
#define FLAG_INLET_OUTLET ENABLE_INLET_OUTLET
#else
#define FLAG_INLET_OUTLET 0
#endif

#if ISENUM_EQ(density_calculation,density_sum,enable) || ISDEF(special_boundary,open_bnd_type_VALS)
#define FLAG_DENSITY_SUM ENABLE_DENSITY_SUM
#else
#define FLAG_DENSITY_SUM 0
#endif

#if ISENUM_EQ(boundaries,moving_bodies,enable) || ISDEF(special_boundary,collisions_file_VALS) \
	|| ISDEF(special_boundary,rotation_vel_x_VALS)\
|| ISDEF(special_boundary,rotation_vel_y_VALS)\
|| ISDEF(special_boundary,rotation_vel_z_VALS)\
|| ISDEF(special_boundary,translation_vel_x_VALS)\
|| ISDEF(special_boundary,translation_vel_y_VALS)\
|| ISDEF(special_boundary,translation_vel_z_VALS)
#define FLAG_MOVING_BODIES ENABLE_MOVING_BODIES
#else
#define FLAG_MOVING_BODIES 0
#endif

#if ISENUM_EQ(time,variable_dt,enable)
#define FLAG_DTADAPT ENABLE_DTADAPT
#else
#define FLAG_DTADAPT 0
#endif

#if ISENUM_EQ(density_calculation,xsph,enable)
#define FLAG_XSPH ENABLE_XSPH
#else
#define FLAG_XSPH 0
#endif

#if ISENUM_EQ(initialisation,repacking,enable)
#define FLAG_REPACKING ENABLE_REPACKING
#else
#define FLAG_REPACKING 0
#endif

#if ISENUM_EQ(boundaries,gamma_quadrature,enable)
#define FLAG_GAMMA_QUADRATURE ENABLE_GAMMA_QUADRATURE
#else
#define FLAG_GAMMA_QUADRATURE 0
#endif

#if ISENUM_EQ(output,internal_energy,enable)
#define FLAG_INTERNAL_ENERGY ENABLE_INTERNAL_ENERGY
#else
#define FLAG_INTERNAL_ENERGY 0
#endif

#if ISENUM_EQ(discretisation,sph_formulation,Multi_fluid) || \
	ISENUM_EQ(discretisation,sph_formulation,Multi_fluid_Grenier)
#define FLAG_MULTIFLUID_SUPPORT ENABLE_MULTIFLUID
#else
#define FLAG_MULTIFLUID_SUPPORT 0
#endif

#define FLAGS_LIST ENABLE_WATER_DEPTH | FLAG_INLET_OUTLET | FLAG_DENSITY_SUM \
	| FLAG_DTADAPT | FLAG_MOVING_BODIES | FLAG_GAMMA_QUADRATURE \
	| FLAG_INTERNAL_ENERGY | FLAG_MULTIFLUID_SUPPORT | FLAG_XSPH \
  | FLAG_REPACKING

#define open_boundary GT_OPENBOUNDARY
#define floating_body GT_FLOATING_BODY
#define moving_body GT_MOVING_BODY
#define free_surface GT_FREE_SURFACE

#define pressure_driven 0
#define velocity_driven 1

/*! IMPOSE_WATER_LEVEL and IMPOSE_VELOCITY macros */
#if ISDEF(special_boundary,open_bnd_type_VALS)
#if ISDEF(special_boundary,open_bnd_water_level_VALS)
#define IMPOSE_WATER_LEVEL \
{ \
	const double wlevel[] = { PVALS( special_boundary, open_bnd_water_level ) }; \
	int aBndType[] = { PINTVALS( special_boundary, type ) }; \
	int anOpenBndType[] = { PINTVALS( special_boundary, open_bnd_type ) }; \
	for (uint i=0; i<NB_SECTIONS(special_boundary); i++) \
	{ \
		if ( aBndType[i] == open_boundary && anOpenBndType[i] == pressure_driven )  \
		{ \
			waterlevel = wlevel[i]; \
		} \
	} \
}
#else
#define IMPOSE_WATER_LEVEL
#endif

#if ISDEF(special_boundary,open_bnd_vel_x_VALS) \
	|| ISDEF(special_boundary,open_bnd_vel_y_VALS) \
|| ISDEF(special_boundary,open_bnd_vel_z_VALS)
#if ISDEF(special_boundary,open_bnd_vel_x_VALS)
#define __OPEN_BND_VEL_X  PVALS(special_boundary,open_bnd_vel_x)
#else
#define __OPEN_BND_VEL_X  0
#endif

#if ISDEF(special_boundary,open_bnd_vel_y_VALS)
#define __OPEN_BND_VEL_Y  PVALS(special_boundary,open_bnd_vel_y)
#else
#define __OPEN_BND_VEL_Y  0
#endif

#if ISDEF(special_boundary,open_bnd_vel_z_VALS)
#define __OPEN_BND_VEL_Z  PVALS(special_boundary,open_bnd_vel_z)
#else
#define __OPEN_BND_VEL_Z  0
#endif

#define IMPOSE_VELOCITY \
{ \
	const double vel_x[NB_SECTIONS(special_boundary)]={__OPEN_BND_VEL_X};\
	const double vel_y[NB_SECTIONS(special_boundary)]={__OPEN_BND_VEL_Y};\
	const double vel_z[NB_SECTIONS(special_boundary)]={__OPEN_BND_VEL_Z};\
	int aBndType[] = { PINTVALS( special_boundary, type ) }; \
	int anOpenBndType[] = { PINTVALS( special_boundary, open_bnd_type ) }; \
	for (uint i=0; i<NB_SECTIONS(special_boundary); i++) \
	{ \
		if (aBndType[i] == open_boundary && anOpenBndType[i] == velocity_driven )  \
		{ \
			eulerVel.x = vel_x[i]; \
			eulerVel.y = vel_y[i]; \
			eulerVel.z = vel_z[i]; \
		} \
	} \
}
#else
#define IMPOSE_VELOCITY
#endif

#else
#define IMPOSE_WATER_LEVEL
#define IMPOSE_VELOCITY
#endif

/*!
 * Provides the base template for the solver source generated by ProblemBuilder.
 */
class GenericProblem: public Problem {
	public:
		GenericProblem(GlobalData *);

		GPUSPH_USER_FUNCTIONS

	private:

		/* Pseudo parameters definitions */
		/* These are not real parameters, but are used to define some parameters
		 * for the user interface that will be used in GenericProblem to
		 * set actual Problem configuration parameters
		 */

		/**@inpsection{output}
		 * @label{VTK_WRITER_INTERVAL}
		 * @default{1.0}
		 * TLT_VTK_WRITER_INTERVAL
		 */
		double vtk_frequency;

		/**@inpsection{output}
		 * @label{COMMON_WRITER_INTERVAL}
		 * @default{}
		 * TLT_COMMON_WRITER_INTERVAL
		 */
		double commonwriter;

		/**@inpsection{discretisation}
		 * @label{PARTICLES_MAX_FACTOR}
		 * @default{1}
		 * TLT_PARTICLES_MAX_FACTOR
		 */
		double particles_max_factor;

		/* \inpsection{periodicity, enable}
		* \label{X}
		* \default{false}
		* TLT_PERIODICITY_X
		*/
		bool periodicity_x;

		/* \inpsection{periodicity, enable}
		* \label{Y}
		* \default{false}
		* TLT_PERIODICITY_Y
		*/
		bool periodicity_y;

		/* \inpsection{periodicity, enable}
		* \label{Z}
		* \default{false}
		* TLT_PERIODICITY_Z
		*/
		bool periodicity_z;

		/* \inpsection{probe}
		* \label{PROBE_X}
		* \default{0.0}
		* TLT_PROBE_X
		*/
		std::vector<float> x;

		/* \inpsection{probe}
		* \label{PROBE_Y}
		* \default{0.0}
		* TLT_PROBE_Y
		*/
		std::vector<float> y;

		/* \inpsection{probe}
		* \label{PROBE_Z}
		* \default{0.0}
		* TLT_PROBE_Z
		*/
		std::vector<float> z;

		/* \inpsection{wave_gage}
		* \label{WAVE_GAGE_X}
		* \default{0.0}
		* TLT_WAVE_GAGE_X
		*/
		std::vector<float> gage_x;

		/* \inpsection{wave_gage}
		* \label{WAVE_GAGE_Y}
		* \default{0.0}
		* TLT_WAVE_GAGE_Y
		*/
		std::vector<float> gage_y;

		/* \inpsection{wave_gage}
		* \label{WAVE_GAGE_Z}
		* \default{0.0}
		* TLT_WAVE_GAGE_Z
		*/
		std::vector<float> gage_z;

		/* End of pseudo parameters definition */

#ifdef GPUSPH_special_boundary_SECTIONS

#if ISDEF(special_boundary,start_time_VALS)
			double m_bndtstart[ NB_SECTIONS(special_boundary) ];
#endif

#if ISDEF(special_boundary,end_time_VALS)
			double m_bndtend[ NB_SECTIONS(special_boundary) ];
#endif

#endif

};
#endif	/* _GENERICPROBLEM_H */

