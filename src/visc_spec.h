/*  Copyright 2018 Giuseppe Bilotta, Alexis Hérault, Robert A.
 	Dalrymple, Eugenio Rustico, Ciro Del Negro

	Conservatoire National des Arts et Metiers, Paris, France

	Istituto Nazionale di Geofisica e Vulcanologia,
    Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

/*! \file
 * Viscosity-related definitions and auxiliary functions
 */

#ifndef _VISC_SPEC_H
#define _VISC_SPEC_H

#include "option_range.h"

#include "simflags.h"

#include "average.h"

//! Rheology of the fluid(s)
/*! For the time being, we only support NEWTONIAN, but
 * this will be extended to include things such as temperature dependency
 * and generalized Newtonian rheologies.
 */
/** @defpsubsection{rheology, RHEOLOGY}
 * @inpsection{viscous_options}
 * @default{Newtonian}
 * @values{inviscid, Newtonian, Bingham, Papanastasious, Power Law, Herschel–Bulkley, Alexandrou, DeKee & Turcotte, Zhu}
 * TLT_RHEOLOGY
 */
enum RheologyType {
	INVISCID, ///< No (laminar) viscosity
	NEWTONIAN, ///< Viscosity independent of strain rate
	BINGHAM, ///< Bingham model (Newtonian + yield strength)
	PAPANASTASIOU, ///< Regularized Bingham model
	POWER_LAW, ///< Viscosity depends on a power of the strain rate
	HERSCHEL_BULKLEY, ///< Power law + yield strength
	ALEXANDROU, ///< Regularized Herschel–Bulkley
	DEKEE_TURCOTTE, ///< Exponential + yield strength
	ZHU, ///< Regularized De Kee and Turcotte
	GRANULAR ///< Viscosity dependant of strain rate
};

//! Name of the rheology type
#ifndef GPUSPH_MAIN
extern
#endif
const char* RheologyName[GRANULAR+1]
#ifdef GPUSPH_MAIN
= {
	"Inviscid",
	"Newtonian",
	"Bingham",
	"Papanastasiou",
	"Power-law",
	"Herschel–Bulkley",
	"Alexandrou",
	"De Kee & Turcotte",
	"Zhu",
	"Granular"
}
#endif
;

DEFINE_OPTION_RANGE(RheologyType, RheologyName, INVISCID, ZHU);

/** Macros and function to programmatically determine rheology traits
 * @{
 */

//! Check if an effective viscosity needs to be computed
#define NEEDS_EFFECTIVE_VISC(rheology) ((rheology) > NEWTONIAN)

//! Check if the rheology is non-linear (aside from the yield strength)
#define NONLINEAR_RHEOLOGY(rheology) ((rheology) >= POWER_LAW)

//! Check if the rheology has a yield strength
#define YIELDING_RHEOLOGY(rheology) (NEEDS_EFFECTIVE_VISC(rheology) && ((rheology) != POWER_LAW))

//! Check if the rheology uses the regularization parameter
#define REGULARIZED_RHEOLOGY(rheology) ( \
	((rheology) == PAPANASTASIOU) || \
	((rheology) == ALEXANDROU) || \
	((rheology) == ZHU) \
	)

//! Check if the rheology has an exponential behavior
#define EXPONENTIAL_RHEOLOGY(rheology) ((rheology) >= DEKEE_TURCOTTE)

//! Check if the rheology has a power-law behavior
/*! This includes rheology with yield strength
 */
#define POWERLAW_RHEOLOGY(rheology) (NONLINEAR_RHEOLOGY(rheology) && (rheology) < DEKEE_TURCOTTE)

//! Forms of yield strength contribution
/** We have three forms for the yield strengt contribution:
 * - no contribution (e.g. from power law)
 * - standard contribution (y_s/\dot\gamma), which can become infinite
 * - regularized contribution (Papanastasiou etc)
 */
enum YsContrib
{
	NO_YS, ///< no yield strength
	STD_YS, ///< standard form
	REG_YS ///< regularized
};

//! Statically determine the yield strength contribution for the given rheological model
template<RheologyType rheologytype>
__host__ __device__ __forceinline__
constexpr YsContrib
yield_strength_type()
{
	return
		REGULARIZED_RHEOLOGY(rheologytype) ? REG_YS : // yield with regularization
		YIELDING_RHEOLOGY(rheologytype) ? STD_YS : // yield without regularization
			NO_YS; // everything else: should be just Newtonian and power-law
}

//! @}

//! Turbulence model
/** @defpsubsection{turbulence, TURBULENCE}
 * @inpsection{viscous_options}
 * @default{disable}
 * @values{disable, artificial_viscosity, k_epsilon, SPS_model}
 * TLT_TURBULENCE
 */
/*!
 * While strictly speaking not a turbulence model, artificial viscosity is considered
 * among the turbulence models, since its behavior can be assimilated to it (i.e.
 * an addition to the viscous model, rather than an alternative to it), even though
 * it's normally only used for inviscid flow.
 */
enum TurbulenceModel {
	LAMINAR_FLOW, ///< No turbulence
	ARTIFICIAL, ///< Artificial viscosity
	SPS, ///< Sub-particle scale turbulence model
	KEPSILON, ///< k-epsilon turbulence model
	INVALID_TURBULENCE
};

//! Name of the turbulence model model
#ifndef GPUSPH_MAIN
extern
#endif
const char* TurbulenceName[INVALID_TURBULENCE+1]
#ifdef GPUSPH_MAIN
= {
	"Pure laminar flow",
	"Artificial viscosity",
	"Sub-particle scale",
	"k-epsilon",
	"(invalid)"
}
#endif
;

DEFINE_OPTION_RANGE(TurbulenceModel, TurbulenceName, LAMINAR_FLOW, KEPSILON);

//! Kind of viscosity used within the simulation
/*! This can be either KINEMATIC or DYNAMIC, depending on whether
 * the preference is to work in terms of the kinematic viscosity ν,
 * or in terms of the dynamic viscosity µ = ρν
 */
/** @defpsubsection{viscosityType, VISCOSITY_TYPE}
 * @inpsection{viscous_options}
 * @default{kinematic}
 * @values{kinematic, dynamic}
 * TLT_VISCOSITY_TYPE
 */
enum ComputationalViscosityType {
	KINEMATIC, ///< Kinematic viscosity (SI units: m²/s)
	DYNAMIC, ///< Dynamic viscosity (SI units: Pa s)
};

//! Name of the viscous model
#ifndef GPUSPH_MAIN
extern
#endif
const char* ComputationalViscosityName[DYNAMIC+1]
#ifdef GPUSPH_MAIN
= {
	"Kinematic",
	"Dynamic",
}
#endif
;

DEFINE_OPTION_RANGE(ComputationalViscosityType, ComputationalViscosityName, KINEMATIC, DYNAMIC);

//! Supported viscous models
/*! Currently only MORRIS is available, with plans to add Monaghan's and
 * Español & Revenga too
 */
/** @defpsubsection{viscousModel, VISCOUS_MODEL}
 * @inpsection{viscous_options}
 * @default{Morris}
 * @values{Morris, Monaghan, Españo & Revenga}
 * TLT_VISCOUS_MODEL
 */
enum ViscousModel {
	MORRIS, ///< Morris et al., JCP 1997
	MONAGHAN, ///< Monaghan & Gingold, JCP 1983
	ESPANOL_REVENGA, ///< Español & Revenga, Phys Rev E 2003
};

//! Name of the viscous model
#ifndef GPUSPH_MAIN
extern
#endif
const char* ViscousModelName[ESPANOL_REVENGA+1]
#ifdef GPUSPH_MAIN
= {
	"Morris 1997",
	"Monaghan & Gingold 1983",
	"Español & Revenga 2003",
}
#endif
;

DEFINE_OPTION_RANGE(ViscousModel, ViscousModelName, MORRIS, ESPANOL_REVENGA);



//! A complete viscous specification includes:
// * a rheological model
// * a turbulence model
// * a computational viscosity specification
// * a viscous model (discretization approach to the viscous operator)
// * an averaging operator
// * knowledge about the presence of multiple fluids
// TODO use the TypeValue and Multiplexer from CUDASimFramework
template<
	RheologyType _rheologytype = NEWTONIAN,
	TurbulenceModel _turbmodel = LAMINAR_FLOW,
	ComputationalViscosityType _compvisc = KINEMATIC,
	ViscousModel _viscmodel = MORRIS,
	AverageOperator _avgop = ARITHMETIC,
	flag_t _simflags = ENABLE_NONE,
	// is this a constant-viscosity formulation?
	bool _is_const_visc = (
		IS_SINGLEFLUID(_simflags) &&
		(_rheologytype == NEWTONIAN) &&
		(_turbmodel != KEPSILON)
	)
>
struct FullViscSpec {
	static constexpr RheologyType rheologytype = _rheologytype;
	static constexpr TurbulenceModel turbmodel = _turbmodel;
	static constexpr ComputationalViscosityType compvisc = _compvisc;
	static constexpr ViscousModel viscmodel = _viscmodel;
	static constexpr AverageOperator avgop = _avgop;
	static constexpr flag_t simflags = _simflags;

	static constexpr bool is_const_visc = _is_const_visc;

	//! Change the rheology type
	template<RheologyType newrheology>
	using with_rheologytype =
		FullViscSpec<newrheology, turbmodel, compvisc, viscmodel, avgop, simflags>;

	//! Change the turbulence model
	template<TurbulenceModel newturb>
	using with_turbmodel =
		FullViscSpec<rheologytype, newturb, compvisc, viscmodel, avgop, simflags>;

	//! Change the computational viscosity type specification
	template<ComputationalViscosityType altcompvisc>
	using with_computational_visc =
		FullViscSpec<rheologytype, turbmodel, altcompvisc, viscmodel, avgop, simflags>;

	//! Change the averaging operator
	template<AverageOperator altavgop>
	using with_avg_operator =
		FullViscSpec<rheologytype, turbmodel, compvisc, viscmodel, altavgop, simflags>;

	//! Force the assumption about constant viscosity
	/*! Sometimes we need to refer to the same viscous specification, but ignoring
	 * (or forcing) the assumption that the viscosity is constant;
	 * this type alias can be used to that effect
	 */
	template<bool is_const_visc>
	using assume_const_visc =
		FullViscSpec<rheologytype, turbmodel, compvisc, viscmodel, avgop, simflags, is_const_visc>;
};

//! Legacy viscosity type
enum LegacyViscosityType {
	ARTVISC = 1,
	KINEMATICVISC, ///< Morris formula, simplified for constant kinematic viscosity and using harmonic averaging of the density
	DYNAMICVISC, ///< Morris formula, with arithmetic averaging of the dynamic viscosity
	SPSVISC, ///< KINEMATICVISC + SPS
	KEPSVISC, ///< DYNAMICVISC + SPS
	GRANULARVISC, ///< Granular rheology and Morris formula with harmonic averaging
	INVALID_VISCOSITY
} ;

//! Name of the viscous model
#ifndef GPUSPH_MAIN
extern
#endif
const char* LegacyViscosityName[INVALID_VISCOSITY+1]
#ifdef GPUSPH_MAIN
= {
	"(null)",
	"Artificial",
	"Kinematic",
	"Dynamic",
	"SPS + kinematic",
	"k-e model",
	"Granular rheology",
	"(invalid)"
}
#endif
;

//! Convert a LegacyViscosityType to a FullViscSpec
/*! A template structure with a typedef 'type' to the corresponding FullViscSpec
 */
template<LegacyViscosityType>
struct ConvertLegacyVisc;

template<>
struct ConvertLegacyVisc<ARTVISC>
{
	/* Inviscid flow with artificial viscosity */
	using type = FullViscSpec<INVISCID, ARTIFICIAL>;
};

template<>
struct ConvertLegacyVisc<KINEMATICVISC>
{
	/* The default, except for the use of harmonic average and assumption of constant
	 * viscosity kinematic viscosity */
	using type = typename FullViscSpec<>::with_avg_operator<HARMONIC>::assume_const_visc<true>;
};

template<>
struct ConvertLegacyVisc<DYNAMICVISC>
{
	/* The default: Morris model with arithmetic mean for a laminar newtonian flow */
	using type = FullViscSpec<>; /* the default! */
};

template<>
struct ConvertLegacyVisc<SPSVISC>
{
	/* KINEMATICVISC + SPS */
	using type = typename ConvertLegacyVisc<KINEMATICVISC>::type::with_turbmodel<SPS>;
};

template<>
struct ConvertLegacyVisc<KEPSVISC>
{
	/* DYNAMICVISC + KEPSILON */
	using type = typename ConvertLegacyVisc<DYNAMICVISC>::type::with_turbmodel<KEPSILON>;
};

template<>
struct ConvertLegacyVisc<GRANULARVISC>
{
	/*  */
	//using type = typename FullViscSpec<>::with_rheologytype<GRANULAR>::with_avg_operator<HARMONIC>;
	using type = typename FullViscSpec<>::with_rheologytype<GRANULAR>::with_avg_operator<ARITHMETIC>;
};

#endif
