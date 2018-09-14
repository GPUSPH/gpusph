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

#include "average.h"

//! Rheology of the fluid(s)
/*! For the time being, we only support NEWTONIAN, but
 * this will be extended to include things such as temperature dependency
 * and generalized Newtonian rheologies.
 */
enum RheologyType {
	NEWTONIAN, ///< Viscosity independent of strain rate
};

//! Name of the rheology type
#ifndef GPUSPH_MAIN
extern
#endif
const char* RheologyTypeName[NEWTONIAN+1]
#ifdef GPUSPH_MAIN
= {
	"Newtonian",
}
#endif
;


//! Kind of viscosity used within the simulation
/*! This can be either KINEMATIC or DYNAMIC, depending on whether
 * the preference is to work in terms of the kinematic viscosity ν,
 * or in terms of the dynamic viscosity µ = ρν
 */
enum ComputationalViscosity {
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


//! Viscous model
//! (TODO this will become a viscous operator type, e.g. Morris vs Monaghan vs Español and Revenga)
enum ViscosityType {
	INVISCID, ///< no laminar viscosity TODO maybe this should become a RheologyType
	KINEMATICVISC, ///< Morris formula, simplified for constant kinematic viscosity and using harmonic averaging of the density
	DYNAMICVISC, ///< Morris formula, with arithmetic averaging of the dynamic density
	INVALID_VISCOSITY
} ;

//! Name of the viscous model
#ifndef GPUSPH_MAIN
extern
#endif
const char* ViscosityName[INVALID_VISCOSITY+1]
#ifdef GPUSPH_MAIN
= {
	"Inviscid",
	"Kinematic",
	"Dynamic",
	"(invalid)"
}
#endif
;

//! Turbulence model
/*!
 * While strictly speaking not a turbulence model, artificial viscosity is considered
 * among the turbulence model, since its behavior can be assimilated to it (i.e.
 * an addition to the viscous model, rather than an alternative to it).
 */
enum TurbulenceModel {
	LAMINAR_FLOW, ///< No turbulence
	ARTVISC, ///< Artificial viscosity
	SPSVISC, ///< Sub-particle scale turbulence model
	KEPSVISC, ///< k-epsilon turbulence model
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

//! Define the default turbulence model for the given viscous model
/*!
 * This is ARTVISC for inviscid flows, and laminar flow (no turbulence) otherwise.
 * (And invalid values map to invalid values).
 */
constexpr TurbulenceModel default_turbulence_for(ViscosityType visctype)
{
	return
		visctype == INVISCID ? ARTVISC :
		visctype >= INVALID_VISCOSITY ? INVALID_TURBULENCE :
			LAMINAR_FLOW;
}

//! Define the legacy laminar model for the given turbulence model
/*!
 * This is inviscid flow for the ARTVISC case, KINEMATICVISC for SPS, and
 * DYNAMICVISC for KEPS. In all other cases we return INVALID_VISCOSITY,
 * since there was no associated viscous model.
 */
constexpr ViscosityType default_laminar_visc_for(TurbulenceModel turbmodel)
{
	return
		turbmodel == ARTVISC ? INVISCID :
		turbmodel == SPSVISC ? KINEMATICVISC :
		turbmodel == KEPSVISC ? DYNAMICVISC :
			INVALID_VISCOSITY;
}

//! Select the default averaging operator for a given legacy viscous operator
constexpr AverageOperator default_avg_op(ViscosityType visctype)
{
	return visctype == KINEMATICVISC ? HARMONIC : ARITHMETIC;
}

//! A complete viscous specification includes:
// * a rheological model
// * a turbulence model
// * a computational viscosity specification
// * a viscosity type (TODO will become the viscous operator)
// * an averaging operator
template<
	RheologyType _rheologytype,
	TurbulenceModel _turbmodel,
	ComputationalViscosity _compvisc= KINEMATIC,
	ViscosityType _visctype = default_laminar_visc_for(_turbmodel),
	AverageOperator _avgop = default_avg_op(_visctype),
	// is this a constant-viscosity formulation?
	// TODO multifluid: we need to specify whether we're using one fluid
	// or more, since for #fluids > 1 we can't assume constant viscosity
	bool _is_const_visc = (_rheologytype == NEWTONIAN && _turbmodel != KEPSVISC)
>
struct FullViscSpec {
	static constexpr RheologyType rheologytype = _rheologytype;
	static constexpr TurbulenceModel turbmodel = _turbmodel;
	static constexpr ComputationalViscosity compvisc = _compvisc;
	static constexpr ViscosityType visctype = _visctype;
	static constexpr AverageOperator avgop = _avgop;

	static constexpr bool is_const_visc = _is_const_visc;

	//! Change the computational viscosity type specification
	/*! Sometimes we need to refer to the same viscous specification, except for the
	 * computational viscosity type; this type alias can be used to that effect
	 */
	template<ComputationalViscosity altcompvisc>
	using change_computational_visc =
		FullViscSpec<rheologytype, turbmodel, altcompvisc, visctype, avgop>;

	//! Force the assumption about constant viscosity
	/*! Sometimes we need to refer to the same viscous specification, but ignoring
	 * (or forcing) the assumption that the viscosity is constant;
	 * this type alias can be used to that effect
	 */
	template<bool is_const_visc>
	using assume_const_visc =
		FullViscSpec<rheologytype, turbmodel, compvisc, visctype, avgop, is_const_visc>;

};

#endif
