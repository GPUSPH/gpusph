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

//! Viscous model
enum ViscosityType {
	INVISCID, ///< no laminar viscosity
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

#endif
