/*  Copyright (c) 2018 INGV, EDF, UniCT, JHU

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
/*! \file
 * Operators to compute m_β*2*A(µ_α, µ_β)/(ρ_αρ_β)
 */

#ifndef VISC_AVG_CU
#define VISC_AVG_CU

#include "visc_spec.h"
#include "average.h"

#include "cpp11_missing.h"

//! Viscous averaging operators
/*! These compute m_β*2*A(µ_α, µ_β)/(ρ_α ρ_β) efficiently,
 * based on the combination of choices for the averaging operator,
 * computational viscous choice and rheology type.
 */

//! Specialization for non-constant dynamic viscosity with arithmetic averaging
template<typename ViscSpec>
_AVG_FUNC_SPEC
enable_if_t<
	!ViscSpec::is_const_visc &&
	ViscSpec::avgop == ARITHMETIC &&
	ViscSpec::compvisc == DYNAMIC,
	float
>
visc_avg(float visc, float neib_visc, float rho, float neib_rho, float neib_mass)
{
	// m_β*2*((µ_α + µ_β)/2)/(ρ_α ρ_β) = 
	// m_β*(µ_α + µ_β)/(ρ_α ρ_β) = 
	return neib_mass*(visc+neib_visc)/(rho*neib_rho);
}

//! Specialization for density-only arithmetic averaging
template<typename ViscSpec>
_AVG_FUNC_SPEC
enable_if_t< ViscSpec::avgop == ARITHMETIC, float >
visc_avg(float rho, float neib_rho, float neib_mass)
{
	// m_β*2*((ρ_α + ρ_β)/2)/(ρ_αρ_β) = 
	// m_β*(ρ_α + ρ_β)/(ρ_αρ_β) = 
	return neib_mass*(rho+neib_rho)/(rho*neib_rho);
}

//! Specialization for non-constant dynamic viscosity with harmonic averaging
template<typename ViscSpec>
_AVG_FUNC_SPEC
enable_if_t<
	!ViscSpec::is_const_visc &&
	ViscSpec::avgop == HARMONIC &&
	ViscSpec::compvisc == DYNAMIC,
	float
>
visc_avg(float visc, float neib_visc, float rho, float neib_rho, float neib_mass)
{
	// m_β*2*(2*(µ_α µ_β)/(µ_α + µ_β))/(ρ_α ρ_β) = 
	// m_β*4*(µ_α µ_β)/(µ_α + µ_β))/(ρ_α ρ_β)
	return 4*neib_mass*(visc*neib_visc)/(visc+neib_visc)/(rho*neib_rho);
}

//! Specialization for density-only harmonic averaging
template<typename ViscSpec>
_AVG_FUNC_SPEC
enable_if_t< ViscSpec::avgop == HARMONIC, float >
visc_avg(float rho, float neib_rho, float neib_mass)
{
	// m_β*2*(2*(ρ_α ρ_β)/(ρ_α + ρ_β))/(ρ_α ρ_β) = 
	// m_β*4/(ρ_α + ρ_β)
	return 4*neib_mass/(rho+neib_rho);
}

//! Specialization for non-constant dynamic viscosity with geometric averaging
template<typename ViscSpec>
_AVG_FUNC_SPEC
enable_if_t<
	!ViscSpec::is_const_visc &&
	ViscSpec::avgop == GEOMETRIC &&
	ViscSpec::compvisc == DYNAMIC,
	float
>
visc_avg(float visc, float neib_visc, float rho, float neib_rho, float neib_mass)
{
	// No simplification for
	// m_β*2*(sqrt(µ_α µ_β))/(ρ_α ρ_β)
	return 2*neib_mass*average<GEOMETRIC>(visc,neib_visc)/(rho*neib_rho);
}

//! Specialization for density-only harmonic averaging
template<typename ViscSpec>
_AVG_FUNC_SPEC
enable_if_t< ViscSpec::avgop == GEOMETRIC, float >
visc_avg(float rho, float neib_rho, float neib_mass)
{
	// m_β*2*(sqrt(ρ_α ρ_β))/(ρ_α ρ_β) = 
	// m_β*2*rsqrt(ρ_α ρ_β)
	return 2*neib_mass*rsqrt(rho*neib_rho);
}

/* Optimizations in case of constant viscosity */

//! Specialization for constant dynamic viscosity
/*! In this case there is simply no averaging to do.
 */
template<typename ViscSpec>
_AVG_FUNC_SPEC
enable_if_t<
	ViscSpec::is_const_visc &&
	ViscSpec::compvisc == DYNAMIC,
	float
>
visc_avg(float visc, float neib_visc, float rho, float neib_rho, float neib_mass)
{
	return 2*neib_mass*visc/(rho*neib_rho);

}


//! Specialization for constant kinematic viscosity
/*! In this case the viscosity can be factored out,
 * and we are left with the densities instead of the dynamic viscosities
 * inside the averaging operator, so we apply the required averaging
 * operator, but in a special form that only gets passed the densities
 */
template<typename ViscSpec>
_AVG_FUNC_SPEC
enable_if_t<
	ViscSpec::is_const_visc &&
	ViscSpec::compvisc == KINEMATIC,
	float
>
visc_avg(float visc, float neib_visc, float rho, float neib_rho, float neib_mass)
{
	return visc*visc_avg<ViscSpec>(rho, neib_rho, neib_mass);
}

//! Specialization for non-constant kinematic viscosity
template<typename ViscSpec>
_AVG_FUNC_SPEC
enable_if_t<
	!ViscSpec::is_const_visc &&
	ViscSpec::compvisc == KINEMATIC,
	float
>
visc_avg(float visc, float neib_visc, float rho, float neib_rho, float neib_mass)
{
	// no simplification possible, just call the dynvisc variant
	using DynSpec = typename ViscSpec::template with_computational_visc<DYNAMIC>;
	return visc_avg<DynSpec>(visc*rho, neib_visc*neib_rho, rho, neib_rho, neib_mass);
}

#endif
