/*  Copyright (c) 2015-2019 INGV, EDF, UniCT, JHU

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
 * Physical constants used by all kernels
 */

#ifndef _PHYS_CORE_
#define _PHYS_CORE_

#include "particledefine.h"

/*
 * SFINAE helpers for particle data structure access
 */

#include "has_member.h"
DECLARE_MEMBER_DETECTOR(relVel, has_relVel)

namespace cuphys {
__constant__ uint	d_numfluids;			///< number of different fluids

__constant__ float	d_sqC0[MAX_FLUID_TYPES];	///< square of sound speed for at-rest density for each fluid

__constant__ float	d_densityDiffCoeff;			///< coefficient for density diffusion

__constant__ float	d_epsinterface;			///< interface epsilon for simplified surface tension in Grenier

// LJ boundary repusion force comuting
__constant__ float	d_dcoeff;
__constant__ float	d_p1coeff;
__constant__ float	d_p2coeff;
__constant__ float	d_r0;

// Monaghan-Kaijar boundary repulsion force constants
__constant__ float	d_MK_K;		///< This is typically the square of the maximum velocity, or gravity times the maximum height
__constant__ float	d_MK_d;		///< This is typically the distance between boundary particles
__constant__ float	d_MK_beta;	///< This is typically the ration between h and the distance between boundary particles

__constant__ float	d_visccoeff[MAX_FLUID_TYPES];	///< viscous coefficient or consistency index for laminar viscosity or lower of kinematic effective viscosity for granular rheology
__constant__ float	d_visc2coeff[MAX_FLUID_TYPES]; ///< second viscous coefficient or bulk viscosity
__constant__ float	d_yield_strength[MAX_FLUID_TYPES]; ///< yield strength for Bingham and Herschel–Bulkley rheology
__constant__ float	d_visc_nonlinear_param[MAX_FLUID_TYPES]; ///< exponent for power-law and Herschel–Bulkley rheology, exponential coefficient for Zhu and DeKee & Turcotte
__constant__ float	d_visc_regularization_param[MAX_FLUID_TYPES]; ///< Papanastasiou regularization parameter

__constant__ float	d_limiting_kinvisc; ///< upper bound to the viscosity
__constant__ float	d_monaghan_visc_coeff; ///< multiplicative coefficient in Monaghan's viscous model

// granular rheology parameters
__constant__ float	d_sinpsi[MAX_FLUID_TYPES];// sinus of internal friction angle for granular rheology (yield stress parameter)
__constant__ float	d_cohesion[MAX_FLUID_TYPES];// sinus of internal friction angle for granular rheology (yield stress parameter)

// Artificial viscosity parameters
__constant__ float	d_artvisccoeff;					///< viscous coefficient for artificial viscosity
__constant__ float	d_epsartvisc;					///< epsilon of artificial viscosity

// Sub-Particle Scale (SPS) Turbulence parameters
__constant__ float	d_smagfactor;
__constant__ float	d_kspsfactor;

__constant__ float	d_partsurf;		///< particle surface (typically particle spacing suared)

// Free surface detection
__constant__ float	d_cosconeanglefluid;
__constant__ float	d_cosconeanglenonfluid;

// physical constants
__constant__ float	d_rho0[MAX_FLUID_TYPES];		// rest density of fluids
__constant__ float3	d_gravity;						// gravity (vector)
// speed of sound constants
__constant__ float	d_bcoeff[MAX_FLUID_TYPES];		// \rho_0 c_0^2 / \gamma
__constant__ float	d_gammacoeff[MAX_FLUID_TYPES];	// \gamma
__constant__ float	d_sscoeff[MAX_FLUID_TYPES];		// c_0
__constant__ float	d_sspowercoeff[MAX_FLUID_TYPES];// (\gamma - 1)/2

// repacking constants
__constant__ float d_repack_alpha; // parameter alpha for repacking, controls the damping
__constant__ float d_repack_a; // parameter a for repacking, controls the mixing
__constant__ float d_repack_v; // parameter V_0.

/********************** Equation of state, speed of sound, repulsive force **********************************/
// Equation of state: pressure from density, where i is the fluid kind, not particle_id

__device__ __forceinline__ float
P(const float rho_tilde, const ushort i)
{
	const float rho_ratio = rho_tilde + 1.0f; // rho/rho0
	return d_bcoeff[i]*(__powf(rho_ratio, d_gammacoeff[i]) - 1.0f);
}

// Inverse equation of state: density from pressure, where i is the fluid kind, not particle_id
//RHO returns rho_tilde = rho/rho0 - 1
__device__ __forceinline__ float
RHO(const float p, const ushort i)
{
	return __powf(p/d_bcoeff[i] + 1.0f, 1.0f/d_gammacoeff[i]) - 1.0f;
}

// Riemann celerity
__device__ float
R(const float rho_tilde, const ushort i)
{
	const float rho_ratio = rho_tilde + 1.0f; // rho/rho0
	return 2.0f/(d_gammacoeff[i]-1.0f)*d_sscoeff[i]*__powf(rho_ratio, 0.5f*d_gammacoeff[i]-0.5f);
}

// Relative density from Riemann celerity
__device__ __forceinline__ float
RHOR(const float r, const ushort i)
{
	return __powf((d_gammacoeff[i]-1.f)*r/(2.f*d_sscoeff[i]), 2.f/(d_gammacoeff[i]-1.f)) -1.0f;
}

// Sound speed computed from density
__device__ __forceinline__ float
soundSpeed(const float rho_tilde, const ushort i)
{
	const float rho_ratio = rho_tilde + 1.0f; // rho/rho0
	return d_sscoeff[i]*__powf(rho_ratio, d_sspowercoeff[i]);
}

// returns physical density from numerical (stored) density
__device__ __forceinline__ float
physical_density(const float rho_tilde, const ushort i)
{
	return (rho_tilde + 1.0f)*d_rho0[i];
}

// Uniform precision on density
// returns the numerical (stored) density from physical density
__device__ __forceinline__ float
numerical_density(const float rho, const ushort i)
{
	return rho/d_rho0[i] - 1.0f;
}

/********************** Particle data structure management **********************************/
/* Several kernels use a particle data structure to carry around particle information after
 * it has been loaded. The members are named quite consistently, so we can provide some wrapper
 * functions to access the desired property directly.
 */

/// Fetch the particle (numerical) density
/*! This is either pdata.vel.w or ndata.relVel.w, depending on whether we're
 * doing it for the central particle or the neighbor. The difference is handled
 * by the specializations, whose SFINAE condition is the presence of a relVel member.
 */
template<typename N>
__device__ __forceinline__
enable_if_t<has_relVel<N>(), float>
particle_density(N const& ndata)
{
	/* has relVel, so it's neighbor data */
	return ndata.relVel.w;
}

template<typename P>
__device__ __forceinline__
enable_if_t<not has_relVel<P>(), float>
particle_density(P const& pdata)
{
	/* no relVel, assume it's central particle data */
	return pdata.vel.w;
}

/// Fetch the physical particle density
template<typename N>
__device__ __forceinline__
float
physical_density(N const& ndata)
{
	return cuphys::physical_density(particle_density(ndata), fluid_num(ndata.info));
}


}
#endif
