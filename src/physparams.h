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

/* Physical parameters for problems */

#ifndef _PHYSPARAMS_H
#define _PHYSPARAMS_H

#include <stdexcept>
#include <vector>
#include <iostream>

#include "particledefine.h"

#include "deprecation.h"

class Problem;
class XProblem;
class GPUWorker;
class GPUSPH;

typedef struct PhysParams {
	/*! Parameters for Cole's equation of state
		\f[
			P(\rho) = B((\rho/\rho_0)^\gamma - 1),
		\f]
		where \f$\rho_0\$ is the at-rest density,
		\f$\gamma\f$ is the adjabatic index and
		\f$B = \rho_0 c_0^2/\gamma\f$ for a given
		at-rest sound speed \f$c_0\f$.
		The sound speed for density \f$\rho\f$ can thus be computed as
		\f[
			c(\rho) = c_0((\rho/\rho_0)^{(\gamma -1)/2},
		\f]
	 */
	//! @{
	std::vector<float> rho0; //< At-rest density
	std::vector<float> bcoeff; //< Pressure parameter
	std::vector<float> gammacoeff; //< Adjabatic index
	std::vector<float> sscoeff; //< Sound speed coefficient ( = sound speed at rest density)
	std::vector<float> sspowercoeff; //< Sound speed exponent ( = (\gamma - 1)/2 )
	//! @}

	//! Fluid viscosity
	//! @{
	float	artvisccoeff;	//< Artificial viscosity coefficient (one, for all fluids)
	std::vector<float>	kinematicvisc;	//< Kinematic viscosity coefficient (ν)
	/*! Viscosity coefficient used in the viscous contribution functions, depends on
		viscosity model:
		* for ARTVSIC: artificial viscosity coefficient
		* for KINEMATICVISC: 4*kinematic viscosity coefficient,
		* for DYNAMICVISC: kinematic viscosity coefficient
		(The choice might seem paradoxical, but with DYNAMICVISC the dynamic viscosity
		 coefficient is obtained multiplying visccoeff by the particle density, while
		 with the KINEMATICVISC model the kinematic coefficient is used directly, in a
		 formula what also includes a harmonic average from which the factor 4 emerges.)
	 */
	std::vector<float>	visccoeff;
	//! @}

	float	partsurf;		// particle area (for surface friction)

	float3	gravity;		// gravity

	// interface epsilon for Grenier's simplified surface tension model
	float	epsinterface;

	// Lennard-Jones boundary coefficients
	float	r0;				// influence radius of boundary repulsive force
	float	dcoeff;
	float	p1coeff;
	float	p2coeff;
	// Monaghan-Kajtar boundary coefficients
	float	MK_K;			// Typically: maximum velocity squared, or gravity times maximum height
	float	MK_d;			// Typically: distance between boundary particles
	float	MK_beta;		// Typically: ratio between h and MK_d

	float	epsartvisc;
	float	epsxsph;		// XSPH correction coefficient

	// offset vector and limits for periodic boundaries:
	// DEPRECATED
	float3	dispvect DEPRECATED_MSG("dispvect is not needed anymore");
	float3	maxlimit DEPRECATED_MSG("maxlimit is not needed anymore");
	float3	minlimit DEPRECATED_MSG("minlimit is not needed anymore");

	float	ewres;			// DEM east-west resolution
	float	nsres;			// DEM north-south resolution
	float	demdx;			// Used for normal compution: displcement in x direction range ]0, exres[
	float	demdy;			// displcement in y direction range ]0, nsres[
	float	demdxdy;		// demdx*demdy
	float	demzmin;		// minimum z in DEM
	float	smagfactor;		// (Cs*∆p)^2
	float	kspsfactor;		// 2/3*Ci*∆p^2
	float	cosconeanglefluid;	     // cos of cone angle for free surface detection (If the neighboring particle is fluid)
	float	cosconeanglenonfluid;	 // cos of cone angle for free surface detection (If the neighboring particle is non_fluid
	float	objectobjectdf;	// damping factor for object-object interaction
	float	objectboundarydf;	// damping factor for object-boundary interaction

	// We have three deprecated members, but we don't need
	// to get a warning about them for the constructor, only
	// when the users actually assign to them
IGNORE_WARNINGS(deprecated-declarations)
	PhysParams(void) :
		artvisccoeff(0.3f),
		partsurf(0),
		gravity(make_float3(0, 0, -9.81)),
		epsinterface(NAN),
		r0(NAN),
		p1coeff(12.0f),
		p2coeff(6.0f),
		epsxsph(0.5f),
		smagfactor(NAN),
		kspsfactor(NAN),
		cosconeanglefluid(0.86f),
		cosconeanglenonfluid(0.5f),
		objectobjectdf(1.0f),
		objectboundarydf(1.0f)
	{};
RESTORE_WARNINGS

	// Problem, XProblem (but not their derivatives —luckily, friendship is not inherited)
	// GPUWorker and GPUSPH should be the only ones
	// that access the physparams manipulator methods
	friend class Problem;
	friend class XProblem;
	friend class GPUWorker;
	friend class GPUSPH;

	size_t numFluids() const
	{ return rho0.size(); }

protected:

	/*! Add a new fluid with given at-rest density
	  @param rho	at-rest density
	 */
	size_t add_fluid(float rho) {
		if (numFluids() == MAX_FLUID_TYPES)
			throw std::runtime_error("too many fluids");
		rho0.push_back(rho);

		// Prime the equation of state arrays, but do not initialize them
		gammacoeff.push_back(NAN);
		bcoeff.push_back(NAN);
		sscoeff.push_back(NAN);
		sspowercoeff.push_back(NAN);

		// Prime the viscosity coefficient arrays, but do not initialize them
		kinematicvisc.push_back(NAN);
		visccoeff.push_back(NAN);

		return rho0.size() - 1;
	}

	//! Change the density of the given fluid
	void set_density(size_t fluid_idx, float _rho0)
	{
		rho0.at(fluid_idx) = _rho0;
	}

	//! Get the density of the given fluid
	float get_density(size_t fluid_idx)
	{
		return rho0.at(fluid_idx);
	}

	/*! Set the equation of state of a given fluid, specifying the adjabatic
	 *  index and speed of sound. A non-finite speed of sound implies
	 *  that it should be autocomputed (currrently supported in XProblem only)
	  @param fluid_idx	fluid index
	  @param gamma	adjabatic index
	  @param c0	sound speed at rest
	  */
	void set_equation_of_state(size_t fluid_idx, float gamma, float c0) {
		if (fluid_idx >= numFluids())
			throw std::out_of_range("trying to set equation of state for a non-existing fluid");
		gammacoeff[fluid_idx] = gamma;
		bcoeff[fluid_idx] = rho0[fluid_idx]*c0*c0/gamma;
		sscoeff[fluid_idx] = c0;
		sspowercoeff[fluid_idx] = (gamma-1)/2;
	}

	/*! Set the kinematic viscosity of the given fluid
	  @param fluid_idx	fluid index
	  @param nu	kinematic viscosity
	  */
	void set_kinematic_visc(size_t fluid_idx, float nu) {
		kinematicvisc.at(fluid_idx) = nu;
	}

	/*! Set the dynamic viscosity of the given fluid
	  @param fluid_idx	fluid index
	  @param mu	dynamic viscosity
	  */
	void set_dynamic_visc(size_t fluid_idx, float mu) {
		set_kinematic_visc(fluid_idx, mu/rho0[fluid_idx]);
	}

	/*! Get the kinematic viscosity for the given fluid
	 * @param fluid_idx	fluid index
	 */
	float get_kinematic_visc(size_t fluid_idx) const {
		return kinematicvisc.at(fluid_idx);
	}

	/*! Set density parameters
	  @param i	index in the array of materials
	  @param rho	base density
	  @param gamma	gamma coefficient
	  @param c0	sound speed for density at rest

	  The number of fluids is automatically increased if set_density()
	  is called with consecutive indices
	 */
	void set_density(uint i, float rho, float gamma, float c0)
	DEPRECATED_MSG("set_density() is deprecated, use add_fluid() + set_equation_of_state() instead")
	{
		if (i == rho0.size()) {
			add_fluid(rho);
			set_equation_of_state(i, gamma, c0);
		}
		else if (i < rho0.size()) {
			std::cerr << "changing properties of fluid " << i << std::endl;
			set_density(i, rho);
			set_equation_of_state(i, gamma, c0);
		} else {
			std::cerr << "setting density for fluid index " << i << " > " << rho0.size() << std::endl;
			throw std::runtime_error("fluid index is growing too fast");
		}
	}
} PhysParams;

#endif
