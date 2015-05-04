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
#include <iostream>

#include "particledefine.h"

#include "deprecation.h"

typedef struct PhysParams {
	float	rho0[MAX_FLUID_TYPES]; // density of various particles

	float	partsurf;		// particle area (for surface friction)

	float3	gravity;		// gravity
	float	bcoeff[MAX_FLUID_TYPES];
	float	gammacoeff[MAX_FLUID_TYPES];
	float	sscoeff[MAX_FLUID_TYPES];
	float	sspowercoeff[MAX_FLUID_TYPES];

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

	float	kinematicvisc[MAX_FLUID_TYPES];	// Kinematic viscosity
	float	artvisccoeff;	// Artificial viscosity coefficient
	// For ARTVSIC: artificial viscosity coefficient
	// For KINEMATICVISC: 4*kinematic viscosity,
	// For DYNAMICVISC: dynamic viscosity
	float	visccoeff[MAX_FLUID_TYPES];
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
	uint	numFluids;      // number of fluids in simulation
	float	cosconeanglefluid;	     // cos of cone angle for free surface detection (If the neighboring particle is fluid)
	float	cosconeanglenonfluid;	 // cos of cone angle for free surface detection (If the neighboring particle is non_fluid
	float	objectobjectdf;	// damping factor for object-object interaction
	float	objectboundarydf;	// damping factor for object-boundary interaction

	// We have three deprecated members, but we don't need
	// to get a warning about them for the constructor, only
	// when the users actually assign to them
IGNORE_WARNINGS(deprecated-declarations)
	PhysParams(void) :
		partsurf(0),
		epsinterface(NAN),
		r0(NAN),
		p1coeff(12.0f),
		p2coeff(6.0f),
		epsxsph(0.5f),
		smagfactor(NAN),
		kspsfactor(NAN),
		numFluids(1),
		cosconeanglefluid(0.86f),
		cosconeanglenonfluid(0.5f),
		objectobjectdf(1.0f),
		objectboundarydf(1.0f)
	{};
RESTORE_WARNINGS

	/*! Set density parameters
	  @param i	index in the array of materials
	  @param rho	base density
	  @param gamma	gamma coefficient
	  @param c0	sound speed for density at rest

	  The number of fluids is automatically increased if set_density()
	  is called with consecutive indices
	 */
	void set_density(uint i, float rho, float gamma, float c0) {
		rho0[i] = rho;
		gammacoeff[i] = gamma;
		bcoeff[i] = rho*c0*c0/gamma;
		sscoeff[i] = c0;
		sspowercoeff[i] = (gamma - 1)/2;
		/* Check if we need to increase numFluids. If the user skipped an index,
		 * we will have i > numFluids, which we assume it's an error; otherwise,
		 * we will have i <= numFluids, and if == we need to increase
		 */
		if (i > numFluids) {
			std::cerr << "setting density for fluid index " << i << " > " << numFluids << std::endl;
			throw std::runtime_error("fluid index is growing too fast");
		}
		if (i == numFluids)
			numFluids++;
	}
} PhysParams;

#endif
