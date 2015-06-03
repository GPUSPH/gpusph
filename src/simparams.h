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

/* Simulation parameters for problems */

#ifndef _SIMPARAMS_H
#define _SIMPARAMS_H

#include <vector>
#include <stdexcept>
#include "Point.h"
#include "simflags.h"
#include "deprecation.h"

typedef std::vector<double3> GageList;

typedef struct SimParams {
	// Options that are set via SimFramework.
	const KernelType		kerneltype;				// kernel type
	const SPHFormulation	sph_formulation;		// formulation to use for density and pressure computation
	const ViscosityType	visctype;				// viscosity type (1 artificial, 2 laminar)
	const BoundaryType	boundarytype;			// boundary force formulation (Lennard-Jones etc)
	const Periodicity		periodicbound;			// periodicity of the domain (combination of PERIODIC_[XYZ], or PERIODIC_NONE)
	const flag_t			simflags;				// simulation flags

	double			sfactor;				// smoothing factor
	double			slength;				// smoothing length (smoothing factor * deltap)
	double			kernelradius;			// kernel radius
	double			influenceRadius;		// influence radius ( = kernelradius * slength)
	double			nlexpansionfactor;		// expand influenceradius by nlexpansionfactor for neib list construction
	double			nlInfluenceRadius;		// extended radius ( = influence radius * nlexpansionfactor)
	double			nlSqInfluenceRadius;	// square influence radius for neib list construction
	float			dt;						// initial timestep
	double			tend;					// simulation end time (0 means run forever)
	float			dtadaptfactor;			// safety factor in the adaptive time step formula
	uint			buildneibsfreq;			// frequency (in iterations) of neib list rebuilding

	float			ferrari;				// coefficient for Ferrari correction
	float			ferrariLengthScale;		// length scale for Ferrari correction

	bool			gcallback;				// true if using a variable gravity in problem
	bool			csvtestpoints;			// true to dump the testpoints also in CSV files
	bool			csvsimplegages;			// true to dump the gages also in CSV files
	bool			calc_energy;			// true if we want to compute system energy at save time
	GageList		gage;					// water gages
	uint			numODEbodies;			// number of bodies which movmement is computed by ODE
	uint			numforcesbodies;		// number of moving bodies on which we need to compute the forces (includes ODE bodies)
	uint			numbodies;				// total number of bodies (ODE + forces + moving)
	uint			maxneibsnum;			// maximum number of neibs (should be a multiple of NEIBS_INTERLEAVE)
	float			epsilon;				// if |r_a - r_b| < epsilon two positions are considered identical
	uint			numOpenBoundaries;				// number of open boundaries

	SimParams(
		KernelType _kernel = WENDLAND,
		SPHFormulation _formulation = SPH_F1,
		ViscosityType _visctype = ARTVISC,
		BoundaryType _btype = LJ_BOUNDARY,
		Periodicity _periodic = PERIODIC_NONE,
		flag_t _simflags = ENABLE_DTADAPT) :

		kerneltype(_kernel),
		sph_formulation(_formulation),
		visctype(_visctype),
		boundarytype(_btype),
		periodicbound(_periodic),
		simflags(_simflags),

		sfactor(1.3f),
		slength(0),
		/* The default kernel radius depends on the kernel choice:
		 * most kernels have radius 2 (and in fact do not support a different
		 * kernel radius currently), but the Gaussian kernel supports arbitrary
		 * kernel radii and uses 3 by default
		 * TODO we should have some centralized way to specify the default kernel
		 * radius for each KernelType
		 */
		kernelradius(_kernel == GAUSSIAN ? 3.0f : 2.0f),
		influenceRadius(0),
		nlexpansionfactor(1.0f),
		nlInfluenceRadius(0),
		nlSqInfluenceRadius(0),
		dt(0),
		tend(0),
		dtadaptfactor(0.3f),
		buildneibsfreq(10),

		ferrari(NAN),
		ferrariLengthScale(NAN),

		gcallback(false),
		csvtestpoints(false),
		csvsimplegages(false),
		calc_energy(true),
		numforcesbodies(0),
		numbodies(0),
		maxneibsnum(0),
		epsilon(5e-5f),
		numOpenBoundaries(0)
	{};

	inline double
	set_smoothing(double smooth, double deltap)
	{
		sfactor = smooth;
		slength = smooth*deltap;

		set_influenceradius();

		return slength;
	}

	// DEPRECATED, use set_kernel_radius instead
	inline double
	set_kernel(KernelType kernel, double radius=0) DEPRECATED
	{
		if (kernel != kerneltype)
			throw std::runtime_error("cannot change kernel type this way anymore");

		// TODO currently all our kernels have radius 2,
		// remember to adjust this when we have kernels
		// with different radii
		set_kernel_radius(radius ? radius :
			kernel == GAUSSIAN ? 3.0 : 2.0);

		return set_influenceradius();
	}

	inline void
	set_kernel_radius(double radius)
	{
		kernelradius = radius;
		set_influenceradius();
	}


	// internal: update the influence radius et al
	inline double
	set_influenceradius() {
		influenceRadius = slength * kernelradius;
		nlInfluenceRadius = nlexpansionfactor * influenceRadius;
		nlSqInfluenceRadius = nlInfluenceRadius * nlInfluenceRadius;

		return influenceRadius;
	}

} SimParams;

#endif

