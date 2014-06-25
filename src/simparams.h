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
#include "Point.h"

typedef struct MbCallBack {
	short			type;
	float			tstart;
	float			tend;
	float3			origin;
	float3			vel;
	float3			disp;
	float			sintheta;
	float			costheta;
	float			dthetadt;
	float			omega;
	float			amplitude;
	float			phase;
} MbCallBack;

typedef std::vector<double3> GageList;

typedef struct SimParams {
	double			sfactor;			// smoothing factor
	double			slength;			// smoothing length (smoothing factor * deltap)
	KernelType		kerneltype;			// kernel type
	double			kernelradius;		// kernel radius
	double			influenceRadius;	// influence radius ( = kernelradius * slength)
	double			nlInfluenceRadius;	// extended radius ( = influence radius * nlexpansionfactor)
	double			nlSqInfluenceRadius;	// square influence radius for neib list construction
	float			dt;					// initial timestep
	float			tend;				// simulation end time (0 means run forever)
	bool			xsph;				// true if XSPH correction
	bool			dtadapt;			// true if adaptive timestep
	float			dtadaptfactor;		// safety factor in the adaptive time step formula
	uint			buildneibsfreq;		// frequency (in iterations) of neib list rebuilding
	uint			shepardfreq;		// frequency (in iterations) of Shepard density filter
	uint			mlsfreq;			// frequency (in iterations) of MLS density filter
	float			ferrari;			// coefficient for Ferrari correction
	ViscosityType	visctype;			// viscosity type (1 artificial, 2 laminar)
	bool			mbcallback;			// true if moving boundary velocity varies
	bool			gcallback;			// true if using a variable gravity in problem
	Periodicity		periodicbound;		// periodicity of the domain (combination of PERIODIC_[XYZ], or PERIODIC_NONE)
	double			nlexpansionfactor;	// increase influcenradius by nlexpansionfactor for neib list construction
	bool			usedem;				// true if using a DEM
	SPHFormulation	sph_formulation;	// formulation to use for density and pressure computation
	BoundaryType	boundarytype;		// boundary force formulation (Lennard-Jones etc)
	bool			vorticity;			// true if we want to save vorticity
	bool			testpoints;			// true if we want to find velocity at testpoints
	bool			csvtestpoints;		// true to dump the testpoints also in CSV files
	bool			csvsimplegages;		// true to dump the gages also in CSV files
	bool			savenormals;		// true if we want to save the normals at free surface
	bool			surfaceparticle;	// true if we want to find surface particles
	bool			calc_energy;		// true if we want to compute system energy at save time
	GageList		gage;				// water gages
	uint			numODEbodies;		// number of floating bodies
	uint			maxneibsnum;		// maximum number of neibs (should be a multiple of NEIBS_INTERLEAVE)
	bool			calcPrivate;		// add the private array for debugging / additional calculation
	float			epsilon;			// if |r_a - r_b| < epsilon two positions are considered identical
	bool			movingBoundaries;	// defines if moving boundaries are present

	SimParams(void) :
		sfactor(1.3),
		slength(0),
		kerneltype(WENDLAND),
		kernelradius(2.0),
		influenceRadius(0),
		nlInfluenceRadius(0),
		nlSqInfluenceRadius(0),
		dt(0),
		tend(0),
		xsph(false),
		dtadapt(true),
		dtadaptfactor(0.3),
		buildneibsfreq(10),
		shepardfreq(0),
		mlsfreq(15),
		ferrari(0),
		visctype(ARTVISC),
		mbcallback(false),
		gcallback(false),
		periodicbound(PERIODIC_NONE),
		nlexpansionfactor(1.0),
		usedem(false),
		sph_formulation(SPH_F1),
		boundarytype(LJ_BOUNDARY),
		vorticity(false),
		testpoints(false),
		csvtestpoints(false),
		csvsimplegages(false),
		savenormals(false),
		surfaceparticle(false),
		calc_energy(true),
		numODEbodies(0),
		maxneibsnum(0),
		calcPrivate(false),
		epsilon(5e-1),
		movingBoundaries(false)
	{};

	inline double
	set_smoothing(double smooth, double deltap)
	{
		sfactor = smooth;
		slength = smooth*deltap;

		set_influenceradius();

		return slength;
	}

	inline double
	set_kernel(KernelType kernel, double radius=0)
	{
		kerneltype = kernel;
		// TODO currently all our kernels have radius 2,
		// remember to adjust this when we have kernels
		// with different radii
		kernelradius = radius ? radius : 2.0;

		return set_influenceradius();
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

