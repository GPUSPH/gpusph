/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Universit√† di Catania, Catania, Italy

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
#include "particledefine.h"
#include "simflags.h"
// #include "deprecation.h"

typedef std::vector<double4> GageList;

/// Structure holding all simulation related parameters
/*! This structure holds all the simulation related parameters
 *  along with some basic initialization functions.
 *
 *	\ingroup datastructs
 */
typedef struct SimParams {
	/** \name Options that are set via SimFramework
	 * @{ */
	const KernelType		kerneltype;					///< Kernel type
	const SPHFormulation	sph_formulation;			///< SPH formulation to use
	const DensityDiffusionType densitydiffusiontype; 	///< Type of density diffusion corrective term
	const ViscosityType		visctype;					///< Viscosity type (artificial, laminar, ...)
	const BoundaryType		boundarytype;				///< Boundary type (Lennard-Jones, SA? ...)
	const Periodicity		periodicbound;				///< Periodicity of the domain (combination of PERIODIC_[XYZ], or PERIODIC_NONE)
	const flag_t			simflags;					///< Simulation flags
	/** @} */

	/** \name Kernel and neighbor list related parameters
	 * @{ */
	double			sfactor;				///< Smoothing factor
	double			slength;				///< Smoothing length \f$ h \f$ (smoothing factor * \f$ \Delta p \f$)
	double			kernelradius;			///< Kernel radius \f$ \kappa \f$
	double			influenceRadius;		///< Influence radius \f$ \kappa h \f$
	double			nlexpansionfactor;		///< Expansion factor to apply to influenceradius for the neighbor list construction
	double			nlInfluenceRadius;		///< Influence radius ( = \f$ \kappa h \f$ * nlexpansionfactor) used in neighbor list construction
	double			nlSqInfluenceRadius;	///< Square influence radius for neighbor list construction
	uint			buildneibsfreq;			///< Frequency (in iterations) of neighbor list rebuilding
	uint			neiblistsize;			///< Total size of the neighbor list (per particle)
	uint			neibboundpos;			///< Marker for boundary parts section of the neighbor list
	/** @} */

	/** \name Time related parameters
	 * @{ */
	float			dt;						///< Time step (initial when using adaptive time stepping)
	double			tend;					///< Simulation end time (0 means run forever)
	float			dtadaptfactor;			///< Safety factor used in adaptive time step computation
	/** @} */

	/** \name Density diffusion related parameters
	 * @{ */
	float			densityDiffCoeff;		///< Coefficient for density diffusion TODO: be more precise
	float			ferrariLengthScale;		///< Length scale for Ferrari correction
	/** @} */

	/** \name Call back and post-processing related parameters
	 * @{ */
	bool			gcallback;				///< True if using a variable gravity set trough a callback function
	bool			calc_energy;			///< True if we want to compute system energy at save time
	GageList		gage;					///< Water gages list
	/** @} */

	/** \name Floating/moving bodies related parameters
	 * @{ */
	uint			numODEbodies;			///< Number of bodies which movement is computed by ODE
	uint			numforcesbodies;		///< Number of moving bodies on which we need to compute the forces on (includes ODE bodies)
	uint			numbodies;				///< Total number of bodies (ODE + forces + moving)
	/** @} */

	/** \name I/O boundaries related parameters
	 * @{ */
	uint			numOpenBoundaries;		///< Number of open boundaries
	/** @} */

	/** \name Other parameters
	 * @{ */
	float			epsilon;				///< If \f$ |r_a - r_b| < \epsilon \f$ two positions are considered identical. TODO: check that the test is done on a relative quantity
	/** @} */

	SimParams(
		KernelType _kernel = WENDLAND,
		SPHFormulation _formulation = SPH_F1,
		DensityDiffusionType _densitydiffusiontype = DENSITY_DIFFUSION_NONE,
		ViscosityType _visctype = ARTVISC,
		BoundaryType _btype = LJ_BOUNDARY,
		Periodicity _periodic = PERIODIC_NONE,
		flag_t _simflags = ENABLE_DTADAPT) :

		kerneltype(_kernel),
		sph_formulation(_formulation),
		densitydiffusiontype(_densitydiffusiontype),
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

		densityDiffCoeff(NAN),
		ferrariLengthScale(NAN),

		gcallback(false),
		calc_energy(true),
		numforcesbodies(0),
		numbodies(0),
		neiblistsize(0),
		neibboundpos(0),
		epsilon(5e-5f),
		numOpenBoundaries(0)
	{};

	/** \name Kernel parameters related methods
	 * @{ */
	/// Set Kernel and neighbor list influence radius parameters
	/*! This function set the kernel parameters (smoothing factor, \f$ h \f$, ...) along
	 *  with the influence radius used for the neighbor list construction.
	 *
	 *  \return smoothing length \f$ h \f$
	 */
	inline double
	set_smoothing(
			double smooth, 	///< [in] smoothing factor
			double deltap	///< [in] particle spacing \f$ \Delta p \f$
			)
	{
		sfactor = smooth;
		slength = smooth*deltap;

		set_influenceradius();

		return slength;
	}

	/// Set Kernel radius
	/*! This function set the radius of the currently used Kernel and update the
	 *  related variables.
	 */
	inline void
	set_kernel_radius(
			double radius	///< Influence radius \f$ \kappa \f$
			)
	{
		kernelradius = radius;
		set_influenceradius();
	}

	/// Set neighbor list expansion factor
	/*! This function set the expansion factor used to determine the influence radius used
	 *  during neighbor search.
	 */
	inline double
	set_neiblist_expansion(double _nlfactor)
	{
		nlexpansionfactor = _nlfactor;
		set_influenceradius();
		return nlInfluenceRadius;
	}


	/// Update Kernel radius related parameters
	/*! This the Kernel radius related parameters (influenceRadius, nlInfluenceRadius, ...).
	 *  Only used internally.
	 */
	inline double
	set_influenceradius() {
		influenceRadius = slength * kernelradius;
		nlInfluenceRadius = nlexpansionfactor * influenceRadius;
		nlSqInfluenceRadius = nlInfluenceRadius * nlInfluenceRadius;

		return influenceRadius;
	}

	/// Return the number of layers of particles necessary to cover the influence radius
	/*!
	 *  \return layers number
	 */
	inline int
	get_influence_layers() const
	{ return (int)ceil(sfactor*kernelradius); }
	/** @} */

} SimParams;

#endif

