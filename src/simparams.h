/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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
 * Simulation parameters for problems
 */

#ifndef _SIMPARAMS_H
#define _SIMPARAMS_H

#include <vector>
#include <stdexcept>
#include "particledefine.h"
#include "simflags.h"
// #include "deprecation.h"

typedef std::vector<double4> GageList;

/** The SimParams structure holds all the simulation related parameters
 *  along with some basic initialization functions.
 *
 *	\ingroup datastructs
 */
typedef struct SimParams {
	/** \name Options that are set via SimFramework
	 * @{ */
	const KernelType		kerneltype;					///< Kernel type
	const SPHFormulation	sph_formulation;			///< SPH formulation to use
	const DensityDiffusionType densitydiffusiontype;	///< Type of density diffusion corrective term

	const RheologyType		rheologytype;				///< Rheology type (Newtonian etc)
	const TurbulenceModel	turbmodel;					///< Turbulence model
	const ComputationalViscosityType compvisc;			///< Viscosity kind used in computations (kinematic vs dynamic)
	const ViscousModel		viscmodel;					///< Discretization model for the viscous operator (e.g. Morris vs Monaghan)
	const AverageOperator	viscavgop;					///< Averaging operator used in the viscous model (e.g. arithmetic vs harmonic)
	const bool				is_const_visc;				///< Are we assuming constant viscosity?

	const BoundaryType		boundarytype;				///< Boundary type (Lennard-Jones, SA? ...)
	const Periodicity		periodicbound;				///< Periodicity of the domain (combination of PERIODIC_[XYZ], or PERIODIC_NONE)
	const flag_t			simflags;					///< Simulation flags
	/** @} */

	/* \note
	 * simparams.h is scanned by the SALOME user interface.
	 * To change the user interface, it is only necessary to
	 * modify the appropriate comments in simparams.h, physparams.h,
	 * Problem.h, XProblem.h, particledefine.h and simflags.h
	 * The variable labels and tooltips are
	 * defined in the user interface files themselves, so
	 * ease follow the convention we adopted: use placeholders
	 * in the GPUSPH files and define them in GPUSPHGUI.
	 * The tooltips are the comments appearing when sliding
	 * the mouse over a variable in the interface. They are
	 * contained in the TLT_ variables. All the placeholders
	 * contents are defined in:
	 * gpusphgui/SGPUSPH_SRC/src/SGPUSPHGUI/resources/SGPUSPH_msg_en.ts
	 * The sections to be used in the user interface are
	 * defined in gpusphgui/SGPUSPH/resources/params.xml.
	 * To assign a parameter to a section, the command
	 * \inpsection is used.
	 * Please consult this file for the list of sections.
	 */

	/** \name Kernel and neighbor list related parameters
	 * @{ */
	/**
	 * \inpsection{discretisation}
	 * \default{1.3}
	 * \label{SMOOTHING_FACTOR}
	 * TLT_SMOOTHING_FACTOR
	 */
	double			sfactor;				///< Smoothing factor
	double			slength;				///< Smoothing length \f$ h \f$ (smoothing factor * \f$ \Delta p \f$)
	double			kernelradius;			///< Kernel radius \f$ \kappa \f$
	double			influenceRadius;		///< Influence radius \f$ \kappa h \f$
	double			nlexpansionfactor;		///< Expansion factor to apply to influenceradius for the neighbor list construction
	double			nlInfluenceRadius;		///< Influence radius ( = \f$ \kappa h \f$ * nlexpansionfactor) used in neighbor list construction
	double			nlSqInfluenceRadius;	///< Square influence radius for neighbor list construction
	/*!
	 * \inpsection{neighbours}
	 * \default{1}
	 * \label{NEIB_FREQ}
	 * TLT_NEIB_FREQ
	 */
	uint			buildneibsfreq;			///< Frequency (in iterations) of neighbor list rebuilding
	/*!
	 * \inpsection{neighbours}
	 * \default{256}
	 * \label{NEIB_LIST_SIZE}
	 * TLT_NEIB_LIST_SIZE
	 */
	uint			neiblistsize;			///< Total size of the neighbor list (per particle)
	/*!
	 * \inpsection{neighbours}
	 * \default{256}
	 * \label{NEIB_BOUND_POS}
	 * TLT_NEIB_BOUND_POS
	 */
	uint			neibboundpos;			///< Marker for boundary parts section of the neighbor list
	/** @} */

	/*! \inpsection{variable_dt, disable}
	 * \label{DT}
	 * \default{1e-5}
	 * TLT_DT
	 */
	float			dt;						///< Time step (initial when using adaptive time stepping)

	/*! \inpsection{time}
	 * \label{SIMULATION_END_TIME}
	 * \default{10.}
	 * TLT_SIMULATION_END_TIME
	 */
	double			tend;					///< Simulation end time (0 means run forever)

	/** \name FEA-related parameterp
	 * TODO need UI parameters
	 * @{ */
	double			t_fea_start;					///<  FEM analysis starting time
	float			fea_write_every;				///<  FEM nodes written to file every
	int			numNodesToWrite;				///<  Number of FEM nodes to write
	int			numConstraintsToWrite;				///<  Number of FEM constraints for which to write force and torque
	uint			feaSph_iterations_ratio;	///<  FEM analysis is computed every feaSph_iterations_ratio sph iterations
	/** @} */

	/*! \inpsection{variable_dt, enable}
	 * \label{DT_FACTOR}
	 * \default{0.3}
	 * \max{1.}
	 * TLT_DT_FACTOR
	 */
	float			dtadaptfactor;			///< Safety factor used in adaptive time step computation

	/** \name Density diffusion related parameters
	 * @{ */
	/*!
	 * \inpsection{density_diff_type, Colagrossi}
	 * \label{DENSITY_DIFFUSION_COEF}
	 * \default{0.1}
	 * \min{0}
	 * \max{1}
	 * TLT_DENSITY_DIFF_COEF
	 */
	float			densityDiffCoeff;		///< Coefficient for density diffusion TODO: be more precise
	/*!
	 * \inpsection{density_diff_type, Brezzi}
	 * \label{DENSITY_DIFFUSION_COEF}
	 * \default{0.1}
	 * \min{0}
	 * \max{1}
	 * TLT_DENSITY_DIFF_COEF
	 */
	// For brezziDiffCoeff to appear in the GUI we need to define it here explicitely.
	// However, do not try to set its value in your Problem: it is not used by GPUSPH, only
	// densityDiffCoeff is used, for all types of densityDiffusion.
	// In GenericProblem, densityDiffCoeff is set equal to brezziDiffCoeff.
	float			brezziDiffCoeff; //< Dummy coefficient for Brezzi density diffusion, only for the user interface
	/*!
	 * \inpsection{density_diff_type, Ferrari}
	 * \label{DENSITY_DIFFUSION_COEF}
	 * \default{0.1}
	 * \min{0}
	 * \max{1}
	 * TLT_DENSITY_DIFF_COEF
	 */
	// For ferrariDiffCoeff to appear in the GUI we need to define it here explicitely.
	// However, do not try to set its value in your Problem: it is not used by GPUSPH, only
	// densityDiffCoeff is used, for all types of densityDiffusion.
	// In GenericProblem, densityDiffCoeff is set equal to ferrariDiffCoeff.
	float			ferrariDiffCoeff;		///< Dummy coefficient for Ferrari density diffusion, only for the user interface

	float			ferrariLengthScale;		///< Length scale for Ferrari correction
	/** @} */

	/** \name Call back and post-processing related parameters
	 * @{ */
	bool			gcallback;		///< True if using a variable gravity set through a callback function
	bool			calc_energy;	///< True if we want to compute system energy at save time
	GageList		gage;			///< Water gages list
	bool			fcallback;		///< True if using a GT_FEA_FORCE set through a callback function
	uint			numForceNodes;			///< number of nodes with force applied by GT_FEA_FORCE TODO here we are assuming only one force box is used. To define more this should be an array
	/** @} */

	/** \name Floating/moving bodies related parameters
	 * @{ */
	uint			numODEbodies;			///< Number of bodies which movement is computed by ODE
	uint			numforcesbodies;		///< Number of moving bodies on which we need to compute the forces on (includes ODE bodies)
	uint			numbodies;				///< Total number of bodies (ODE + forces + moving)
	/** @} */

	/** \name Deformable bodies related parameters
	 * @{ */
	uint			numfeabodies;			///< Number of bodies on which we perform FEA 
	/** @} */

	/** \name I/O boundaries related parameters
	 * @{ */
	uint			numOpenBoundaries;		///< Number of open boundaries
	/** @} */

	/** \name Other parameters
	 * @{ */
	float			epsilon;				///< If \f$ |r_a - r_b| < \epsilon \f$ two positions are considered identical. TODO: check that the test is done on a relative quantity
	/** @} */
	/** \name Repacking parameters
	 * @{ */
	/*!
	 * \inpsection{repacking, enable}
	 * \label{REPACK_MAX_ITER}
	 * \default{200}
	 * TLT_REPACK_MAX_ITER
	 */
	uint			repack_maxiter; //< maximum number of iterations for repacking
	/*!
	 * \inpsection{repacking, enable}
	 * \label{REPACK_A}
	 * \default{0.1}
	 * TLT_REPACK_A
	 */
	float			repack_a;	//< repacking parameter 'a' for mixing intensity, recommended value: 1
	/*!
	 * \inpsection{repacking, enable}
	 * \label{REPACK_ALPHA}
	 * \default{0.01}
	 * TLT_REPACK_ALPHA
	 */
	float			repack_alpha;	//< repacking parameter 'alpha' for velocity damping, recommended value: 0.1
	/** @} */
	/** \name Jacobi parameters
	 * @{ */
	/*!
	 * \inpsection{jacobi, enable}
	 * \label{JACOBI_MAX_ITER}
	 * \default{1000}
	 * TLT_JACOBI_MAX_ITER
	 */
	uint			jacobi_maxiter; //< maximum number of iterations for effective pressure Jacobi solver
	/*!
	 * \inpsection{repacking, enable}
	 * \label{JACOBI_BACKERR}
	 * \default{0.00001}
	 * TLT_JACOBI_BACKERR
	 */
	float			jacobi_backerr;	//< backward error threshold for boundary particles (only vertex for SA) effective pressure convergence
	/*!
	 * \inpsection{repacking, enable}
	 * \label{JACOBI_RESIDUAL}
	 * \default{0.000001}
	 * TLT_JACOBI_RESIDUAL
	 */
	float			jacobi_residual;	//< residual threshold for fluid particles effective pressure convergence
	/** @} */

	template<typename Framework>
	SimParams(Framework *simframework) :
		kerneltype(Framework::kerneltype),
		sph_formulation(Framework::sph_formulation),
		densitydiffusiontype(Framework::densitydiffusiontype),
		rheologytype(Framework::rheologytype),
		turbmodel(Framework::turbmodel),
		compvisc(Framework::compvisc),
		viscmodel(Framework::viscmodel),
		viscavgop(Framework::viscavgop),
		is_const_visc(Framework::is_const_visc),
		boundarytype(Framework::boundarytype),
		periodicbound(Framework::periodicbound),
		simflags(Framework::simflags),

		sfactor(1.8f),
		slength(0),
		/* The default kernel radius depends on the kernel choice:
		 * most kernels have radius 2 (and in fact do not support a different
		 * kernel radius currently), but the Gaussian kernel supports arbitrary
		 * kernel radii and uses 3 by default
		 * TODO we should have some centralized way to specify the default kernel
		 * radius for each KernelType
		 */
		kernelradius(kerneltype == GAUSSIAN ? 3.0f : 2.0f),
		influenceRadius(0),
		nlexpansionfactor(1.0f),
		nlInfluenceRadius(0),
		nlSqInfluenceRadius(0),
		buildneibsfreq(10),
		neiblistsize(0),
		neibboundpos(0),

		dt(0),
		tend(0),
		t_fea_start(0),
		fea_write_every(NAN),
		numNodesToWrite(0),
		numConstraintsToWrite(0),
		feaSph_iterations_ratio(1),
		dtadaptfactor(0.3f),

		densityDiffCoeff(NAN),
		ferrariLengthScale(NAN),

		gcallback(false),
		calc_energy(true),
		fcallback(false),
		numForceNodes(0),
		numODEbodies(0),
		numforcesbodies(0),
		numbodies(0),
		numfeabodies(0),
		numOpenBoundaries(0),
		epsilon(5e-5f),
		repack_maxiter(2000),
		repack_a(0.1f),
		repack_alpha(0.01f),
		jacobi_maxiter(1000),
		jacobi_backerr(0.00001),
		jacobi_residual(0.000001)
	{}

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
			double smooth, 	///< [in] smoothing factor
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

	/// return true if the boundary type requires multiple layers
	inline bool
	boundary_is_multilayer() const
	{ return ::boundary_is_multilayer(boundarytype); }
	/** @} */

} SimParams;

#endif

