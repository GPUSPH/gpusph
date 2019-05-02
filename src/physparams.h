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

/*! \file
 * Physical parameters for problems
 */

/* \note
 * The sections to be used in the user interface are
 * defined in gpusphgui/SGPUSPH/resources/params.xml.
 * Please consult this file for the list of sections.
*/

#ifndef _PHYSPARAMS_H
#define _PHYSPARAMS_H

#include <stdexcept>
#include <vector>
#include <iostream>

#include "particledefine.h"

#include "visc_spec.h"

#include "deprecation.h"

class ProblemCore;
template<int version>
class ProblemAPI;
class GPUWorker;
class GPUSPH;

/* The PhysParams structure holds all the physical parameters needed by the simulation
 *  along with some basic initialization functions.
 *
 *	\ingroup datastructs
 */
typedef struct PhysParams {
	/* \note
	 * physparams.h is scanned by the SALOME user interface.
	 * To change the user interface, it is only necessary to
	 * modify the appropriate comments in simparams.h, physparams.h,
	 * ProblemCore.h, XProblem.h, particledefine.h and simflags.h
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

	//! Rheology type
	/*! This is initialized at structure construction, and is used to determine how
	 * to use some of the viscous options
	 */
	const RheologyType rheologytype;

	/*! Equation of state related parameters
		The relation between pressure and density is given for Cole's equation of state
		\f[
		P(\rho) = B\left(\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right)
		\f]
		where \f$\rho_0\f$ is the at-rest density,
		\f$\gamma\f$ is the adiabatic index and
		\f$B = \rho_0 c_0^2/\gamma\f$ for a given
		at-rest sound speed \f$c_0\f$.
		The sound speed for a given density \f$\rho\f$ can thus be computed as
		\f[
		c(\rho) = c_0 \left(\frac{\rho}{\rho_0}\right)^{(\gamma -1)/2},
		\f]
		Obviously all of the coefficients below are fluid dependent and are then stored in an
		STL vector.
	 */
	//! @{

  /*!
	 * \inpsection{fluid}
	 * \label{FLUID_DENSITY}
	 * \default{1000}
	 */
	std::vector<float> rho0; 			///< At-rest density, \f$\rho_0\f$

	std::vector<float> bcoeff; 			///< Pressure parameter, \f$ B \f$
  /*!
	 * \inpsection{fluid}
	 * \label{EOS_EXPONENT}
	 * \default{7}
   * TLT_EOS_EXPONENT
	 */
	std::vector<float> gammacoeff;		///< Adiabatic index, \f$\gamma\f$
  /*!
	 * \defpsubsection{c0_input_method, C0_INPUT_METHOD}
	 * \inpsection{fluid,0}
	 * \values{direct_input, calculation}
	 * \default{direct_input}
	 * TLT_C0_INPUT_METHOD
	 */
	/*!
	 * \inpsection{c0_input_method, direct_input}
	 * \label{FLUID_C0}
	 * \default{0}
	 * TLT_FLUID_C0
	 */
	std::vector<float> sscoeff; 		///< Sound speed coefficient ( = sound speed at rest density, \f$ c_0 \f$)

	std::vector<float> sspowercoeff; 	///< Sound speed equation exponent ( = \f$(\gamma -1)/2\f$ )
	/** @} */

	/** \name Viscosity related parameters
	 */
	//! Artifical viscosity
	//! @{
	/*!
	 * \inpsection{turbulence, artificial_viscosity}
	 * \label{ARTIFICIAL_VISCOSITY_VALUE}
	 * \default{0.3}
	 * TLT_ARIFICIAL_VISCOSITY_VALUE
	 */
	float	artvisccoeff;				///< Artificial viscosity coefficient (one, for all fluids)

	//! Small coefficient used to avoid singularity in artificial viscosity computation
	/*! Note that this is also used in other circumstances where we want avoid a division by
	 * zero that depends on the inter-particle distance.
	 * \todo rename variable, and differentiate between one which is based on slength
	 * and one that is based on its square
	 */
	/*!
	 * \inpsection{turbulence, artificial_viscosity}
	 * \label{EPSILON_ARTVISC_VALUE}
	 * \default{NAN}
	 * TLT_EPSILON_ARTVISC_VALUE
	 */
	float	epsartvisc;

	/** \name Newtonian fluids
	 * @{ */
	/*!
	 * \inpsection{fluid}
	 * \label{FLUID_VISCOSITY}
	 * \default{10e-6}
	 * TLT_FLUID_VISCOSITY
	 */
	std::vector<float>	kinematicvisc;	///< Kinematic viscosity (\f$ \nu \f$). SI units: m²/s
	std::vector<float>	bulkvisc;	///< Bulk viscosity used with ESPANOL_REVENGA
	/** @} */

	/** \name Generalized Newtonian parameters
	 *  Note that we there is a small overlap: when modelling Newtonian fluids with a more general model,
	 *  the consistency index corresponds to the dynamic viscous viscosity (\f$ \mu \f$, SI units Pa s), that
	 *  the user could also set via set_kinematic_visc.
	 *  Consistency between the two indices is preserved in the wrapper functions.
	 * @{ */
	std::vector<float>	yield_strength;	///< Yield strength for Bingham, Herschel–Bulkley and Zhu rheological models. SI units: Pa s

	//! Viscous non-linearity
	/*! This is either the exponent n for the power-law and Herschel–Bulkley model (pure number):
	 * or the exponential coefficient of De Kee and Turcotte (SI units: s).
	 *
	 * With Herschel–Bulkley, the effective viscosity is computed as:
	 *
	 * k \dot\gamma^(n-1) + τ_0/\dot\gamma
	 *
	 * for De Kee and Turcotte as:
	 *
	 * mu exp(-t1 \dot\gamma) + τ_0/\dot\gamma
	 *
	 * n = 1 or t1 = 0 give the Newtonian rheology.
	 * n < 1 or t1 > 0 give shear-thinning behavior (pseudo-plastic).
	 * n > 1 or t1 < 0 give shear-thickening behavior (dilatant).
	 */
	std::vector<float>	visc_nonlinear_param;	///< Exponent for power-law and Herschel–Bulkley rheology

	//! Viscosity consistency index
	/*! This is the dynamic viscosity in the Newtonian case, the consistency index for power-law and Herschel–Bulkley,
	 * and the exponential coefficient for the De Kee & Turcotte, and Zhu models.
	 * SI units: Pa s^n in the power-law and HB cases, Pa s in all other cases
	 */
	std::vector<float>	visc_consistency;

	//! Papanastasiou regularization parameter
	/*! The Bingham model is regularized following the Papanastasiou model, with an effective viscosity computed as
	 *
	 * mu + τ_0 (1 - exp(-m \dot\gamma))/\dot\gamma
	 *
	 * instead of
	 *
	 * mu + τ_0/\dot\gamma
	 *
	 * so that the limit for \dot\gamma \to 0 is mu + τ_0 m instead of producing an infinite viscosity.
	 *
	 * The same regularization for the yield strength component can be applied to the Herschel–Bulkley model,
	 * as proposed by Alexandrou et al 2001, that computes the effective viscosity as:
	 *
	 * k \dot\gamma^(n-1) + τ_0 (1 - exp(-m \dot\gamma))/\dot\gamma
	 *
	 * However, this does not prevent infinite viscosity for shear-thinning materials, which is solved by the Zhu model,
	 * which combines Papanastasiou with De Kee and Turcotte:
	 *
	 * mu exp(-t1 \dot\gamma) + τ_0 (1 - exp(-m \dot\gamma))/\dot\gamma
	 *
	 * \see{visc_nonlinear_param, limiting_kinvisc}
	 */
	std::vector<float>	visc_regularization_param;

	//! Upper limit to the effective kinematic viscosity. SI units: m²/s
	/*! This is used to prevent infinite viscosity for shear rates close to zero
	 * when the model is not regularized.
	 *
	 * \note this is fluid-independent
	 */
	float limiting_kinvisc;
	/** @} */

	std::vector<float>	visccoeff;		///< Viscosity coefficient
	std::vector<float>	visc2coeff;		///< Second viscosity coefficient, for Español & Revenga

	//! Multiplicative coefficient in Monaghan's viscous model
	/*! The Monaghan viscous operator can be described as
	 *
	 * \f[
	 *  K \sum_b \frac{m_b}{\rho_a \rho_b} 2 A(\mu_a, \mu_b) F(r) \frac{r\cdot v}{r\cdor r} r
	 * \f]
	 *
	 * where K is a (usually problem-dependent) constant, r is the particle distance vector, and v the
	 * relative velocity vector. The main difference between Monaghan's and Morris' viscous operators
	 * is therefore that Morris' is directed along the relative velocity, while Monaghan's is directed
	 * along the relative particle position.
	 * The constant K has to be determined experimentally, but K = 8 is a good default for 2D problems,
	 * and K = 10 is a good default for 3D problems (K = 2(d+2))
	 */
	float monaghan_visc_coeff;
	/** @} */

	/** \name Lennard-Jones (LJ) boundary related parameters
	 *  With LJ boundary the boundary particle interact with fluid one only trough a repulsive force \f$ {\bf{f}}({\bf{r}}) \f$
	 *  defined by :
	 *	\f[
	 *		{\bf{f}}({\bf{r}}) =
	 *		\begin{cases}
 	 *		D\left( \left( \frac{r_0 }{||{\bf{r}}||} \right)^{p_1}  - \left( \frac{r_0 }{||{\bf{r}}||} \right)^{p_2 } \right)\frac{{\bf{r}}}{||{\bf{r}}||}&\text{for $||{\bf{r}}|| \le r_0$}\\
 0&\text{if $||{\bf{r}}|| > r_0$}
	*	\end{cases}
	 *	\f]
	 *	where \f$ {\bf{f}} \f$ is the relative position between the boundary particle and the fluid particle, \f$ r_0 \f$
	 *	the influence radius of the repulsive force (typically equal to initial inter-particle distance \f$ \Delta p \f$),
	 *	usually \f$ p_1 = 12 \f$, \f$p_2 = 6\f$ and \f$ D \f$ is a problem dependent parameter.
	 * @{ */
  /*!
   * \defpsubsection{Lennard_Jones_formulation, LENNARD_JONES_PARAMETERS}
   * \inpsection{boundaries}
	 * \values{enable, disable}
	 * \default{disable}
   * TLT_LENNARD_JONES_PARAMETERS
	 */
  /*!
   * \inpsection{Lennard_Jones_formulation, enable}
	 * \label{LJ_R0}
	 * \default{0.}
	 * TLT_LJ_R0
   */
	float	r0;			///< Influence radius of LJ repulsive force, \f$ r_0 \f$

	float	dcoeff;		///< \f$ D \f$
  /*!
   * \inpsection{Lennard_Jones_formulation, enable}
	 * \label{LJ_P1_COEFF}
	 * \default{12.}
	 * TLT_LJ_P1COEFF
   */
	float	p1coeff;	///< \f$ p_1 \f$
  /*!
   * \inpsection{Lennard_Jones_formulation, enable}
	 * \label{LJ_P2_COEFF}
	 * \default{6.}
	 * TLT_LJ_P2COEFF
   */
	float	p2coeff;	///< \f$ p_2 \f$
	/** @} */


	/** \name Geometrical DEM and LJ boundary related parameters
	 *  When boundaries can be built using a set of plane or when simulating flows over a real topography
	 *  we can directly compute a boundary normal repulsive force without using boundary particles.
	 *  With a real topography we directly use the Digital Elevation Model (DEM).
	 *  The parameters needed in those cases are described below.
	 * @{ */
	float	ewres;			///< DEM east-west resolution
	float	nsres;			///< DEM north-south resolution
	float	demdx;			///< Displacement in x direction (in ]0, ewres[) used for normal computation
	float	demdy;			///< Displacement in y direction (in ]0, nsres[) used for normal computation
	float	demdxdy;		///< demdx*demdy
	float	demzmin;		///< Minimum elevation of the terrain
	float	partsurf;		///< Particle surface (used to compute surface friction)
	/** @} */

	/** \name Monaghan-Kajtar (MK) boundary related parameters
	 *  With MK boundary the boundary particle interact with fluid one only trough a repulsive force \f$ {\bf{f}}({\bf{r}}) \f$
	 *  defined by :
	 *	TODO.
	 * @{ */
	float	MK_K;			///< Typically: maximum velocity squared, or gravity times maximum height
	float	MK_d;			///< Typically: distance between boundary particles
	float	MK_beta;		///< Typically: ratio between h and MK_d
	/** @} */

	/** \name XSPH related parameter
	 * @{ */
  /*!
   * \inpsection{xsph, enable}
	 * \label{XSPH_COEFF}
	 * \default{0.5}
	 * TLT_XSPH_COEFF
   */
	float	epsxsph;		///< \f$ \epsilon \f$ coefficient for XSPH correction
	/** @} */

	/** \name Parameters for Sub-Particle Scale model for Large Eddy Simulation (LES)
	 *  As described in Dalrymple & Rogers (2006).
	 * @{ */
	float	smagorinsky_constant; ///< Smagorinsky constant C_s for the SPS turbulence model
	float	isotropic_sps_constant; ///< C_i constant for the isotropic part of the sub-scale stress tensor

	// TODO FIXME These should be accessible only by the GPUSPH internals
	float	smagfactor;		///< Smagorinsky factor. This is equal to (C_s*∆p)^2 where C_s is the Smagorinsky constant
	float	kspsfactor;		///< Isotropic SPS factor. This is equal to (2*C_i/3)*∆p^2
	/** @} */

	/** \name Surface tension related parameter
	 * @{ */
	float	epsinterface;	///< Interface epsilon for Grenier's simplified surface tension model
	/** @} */

	/** \name Free surface detection related parameters
	 * @{ */
	float	cosconeanglefluid;	     ///< Cosine of cone angle for free surface detection (If the neighboring particle is fluid)
	float	cosconeanglenonfluid;	 ///< Cosine of cone angle for free surface detection (If the neighboring particle is non_fluid
	/** @} */

	/** \name Other parameters
	 * @{ */
  /*!
   * \inpsection{physics}
   * \default{0,0,-9.81}
   * \label{GRAVITY}
	 * \mandatory
	 * TLT_GRAVITY
   */
	float3	gravity;		///< Gravity
	/** @} */

	// We have 2 deprecated members, but we don't need
	// to get a warning about them for the constructor, only
	// when the users actually assign to them
IGNORE_WARNINGS(deprecated-declarations)
	PhysParams(RheologyType _rheologytype) :
		rheologytype(_rheologytype),
		artvisccoeff(0.3f),
		epsartvisc(NAN),

		limiting_kinvisc(1.0e3),
		monaghan_visc_coeff(10.0f), /* should be 2*(d+2) with d problem dimension; but currently we only support d=3 */

		r0(NAN),
		dcoeff(NAN),
		p1coeff(12.0f),
		p2coeff(6.0f),

		partsurf(0),

		MK_K(NAN),
		MK_d(NAN),
		MK_beta(2),

		epsxsph(0.5f),

		smagorinsky_constant(0.12),
		isotropic_sps_constant(0.0066),
		smagfactor(NAN),
		kspsfactor(NAN),

		epsinterface(NAN),

		cosconeanglefluid(0.86f),
		cosconeanglenonfluid(0.5f),

		gravity(make_float3(0, 0, -9.81))
	{};

	// ProblemCore and all ProblemAPI specialization
	// (but not their derivatives —luckily, friendship is not inherited)
	// GPUWorker and GPUSPH should be the only ones
	// that access the physparams manipulator methods
	friend class ProblemCore;
	template<int version> friend class ProblemAPI;
	friend class GPUWorker;
	friend class GPUSPH;

	/// Returns the number of fluids in the simulation
	/*! This function return the current number of fluids in the simulation
	 *
	 *	\return number of fluids in the simualtion
	 */
	size_t numFluids() const
	{ return rho0.size(); }

protected:
	/** \name Density - equation of state related methods
	 * @{ */
	/// Add a new fluid of given density in the simulation
	/*! This function add a new fluid of at-rest density \f$ \rho_0 \f$ in the
	 *  simulation and prime the equation of state vectors WITHOUT initializing them:
	 *  i.e. \f$ \gamma \f$, ... ARE NOT SET. A call to add_fluid should then be followed
	 *  by a call to set_equation_of_state.
	 */
	size_t add_fluid(
			float rho	///< [in] at-rest fluid density \f$ \rho_0 \f$
			) {
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
		bulkvisc.push_back(NAN);
		visccoeff.push_back(NAN);
		visc2coeff.push_back(NAN);

		// We do initialize the generalized Newtonian parameters, with the values
		// we would need to reduce back to a Newtonian rheology
		yield_strength.push_back(0.0f);

		/* Newtonian behavior is achieved with a power-law exponent of 1 or an exponential coefficient of 0,
		 * so it depends on the rheology type
		 */
		visc_nonlinear_param.push_back(EXPONENTIAL_RHEOLOGY(rheologytype) ? 0.0f : 1.0f);
		visc_consistency.push_back(NAN); // this must be kept in sync with kinematicvisc

		// Default regularization param: 1000
		visc_regularization_param.push_back(1000.0f);

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

	/// Set the equation of state of a given fluid
	/*! Set the parameters of the equation of state of a given fluid, specifying the adiabatic
	 *  index and speed of sound. A non-finite speed of sound implies that it should be autocomputed
	 *  (currrently supported in XProblem only). \f$ B \f$ is automatically computed as \f$ \frac{\rho_0 c_0^2}{\gamma} \f$.
	 */
	void set_equation_of_state(
			size_t fluid_idx, 	///< [in] fluid number
			float gamma, 		///< [in] \f$ \gamma \f$
			float c0			///< [in] \f$ c_0 \f$
			) {
		if (fluid_idx >= numFluids())
			throw std::out_of_range("trying to set equation of state for a non-existing fluid");
		gammacoeff[fluid_idx] = gamma;
		bcoeff[fluid_idx] = rho0[fluid_idx]*c0*c0/gamma;
		sscoeff[fluid_idx] = c0;
		sspowercoeff[fluid_idx] = (gamma-1)/2;
	}

	/// Set density and equation of state of a given fluid
	/*! \deprecated
	 *  Set the density and the parameters of the equation of state of a given fluid, specifying
	 *  the at-rest density, the adiabatic index and speed of sound.\f$ B \f$ is automatically
	 *  computed as \f$ \frac{\rho_0 c_0^2}{\gamma} \f$.
	 *  The number of fluids is automatically increased if set_density() is called with a fluid number
	 *  equals to actual number of fluids.
	 */
	void set_density(
			uint i, 		///< [in] fluid number
			float rho, 		///< [in] at-rest fluid density \f$ \rho_0 \f$
			float gamma, 	///< [in] \f$ \gamma \f$
			float c0		///< [in] \f$ c_0 \f$
			)
	/*! \cond */
	DEPRECATED_MSG("set_density() is deprecated, use add_fluid() + set_equation_of_state() instead")
	/*! \endcond */
	{
		if (i == rho0.size()) {
			add_fluid(rho);
			set_equation_of_state(i, gamma, c0);
		}
		else if (i < rho0.size()) {
			std::cerr << "changing properties of fluid " << i << std::endl;
			rho0[i] = rho;
			set_equation_of_state(i, gamma, c0);
		} else {
			std::cerr << "setting density for fluid index " << i << " > " << rho0.size() << std::endl;
			throw std::runtime_error("fluid index is growing too fast");
		}
	}
	/** @} */

	/** \name Methods to set the Large Eddy Simulation (LES) turbulent model parameters
	 * @{ */
	//! Set the C_s (Smagorinsky) and C_i (isotropic) constants of the SPS model
	void set_sps_parameters(float C_s, float C_i)
	{
		smagorinsky_constant = C_s;
		isotropic_sps_constant = C_i;
	}
	/** @} */

	/** \name Rheological model checks
	 * @{ */

	//! Check if using a power-law rheology
	bool is_power_law_rheology() const
	{ return POWERLAW_RHEOLOGY(rheologytype); }

	//! Throw if not using a power-law rheology
	void must_be_power_law_rheology() const
	{
		if (!is_power_law_rheology())
			throw std::invalid_argument("The " + std::string(RheologyName[rheologytype]) + " rheological model is not power-law");
	}

	//! Check if using an exponential rheology
	bool is_exponential_rheology() const
	{ return EXPONENTIAL_RHEOLOGY(rheologytype); }

	//! Throw if not using an exponential rheology
	void must_be_exponential_rheology() const
	{
		if (!is_exponential_rheology())
			throw std::invalid_argument("The " + std::string(RheologyName[rheologytype]) + " rheological model is not exponential");
	}

	/** @} */

	/** \name Viscosity related methods
	 * @{ */

	void set_artificial_visc(float artvisc)
	{ artvisccoeff = artvisc; }

	/// Raise the limiting viscosity if necessary to take into account the new settings for the given fluid.
	void update_limiting_kinvisc(int fluid_idx)
	{
		float new_limit = yield_strength.at(fluid_idx)*visc_regularization_param[fluid_idx] + visc_consistency[fluid_idx];
		limiting_kinvisc = fmaxf(limiting_kinvisc, new_limit);
	}

	/// Set the kinematic viscosity of a given fluid
	/*! This function set the kinematic viscosity \f$ \nu \f$ of a given fluid
	 * \todo This should check that fluid is Newtonian
	 */
	void set_kinematic_visc(
			size_t fluid_idx,	///< [in] fluid number
			float nu			///< [in] kinematic viscosity \f$ \nu \f$
			) {
		kinematicvisc.at(fluid_idx) = nu;
		visc_consistency.at(fluid_idx) = nu*rho0[fluid_idx];
		update_limiting_kinvisc(fluid_idx);
	}

	/// Set the dynamic viscosity of a given fluid
	/*! This set the dynamic viscosity \f$ \mu \f$ of a given fluid
	 * \todo This should check that fluid is Newtonian
	*/
	void set_dynamic_visc(
			size_t fluid_idx,	///< [in] fluid number
			float mu			///< [in] dynamic viscosity \f$ \mu \f$
			) {
		kinematicvisc.at(fluid_idx) = mu/rho0[fluid_idx];
		visc_consistency.at(fluid_idx) = mu;
		update_limiting_kinvisc(fluid_idx);
	}

	/// Se the bulk viscosity of a given fluid
	void set_bulk_visc(
			size_t fluid_idx,	///< [in] fluid number
			float zeta			///< [in] bulk viscosity \f$ \zeta \f$
			) {
		bulkvisc.at(fluid_idx) = zeta;
	}

	/// Set the consistency index of a given fluid
	/*! This is equivalent to setting the dynamic viscosity for Newtonian fluids
	 */
	void set_consistency_index(
			size_t fluid_idx,	///< [in] fluid number
			float k			///< [in] viscous consistency index \f$ \mu \f$
			) {
		set_dynamic_visc(fluid_idx, k);
	}

	/// Set the yield strength of a given fluid
	void set_yield_strength(
			size_t fluid_idx,	///< [in] fluid number
			float ys			///< [in] viscous consistency index \f$ y_s \f$
			) {
		yield_strength.at(fluid_idx) = ys;
		update_limiting_kinvisc(fluid_idx);
	}

	/// Set the power law exponent of a given fluid
	void set_visc_power_law(
			size_t fluid_idx,	///< [in] fluid number
			float n			///< [in] viscous power law index \f$ n \f$
			) {
		must_be_power_law_rheology();
		visc_nonlinear_param.at(fluid_idx) = n;
	}

	/// Set the exponential law coefficient of a given fluid
	void set_visc_exponential_coeff(
			size_t fluid_idx,	///< [in] fluid number
			float t1			///< [in] viscous exponential law coefficient \f$ t_1 \f$
			) {
		must_be_exponential_rheology();
		visc_nonlinear_param.at(fluid_idx) = t1;
	}

	/// Set the regularization parameter for the yield strength component of the apparent viscosity
	void set_visc_regularization_param(
			size_t fluid_idx,	///< [in] fluid number
			float m			///< [in] regularization parameter \f$ m \f$
			) {
		visc_regularization_param.at(fluid_idx) = m;
	}

	/// Set the maximum allowed viscosity
	void set_limiting_kinviscosity(
			float max_visc		///< [in] maximum allowed viscosity
		) {
		limiting_kinvisc = max_visc;
	}

	/// Return the kinematic viscosity of a given fluid
	/*! This function returns the kinematic viscosity \f$ \nu \f$ of a given fluid
	 *
	 * \return \f$ \nu \f$
	*/
	float get_kinematic_visc(
			size_t fluid_idx	///< [in] fluid number
			) const {
		return kinematicvisc.at(fluid_idx);
	}

	/// Return the dynamic viscosity of a given fluid
	float get_dynamic_visc(
			size_t fluid_idx	///< [in] fluid number
			) const {
		return visc_consistency.at(fluid_idx);
	}

	/// Return the viscous consistency index of a given fluid
	float get_consistency_index(
			size_t fluid_idx	///< [in] fluid number
			) const {
		return visc_consistency.at(fluid_idx);
	}

	/// Return the yield strength of a given fluid
	float get_yield_strength(
			size_t fluid_idx	///< [in] fluid number
			) const {
		return yield_strength.at(fluid_idx);
	}

	/// Return the viscous power law exponent of a given fluid
	float get_visc_power_law(
			size_t fluid_idx	///< [in] fluid number
			) const {
		must_be_power_law_rheology();
		return visc_nonlinear_param.at(fluid_idx);
	}

	/// Return the viscous exponential law coefficient of a given fluid
	float get_visc_exponential_coeff(
			size_t fluid_idx	///< [in] fluid number
			) const {
		must_be_exponential_rheology();
		return visc_nonlinear_param.at(fluid_idx);
	}

	/// Return the regularization paramter for the yield strength component of the apparent viscosity
	float get_visc_regularization_param(
			size_t fluid_idx	///< [in] fluid number
			) const {
		return visc_regularization_param.at(fluid_idx);
	}

	/** @} */

} PhysParams;

#endif
