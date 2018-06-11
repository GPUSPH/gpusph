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

/// Structure holding all physical parameters
/*! This structure holds all the physical parameters needed by the simulation
 *  along with some basic initialization functions.
 *
 *	\ingroup datastructs
 */
typedef struct PhysParams {
	/** \name Equation of state related parameters
	 *  The relation between pressure and density is given for Cole's equation of state
	 *	\f[
	 *		P(\rho) = B\left(\left(\frac{\rho}{\rho_0}\right)^\gamma - 1\right)
	 *	\f]
	 *	where \f$\rho_0\f$ is the at-rest density,
	 *	\f$\gamma\f$ is the adiabatic index and
	 *	\f$B = \rho_0 c_0^2/\gamma\f$ for a given
	 *	at-rest sound speed \f$c_0\f$.
	 *	The sound speed for a given density \f$\rho\f$ can thus be computed as
	 *	\f[
	 *		c(\rho) = c_0 \left(\frac{\rho}{\rho_0}\right)^{(\gamma -1)/2},
	 *	\f]
	 *	Obviously all of the coefficients below are fluid dependent and are then stored in an
	 *	STL vector.
	 *  @{ */
	std::vector<float> rho0; 			///< At-rest density, \f$\rho_0\f$
	std::vector<float> bcoeff; 			///< Pressure parameter, \f$ B \f$
	std::vector<float> gammacoeff;		///< Adiabatic index, \f$\gamma\f$
	std::vector<float> sscoeff; 		///< Sound speed coefficient ( = sound speed at rest density, \f$ c_0 \f$)
	std::vector<float> sspowercoeff; 	///< Sound speed equation exponent ( = \f$(\gamma -1)/2\f$ )
	/** @} */

	/** \name Viscosity related parameters
	 *  Viscosity coefficient used in the viscous contribution functions, depends on
	 *  viscosity model:
	 *		- for ARTVSIC: artificial viscosity coefficient
	 *  	- for KINEMATICVISC: 4xkinematic viscosity,
	 *   	- for DYNAMICVISC: kinematic viscosity
	 *
	 *  (The choice might seem paradoxical, but with DYNAMICVISC the dynamic viscosity
	 *  coefficient is obtained multiplying visccoeff by the particle density, while
	 *  with the KINEMATICVISC model the kinematic viscosity is used directly, in a
	 *  formula what also includes a harmonic average from which the factor 4 emerges).
	 *
	 *  Obviously the fluid dependent coefficients below are stored in an STL vector.
	 * @{ */
	float	artvisccoeff;				///< Artificial viscosity coefficient (one, for all fluids)
	float	epsartvisc;					///< Small coefficient used to avoid singularity in artificial viscosity computation
	std::vector<float>	kinematicvisc;	///< Kinematic viscosity (\f$ \nu \f$)
	std::vector<float>	visccoeff;		///< Viscosity coefficient
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
	float	r0;			///< Influence radius of LJ repulsive force, \f$ r_0 \f$
	float	dcoeff;		///< \f$ D \f$
	float	p1coeff;	///< \f$ p_1 \f$
	float	p2coeff;	///< \f$ p_2 \f$
	/** @} */


	/** \name Geometrical LJ boundary related parameters
	 *  When boundaries can be built using a set of plane or when simulating flows over a real topography
	 *  we can directly compute a boundary normal repulsive force without using boundary particles.
	 *  With a real topography we directly use the Digital Elevation Model (DEM).
	 *  The parameters needed in those case are describe below.
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
	float	epsxsph;		///< \f$ \epsilon \f$ coefficient for XSPH correction
	/** @} */

	/** \name Large Eddy Simulation (LES) related parameters
	 *  The implemented Smagorinsky LES model depends on two parameters
	 * @{ */
	float	smagfactor;		///< TDOD
	float	kspsfactor;		///< TDOD
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
	float3	gravity;		///< Gravity
	/** @} */

	/** \name Deprecated parameters
	 * @{ */
	float	objectobjectdf DEPRECATED_MSG("objectobjectdf is not needed anymore");	///< Damping factor for object-object interaction
	float	objectboundarydf DEPRECATED_MSG("objectboundarydf is not needed anymore");	///< Damping factor for object-boundary interaction
	/** @} */

	// We have 2 deprecated members, but we don't need
	// to get a warning about them for the constructor, only
	// when the users actually assign to them
IGNORE_WARNINGS(deprecated-declarations)
	PhysParams(void) :
		artvisccoeff(0.3f),
		epsartvisc(NAN),

		r0(NAN),
		p1coeff(12.0f),
		p2coeff(6.0f),

		partsurf(0),

		epsxsph(0.5f),

		smagfactor(NAN),
		kspsfactor(NAN),

		epsinterface(NAN),

		cosconeanglefluid(0.86f),
		cosconeanglenonfluid(0.5f),

		gravity(make_float3(0, 0, -9.81)),

		objectobjectdf(1.0f),
		objectboundarydf(1.0f)
	{};

	// Problem, XProblem (but not their derivatives —luckily, friendship is not inherited)
	// GPUWorker and GPUSPH should be the only ones
	// that access the physparams manipulator methods
	friend class Problem;
	friend class XProblem;
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

	/** \name Viscosity related methods
	 * @{ */
	/// Set the kinematic viscosity of a given fluid
	/*! This function set the kinematic viscosity \f$ \nu \f$ of a given fluid
	 */
	void set_kinematic_visc(
			size_t fluid_idx,	///< [in] fluid number
			float nu			///< [in] kinematic viscosity \f$ \nu \f$
			) {
		kinematicvisc.at(fluid_idx) = nu;
	}

	/// Set the dynamic viscosity of a given fluid
	/*! This set the dynamic viscosity \f$ \mu \f$ of a given fluid
	*/
	void set_dynamic_visc(
			size_t fluid_idx,	///< [in] fluid number
			float mu			///< [in] dynamic viscosity viscosity \f$ \mu \f$
			) {
		set_kinematic_visc(fluid_idx, mu/rho0[fluid_idx]);
	}

	/// Return the kinematic viscosity of a given fluid
	/*! This function return kinematic viscosity \f$ \nu \f$ of a given fluid
	 *
	 * \return \f$ \nu \f$
	*/
	float get_kinematic_visc(
			size_t fluid_idx	///< [in] fluid number
			) const {
		return kinematicvisc.at(fluid_idx);
	}
	/** @} */

} PhysParams;

#endif
