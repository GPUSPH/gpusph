/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _EULER_PARAMS_H
#define _EULER_PARAMS_H

#include "particledefine.h"
#include "simflags.h"

/* To achieve good performance, each combination of kernel, boundary, formulation
   etc is a different specialization, something which in itself makes for a huge
   number of kernels.
   To make things more complicated, some incarnations require a different set
   of parameters compared to other, and instantiate different sets of variables.
   All of this should be managed as automagically as possible while trying to
   rely only on the most readable features of C++ and, if possible, using the
   preprocessor as little as possible.

   To this end, we set up a mechanism that allows us to build structure
   templates in which the number and quality of members depends on the
   specialization.
*/

#include "cond_params.h"

#include "common_params.h"

// We now have the tools to assemble the structure that will be used to pass parameters to the euler kernel

/* Now we define structures that hold the parameters to be passed
   to the euler kernel. These are defined in chunks, and then ‘merged’
   into the template structure that is actually used as argument to the kernel.
   Each struct must define an appropriate constructor / initializer for its const
   members
*/

/// Parameters common to all forces kernel specializations
struct common_euler_params :
	Pos_params<>,
	Vel_params<>
{
	const	hashKey	* __restrict__ particleHash;	///< particle's hash (in)
	const	particleinfo	* __restrict__ info;		///< particle's information
	const	float4	* __restrict__ forces;			///< derivative of particle's velocity and density (in)
	const	uint	numParticles;			///< total number of particles
	const	float	dt;			///< time step (dt or dt/2, depending on integrator step)
	const	float	t;				///< simulation time

	// Constructor / initializer
	common_euler_params(
		BufferList const&	bufread,
		BufferList&			bufwrite,
		const	uint		_numParticles,
		const	float		_dt,
		const	float		_t)
	:
		Pos_params<>(bufread, bufwrite),
		Vel_params<>(bufread, bufwrite),
		particleHash(bufread.getData<BUFFER_HASH>()),
		info(bufread.getData<BUFFER_INFO>()),
		forces(bufread.getData<BUFFER_FORCES>()),
		numParticles(_numParticles),
		dt(_dt),
		t(_t)
	{}
};

/// Additional parameters passed only to kernels with XSPH enabled
struct xsph_euler_params
{
	const	float4	* __restrict__ xsph;
	xsph_euler_params(BufferList const& bufread) :
		xsph(bufread.getData<BUFFER_XSPH>())
	{}
};

/// Additional parameters passed only to kernels with KEPSILON
struct keps_euler_params :
	TKE_params<>, ///< old and new k, for k-e model
	Eps_params<>  ///< old and new e, for k-emodel
{
	float			* __restrict__ newTurbVisc; ///< updated value of the eddy viscosity (out)
	const	float3	* __restrict__ keps_dkde;	///< derivative of ??? (in)

	// Constructor / initializer
	keps_euler_params(
		BufferList const&	bufread,
		BufferList&			bufwrite)
	:
		TKE_params<>(bufread, bufwrite),
		Eps_params<>(bufread, bufwrite),
		newTurbVisc(bufwrite.getData<BUFFER_TURBVISC>()),
		keps_dkde(bufread.getData<BUFFER_DKDE>())
	{}
};

/// Additional parameters passed only to kernels with ENABLE_INTERNAL_ENERGY
struct energy_euler_params : Energy_params<>
{
	const	float	* __restrict__ DEDt;				///< internal energy derivative with respect to time (in)

	// Constructor / initializer
	energy_euler_params(
		BufferList const&	bufread,
		BufferList&			bufwrite)
	:
		Energy_params<>(bufread, bufwrite),
		DEDt(bufread.getData<BUFFER_INTERNAL_ENERGY_UPD>())
	{}
};

/// The actual euler_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	BoundaryType _boundarytype,
	typename _ViscSpec,
	flag_t _simflags,
	int _step,
	RunMode _run_mode = SIMULATE,
	bool _repacking = (_run_mode == REPACK),
	bool _has_keps = _ViscSpec::turbmodel == KEPSILON && !_repacking,
	bool _has_eulerVel =
		(_has_keps || (_boundarytype == SA_BOUNDARY && (_simflags & ENABLE_INLET_OUTLET)))
		&& !_repacking,
	typename eulerVel_params = typename
		COND_STRUCT(_has_eulerVel, EulerVel_params<>),
	typename sa_boundary_moving_params = typename
		COND_STRUCT(_boundarytype == SA_BOUNDARY && (_simflags & ENABLE_MOVING_BODIES) &&
				!_repacking, BoundElement_params<>),
	typename grenier_params = typename
		COND_STRUCT(_sph_formulation == SPH_GRENIER && !_repacking, Vol_params<>)
	>
struct euler_params :
	common_euler_params,
	COND_STRUCT(_simflags & ENABLE_XSPH && !_repacking, xsph_euler_params),
	eulerVel_params,
	sa_boundary_moving_params,
	COND_STRUCT(_has_keps, keps_euler_params),
	grenier_params,
	COND_STRUCT(_simflags & ENABLE_INTERNAL_ENERGY, energy_euler_params)
{
	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr SPHFormulation sph_formulation = _sph_formulation;
	static constexpr BoundaryType boundarytype = _boundarytype;
	using ViscSpec = _ViscSpec;
	static constexpr flag_t simflags = _simflags;
	static constexpr int step = _step;
	static const RunMode run_mode = _run_mode;
	static const bool repacking = _repacking;
	static constexpr bool has_keps = _has_keps;
	static constexpr bool has_eulerVel = _has_eulerVel;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the euler kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	euler_params(
		BufferList const&	bufread,
		BufferList&			bufwrite,
		const	uint		_numParticles,
		const	float		_dt,
		const	float		_t)
	:
		common_euler_params(bufread, bufwrite, _numParticles, _dt, _t),
		COND_STRUCT((simflags & ENABLE_XSPH) && !_repacking, xsph_euler_params)(bufread),
		eulerVel_params(bufread, bufwrite),
		sa_boundary_moving_params(bufread, bufwrite),
		COND_STRUCT(has_keps, keps_euler_params)(bufread, bufwrite),
		grenier_params(bufread, bufwrite),
		COND_STRUCT(simflags & ENABLE_INTERNAL_ENERGY, energy_euler_params)(bufread, bufwrite)
	{}
};

template<KernelType _kerneltype,
	BoundaryType _boundarytype,
	flag_t _simflags,
	int _step>
using euler_repack_params = euler_params<_kerneltype, SPH_F1,
	  _boundarytype, repackViscSpec<_simflags>, _simflags, _step, REPACK>;

#endif // _EULER_PARAMS_H

