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

// We now have the tools to assemble the structure that will be used to pass parameters to the euler kernel

/* Now we define structures that hold the parameters to be passed
   to the euler kernel. These are defined in chunks, and then ‘merged’
   into the template structure that is actually used as argument to the kernel.
   Each struct must define an appropriate constructor / initializer for its const
   members
*/

/// Parameters common to all forces kernel specializations
struct common_euler_params
{
			float4	* __restrict__ newPos;		///< updated particle's position (out)
			float4	* __restrict__ newVel;			///< updated particle's velocity (out)
	const	float4	* __restrict__ oldPos;			///< previous particle's position (in)
	const	hashKey	* __restrict__ particleHash;	///< particle's hash (in)
	const	float4	* __restrict__ oldVel;			///< previous particle's velocity (in/out)
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
		newPos(bufwrite.getData<BUFFER_POS>()),
		newVel(bufwrite.getData<BUFFER_VEL>()),
		oldPos(bufread.getData<BUFFER_POS>()),
		particleHash(bufread.getData<BUFFER_HASH>()),
		oldVel(bufread.getData<BUFFER_VEL>()),
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

/// Additional parameters passed only to kernels with SA_BOUNDARY
struct euler_vel_euler_params
{
			float4	* __restrict__ newEulerVel;
	const	float4	* __restrict__ oldEulerVel;

	// Constructor / initializer
	euler_vel_euler_params(
		BufferList const&	bufread,
		BufferList&			bufwrite)
	:
		newEulerVel(bufwrite.getData<BUFFER_EULERVEL>()),
		oldEulerVel(bufread.getData<BUFFER_EULERVEL>())
	{}
};

/// Additional parameters passed only to kernels with SA_BOUNDARY and moving objects
struct sa_boundary_moving_euler_params
{
			float4	* __restrict__ newBoundElement;
	const	float4	* __restrict__ oldBoundElement;

	sa_boundary_moving_euler_params(
		BufferList const&	bufread,
		BufferList&			bufwrite)
	:
		newBoundElement(bufwrite.getData<BUFFER_BOUNDELEMENTS>()),
		oldBoundElement(bufread.getData<BUFFER_BOUNDELEMENTS>())
	{}
};


/// Additional parameters passed only to kernels with KEPSILON
struct keps_euler_params
{
	float			* __restrict__ newTKE;	///< updated values of k, for k-e model (out)
	float			* __restrict__ newEps;	///< updated values of e, for k-e model (out)
	float			* __restrict__ newTurbVisc; ///< updated value of the eddy viscosity (out)
	const	float	* __restrict__ oldTKE;		///< previous values of k, for k-e model (in)
	const	float	* __restrict__ oldEps;		///< previous values of e, for k-e model
	const	float3	* __restrict__ keps_dkde;	///< derivative of ??? (in)

	// Constructor / initializer
	keps_euler_params(
		BufferList const&	bufread,
		BufferList&			bufwrite)
	:
		newTKE(bufwrite.getData<BUFFER_TKE>()),
		newEps(bufwrite.getData<BUFFER_EPSILON>()),
		newTurbVisc(bufwrite.getData<BUFFER_TURBVISC>()),
		oldTKE(bufread.getData<BUFFER_TKE>()),
		oldEps(bufread.getData<BUFFER_EPSILON>()),
		keps_dkde(bufread.getData<BUFFER_DKDE>())
	{}
};


/// Additional parameters passed only to kernels with SPH_GRENIER formulation
struct grenier_euler_params
{
			float4	* __restrict__ newVol;			///< updated particle's voume (out)
	const	float4	* __restrict__ oldVol;			///< previous particle's volume (in)

	// Constructor / initializer
	grenier_euler_params(
		BufferList const&	bufread,
		BufferList&			bufwrite)
	:
		newVol(bufwrite.getData<BUFFER_VOLUME>()),
		oldVol(bufread.getData<BUFFER_VOLUME>())
	{}
};

/// Additional parameters passed only to kernels with ENABLE_INTERNAL_ENERGY
struct energy_euler_params
{
			float	* __restrict__ newEnergy;			///< updated particle's internal energy (out)
	const	float	* __restrict__ oldEnergy;			///< previous particle's internal energy (in)
	const	float	* __restrict__ DEDt;				///< internal energy derivative with respect to time (in)

	// Constructor / initializer
	energy_euler_params(
		BufferList const&	bufread,
		BufferList&			bufwrite)
	:
		newEnergy(bufwrite.getData<BUFFER_INTERNAL_ENERGY>()),
		oldEnergy(bufread.getData<BUFFER_INTERNAL_ENERGY>()),
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
	bool _has_keps = _ViscSpec::turbmodel == KEPSILON,
	bool _has_eulerVel =
		_has_keps || (_boundarytype == SA_BOUNDARY && (_simflags & ENABLE_INLET_OUTLET))
	>
struct euler_params :
	common_euler_params,
	COND_STRUCT(_simflags & ENABLE_XSPH, xsph_euler_params),
	COND_STRUCT(_has_eulerVel, euler_vel_euler_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY && (_simflags & ENABLE_MOVING_BODIES),
		sa_boundary_moving_euler_params),
	COND_STRUCT(_has_keps, keps_euler_params),
	COND_STRUCT(_sph_formulation == SPH_GRENIER, grenier_euler_params),
	COND_STRUCT(_simflags & ENABLE_INTERNAL_ENERGY, energy_euler_params)
{
	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr SPHFormulation sph_formulation = _sph_formulation;
	static constexpr BoundaryType boundarytype = _boundarytype;
	using ViscSpec = _ViscSpec;
	static constexpr flag_t simflags = _simflags;
	static constexpr int step = _step;
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
		COND_STRUCT(simflags & ENABLE_XSPH, xsph_euler_params)(bufread),
		COND_STRUCT(has_eulerVel, euler_vel_euler_params)(bufread, bufwrite),
		COND_STRUCT(_boundarytype == SA_BOUNDARY && (_simflags & ENABLE_MOVING_BODIES), sa_boundary_moving_euler_params)
			(bufread, bufwrite),
		COND_STRUCT(has_keps, keps_euler_params)(bufread, bufwrite),
		COND_STRUCT(sph_formulation == SPH_GRENIER, grenier_euler_params)(bufread, bufwrite),
		COND_STRUCT(simflags & ENABLE_INTERNAL_ENERGY, energy_euler_params)(bufread, bufwrite)
	{}
};

#endif // _EULER_PARAMS_H

