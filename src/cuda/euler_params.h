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
	const	float	full_dt;			///< time step (dt)
	const	float	half_dt;			///< half of time step (dt/2)
	const	float	t;				///< simulation time
	const	uint		step;			///< integrator step //parametro template di euler params struttura collettiva

	// Constructor / initializer
	common_euler_params(
				float4		* __restrict__ _newPos,
				float4		* __restrict__ _newVel,
		const	float4		* __restrict__ _oldPos,
		const	hashKey		* __restrict__ _particleHash,
		const	float4		* __restrict__ _oldVel,
		const	particleinfo	* __restrict__ _info,
		const	float4		* __restrict__ _forces,
		const	uint			_numParticles,
		const	float		_full_dt,
		const	float		_half_dt,
		const	float		_t,
		const	uint			_step) :
		newPos(_newPos),
		newVel(_newVel),
		oldPos(_oldPos),
		particleHash(_particleHash),
		oldVel(_oldVel),
		info(_info),
		forces(_forces),
		numParticles(_numParticles),
		full_dt(_full_dt),
		half_dt(_half_dt),
		t(_t),
		step(_step)
	{}
};

/// Additional parameters passed only to kernels with XSPH enabled
struct xsph_euler_params
{
	const	float4	* __restrict__ xsph;
	xsph_euler_params(const	float4 * __restrict__ _xsph) :
		xsph(_xsph)
	{}
};

/// Additional parameters passed only to kernels with SA_BOUNDARY
struct sa_boundary_euler_params
{
			float4	* __restrict__ newEulerVel;
	const	float2	* __restrict__ vertPos0;
	const	float2	* __restrict__ vertPos1;
	const	float2	* __restrict__ vertPos2;
	const	float4	* __restrict__ oldEulerVel;
	const	float	slength;
	const	float	influenceradius;
	const	neibdata	* __restrict__ neibsList;
	const	uint	* __restrict__ cellStart;

	// Constructor / initializer
	sa_boundary_euler_params(
				float4	* __restrict__ _newEulerVel,
		const	float2	* __restrict__  const _vertPos[],
		const	float4	* __restrict__ _oldEulerVel,
		const	float	_slength,
		const	float	_influenceradius,
		const	neibdata	* __restrict__ _neibsList,
		const	uint	* __restrict__ _cellStart) :
		newEulerVel(_newEulerVel),
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2]),
		oldEulerVel(_oldEulerVel),
		slength(_slength),
		influenceradius(_influenceradius),
		neibsList(_neibsList),
		cellStart(_cellStart)
	{}
};

/// Additional parameters passed only to kernels with SA_BOUNDARY and moving objects
struct sa_boundary_moving_euler_params
{
			float4	* __restrict__ newBoundElement;
	const	float4	* __restrict__ oldBoundElement;

	sa_boundary_moving_euler_params(
				float4	* __restrict__ _newBoundElement,
		const	float4	* __restrict__ _oldBoundElement)
	:
		newBoundElement(_newBoundElement),
		oldBoundElement(_oldBoundElement)
	{}
};


/// Additional parameters passed only to kernels with KEPSVISC
struct kepsvisc_euler_params
{
	float			* __restrict__ newTKE;	///< updated values of k, for k-e model (out)
	float			* __restrict__ newEps;	///< updated values of e, for k-e model (out)
	float			* __restrict__ newTurbVisc; ///< updated value of the eddy viscosity (out)
	const	float	* __restrict__ oldTKE;		///< previous values of k, for k-e model (in)
	const	float	* __restrict__ oldEps;		///< previous values of e, for k-e model
	const	float3	* __restrict__ keps_dkde;	///< derivative of ??? (in)

	// Constructor / initializer
	kepsvisc_euler_params(
			float		* __restrict__ _newTKE,
			float		* __restrict__ _newEps,
			float		* __restrict__ _newTurbVisc,
			const float	* __restrict__ _oldTKE,
			const float	* __restrict__ _oldEps,
			const float3	* __restrict__ _keps_dkde):
			newTKE(_newTKE),
			newEps(_newEps),
			newTurbVisc(_newTurbVisc),
			oldTKE(_oldTKE),
			oldEps(_oldEps),
			keps_dkde(_keps_dkde)
	{}
};


/// Additional parameters passed only to kernels with SPH_GRENIER formulation
struct grenier_euler_params
{
			float4	* __restrict__ newVol;			///< updated particle's voume (out)
	const	float4	* __restrict__ oldVol;			///< previous particle's volume (in)

	// Constructor / initializer
	grenier_euler_params(
				float4 * __restrict__ _newVol,
			const float4 * __restrict__ _oldVol) :
			newVol(_newVol),
			oldVol(_oldVol)
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
				float * __restrict__ _newEnergy,
		const	float * __restrict__ _oldEnergy,
		const	float * __restrict__ _DEDt) :
			newEnergy(_newEnergy),
			oldEnergy(_oldEnergy),
			DEDt(_DEDt)
	{}
};

/// The actual euler_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	BoundaryType _boundarytype,
	ViscosityType _visctype,
	TurbulenceModel _turbmodel,
	flag_t _simflags>
struct euler_params :
	common_euler_params,
	COND_STRUCT(_simflags & ENABLE_XSPH, xsph_euler_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_boundary_euler_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY && (_simflags & ENABLE_MOVING_BODIES), sa_boundary_moving_euler_params),
	COND_STRUCT(_turbmodel == KEPSVISC, kepsvisc_euler_params),
	COND_STRUCT(_sph_formulation == SPH_GRENIER, grenier_euler_params),
	COND_STRUCT(_simflags & ENABLE_INTERNAL_ENERGY, energy_euler_params)
{
	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr SPHFormulation sph_formulation = _sph_formulation;
	static constexpr BoundaryType boundarytype = _boundarytype;
	static constexpr ViscosityType visctype = _visctype;
	static constexpr TurbulenceModel turbmodel = _turbmodel;
	static constexpr flag_t simflags = _simflags;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the euler kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	euler_params(
		// common
				float4		* __restrict__ _newPos,
				float4		* __restrict__ _newVel,
		const	float4		* __restrict__ _oldPos,
		const	hashKey		* __restrict__ _particleHash,
		const	float4		* __restrict__ _oldVel,
		const	particleinfo	* __restrict__ _info,
		const	float4		* __restrict__ _forces,
		const	uint			_numParticles,
		const	float		_full_dt,
		const	float		_half_dt,
		const	float		_t,
		const	uint			_step,

		// XSPH
		const	float4	* __restrict__ _xsph,

		// SA_BOUNDARY
				float4	* __restrict__ _newEulerVel,
				float4	* __restrict__ _newBoundElement,
		const	float2	* __restrict__  const _vertPos[],
		const	float4	* __restrict__ _oldEulerVel,
		const	float4	* __restrict__ _oldBoundElement,
		const	float	_slength,
		const	float	_influenceradius,
		const	neibdata	* __restrict__ _neibsList,
		const	uint	* __restrict__ _cellStart,

		// KEPSVISC
				float	* __restrict__ _newTKE,
				float	* __restrict__ _newEps,
				float	* __restrict__ _newTurbVisc,
		const	float	* __restrict__ _oldTKE,
		const	float	* __restrict__ _oldEps,
		const	float3	* __restrict__ _keps_dkde,

		// SPH_GRENIER
				float4	* __restrict__ _newVol,
		const	float4	* __restrict__ _oldVol,

		// ENABLE_INTERNAL_ENERGY
				float	* __restrict__ _newEnergy,
		const	float	* __restrict__ _oldEnergy,
		const	float	* __restrict__ _DEDt) :

		common_euler_params(_newPos, _newVel, _oldPos, _particleHash,
			_oldVel, _info, _forces, _numParticles, _full_dt, _half_dt, _t, _step),
		COND_STRUCT(simflags & ENABLE_XSPH, xsph_euler_params)(_xsph),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_euler_params)
			(_newEulerVel, _vertPos, _oldEulerVel, _slength, _influenceradius, _neibsList, _cellStart),
		COND_STRUCT(_boundarytype == SA_BOUNDARY && (_simflags & ENABLE_MOVING_BODIES), sa_boundary_moving_euler_params)
			(_newBoundElement, _oldBoundElement),
		COND_STRUCT(turbmodel == KEPSVISC, kepsvisc_euler_params)(_newTKE, _newEps, _newTurbVisc, _oldTKE, _oldEps, _keps_dkde),
		COND_STRUCT(sph_formulation == SPH_GRENIER, grenier_euler_params)(_newVol, _oldVol),
		COND_STRUCT(simflags & ENABLE_INTERNAL_ENERGY, energy_euler_params)(_newEnergy, _oldEnergy, _DEDt)

	{}
};

#endif // _EULER_PARAMS_H

