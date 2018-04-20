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

#ifndef _DENSITY_SUM_PARAMS_H
#define _DENSITY_SUM_PARAMS_H

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

// We now have the tools to assemble the structure that will be used to pass parameters to the density_sum kernel

/* Now we define structures that hold the parameters to be passed
   to the density_sum kernel. These are defined in chunks, and then ‘merged’
   into the template structure that is actually used as argument to the kernel.
   Each struct must define an appropriate constructor / initializer for its const
   members
*/

/// Parameters common to all density_sum kernel specializations
struct common_density_sum_params
{
	const	float4	*oldPos;			///< previous particle's position (in)
	const	float4	*newPos;			///< updated particle's position (in)
	const	float4	*oldVel;			///< previous particle's velocity (in/out)
			float4	*newVel;			///< updated particle's velocity (out)
			float4	*oldgGam;
			float4	*newgGam;
	const	float4	*oldEulerVel;
			float4	*newEulerVel;
	const	hashKey	*particleHash;		///< particle's hash (in)
	const	particleinfo	*info;		///< particle's information
			float4	*forces;			///< derivative of particle's velocity and density (in/out)
	const	uint	numParticles;		///< total number of particles
	const	float	full_dt;			///< time step (dt)
	const	float	half_dt;			///< half of time step (dt/2)
	const	float	t;					///< simulation time
	const	uint	step;			///< integrator step //parametro template di euler params struttura collettiva
	const	float	slength;
	const	float	influenceradius;
	const	neibdata	*neibsList;
	const	uint	*cellStart;

	// Constructor / initializer
	common_density_sum_params(
		const	float4		*_oldPos,
		const	float4		*_newPos,
		const	float4		*_oldVel,
				float4		*_newVel,
				float4		*_oldgGam,
				float4		*_newgGam,
		const	float4		*_oldEulerVel,
				float4		*_newEulerVel,
		const	hashKey		*_particleHash,
		const	particleinfo	*_info,
				float4		*_forces,
		const	uint		_numParticles,
		const	float		_full_dt,
		const	float		_half_dt,
		const	float		_t,
		const	uint		_step,
		const	float		_slength,
		const	float		_influenceradius,
		const	neibdata	*_neibsList,
		const	uint		*_cellStart) :
		oldPos(_oldPos),
		newPos(_newPos),
		oldVel(_oldVel),
		newVel(_newVel),
		oldgGam(_oldgGam),
		newgGam(_newgGam),
		oldEulerVel(_oldEulerVel),
		newEulerVel(_newEulerVel),
		particleHash(_particleHash),
		info(_info),
		forces(_forces),
		numParticles(_numParticles),
		full_dt(_full_dt),
		half_dt(_half_dt),
		t(_t),
		step(_step),
		slength(_slength),
		influenceradius(_influenceradius),
		neibsList(_neibsList),
		cellStart(_cellStart)
	{}
};

/// Additional parameters passed only to the kernel with BOUNDARY neighbors
struct boundary_density_sum_params
{
			float4	* __restrict__ newBoundElement;
	const	float2	* __restrict__ vertPos0;
	const	float2	* __restrict__ vertPos1;
	const	float2	* __restrict__ vertPos2;

	// Constructor / initializer
	boundary_density_sum_params(
				float4	*_newBoundElement,
		const	float2	* const _vertPos[]) :
		newBoundElement(_newBoundElement),
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2])
	{}
};

/// The actual density_sum_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	ParticleType _ntype,
	flag_t _simflags>
struct density_sum_params :
	common_density_sum_params,
	COND_STRUCT(_ntype == PT_BOUNDARY, boundary_density_sum_params)
{
	static const KernelType kerneltype = _kerneltype;
	static const ParticleType ntype = _ntype;
	static const flag_t simflags = _simflags;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the density_sum kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	density_sum_params(
		// common
		const	float4		*_oldPos,
		const	float4		*_newPos,
		const	float4		*_oldVel,
				float4		*_newVel,
				float4		*_oldgGam,
				float4		*_newgGam,
		const	float4		*_oldEulerVel,
				float4		*_newEulerVel,
		const	hashKey		*_particleHash,
		const	particleinfo	*_info,
				float4		*_forces,
		const	uint		_numParticles,
		const	float		_full_dt,
		const	float		_half_dt,
		const	float		_t,
		const	uint		_step,
		const	float		_slength,
		const	float		_influenceradius,
		const	neibdata	*_neibsList,
		const	uint		*_cellStart,

		// SA_BOUNDARY
				float4*		_newBoundElement,
		const	float2*		const _vertPos[]) :

		common_density_sum_params(_oldPos, _newPos, _oldVel, _newVel, _oldgGam, _newgGam, _oldEulerVel, _newEulerVel,
			_particleHash, _info, _forces, _numParticles, _full_dt, _half_dt, _t, _step, _slength, _influenceradius, _neibsList, _cellStart),
		COND_STRUCT(_ntype == PT_BOUNDARY, boundary_density_sum_params)
			(_newBoundElement, _vertPos)
	{}
};

/// Common params for integrateGammaDevice
struct common_integrate_gamma_params
{
	const	float4	* __restrict__ oldPos; ///< positions at step n
	const	float4	* __restrict__ newPos; ///< positions at step n+1
	const	float4	* __restrict__ oldVel; ///< velocities at step n
	const	float4	* __restrict__ newVel; ///< velocities at step n+1
	const	particleinfo * __restrict__ info; ///< particle info
	const	hashKey	* __restrict__ particleHash; ///< particle hash
	const	float4	* __restrict__ oldgGam; ///< previous gamma and its gradient
			float4	* __restrict__ newgGam; ///< [out] new gamma and its gradient
	const	float4	* __restrict__ oldBoundElement; ///< boundary elements at step n
	const	float4	* __restrict__ newBoundElement; ///< boundary elements at step n+1
	const	float2	* __restrict__ vertPos0;
	const	float2	* __restrict__ vertPos1;
	const	float2	* __restrict__ vertPos2;
	const	neibdata *__restrict__ neibsList;
	const	uint	* __restrict__ cellStart;
	const	uint	particleRangeEnd; ///< max number of particles
	const	float	full_dt; ///< time step (dt)
	const	float	half_dt; ///< half of time step (dt/2)
	const	float	t; ///< simulation time
	const	uint	step; ///< integrator step
	const	float	slength;
	const	float	influenceradius;

	common_integrate_gamma_params(
		const	float4	* __restrict__ _oldPos, ///< positions at step n
		const	float4	* __restrict__ _newPos, ///< positions at step n+1
		const	float4	* __restrict__ _oldVel, ///< velocities at step n
		const	float4	* __restrict__ _newVel, ///< velocities at step n+1
		const	particleinfo * __restrict__ _info, ///< particle info
		const	hashKey	* __restrict__ _particleHash, ///< particle hash
		const	float4	* __restrict__ _oldgGam, ///< previous gamma and its gradient
				float4	* __restrict__ _newgGam, ///< [out] new gamma and its gradient
		const	float4	* __restrict__ _oldBoundElement, ///< boundary elements at step n
		const	float4	* __restrict__ _newBoundElement, ///< boundary elements at step n+1
		const	float2	* const _vertPos[],
		const	neibdata *__restrict__ _neibsList,
		const	uint	* __restrict__ _cellStart,
		const	uint	_particleRangeEnd, ///< max number of particles
		const	float	_full_dt, ///< time step (dt)
		const	float	_half_dt, ///< half of time step (dt/2)
		const	float	_t, ///< simulation time
		const	uint	_step, ///< integrator step
		const	float	_slength,
		const	float	_influenceradius)
	:
		oldPos(_oldPos),
		newPos(_newPos),
		oldVel(_oldVel),
		newVel(_newVel),
		info(_info),
		particleHash(_particleHash),
		oldgGam(_oldgGam),
		newgGam(_newgGam),
		oldBoundElement(_oldBoundElement),
		newBoundElement(_newBoundElement),
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2]),
		neibsList(_neibsList),
		cellStart(_cellStart),
		particleRangeEnd(_particleRangeEnd),
		full_dt(_full_dt), half_dt(_half_dt), t(_t), step(_step),
		slength(_slength), influenceradius(_influenceradius)
	{}
};

/// integrateGammaDevice parameters specific for I/O
struct io_integrate_gamma_params
{
	const float4 * __restrict__ oldEulerVel;
	const float4 * __restrict__ newEulerVel;

	io_integrate_gamma_params(
		const float4 * __restrict__ _oldEulerVel,
		const float4 * __restrict__ _newEulerVel)
	:
		oldEulerVel(_oldEulerVel),
		newEulerVel(_newEulerVel)
	{}
};

template<flag_t _simflags>
struct integrate_gamma_params :
	common_integrate_gamma_params,
	COND_STRUCT(_simflags & ENABLE_INLET_OUTLET, io_integrate_gamma_params)
{
	static constexpr flag_t simflags = _simflags;

	integrate_gamma_params(
		const	float4	* __restrict__ _oldPos, ///< positions at step n
		const	float4	* __restrict__ _newPos, ///< positions at step n+1
		const	float4	* __restrict__ _oldVel, ///< velocities at step n
		const	float4	* __restrict__ _newVel, ///< velocities at step n+1
		const	particleinfo * __restrict__ _info, ///< particle info
		const	hashKey	* __restrict__ _particleHash, ///< particle hash
		const	float4	* __restrict__ _oldgGam, ///< previous gamma and its gradient
				float4	* __restrict__ _newgGam, ///< [out] new gamma and its gradient
		const	float4	* __restrict__ _oldBoundElement, ///< boundary elements at step n
		const	float4	* __restrict__ _newBoundElement, ///< boundary elements at step n+1
		const	float4	* __restrict__ _oldEulerVel,
		const	float4	* __restrict__ _newEulerVel,
		const	float2	* const _vertPos[],
		const	neibdata *__restrict__ _neibsList,
		const	uint	* __restrict__ _cellStart,
		const	uint	_particleRangeEnd, ///< max number of particles
		const	float	_full_dt, ///< time step (dt)
		const	float	_half_dt, ///< half of time step (dt/2)
		const	float	_t, ///< simulation time
		const	uint	_step, ///< integrator step
		const	float	_slength,
		const	float	_influenceradius)
	:
		common_integrate_gamma_params(
				_oldPos, _newPos, _oldVel, _newVel,
				_info, _particleHash,
				_oldgGam, _newgGam, _oldBoundElement, _newBoundElement,
				_vertPos,
				_neibsList, _cellStart,
				_particleRangeEnd,
				_full_dt, _half_dt, _t, _step,
				_slength, _influenceradius),
		COND_STRUCT(simflags & ENABLE_INLET_OUTLET, io_integrate_gamma_params)(_oldEulerVel, _newEulerVel)
	{}
};

#endif // _DENSITY_SUM_PARAMS_H

