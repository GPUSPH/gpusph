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

#ifndef _REPACK_PARAMS_H
#define _REPACK_PARAMS_H

#include "particledefine.h"
#include "simflags.h"

/* The forces computation kernel is probably the most complex beast in GPUSPH.
   To achieve good performance, each combination of kernel, boundary, formulation
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

// We now have the tools to assemble the structure that will be used to pass parameters to the forces kernel

/* Now we define structures that hold the parameters to be passed
   to the forces kernel. These are defined in chunks, and then ‘merged’
   into the template structure that is actually used as argument to the kernel.
   Each struct must define an appropriate constructor / initializer for its const
   members
*/

/// Parameters common to all repack kernel specializations
struct common_repack_params
{
			float4	* __restrict__ forces;
	const	float4	* __restrict__ posArray;
	const	hashKey * __restrict__ particleHash;
	const	uint	* __restrict__ cellStart;
	const	neibdata	* __restrict__ neibsList;

	// Particle range to work on. toParticle is _exclusive_
	const	uint	fromParticle;
	const	uint	toParticle;

	// TODO these should probably go into constant memory
	const	float	deltap;
	const	float	slength;
	const	float	influenceradius;
	const	float	dt;

	// Constructor / initializer
	common_repack_params(
				float4	* __restrict__ _forces,
		const	float4	* __restrict__ _posArray,
		const	hashKey * __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,
		const	uint	_fromParticle,
		const	uint	_toParticle,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius,
		const	float	_dt) :
		forces(_forces),
		posArray(_posArray),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		fromParticle(_fromParticle),
		toParticle(_toParticle),
		deltap(_deltap),
		slength(_slength),
		influenceradius(_influenceradius),
		dt(_dt)
	{}
};

/// Parameters common to all finalize forces kernel specializations
struct common_finalize_repack_params
{
			float4	*forces;
	const	float4	*posArray;
	const	float4	*velArray;
	const	hashKey *particleHash;
	const	uint	*cellStart;

	// Particle range to work on. toParticle is _exclusive_
	const	uint	fromParticle;
	const	uint	toParticle;

	const	float	slength;
	const	float	deltap;

	// Constructor / initializer
	common_finalize_repack_params(
				float4	*_forces,
		const	float4	*_posArray,
		const	float4	*_velArray,
		const	hashKey *_particleHash,
		const	uint	*_cellStart,
		const	uint	_fromParticle,
		const	uint	_toParticle,
		const	float	_slength,
		const float _deltap) :
		forces(_forces),
		posArray(_posArray),
		velArray(_velArray),
		particleHash(_particleHash),
		cellStart(_cellStart),
		fromParticle(_fromParticle),
		toParticle(_toParticle),
		slength(_slength),
		deltap(_deltap)
	{}
};

/// Additional parameters passed only to kernels with dynamic timestepping
struct dyndt_finalize_repack_params
{
	float	* __restrict__ cfl_forces;
	float	* __restrict__ cfl_gamma;
	uint	cflOffset;
	uint	cflGammaOffset;

	dyndt_finalize_repack_params(float * __restrict__ _cfl_forces, float * __restrict__ _cfl_gamma,
		uint _numParticles, uint _cflOffset) :
		cfl_forces(_cfl_forces), cfl_gamma(_cfl_gamma), cflOffset(_cflOffset),
		cflGammaOffset(round_up(_numParticles, 4U) + cflOffset)
	{}
};

/// Additional parameters passed only to kernels with SA_BOUNDARY
/// in case of of a fluid/boundary interaction
struct sa_boundary_repack_params
{
			float	* __restrict__ cfl_gamma;
	const	float2	* __restrict__ vertPos0;
	const	float2	* __restrict__ vertPos1;
	const	float2	* __restrict__ vertPos2;
	const	float	epsilon;

	// Constructor / initializer
	sa_boundary_repack_params(
				float	* __restrict__ _cfl_gamma,
		const	float2	* __restrict__  const _vertPos[],
		const	float	_epsilon) :
		cfl_gamma(_cfl_gamma),
		epsilon(_epsilon)
	{
		if (_vertPos) {
			vertPos0 = _vertPos[0];
			vertPos1 = _vertPos[1];
			vertPos2 = _vertPos[2];
		} else {
			vertPos0 = vertPos1 = vertPos2 = NULL;
		}
	}
};

/// Additional parameters passed only finalize repack with SA_BOUNDARY formulation
struct sa_finalize_repack_params
{
	const	float4	*gGam;

	// Constructor / initializer
	sa_finalize_repack_params(const	float4	*_gGam) :
		gGam(_gGam)
	{}
};

/// The actual repack_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	BoundaryType _boundarytype,
	flag_t _simflags,
	ParticleType _cptype,
	ParticleType _nptype>
struct repack_params :
	common_repack_params,
	COND_STRUCT(_boundarytype == SA_BOUNDARY && _cptype != _nptype, sa_boundary_repack_params)
{
	static const KernelType kerneltype = _kerneltype;
	static const BoundaryType boundarytype = _boundarytype;
	static const flag_t simflags = _simflags;
	static const ParticleType cptype = _cptype;
	static const ParticleType nptype = _nptype;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	repack_params(
		// common
				float4	* __restrict__ _forces,
		const	float4	* __restrict__ _pos,
		const	hashKey	* __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata* __restrict__ _neibsList,
				uint	_fromParticle,
				uint	_toParticle,

		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius,
		const	float	_dt,

		// SA_BOUNDARY
				float	* __restrict__ _cfl_gamma,
		const	float2	* __restrict__ const _vertPos[],
		const	float	_epsilon

		) :
		common_repack_params(_forces,
			_pos, _particleHash, _cellStart,
			_neibsList, _fromParticle, _toParticle,
			_deltap, _slength, _influenceradius, _dt),
		COND_STRUCT(boundarytype == SA_BOUNDARY && cptype != nptype, sa_boundary_repack_params)
			(_cfl_gamma, _vertPos, _epsilon)
	{}
};


/// The actual finalize_forces_params struct, which concatenates all of the above, as appropriate.
template<BoundaryType _boundarytype,
	flag_t _simflags>
struct finalize_repack_params :
	common_finalize_repack_params,
	COND_STRUCT(_simflags & ENABLE_DTADAPT, dyndt_finalize_repack_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_finalize_repack_params)
{
	static const BoundaryType boundarytype = _boundarytype;
	static const flag_t simflags = _simflags;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the finalize forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	finalize_repack_params(
		// common
				float4	*_forces,
		const	float4	*_posArray,
		const	float4	*_velArray,
		const	hashKey	*_particleHash,
		const	uint	*_cellStart,
				uint	_numParticles,
				uint	_fromParticle,
				uint	_toParticle,

		const	float	_slength,
		const float _deltap,
		// dyndt
				float	*_cfl_forces,
				float	*_cfl_gamma,
				uint	_cflOffset,

		// SA_BOUNDARY
		const	float4	*_gGam
		) :
		common_finalize_repack_params(_forces,
			_posArray, _velArray, _particleHash, _cellStart,
			 _fromParticle, _toParticle, _slength,_deltap),
		COND_STRUCT(simflags & ENABLE_DTADAPT, dyndt_finalize_repack_params)
			(_cfl_forces, _cfl_gamma, _numParticles, _cflOffset),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_finalize_repack_params) (_gGam)
	{}
};

#endif // _REPACK_PARAMS_H

