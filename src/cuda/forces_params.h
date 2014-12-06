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

#ifndef _FORCES_PARAMS_H
#define _FORCES_PARAMS_H

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

/// Parameters common to all forces kernel specializations
struct common_forces_params
{
			float4	*forces;
			float2	*contupd;
			float4	*rbforces;
			float4	*rbtorques;
	const	float4	*posArray;
	const	hashKey *particleHash;
	const	uint	*cellStart;
	const	neibdata	*neibsList;

	// Particle range to work on. toParticle is _exclusive_
	const	uint	fromParticle;
	const	uint	toParticle;

	// TODO these should probably go into constant memory
	const	float	deltap;
	const	float	slength;
	const	float	influenceradius;
	const	bool	usedem;

	// Constructor / initializer
	common_forces_params(
				float4	*_forces,
				float2	*_contupd,
				float4	*_rbforces,
				float4	*_rbtorques,
		const	float4	*_posArray,
		const	hashKey *_particleHash,
		const	uint	*_cellStart,
		const	neibdata	*_neibsList,
		const	uint	_fromParticle,
		const	uint	_toParticle,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius,
		const	bool	_usedem) :
		forces(_forces),
		contupd(_contupd),
		rbforces(_rbforces),
		rbtorques(_rbtorques),
		posArray(_posArray),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		fromParticle(_fromParticle),
		toParticle(_toParticle),
		deltap(_deltap),
		slength(_slength),
		influenceradius(_influenceradius),
		usedem(_usedem)
	{}
};

/// Additional parameters passed only to kernels with dynamic timestepping
struct dyndt_forces_params
{
	float	*cfl;
	float	*cfltvisc;
	uint	cflOffset;

	dyndt_forces_params(float *_cfl, float *_cfltvisc, uint _cflOffset) :
		cfl(_cfl), cfltvisc(_cfltvisc), cflOffset(_cflOffset)
	{}
};

/// Additional parameters passed only to kernels with XSPH enabled
struct xsph_forces_params
{
	float4	*xsph;
	xsph_forces_params(float4 *_xsph) :
		xsph(_xsph)
	{}
};

/// Additional parameters passed only to kernels with SA_BOUNDARY
struct sa_boundary_forces_params
{
			float4	*newGGam;
	const	float2	*vertPos0;
	const	float2	*vertPos1;
	const	float2	*vertPos2;
	const	float	epsilon;
	const	bool	movingBoundaries;
	// TODO move into separate struct for inoutBoundaries
			uint	*IOwaterdepth;
			bool	ioWaterdepthComputation;

	// Constructor / initializer
	sa_boundary_forces_params(
				float4	*_newGGam,
		const	float2	* const _vertPos[],
		const	float	_epsilon,
		const	bool	_movingBoundaries,
				uint	*_IOwaterdepth,
		const	bool	_ioWaterdepthComputation) :
		newGGam(_newGGam),
		epsilon(_epsilon),
		movingBoundaries(_movingBoundaries),
		IOwaterdepth(_IOwaterdepth),
		ioWaterdepthComputation(_ioWaterdepthComputation)
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

/// Additional parameters passed only to kernels with KEPSVISC
struct kepsvisc_forces_params
{
	float3	*keps_dkde;
	float	*turbvisc;
	kepsvisc_forces_params(float3 *_keps_dkde, float *_turbvisc) :
		keps_dkde(_keps_dkde),
		turbvisc(_turbvisc)
	{}
};

/// The actual forces_params struct, which concatenates all of the above, as appropriate.
template<KernelType kerneltype,
	BoundaryType boundarytype,
	ViscosityType visctype,
	bool dyndt,
	bool usexsph,
	bool inoutBoundaries>
struct forces_params :
	common_forces_params,
	COND_STRUCT(dyndt, dyndt_forces_params),
	COND_STRUCT(usexsph, xsph_forces_params),
	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_forces_params),
	COND_STRUCT(visctype == KEPSVISC, kepsvisc_forces_params)
{
	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	forces_params(
		// common
				float4	*_forces,
				float2	*_contupd,
				float4	*_rbforces,
				float4	*_rbtorques,
		const	float4	*_pos,
		const	hashKey	*_particleHash,
		const	uint	*_cellStart,
		const	neibdata*_neibsList,
				uint	_fromParticle,
				uint	_toParticle,

				float	_deltap,
				float	_slength,
				float	_influenceradius,
				bool	_usedem,

		// dyndt
				float	*_cfl,
				float	*_cflTVisc,
				uint	_cflOffset,

		// XSPH
				float4	*_xsph,

		// SA_BOUNDARY
				float4	*_newGGam,
		const	float2	* const _vertPos[],
		const	float	_epsilon,
		const	bool	_movingBoundaries,
				uint	*_IOwaterdepth,
		const	bool	_ioWaterdepthComputation,

		// KEPSVISC
				float3	*_keps_dkde,
				float	*_turbvisc
		) :
		common_forces_params(_forces, _contupd, _rbforces, _rbtorques,
			_pos, _particleHash, _cellStart,
			_neibsList, _fromParticle, _toParticle,
			_deltap, _slength, _influenceradius, _usedem),
		COND_STRUCT(dyndt, dyndt_forces_params)(_cfl, _cflTVisc, _cflOffset),
		COND_STRUCT(usexsph, xsph_forces_params)(_xsph),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_forces_params)
			(_newGGam, _vertPos, _epsilon, _movingBoundaries, _IOwaterdepth, _ioWaterdepthComputation),
		COND_STRUCT(visctype == KEPSVISC, kepsvisc_forces_params)(_keps_dkde, _turbvisc)
	{}
};

#endif // _FORCES_PARAMS_H

