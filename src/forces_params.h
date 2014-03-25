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

   These template structures depend on std::conditional to select which
   struct members are allowed in each specialization of the options.
   This is part of the C++11 stdlib, but it can be easily implemented in older
   C++ revisions too.
*/

#if __cplusplus <= 199711L
template<bool B, typename T, typename F>
struct conditional { typedef T type; };

template<typename T, typename F>
struct conditional<false, T, F> { typedef F type; };
#else
#include <type_traits>
using std::conditional;
#endif

/* The general idea is that each group of members of the structure is defined as
   a specific (non-template, usually) structure, and then the actual template
   structure is defined as a derived class of each of the specific (sub) structures,
   if the appropriate template parameters are set. When the template parameter values
   are not correct, an empty structure is included in place of the actual one.
   While the structure itself is empty, it must provide a constructor that acts as
   fallback for each constructor used in the structure it is used as a replacement of.
*/

template<typename>
struct empty
{
	// constructors
	__host__ __device__ __forceinline__
	empty() {}

	template<typename T1>
	__host__ __device__ __forceinline__
	empty(T1) {}

	template<typename T1, typename T2>
	__host__ __device__ __forceinline__
	empty(T1, T2) {}

	template<typename T1, typename T2, typename T3>
	__host__ __device__ __forceinline__
	empty(T1, T2, T3) {}

	template<typename T1, typename T2, typename T3, typename T4>
	__host__ __device__ __forceinline__
	empty(T1, T2, T3, T4) {}
};


/* Inclusion of struct_to_include under a given condition would be achieved by
   deriving the complete class from:
   > conditional<boolean_condition, struct_to_include, empty<struct_to_include> >::type
   for example:
   > conditional<boundarytype == SA_BOUNDARY, sa_boundary_forces_params, empty<sa_boundary_forces_params> >::type
   which is very verbose, so we define a macro COND_STRUCT(boolean_condition,
   struct_to_include) to do the job for us
*/

#define COND_STRUCT(some_cond, some_struct) \
	conditional<some_cond, some_struct, empty<some_struct> >::type


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
			float4	*rbforces;
			float4	*rbtorques;
	const	float4	*posArray;
	const	hashKey *particleHash;
	const	uint	*cellStart;
	const	neibdata	*neibsList;
	const	uint	numParticles;

	// TODO these should probably go into constant memory
	const	float	deltap;
	const	float	slength;
	const	float	influenceradius;

	// Constructor / initializer
	common_forces_params(
				float4	*_forces,
				float4	*_rbforces,
				float4	*_rbtorques,
		const	float4	*_posArray,
		const	hashKey *_particleHash,
		const	uint	*_cellStart,
		const	neibdata	*_neibsList,
		const	uint	_numParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius) :
		forces(_forces),
		rbforces(_rbforces),
		rbtorques(_rbtorques),
		posArray(_posArray),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		numParticles(_numParticles),
		deltap(_deltap),
		slength(_slength),
		influenceradius(_influenceradius)
	{}
};

/// Additional parameters passed only to kernels with dynamic timestepping
struct dyndt_forces_params
{
	float	*cfl;
	float	*cfltvisc;

	dyndt_forces_params(float *_cfl, float *_cfltvisc) :
		cfl(_cfl), cfltvisc(_cfltvisc)
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
// TODO FIXME for the time being this is _always_ included
struct sa_boundary_forces_params
{
			float4	*newGGam;
	const	float2	*vertPos0;
	const	float2	*vertPos1;
	const	float2	*vertPos2;
	const	float	epsilon;
	const	bool	movingBoundaries;

	// Constructor / initializer
	sa_boundary_forces_params(
				float4	*_newGGam,
		const	float2	* const _vertPos[],
		const	float	_epsilon,
		const	bool	_movingBoundaries) :
		newGGam(_newGGam),
		epsilon(_epsilon),
		movingBoundaries(_movingBoundaries)
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
	float2	*keps_dkde;
	float	*turbvisc;
	kepsvisc_forces_params(float2 *_keps_dkde, float *_turbvisc) :
		keps_dkde(_keps_dkde),
		turbvisc(_turbvisc)
	{}
};

/// The actual forces_params struct, which concatenates all of the above, as appropriate.
template<KernelType kerneltype,
	BoundaryType boundarytype,
	ViscosityType visctype,
	bool dyndt,
	bool usexsph>
struct forces_params :
	common_forces_params,
	COND_STRUCT(dyndt, dyndt_forces_params),
	COND_STRUCT(usexsph, xsph_forces_params),
#if 0 // TODO FIXME for the time being this is included unconditionally
	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_forces_params),
#else
	sa_boundary_forces_params,
#endif
	COND_STRUCT(visctype == KEPSVISC, kepsvisc_forces_params)
{
	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	forces_params(
		// common
				float4	*_forces,
				float4	*_rbforces,
				float4	*_rbtorques,
		const	float4	*_pos,
		const	hashKey	*_particleHash,
		const	uint	*_cellStart,
		const	neibdata*_neibsList,
				uint	_numParticles,

				float	_deltap,
				float	_slength,
				float	_influenceradius,

		// dyndt
				float	*_cfl,
				float	*_cflTVisc,

		// XSPH
				float4	*_xsph,

		// SA_BOUNDARY
				float4	*_newGGam,
		const	float2	* const _vertPos[],
		const	float	_epsilon,
		const	bool	_movingBoundaries,

		// KEPSVISC
				float2	*_keps_dkde,
				float	*_turbvisc
		) :
		common_forces_params(_forces, _rbforces, _rbtorques,
			_pos, _particleHash, _cellStart,
			_neibsList, _numParticles,
			_deltap, _slength, _influenceradius),
		COND_STRUCT(dyndt, dyndt_forces_params)(_cfl, _cflTVisc),
		COND_STRUCT(usexsph, xsph_forces_params)(_xsph),
#if 0 // TODO FIXME for the time being this is included unconditionally
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_forces_params)
#else
		sa_boundary_forces_params
#endif
			(_newGGam, _vertPos, _epsilon, _movingBoundaries),
		COND_STRUCT(visctype == KEPSVISC, kepsvisc_forces_params)(_keps_dkde, _turbvisc)
	{}
};



#endif // _FORCES_PARAMS_H

