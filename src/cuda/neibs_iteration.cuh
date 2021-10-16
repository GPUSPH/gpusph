/*  Copyright (c) 2017-2019 INGV, EDF, UniCT, JHU

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

#ifndef NEIBS_ITERATION_CUH
#define NEIBS_ITERATION_CUH

#include <tuple>

#include "cpp11_missing.h"

#include "particledefine.h" // BoundaryType

#include "cellgrid.cuh"
#include "posvel_struct.h" // pos_mass

#include "neibs_list_layout.h"

namespace cuneibs
{
/** \addtogroup neibs_device_constants Device constants
 * 	\ingroup neibs
 *  Device constants used in neighbor list construction
 *  @{ */
__constant__ uint d_neibboundpos;		///< Starting pos of boundary particle in neib list
__constant__ uint d_neiblistsize;		///< Total neib list size
__constant__ idx_t d_neiblist_stride;	///< Stride dimension for NL_INTERLEAVE (= number of allocated particles)
__constant__ idx_t d_neiblist_end;		///< maximum number of neighbors * number of allocated particles
/** @} */


/** \addtogroup neibs_iteration Neighbor list iteration functions
 *  \ingroup neibs
 *  Templatized functions to iterate over neighbors of a given kind.
 */

/// Location of the first index for a neighbor of the given type.
/*! This is the location in the neighbor list of a given particle,
 *  without taking the strided structure into account.
 *  \see neib_list_start
 */
template<ParticleType ptype>
__device__ __forceinline__
uint first_neib_loc() {
	return	(ptype == PT_FLUID) ? 0 :
			(ptype == PT_BOUNDARY) ? d_neibboundpos :
			/* ptype == PT_VERTEX */ d_neibboundpos+1;
}

/// Start of the list of neighbors of the given type
template<ParticleType ptype>
__device__ __forceinline__
idx_t neib_list_start() {
#if NL_INTERLEAVED
	return	idx_t(first_neib_loc<ptype>())*d_neiblist_stride;
#else
	return	idx_t(first_neib_loc<ptype>());
#endif
}

/// Increment for the neighbor list of the given type
template<ParticleType ptype>
__device__ __forceinline__
constexpr idx_t neib_list_step() {
#if NL_INTERLEAVED
	return ptype == PT_BOUNDARY ? -d_neiblist_stride : d_neiblist_stride;
#else
	return ptype == PT_BOUNDARY ? -1 : 1;
#endif
}

/// Base neighbor list traversal class
/*! This class holds the variables and methods common to all
 *  neighbor list traversal iterator classes
 */
class neiblist_iterator_core : protected pos_mass
{
protected:
	// This class is not actually associated to any specific particle type,
	// and is instead used as a base class and a terminator, identified
	// by having an associated particle type of PT_NONE
	static constexpr ParticleType ptype = PT_NONE;

	const	uint*	cellStart;	///< cells first particle index
	const	neibdata* neibsList; ///< neighbors list
	const	int3	gridPos;	///< current particle cell index
	const	uint	index;		///< current particle index

	// Persistent variables across getNeibData calls
	float3 pos_corr;
	idx_t i;
	uint neib_cell_base_index;
	uchar neib_cellnum;

	uint _neib_index; ///< index of the current neighbor

	//! Current particle type being processed,
	//! in the case of multi-type iterators
	ParticleType current_type;

	__device__ __forceinline__
	void update_neib_index(neibdata neib_data)
	{
		_neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
			neib_cellnum, neib_cell_base_index);
	}

public:

	//! Neighbor sequential offset
	/*! Any structure which is addressed like the neighbor list can find
	 * the current neighbor data adding the central particle index to the neib_list_offset()
	 */
	__device__ __forceinline__
	idx_t neib_list_offset() const
	{ return i; }

	//! Neighbor index (i.e. the particle index of the neighbor)
	__device__ __forceinline__
	uint const& neib_index() const {
		return _neib_index;
	}

	/// Compute the relative distance to a neighbor with the given local position
	/*! The neighbor position should be local to the current neighbor cell being
	 *  traversed.
	 */
	__device__ __forceinline__
	relPos_mass relPos(float4 const& neibPos) const {
		return pos_corr - pos_mass(neibPos);
	}

	__device__ __forceinline__
	neiblist_iterator_core(uint _index, pos_mass const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList)
	:
		pos_mass(_pos),
		cellStart(_cellStart), neibsList(_neibsList),
		gridPos(_gridPos), index(_index),
		pos_corr(make_float3(0.0f)),
		neib_cell_base_index(0),
		neib_cellnum(0)
	{}

	__device__ __forceinline__
	neiblist_iterator_core(uint index, float4 const& pos, int3 const& gridPos,
		const uint *cellStart, const neibdata *neibsList)
	:
		neiblist_iterator_core(index, pos_mass(pos), gridPos, cellStart, neibsList)
	{}
};

/// Iterator class to traverse the neighbor list for one or more types
template<ParticleType ptype_, typename NextIterator = neiblist_iterator_core>
class neiblist_iterator_nest : public NextIterator
{
protected:
	using core = neiblist_iterator_core;

	static constexpr ParticleType ptype = ptype_;

	static constexpr bool lastIterator = NextIterator::ptype == PT_NONE;

	//! Fetch the next neighbor
	//! There are two instances of this member function.
	//! This one is used in the last (or only) iterator case,
	//! and simply loads a neighbor, and if not found returns false.
	//!
	//! \note the reason for the argument is that we cannot specialize a member function
	//! in C++ simply by return type, so we pass a dummy argument.
	//! This could be solved using more recent C++ features such as if constexpr
	//!
	//! \return false if not found, true otherwise
	__device__ __forceinline__
	bool fetch_next()
	{
		core::i += neib_list_step<ptype>();
		neibdata neib_data = core::neibsList[ITH_STEP_NEIGHBOR(core::index, core::i, d_neiblistsize)];
		if (neib_data == NEIBS_END) return false;

		core::update_neib_index(neib_data);
		return true;
	}

	//! Fetch the next neighbor, assumingno next iterators
	//! There are two instances of this member function.
	//! This one is used in the last (or only) iterator case,
	//! and simply loads a neighbor, and if not found returns false.
	//!
	//! \note the reason for the argument is that we cannot specialize a member function
	//! in C++ simply by return type, so we pass a dummy argument.
	//! This could be solved using more recent C++ features such as if constexpr
	template<typename IsLastIterator>
	__device__ __forceinline__
	enable_if_t<IsLastIterator::value == true, bool>
	_next(IsLastIterator const& bool_class)
	{ return fetch_next(); }

	//! Fetch the next neighbor or delegate to the next iterator
	//! This instance is used when there is a next iterator.
	//! \note see the other instance for the meaning of the argument.
	template<typename IsLastIterator>
	__device__ __forceinline__
	enable_if_t<IsLastIterator::value == false, bool>
	_next(IsLastIterator const& bool_class)
	{
		if (core::current_type == ptype) {
			bool ret = fetch_next();
			if (ret) return true;
			// finished current type, switch to the next
			NextIterator::reset();
		}
		// current type has finished, delegate the call
		// to the next type
		return NextIterator::next();
	}

public:
	__device__ __forceinline__
	void reset()
	{
		// We start “behind” the official start, because every fetch
		// does an increment first
		// TODO FIXME this is ugly, but can work as a stop-gap measure
		// until we find the best way to map the complexities of neighbors
		// iteration to the STL iterator interface; C++20 with the Sentinel
		// concept should make it much easier.
		core::i = neib_list_start<ptype>() - neib_list_step<ptype>();
		core::current_type = ptype;
	}

	__device__ __forceinline__
	bool next()
	{
		using next_selector = bool_constant<!!lastIterator>;
		return _next(next_selector());
	}

	template<typename PosMass, // float4 or pos_mass
		typename IsFirstIterator = std::false_type
		>
	__device__ __forceinline__
	neiblist_iterator_nest(uint _index, PosMass const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList,
		IsFirstIterator const& condition = IsFirstIterator())
	:
		NextIterator(_index, _pos, _gridPos, _cellStart, _neibsList)
	{
		// reset only if this is the first neighbor
		if (condition.value) reset();
	}
};

//! More practical way to specify the list of types
template<ParticleType... types>
class neiblist_iterator;

template<ParticleType ptype, ParticleType... other_types>
class neiblist_iterator<ptype, other_types...> :
	public neiblist_iterator_nest<ptype,
		// the next iterator is chosen based on how many remain,
		// to properly terminate the list
		typename std::conditional<sizeof...(other_types) == 0,
			neiblist_iterator_core,
			neiblist_iterator<other_types...>
		>::type>
{
	using base = neiblist_iterator_nest<ptype,
		// the next iterator is chosen based on how many remain,
		// to properly terminate the list
		typename std::conditional<sizeof...(other_types) == 0,
			neiblist_iterator_core,
			neiblist_iterator<other_types...>
		>::type>;

public:
	//! Constructor: delegate to the nested iterator class,
	//! passing a true_type to ensure that the first iterator in the series
	//! gets properly initialized
	template<typename PosMass> // float4 or pos_mass
	__device__ __forceinline__
	neiblist_iterator(uint _index, PosMass const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList)
	:
		base(_index, _pos, _gridPos, _cellStart, _neibsList, std::true_type())
	{}
};

/// Iterator over all types allowed by the given formulation
/*! This iterates over all PT_FLUID and PT_BOUNDARY types,
 * and over all PT_VERTEX types if the boundarytype is SA_BOUNDARY
 */
template<BoundaryType boundarytype>
using allneibs_iterator = neiblist_iterator<PT_FLUID, PT_BOUNDARY,
	  boundarytype == SA_BOUNDARY ? PT_VERTEX : PT_NONE
>;

/// A practical macro to iterate over all neighbours of a given type
/*! This instantiates a neiblist_iterator of the proper type, called neib_iter,
 *  in the scope of a for () that terminates when there are no more neighbors of the
 *  given type.
 */
#define for_each_neib(ptype, index, pos, gridPos, cellStart, neibsList) \
	for ( \
		neiblist_iterator<ptype> neib_iter(index, pos, gridPos, cellStart, neibsList) ; \
		neib_iter.next() ; \
	)

#define for_each_neib2(ptype1, ptype2, index, pos, gridPos, cellStart, neibsList) \
	for ( \
		neiblist_iterator<ptype1, ptype2> neib_iter(index, pos, gridPos, cellStart, neibsList) ; \
		neib_iter.next() ; \
	)

#define for_each_neib3(ptype1, ptype2, ptype3, index, pos, gridPos, cellStart, neibsList) \
	for ( \
		neiblist_iterator<ptype1, ptype2, ptype3> neib_iter(index, pos, gridPos, cellStart, neibsList) ; \
		neib_iter.next() ; \
	)

/// A practical macro to iterate over all neighbors of all types allowed by the formulation
#define for_every_neib(boundarytype, index, pos, gridPos, cellStart, neibsList) \
	for ( \
		allneibs_iterator<boundarytype> neib_iter(index, pos, gridPos, cellStart, neibsList) ; \
		neib_iter.next() ; \
	)

}

#endif

/* vim: set ft=cuda: */
