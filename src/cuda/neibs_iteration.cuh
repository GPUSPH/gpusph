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

#include "cellgrid.cuh"
#include <tuple>

#ifndef NEIBS_ITERATION_CUH
#define NEIBS_ITERATION_CUH

namespace cuneibs
{
/** \addtogroup neibs_device_constants Device constants
 * 	\ingroup neibs
 *  Device constants used in neighbor list construction
 *  @{ */
__constant__ uint d_neibboundpos;		///< Starting pos of boundary particle in neib list
__constant__ uint d_neiblistsize;		///< Total neib list size
__constant__ idx_t d_neiblist_stride;	///< Stride dimension
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
	return	idx_t(first_neib_loc<ptype>())*d_neiblist_stride;
}

/// Increment for the neighbor list of the given type
template<ParticleType ptype>
__device__ __forceinline__
constexpr idx_t neib_list_step() {
	return ptype == PT_BOUNDARY ? -d_neiblist_stride : d_neiblist_stride;
}

/// Base neighbor list traversal class
/*! This class holds the variables and methods common to all
 *  neighbor list traversal iterator classes
 */
class neiblist_iterator_core
{
protected:

	const	uint*	cellStart;	///< cells first particle index
	const	neibdata* neibsList; ///< neighbors list
	float4	const	pos;		///< current particle cell-relative position
	int3	const	gridPos;	///< current particle cell index
	const	uint	index;		///< current particle index

	// Persistent variables across getNeibData calls
	float3 pos_corr;
	idx_t i;
	uint neib_cell_base_index;
	char neib_cellnum;

	uint _neib_index; ///< index of the current neighbor

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
	float4 relPos(float4 const& neibPos) const {
		return pos_corr - neibPos;
	}

	__device__ __forceinline__
	neiblist_iterator_core(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		cellStart(_cellStart), neibsList(_neibsList),
		pos(_pos), gridPos(_gridPos), index(_index),
		pos_corr(make_float3(0.0f)),
		neib_cell_base_index(0),
		neib_cellnum(0)
	{}
};

/// Iterator class to traverse the neighbor list for a single type
template<ParticleType _ptype>
class neiblist_iterator_simple :
	// Note the _virtual_ dependency from the core: this will allow
	// multiple neiblist_iterators to be subclassed together, sharing
	// a single core
	virtual public neiblist_iterator_core
{
protected:
	using core = neiblist_iterator_core;
	static constexpr ParticleType ptype = _ptype;

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
		i = neib_list_start<ptype>() - neib_list_step<ptype>();
	}

	__device__ __forceinline__
	bool next()
	{
		i += neib_list_step<ptype>();
		neibdata neib_data = neibsList[i + index];
		if (neib_data == NEIBS_END) return false;

		update_neib_index(neib_data);
		return true;
	}

	__device__ __forceinline__
	neiblist_iterator_simple(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList)
	{ reset(); }
};

/// Specialization of neiblist_iterator_simple for the PT_NONE case
/*! This can be used to skip iterating on neighbors in specific cases
 */
template<>
class neiblist_iterator_simple<PT_NONE> :
	virtual public neiblist_iterator_core
{
protected:
	using core = neiblist_iterator_core;
	static constexpr ParticleType ptype = PT_NONE;

public:
	__device__ __forceinline__
	void reset() {}

	__device__ __forceinline__
	bool next()
	{ return false; }

	__device__ __forceinline__
	neiblist_iterator_simple(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList) {}
};

/// Iterator class to traverse the neighbor list for an arbitrary number of types.
/*! The type-specific sections of the neighbor list are traversed
 *  in the given order. e.g. neiblist_iterators<PT_FLUID, PT_BOUNDARY>
 *  would first go through all fluid neighbors, and then through
 *  all PT_BOUNDARY neighbors
 */
template<ParticleType ...ptypes>
class neiblist_iterator :
	// Subclass all relevant single-type neighbor list iterators.
	// Due to the virtual subclassing of core, we'll only get
	// a single shared core.
	public neiblist_iterator_simple<ptypes>...
{
	// A tuple of our base classes, allowing us to access
	// individual type iterartos by positional index
	using iterators = std::tuple<neiblist_iterator_simple<ptypes>...>;
	template<int i>
	using iterator = typename std::tuple_element<i, iterators>::type;

	// Number of types
	enum { size = sizeof...(ptypes) };

	// Current type
	ParticleType current_type;

	// Switch to next type:
	// this selects the next type and resets the corresponding
	// iterator
	template<ParticleType _next, ParticleType ...other_types>
	__device__ __forceinline__
	void next_type()
	{
		current_type = _next;
		neiblist_iterator_simple<_next>::reset();
	}

	// Get the next neighbor of the current type.
	// If not found, switch to the next type and try again.
	template<ParticleType try_type, ParticleType ...other_types>
	__device__ __forceinline__
	enable_if_t<sizeof...(other_types) != 0, bool>
	try_next() {
		if (try_type == current_type) {
			if (neiblist_iterator_simple<try_type>::next())
				return true;
			next_type<other_types...>();
		}
		return try_next<other_types...>();
	}

	// For the last type, just return whatever the next neighbor is
	// (or none if we're done)
	template<ParticleType try_type>
	__device__ __forceinline__
	bool try_next() {
		return neiblist_iterator_simple<try_type>::next();
	}


public:
	using core = typename iterator<0>::core;

	__device__ __forceinline__
	bool next() {
		return try_next<ptypes...>();
	}

	__device__ __forceinline__
	neiblist_iterator(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList),
		neiblist_iterator_simple<ptypes>(_index, _pos, _gridPos, _cellStart, _neibsList)...,
		current_type(iterator<0>::ptype)
	{
		iterator<0>::reset();
	}
};

/// Specialization of neiblist_iterator for a single type
/*! In this case we don't need anything else than what neiblist_iterator_simple provides,
 *  so just depend directly from it.
 */
template<ParticleType ptype>
class neiblist_iterator<ptype> : public neiblist_iterator_simple<ptype>
{
public:
	using base = neiblist_iterator_simple<ptype>;
	using core = typename base::core;

	__device__ __forceinline__
	neiblist_iterator(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList),
		base(_index, _pos, _gridPos, _cellStart, _neibsList) {}
};

/// Specialization of neiblist_iterator for a single type followed by PT_NONE
/*! In this case too we can just depend directly on neiblist_iterator_simple
 *  with no extra machinery.
 */
template<ParticleType ptype>
class neiblist_iterator<ptype, PT_NONE> : public neiblist_iterator_simple<ptype>
{
public:
	using base = neiblist_iterator_simple<ptype>;
	using core = typename base::core;

	__device__ __forceinline__
	neiblist_iterator(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList),
		base(_index, _pos, _gridPos, _cellStart, _neibsList) {}
};

/// Specialization of neiblist_iterator to discard a final PT_NONE
template<ParticleType ptype1, ParticleType ptype2>
class neiblist_iterator<ptype1, ptype2, PT_NONE> : public neiblist_iterator<ptype1, ptype2>
{
public:
	using base = neiblist_iterator<ptype1, ptype2>;
	using core = typename base::core;

	__device__ __forceinline__
	neiblist_iterator(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList),
		base(_index, _pos, _gridPos, _cellStart, _neibsList) {}
};

/// Iterator over all types allowed by the given formulation
/*! This iterates over all PT_FLUID and PT_BOUNDARY types,
 * and over all PT_VERTEX types if the boundarytype is SA_BOUNDARY
 */
template<BoundaryType boundarytype, typename
	_base = neiblist_iterator<PT_FLUID, PT_BOUNDARY,
		boundarytype == SA_BOUNDARY ? PT_VERTEX : PT_NONE>
>
class allneibs_iterator : public _base
{
public:
	using base = _base;
	using core = typename base::core;

	__device__ __forceinline__
	allneibs_iterator(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList),
		base(_index, _pos, _gridPos, _cellStart, _neibsList) {}
};

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
