/*  Copyright 2017 Alexis Herault, Giuseppe Bilotta, Robert A.
 	Dalrymple, Eugenio Rustico, Ciro Del Negro

	Conservatoire National des Arts et Metiers, Paris, France

	Istituto Nazionale di Geofisica e Vulcanologia,
    Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

#include <tuple>

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

	const	uint	index;		///< current particle index
	float4	const&	pos;		///< current particle cell-relative position
	int3	const&	gridPos;	///< current particle cell index
	const	uint*	cellStart;	///< cells first particle index
	const	neibdata* neibsList; ///< neighbors list

	// Persistent variables across getNeibData calls
	char neib_cellnum;
	uint neib_cell_base_index;
	float3 pos_corr;
	idx_t i;

	uint _neib_index; ///< index of the current neighbor

	__device__ __forceinline__
	void update_neib_index(neibdata neib_data)
	{
		_neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
			neib_cellnum, neib_cell_base_index);
	}

public:

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
		index(_index), pos(_pos), gridPos(_gridPos),
		cellStart(_cellStart), neibsList(_neibsList),
		neib_cellnum(0),
		neib_cell_base_index(0),
		pos_corr(make_float3(0.0f))
	{}
};

/// Iterator class to traverse the neighbor list for a specific type
template<ParticleType _ptype>
class neiblist_iterator :
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
	{ i = neib_list_start<ptype>(); }

	__device__ __forceinline__
	bool next()
	{
		neibdata neib_data = neibsList[i + index];
		if (neib_data == NEIBS_END) return false;
		i += neib_list_step<ptype>();

		update_neib_index(neib_data);
		return true;
	}

	__device__ __forceinline__
	neiblist_iterator(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList)
	{ reset(); }
};

/// Specialization for the PT_NONE case, used to skip iterating on neighbors
/// in specific cases
template<>
class neiblist_iterator<PT_NONE> :
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
	neiblist_iterator(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList) {}
};

/// Iterator class to traverse the neighbor list for more than one type.
/*! The type-specific sections of the neighbor list are traversed
 *  in the given order. e.g. neiblist_iterators<PT_FLUID, PT_BOUNDARY>
 *  would first go through all fluid neighbors, and then through
 *  all PT_BOUNDARY neighbors
 */
template<ParticleType ...ptypes>
class neiblist_iterators :
	// Subclass all relevant single-type neighbor list iterators.
	// Due to the virtual subclassing of core, we'll only get
	// a single shared core.
	public neiblist_iterator<ptypes>...
{
	// A tuple of our base classes, allowing us to access
	// individual type iterartos by positional index
	using iterators = std::tuple<neiblist_iterator<ptypes>...>;
	template<int i>
	using iterator = typename std::tuple_element<i, iterators>::type;
	using core = typename iterator<0>::core;

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
		neiblist_iterator<_next>::reset();
	}

	// Get the next neighbor of the current type.
	// If not found, switch to the next type and try again.
	template<ParticleType try_type, ParticleType ...other_types>
	__device__ __forceinline__
	bool try_next(ParticleType current) {
		if (try_type == current) {
			if (neiblist_iterator<try_type>::next())
				return true;
			next_type<other_types...>();
		}
		return try_next<other_types...>(current);
	}

	// For the last type, just return whatever the next neighbor is
	// (or none if we're done)
	template<ParticleType try_type>
	__device__ __forceinline__
	bool try_next(ParticleType current) {
		return neiblist_iterator<try_type>::next();
	}


public:
	__device__ __forceinline__
	bool next() {
		return try_next<ptypes...>(current_type);
	}

	__device__ __forceinline__
	neiblist_iterators(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		core(_index, _pos, _gridPos, _cellStart, _neibsList),
		neiblist_iterator<ptypes>(_index, _pos, _gridPos, _cellStart, _neibsList)...,
		current_type(iterator<0>::ptype)
	{
		iterator<0>::reset();
	}
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
		neiblist_iterators<ptype1, ptype2> neib_iter(index, pos, gridPos, cellStart, neibsList) ; \
		neib_iter.next() ; \
	)

#define for_each_neib3(ptype1, ptype2, ptype3, index, pos, gridPos, cellStart, neibsList) \
	for ( \
		neiblist_iterators<ptype1, ptype2, ptype3> neib_iter(index, pos, gridPos, cellStart, neibsList) ; \
		neib_iter.next() ; \
	)

/// Specialization of for_each_neib3 in 'standard order' (FLUID, BOUNDARY, VERTEX)
#define for_every_neib(index, pos, gridPos, cellStart, neibsList) \
	for_each_neib3(PT_FLUID, PT_BOUNDARY, PT_VERTEX)

/* vim: set ft=cuda: */