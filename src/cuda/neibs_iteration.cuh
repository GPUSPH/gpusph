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

template<ParticleType ptype>
class neiblist_iterator {
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

	uint _neib_index;
public:

	__device__ __forceinline__
	uint const& neib_index() const {
		return _neib_index;
	}

	__device__ __forceinline__
	float4 relPos(float4 const& neibPos) const {
		return pos_corr - neibPos;
	}

	__device__ __forceinline__
	bool next()
	{
		neibdata neib_data = neibsList[i + index];
		if (neib_data == NEIBS_END) return false;
		i += neib_list_step<ptype>();

		_neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
			neib_cellnum, neib_cell_base_index);
		return true;
	}

	__device__ __forceinline__
	neiblist_iterator(uint _index, float4 const& _pos, int3 const& _gridPos,
		const uint *_cellStart, const neibdata *_neibsList) :
		index(_index), pos(_pos), gridPos(_gridPos),
		cellStart(_cellStart), neibsList(_neibsList),
		neib_cellnum(0),
		neib_cell_base_index(0),
		pos_corr(make_float3(0.0f)),
		i(neib_list_start<ptype>())
	{}
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
