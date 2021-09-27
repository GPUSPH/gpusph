/*  Copyright (c) 2020-2021 INGV, EDF, UniCT, JHU

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

/*! \file
 * Parameters for the reorderDataAndFindCellStartDevice kernel
 */

#ifndef REORDER_PARAMS_H
#define REORDER_PARAMS_H

#include "particledefine.h"

#include "buffer.h"
#include "define_buffers.h"
#include "cond_params.h"

// In most cases, when a buffer is included in the reorder_params structure,
// it MUST be present. Some buffers, however (e.g. BUFFER_GRADGAMMA and BUFFER_DUMMY_VEL)
// may be missing (in the initial step).

template<flag_t BufferKey>
struct reorder_optional
{
	static constexpr bool value =
		BufferKey == BUFFER_DUMMY_VEL ||
		BufferKey == BUFFER_GRADGAMMA ;
};

// Define a reorder_data<BUFFER_SOMETHING> struct template,
// that extracts the correct buffers from both the sorted and unsorted list,
// and provides a method to reorder the data belonging to that buffer
template<flag_t BufferKey, bool optional=reorder_optional<BufferKey>::value>
struct reorder_data;

// non-optional case
template<flag_t BufferKey>
struct reorder_data<BufferKey, false>
{
	using	data_t = typename BufferTraits<BufferKey>::element_type;
	const	data_t * __restrict__ unsorted;
			data_t * __restrict__ sorted;

	// initialize the pointers from the buffer lists. Since REORDER requires
	// buffer validation, this will error out if either buffer is not present
	reorder_data(BufferList& sorted_buffers, BufferList const& unsorted_buffers) :
		unsorted(unsorted_buffers.getData<BufferKey>()),
		sorted(sorted_buffers.getData<BufferKey>())
	{}

	__device__ __forceinline__ void
	reorder(const uint index, const uint sortedIndex, particleinfo const& info)
	{ sorted[index] = unsorted[sortedIndex]; }
};

// optional case
template<flag_t BufferKey>
struct reorder_data<BufferKey, true>
{
	using	data_t = typename BufferTraits<BufferKey>::element_type;
	const	data_t * __restrict__ unsorted;
			data_t * __restrict__ sorted;

	// in the case of optional buffers, the constructor initializes the
	// pointers to be null, and only sets them if the key is present in
	// the sorted_buffer list. Note that since REORDER requires buffer validation,
	// this will error out if the other buffer is not present too
	reorder_data(BufferList& sorted_buffers, BufferList const& unsorted_buffers) :
		unsorted(nullptr),
		sorted(nullptr)
	{
		if (sorted_buffers.has(BufferKey)) {
			unsorted = unsorted_buffers.getData<BufferKey>();
			sorted = sorted_buffers.getData<BufferKey>();
		}
	}

	// since the buffers may be missing, the reordering is conditional to
	// the sorted pointer being nonzero
	__device__ __forceinline__ void
	reorder(const uint index, const uint sortedIndex, particleinfo const& info)
	{
		if (sorted)
			sorted[index] = unsorted[sortedIndex];
	}
};

// Sorting specialization for BUFFER_VERTICES, that only copies data over
// if the particle is a boundary particle
template<>
__device__ __forceinline__
void reorder_data<BUFFER_VERTICES>::reorder(const uint index, const uint sortedIndex, particleinfo const& info)
{
	sorted[index] = BOUNDARY(info) ? unsorted[sortedIndex] : make_vertexinfo(0, 0, 0, 0);
}

// Mock-up of the reorder_data for when the buffer is not there
template<flag_t BufferKey>
struct no_reorder_data
{
	no_reorder_data(BufferList& sorted_buffers, BufferList const& unsorted_buffers) {}
	__device__ __forceinline__ void reorder(const uint index, const uint sortedIndex, particleinfo const& info) {}
};

// Conditional structure that maps to reorder_data or no_reorder_data based on condition
template<bool condition, flag_t BufferKey>
using conditional_reorder = typename conditional<condition, reorder_data<BufferKey>, no_reorder_data<BufferKey>>::type;

template<SPHFormulation sph_formulation_, typename ViscSpec, BoundaryType boundarytype_, flag_t simflags_
	, typename reorderPos = reorder_data<BUFFER_POS>
	, typename reorderVel = reorder_data<BUFFER_VEL>
	// SPH_GRENIER
	, typename reorderVol = conditional_reorder<sph_formulation_ == SPH_GRENIER, BUFFER_VOLUME>
	// ENABLE_INTERNAL_ENERGY
	, typename reorderEnergy = conditional_reorder< HAS_INTERNAL_ENERGY(simflags_), BUFFER_INTERNAL_ENERGY>
	// SA_BOUNDARY
	, typename reorderBoundElements = conditional_reorder<boundarytype_ == SA_BOUNDARY, BUFFER_BOUNDELEMENTS>
	, typename reorderGradGamma = conditional_reorder<boundarytype_ == SA_BOUNDARY, BUFFER_GRADGAMMA>
	, typename reorderVertices = conditional_reorder<boundarytype_ == SA_BOUNDARY, BUFFER_VERTICES>
	// KEPSILON
	, typename reorderTKE = conditional_reorder<ViscSpec::turbmodel == KEPSILON, BUFFER_TKE>
	, typename reorderEpsilon = conditional_reorder<ViscSpec::turbmodel == KEPSILON, BUFFER_EPSILON>
	, typename reorderTurbVisc = conditional_reorder<ViscSpec::turbmodel == KEPSILON, BUFFER_TURBVISC>
	// GRANULAR
	, typename reorderEffPres = conditional_reorder<ViscSpec::rheologytype == GRANULAR, BUFFER_EFFPRES>
	// SA_BOUNDARY and ENABLE_INLET_OUTLET or KEPSILON
	, typename reorderEulerVel = conditional_reorder<
		boundarytype_ == SA_BOUNDARY && (HAS_INLET_OUTLET(simflags_) || ViscSpec::turbmodel == KEPSILON),
		BUFFER_EULERVEL>
	, typename reorderNextID = conditional_reorder<HAS_INLET_OUTLET(simflags_), BUFFER_NEXTID>
	, typename reorderDummyVel = conditional_reorder<boundarytype_ == DUMMY_BOUNDARY, BUFFER_DUMMY_VEL>
>
struct reorder_params
	: reorderPos
	, reorderVel
	, reorderVol
	, reorderEnergy
	, reorderBoundElements
	, reorderGradGamma
	, reorderVertices
	, reorderTKE
	, reorderEpsilon
	, reorderTurbVisc
	, reorderEffPres
	, reorderEulerVel
	, reorderNextID
	, reorderDummyVel
{
	reorder_params(BufferList& sorted_buffers, BufferList const& unsorted_buffers)
	: reorderPos(sorted_buffers, unsorted_buffers)
	, reorderVel(sorted_buffers, unsorted_buffers)
	, reorderVol(sorted_buffers, unsorted_buffers)
	, reorderEnergy(sorted_buffers, unsorted_buffers)
	, reorderBoundElements(sorted_buffers, unsorted_buffers)
	, reorderGradGamma(sorted_buffers, unsorted_buffers)
	, reorderVertices(sorted_buffers, unsorted_buffers)
	, reorderTKE(sorted_buffers, unsorted_buffers)
	, reorderEpsilon(sorted_buffers, unsorted_buffers)
	, reorderTurbVisc(sorted_buffers, unsorted_buffers)
	, reorderEffPres(sorted_buffers, unsorted_buffers)
	, reorderEulerVel(sorted_buffers, unsorted_buffers)
	, reorderNextID(sorted_buffers, unsorted_buffers)
	, reorderDummyVel(sorted_buffers, unsorted_buffers)
	{}

	__device__ __forceinline__ void reorder(const uint index, const uint sortedIndex, particleinfo const& info)
	{
	reorderPos::reorder(index, sortedIndex, info);
	reorderVel::reorder(index, sortedIndex, info);
	reorderVol::reorder(index, sortedIndex, info);
	reorderEnergy::reorder(index, sortedIndex, info);
	reorderBoundElements::reorder(index, sortedIndex, info);
	reorderGradGamma::reorder(index, sortedIndex, info);
	reorderVertices::reorder(index, sortedIndex, info);
	reorderTKE::reorder(index, sortedIndex, info);
	reorderEpsilon::reorder(index, sortedIndex, info);
	reorderTurbVisc::reorder(index, sortedIndex, info);
	reorderEffPres::reorder(index, sortedIndex, info);
	reorderEulerVel::reorder(index, sortedIndex, info);
	reorderNextID::reorder(index, sortedIndex, info);
	reorderDummyVel::reorder(index, sortedIndex, info);
	}
};

#endif
