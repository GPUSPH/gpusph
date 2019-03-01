/*  Copyright 2019 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef COMMON_PARAMS_H
#define COMMON_PARAMS_H

/* \file
 *
 * With many kernels adopting the conditional structure template for parameters, we want
 * to simplify their usage by refactoring common substructures and avoid duplicating the initialization
 * sequences.
 *
 * This file collects params structure designed to be used as (sub)components of the actual kernel params
 * structures.
 */

#include "cond_params.h"

/*! \ingroup Common integration structures
 *
 * These are structures that hold two copies (old and new) of the same array. The first is assumed
 * to always be constant, whereas the second can be const or not, depending on a boolean template parameter.
 * @{
 */

template<bool B, typename T>
using writable_type = typename std::conditional<B, T, const T>::type;

/* Since they all have the same structure, we use a macro to define them */

#define DEFINE_PAIR_PARAM(type, member, buffer) \
template<bool writable_new = true> struct member ## _params { \
	using newType = writable_type<writable_new, type>; \
	using writeBufType = writable_type<writable_new, BufferList>; \
	const type * __restrict__ old ## member; \
	newType * __restrict__ new ## member; \
	member ## _params(BufferList const& bufread, \
		writeBufType& bufwrite) \
	: \
		old ## member(bufread.getData<buffer>()), \
		new ## member(bufwrite.template getData<buffer>()) \
	{} \
	member ## _params(member ## _params const&) = default; \
}

// Pos_params oldPos, newPos
DEFINE_PAIR_PARAM(float4, Pos, BUFFER_POS);

// Vel_params oldVel, newVel
DEFINE_PAIR_PARAM(float4, Vel, BUFFER_VEL);

// EulerVel_params oldEulerVel, newEulerVel
DEFINE_PAIR_PARAM(float4, EulerVel, BUFFER_EULERVEL);

// Vol_params oldVol, newVol
DEFINE_PAIR_PARAM(float4, Vol, BUFFER_VOLUME);

// BoundElement_params oldBoundElement, newBoundElement
DEFINE_PAIR_PARAM(float4, BoundElement, BUFFER_BOUNDELEMENTS);

// gGam_params oldgGam, newgGam
DEFINE_PAIR_PARAM(float4, gGam, BUFFER_GRADGAMMA);

// Eps_params oldEps, newEps
DEFINE_PAIR_PARAM(float, Eps, BUFFER_EPSILON);

// TKE_params oldTKE, newTKE
DEFINE_PAIR_PARAM(float, TKE, BUFFER_TKE);

// Energy_params oldEnergy, newEnergy
DEFINE_PAIR_PARAM(float, Energy, BUFFER_INTERNAL_ENERGY);
/*! @} */

template<bool writable = true>
struct vertPos_params
{
	using type = writable_type<writable, float2>;
	using src_ptr_type = typename std::conditional<writable,
		float2 **, const float2 * const *>::type;
	using src_buf_type = writable_type<writable, BufferList>;

	type* __restrict__ vertPos0;
	type* __restrict__ vertPos1;
	type* __restrict__ vertPos2;

	vertPos_params(src_ptr_type vertPos_ptr) :
		vertPos0(vertPos_ptr[0]),
		vertPos1(vertPos_ptr[1]),
		vertPos2(vertPos_ptr[2])
	{}

	vertPos_params(src_buf_type& bufread) :
		vertPos_params(bufread.template getRawPtr<BUFFER_VERTPOS>())
	{}

	vertPos_params(vertPos_params const&) = default;
};

#endif
