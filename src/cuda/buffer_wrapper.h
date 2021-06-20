/*  Copyright (c) 2021 INGV, EDF, UniCT, JHU

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

/*! \file texture_wrapper.h
  Define macros that can be used to quickly define wrapper for texture objects,
  with the associated .fetchSomething */

#ifndef BUFFER_WRAPPER_H
#define BUFFER_WRAPPER_H

#include "buffer_traits.h"

/*** Texture wrappers ***/

/*! This macro defines a struct named name to access the elements of key KEY
    through a texture.
  */
#define DEFINE_BUFFER_WRAPPER_TEXTURE(name, KEY, base, Base) \
struct name \
{ \
private: \
	using element_type = typename BufferTraits<KEY>::element_type; \
	cudaTextureObject_t base##TexObj; \
public: \
	name(BufferList const& bufread) : base##TexObj(getTextureObject<KEY>(bufread)) {} \
\
	__device__ __forceinline__ \
	element_type fetch##Base(const uint index) const \
	{ return tex1Dfetch<element_type>(base##TexObj, index); } \
}

/*! This macro defines a struct named name to access the elements of key KEY
 *  through a texture, assuming KEY defines a tensor
 *  (stored as 3 float2, recovered as a symtensor3).
 */
#define DEFINE_TENSOR_WRAPPER_TEXTURE(name, KEY, base, Base) \
struct name \
{ \
private: \
	using element_type = typename BufferTraits<KEY>::element_type; \
	using return_type = symtensor3; \
	cudaTextureObject_t base##0TexObj; \
	cudaTextureObject_t base##1TexObj; \
	cudaTextureObject_t base##2TexObj; \
public: \
	name(BufferList const& bufread) : \
		base##0TexObj(getTextureObject<KEY>(bufread, 0)), \
		base##1TexObj(getTextureObject<KEY>(bufread, 1)), \
		base##2TexObj(getTextureObject<KEY>(bufread, 2))  \
	{} \
\
	__device__ __forceinline__ \
	return_type fetch##Base(const uint index) const \
	{ \
		return_type base; \
		element_type temp = tex1Dfetch<element_type>(base##0TexObj, index); \
		base.xx = temp.x; \
		base.xy = temp.y; \
		temp = tex1Dfetch<element_type>(base##1TexObj, index); \
		base.xz = temp.x; \
		base.yy = temp.y; \
		temp = tex1Dfetch<element_type>(base##2TexObj, index); \
		base.yz = temp.x; \
		base.zz = temp.y; \
		return base; \
	} \
}

/*** Linear array wrappers ***/

/*! This macro defines a a struct named name to access the elements of key KEY
    through a linear array.
  */
#define DEFINE_BUFFER_WRAPPER_ARRAY(name, KEY, base, Base) \
struct name \
{ \
private: \
	using element_type = typename BufferTraits<KEY>::element_type; \
	const element_type * __restrict__  base##Array; \
public: \
	name(BufferList const& bufread) : base##Array(bufread.getData<KEY>()) {} \
\
	__device__ __forceinline__ \
	element_type fetch##Base(const uint index) const \
	{ return base##Array[index]; } \
}

/*! This macro defines a struct named name to access the elements of key KEY
 *  through a linear array, assuming KEY defines a tensor
 *  (stored as 3 float2, recovered as a symtensor3).
 *  \note the constructor is two-part because we must first extract the
 *  double-pointer with getRawPtr, and then extract each component array
 *  from this.
 */
#define DEFINE_TENSOR_WRAPPER_ARRAY(name, KEY, base, Base) \
struct name \
{ \
private: \
	using element_type = typename BufferTraits<KEY>::element_type; \
	using return_type = symtensor3; \
	using src_ptr_type = const element_type * const * ; \
\
	const element_type * __restrict__ base##0; \
	const element_type * __restrict__ base##1; \
	const element_type * __restrict__ base##2; \
public: \
	name(src_ptr_type base##_ptr) : \
		base##0(base##_ptr[0]), \
		base##1(base##_ptr[1]), \
		base##2(base##_ptr[2])  \
	{} \
\
	name(BufferList const& bufread) : \
		name(bufread.template getRawPtr<KEY>()) \
	{} \
\
	__device__ __forceinline__ \
	return_type fetch##Base(const uint index) const \
	{ \
		return_type base; \
		element_type temp = base##0[index]; \
		base.xx = temp.x; \
		base.xy = temp.y; \
		temp = base##1[index]; \
		base.xz = temp.x; \
		base.yy = temp.y; \
		temp = base##2[index]; \
		base.yz = temp.x; \
		base.zz = temp.y; \
		return base; \
	} \
}


/*** Standard wrappers: default to texture wrapper, can be disabled by setting DISABLE_TEXTURES ***/

#if DISABLE_TEXTURES
#define DEFINE_BUFFER_WRAPPER(...) DEFINE_BUFFER_WRAPPER_ARRAY(__VA_ARGS__)
#define DEFINE_TENSOR_WRAPPER(...) DEFINE_TENSOR_WRAPPER_ARRAY(__VA_ARGS__)
#else
#define DEFINE_BUFFER_WRAPPER(...) DEFINE_BUFFER_WRAPPER_TEXTURE(__VA_ARGS__)
#define DEFINE_TENSOR_WRAPPER(...) DEFINE_TENSOR_WRAPPER_TEXTURE(__VA_ARGS__)
#endif


#endif /* BUFFER_WRAPPER_H */
