/*  Copyright (c) 2013-2019 INGV, EDF, UniCT, JHU

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
 * Specializations of the Buffer class for CUDA device buffers
 */

#ifndef _CUDA_BUFFER_H
#define _CUDA_BUFFER_H

#if _DEBUG_
#include <iostream>
#endif

#include "buffer.h"

#include "define_buffers.h" // BUFFER_NEIBSLIT

#include "cache_preference.h" // DISABLE_TEXTURES

// SAFE_CALL etc
#include "safe_call.h"

/*! Texture object(s) associated with a CUDABuffer.
 * We associate a texture object (bindless texture) with each array
 * of a CUDABuffer (if possible). This can then be used instead of
 * the linear array in contexts where 1D texture access may be
 * preferrable (e.g. on hardware where the texture and L1 cache
 * are separate) without incurring in the runtime cost of
 * binding/unbinding the textures.
 */
template<flag_t Key, int N=BufferTraits<Key>::num_buffers>
class CUDABufferTexture
{
	using element_type = typename Buffer<Key>::element_type;

	// Channel descriptor associated to this buffer
	cudaChannelFormatDesc tex_channel_desc;
	cudaTextureDesc tex_desc;

	// Linear texture objects associated with the arrays of this buffer
	cudaResourceDesc tex_resource_desc[N];
	cudaTextureObject_t tex_obj[N];
public:
	CUDABufferTexture() :
		tex_channel_desc(cudaCreateChannelDesc<element_type>())
	{
		memset(&tex_desc, 0, sizeof(tex_desc));
		tex_desc.addressMode[0] = cudaAddressModeBorder;
		tex_desc.borderColor[0] = tex_desc.borderColor[1] =
		tex_desc.borderColor[2] = tex_desc.borderColor[3] = NAN;
		tex_desc.filterMode = cudaFilterModePoint;
		tex_desc.normalizedCoords = false;
		tex_desc.readMode = cudaReadModeElementType;
		memset(tex_resource_desc, 0, N*sizeof(cudaResourceDesc));
		memset(tex_obj, 0, N*sizeof(cudaTextureObject_t));
	}

	~CUDABufferTexture()
	{
		for (int i = 0 ; i < N; ++i)
		{
			if (tex_obj[i]) try {
				SAFE_CALL(cudaDestroyTextureObject(tex_obj[i]));
			} catch (std::exception const& e) {
				// nothing we can do here anyway
			}
			tex_obj[i] = 0;
		}
	}

	void create(int i, size_t bufmem, element_type *buf)
	{
		tex_resource_desc[i].resType = cudaResourceTypeLinear;
		tex_resource_desc[i].res.linear.desc = tex_channel_desc;
		tex_resource_desc[i].res.linear.devPtr = buf;
		tex_resource_desc[i].res.linear.sizeInBytes = bufmem;

		SAFE_CALL(cudaCreateTextureObject(tex_obj + i, tex_resource_desc + i, &tex_desc, NULL));
	}

	cudaTextureObject_t getTextureObject(int idx = 0) const
	{ return tex_obj[idx]; }
};

/*! CUDABufferTexture interface for buffers that do NOT have an associated texture object
 * This allows us to implement CUDABuffer with a simple conditional dependency,
 * and then using the same interfaces in both cases.
 */
template<flag_t Key>
class CUDABufferNoTexture
{
public:
	void create(...) { /* nothing to do in this case */ }

	cudaTextureObject_t getTextureObject(int idx = 0) const
	{ throw std::invalid_argument("no texture object association for " + std::string(Buffer<Key>::name)); }
};

/*! Implemenetation of the specialization of the Buffer class in the case of CUDA device allocations
 * (i.e. using cudaMalloc/cudaFree/cudaMemset/etc).
 *
 * This is separate from the public CUDABuffer because CUDABuffer must depend on a single
 * template parameter (the Key), whereas we (ab)use the template parameters to determine
 * whether or not we should consider the associated texture objects.
 */
template<flag_t Key,
	// texture objects will be associated only if the element type has a power-of-two size,
	// i.e. if its size has no bits in common with its preceding number.
	// We also exclude the neighbors list because it's huge, and may not fit the
	// maximum linear texture object size on some GPUs (and we never access it via textures anyway).
	size_t element_size_ = sizeof(typename Buffer<Key>::element_type),
	bool has_texture = (Key != BUFFER_NEIBSLIST) && !(element_size_ & (element_size_ - 1)), // quick check for pow2
	typename buffer_texture = typename std::conditional<has_texture && !DISABLE_TEXTURES,
		CUDABufferTexture<Key>, CUDABufferNoTexture<Key>>::type
>
class CUDABufferImplementation : public Buffer<Key>, public buffer_texture
{
	typedef Buffer<Key> baseclass;
public:
	typedef typename baseclass::element_type element_type;

	// constructor: initialize the channel and texture descs
	CUDABufferImplementation(int _init=-1) : Buffer<Key>(_init), buffer_texture() {}

	// destructor: free allocated memory
	virtual ~CUDABufferImplementation() {
		const int N = baseclass::array_count;
		element_type **bufs = baseclass::get_raw_ptr();
		for (int i = 0; i < N; ++i) {
#if _DEBUG_
			//printf("\tfreeing buffer %d\n", i);
#endif
			if (bufs[i]) {
				try {
					SAFE_CALL(cudaFree(bufs[i]));
				} catch (std::exception const& e) {
#if _DEBUG_
					std::cerr << e.what() <<
						" [while freeing buffer " << Key << ":" << i << " ("
						<< BufferTraits<Key>::name << ") @ 0x" << std::hex << bufs[i]
						<< "]" << std::endl;
#endif
				}
				bufs[i] = NULL;
			}
		}
	}

	virtual void clobber() {
		const size_t bufmem = AbstractBuffer::get_allocated_elements()*sizeof(element_type);
		const int N = baseclass::array_count;
		element_type **bufs = baseclass::get_raw_ptr();
		for (int i = 0; i < N; ++i) {
			SAFE_CALL(cudaMemset(bufs[i], baseclass::get_init_value(), bufmem));
		}
	}


	// allocate and clear buffer on device
	virtual size_t alloc(size_t elems) {
		AbstractBuffer::set_allocated_elements(elems);
		const size_t bufmem = elems*sizeof(element_type);
		const int N = baseclass::array_count;
		element_type **bufs = baseclass::get_raw_ptr();
		for (int i = 0; i < N; ++i) {
#ifdef INSPECT_DEVICE_MEMORY
			// If device memory inspection (from host) is enabled,
			// the device buffers are allocated in managed mode,
			// which makes them accessible on host via the same pointer
			// as the device.
			// TODO explore the possibility to make this the default,
			// assessing the performance impact and the hardware
			// and software (esp. CUDA version) requirements.
			SAFE_CALL(cudaMallocManaged(bufs + i, bufmem));
#else
			SAFE_CALL(cudaMalloc(bufs + i, bufmem));
#endif
			SAFE_CALL(cudaMemset(bufs[i], baseclass::get_init_value(), bufmem));

			buffer_texture::create(i, bufmem, bufs[i]);
		}
		return bufmem*N;
	}

	// swap elements at position idx1, idx2 of buffer _buf
	virtual void swap_elements(uint idx1, uint idx2, uint _buf=0) {
		element_type tmp;
		SAFE_CALL(cudaMemcpy(&tmp, this->get_offset_buffer(_buf, idx1), sizeof(element_type),
				cudaMemcpyDeviceToHost));
		SAFE_CALL(cudaMemcpy(
				this->get_offset_buffer(_buf, idx1),
				this->get_offset_buffer(_buf, idx2),
				sizeof(element_type), cudaMemcpyDeviceToDevice));
		SAFE_CALL(cudaMemcpy(this->get_offset_buffer(_buf, idx2), &tmp , sizeof(element_type),
				cudaMemcpyHostToDevice));
	}


	virtual const char* get_buffer_class() const
	{ return "CUDABuffer"; }
};

//! "User-facing” CUDABuffer, that depends on the single Key template parameter
template<flag_t Key>
using CUDABuffer = CUDABufferImplementation<Key>;

//! A function to access the buffer objects of a CUDABuffer directly from a BufferList
template<flag_t Key>
cudaTextureObject_t getTextureObject(const BufferList& list, const uint idx=0)
{
	auto buf = list.get<Key>();
	if (!buf) throw std::runtime_error("no buffer " + std::string(BufferTraits<Key>::name));
	return std::dynamic_pointer_cast<const CUDABuffer<Key>>(buf)->getTextureObject(idx);
}

#endif
