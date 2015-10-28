/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _CUDA_BUFFER_H
#define _CUDA_BUFFER_H

#if _DEBUG_
#include <iostream>
#endif

#include "buffer.h"

/* Specializations of the buffer class that reside on
 * a CUDA device */

// CUDA_SAFE_CALL etc
#include "cuda_call.h"

// a specialization of buffers, with CUDA allocation and free
template<flag_t Key>
class CUDABuffer : public Buffer<Key>
{
	typedef Buffer<Key> baseclass;
public:
	typedef typename baseclass::element_type element_type;

	// constructor: nothing to do
	CUDABuffer(int _init = 0) : Buffer<Key>(_init) {}

	// destructor: free allocated memory
	virtual ~CUDABuffer() {
		const int N = baseclass::array_count;
		element_type **bufs = baseclass::get_raw_ptr();
		for (int i = 0; i < N; ++i) {
#if _DEBUG_
			//printf("\tfreeing buffer %d\n", i);
#endif
			if (bufs[i]) {
				try {
					CUDA_SAFE_CALL(cudaFree(bufs[i]));
				} catch (std::exception &e) {
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

	// allocate and clear buffer on device
	virtual size_t alloc(size_t elems) {
		size_t bufmem = elems*sizeof(element_type);
		const int N = baseclass::array_count;
		element_type **bufs = baseclass::get_raw_ptr();
		for (int i = 0; i < N; ++i) {
			CUDA_SAFE_CALL(cudaMalloc(bufs + i, bufmem));
			CUDA_SAFE_CALL(cudaMemset(bufs[i], baseclass::get_init_value(), bufmem));
		}
		return bufmem*N;
	}

	// swap elements at position idx1, idx2 of buffer _buf
	virtual void swap_elements(uint idx1, uint idx2, uint _buf=0) {
		element_type tmp;
		CUDA_SAFE_CALL(cudaMemcpy(&tmp, this->get_offset_buffer(_buf, idx1), sizeof(element_type),
				cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(
				this->get_offset_buffer(_buf, idx1),
				this->get_offset_buffer(_buf, idx2),
				sizeof(element_type), cudaMemcpyDeviceToDevice));
		CUDA_SAFE_CALL(cudaMemcpy(this->get_offset_buffer(_buf, idx2), &tmp , sizeof(element_type),
				cudaMemcpyHostToDevice));
	}

};

#endif
