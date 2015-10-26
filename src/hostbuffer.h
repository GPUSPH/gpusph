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

/* Specializations of the buffer class for host buffers */

// malloc
#include <cstdlib>
// memset
#include <cstring>

// swap
#include <algorithm>

#include "buffer.h"

/* a specialization of buffers, for the host */

template<flag_t Key>
class HostBuffer : public Buffer<Key>
{
	typedef Buffer<Key> baseclass;
public:
	typedef typename baseclass::element_type element_type;

	// constructor: nothing to do
	HostBuffer(int _init = 0) : Buffer<Key>(_init) {}

	// destructor: free allocated memory
	virtual ~HostBuffer() {
		const int N = baseclass::array_count; // see NOTE for this class
		element_type **bufs = baseclass::get_raw_ptr();
		for (int i = 0; i < N; ++i) {
#if _DEBUG_
			//printf("\tfreeing buffer %d\n", i);
#endif
			if (bufs[i]) {
				free(bufs[i]);
				bufs[i] = NULL;
			}
		}
	}

	// allocate and clear buffer on device
	virtual size_t alloc(size_t elems) {
		size_t bufmem = elems*sizeof(element_type);
		const int N = baseclass::array_count; // see NOTE for this class
		element_type **bufs = baseclass::get_raw_ptr();
		for (int i = 0; i < N; ++i) {
			// malloc instead of calloc since the init
			// value might be nonzero
			bufs[i] = (element_type*)malloc(bufmem);
			memset(bufs[i], baseclass::get_init_value(), bufmem);
		}
		return bufmem*N;
	}

	virtual void swap_elements(uint idx1, uint idx2, uint _buf=0) {
		element_type *buf = baseclass::get_raw_ptr()[_buf];
		std::swap(buf[idx1], buf[idx2]);
	}

};

#endif
