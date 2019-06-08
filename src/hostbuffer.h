/*  Copyright (c) 2013-2018 INGV, EDF, UniCT, JHU

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

#ifndef _HOST_BUFFER_H
#define _HOST_BUFFER_H

/*! \file
 * Specializations of the Buffer class for host buffers
 */

// malloc
#include <cstdlib>
// memset
#include <cstring>

// swap
#include <algorithm>

#include "buffer.h"

/*! Specialize the Buffer class in the case of host allocations
 * (i.e. using malloc/free/memset/etc)
 */
template<flag_t Key>
class HostBuffer : public Buffer<Key>
{
	typedef Buffer<Key> baseclass;
public:
	typedef typename baseclass::element_type element_type;

	// constructor: nothing to do
	HostBuffer(int _init=-1) : Buffer<Key>(_init) {}

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

	virtual void clobber() {
		const size_t bufmem = AbstractBuffer::get_allocated_elements()*sizeof(element_type);
		const int N = baseclass::array_count;
		element_type **bufs = baseclass::get_raw_ptr();
		for (int i = 0; i < N; ++i) {
			memset(bufs[i], baseclass::get_init_value(), bufmem);
		}
	}

	// allocate and clear buffer on device
	virtual size_t alloc(size_t elems) {
		AbstractBuffer::set_allocated_elements(elems);
		const size_t bufmem = elems*sizeof(element_type);
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

	virtual const char* get_buffer_class() const
	{ return "HostBuffer"; }
};

#endif
