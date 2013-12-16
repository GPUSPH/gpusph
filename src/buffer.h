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

#ifndef _BUFFER_H
#define _BUFFER_H

#include <map>

#include <stdexcept>

#include "common_types.h"
#include "buffer_traits.h"

/* BIG FIXME TODO
 * The AbstractBuffer (and related classes) should be templatized
 * over the kind of Worker. The device-specific stuff (alloc, release, copy)
 * should be moved into Worker-specific specializations
 *
 * buffer.h itself should NOT depend on CUDA or anything like that
 * Relevant code parts are marked FIXME TODO CUDA
 *
 * This can wait until different kinds of worker are actually implemented.
 */
// CUDA_SAFE_CALL etc
#include "cuda_call.h"


#if 0
// class to allow covariance between void* and other pointers:
// would allow get_buffer to do typed returns in the
// non-generic Buffer instances
struct PtrCasterBase {
	void *ptr;
};

template<typename T>
struct PtrCaster : PtrCasterBase {
	typedef T type;
};
#endif


/* Base class for the array template class.
 * The base pointer is a pointer to pointer to allow easy management
 * of double-(or more)-buffered arrays.
 */
class AbstractBuffer
{
	void **m_ptr;

protected:
	// constructor that aliases m_ptr to some array of pointers
	AbstractBuffer(void *bufs[]) { m_ptr = bufs; }

public:

	// default constructor: just ensure ptr is null
	AbstractBuffer() : m_ptr(NULL) {}

	// destructor must be virtual
	virtual ~AbstractBuffer() { }

	// element size of the arrays
	// overloaded in subclasses
	virtual size_t get_element_size() const
	{ return 0; }

	// number of arrays
	// overloaded in subclasses
	virtual int get_array_count() const
	{ return 0; }

	virtual const char* get_buffer_name() const
	{
		throw std::runtime_error("AbstractBuffer name queried");
	}

	// allocate buffer on device and return total amount of memory allocated
	virtual size_t device_alloc(size_t elems) {
		throw std::runtime_error("cannot allocate generic buffer");
	}

	// base method to return a specific buffer of the array
	// WARNING: this doesn't check for validity of idx.
	// We have both const and non-const version
	virtual void *get_buffer(uint idx) {
		return m_ptr ? m_ptr[idx] : NULL;
	}
	virtual const void *get_buffer(uint idx) const {
		return m_ptr ? m_ptr[idx] : NULL;
	}

	// as above, plus offset
	virtual void *get_offset_buffer(uint idx, size_t offset) {
		throw runtime_error("can't determine buffer offset in AbstractBuffer");
	}
	virtual const void *get_offset_buffer(uint idx, size_t offset) const {
		throw runtime_error("can't determine buffer offset in AbstractBuffer");
	}

};

/* This class encapsulates type-specific arrays of buffers. 
 * By default the array will have a single buffer, with trivial
 * extensions to N-buffers.
 */
template<typename T, int N=1>
class GenericBuffer : public AbstractBuffer
{
	T *m_bufs[N];

	// initialization value for the arrays
	// NOTE that this is an int, not a T, because initalization
	// is done with a memset
	int m_init;
public:
	typedef T element_type;

	// constructor: ensure all buffers are NULL, set the init value
	GenericBuffer(int _init = 0) : AbstractBuffer((void**)m_bufs) {
		m_init = _init;
		for (int i = 0; i < N; ++i)
			m_bufs[i] = NULL;
	}

	// destructor: free allocated memory
	// FIXME TODO CUDA
	virtual ~GenericBuffer() {
		for (int i = 0; i < N; ++i) {
#if _DEBUG_
			printf("\tfreeing buffer %d\n", i);
#endif
			if (m_bufs[i]) {
				CUDA_SAFE_CALL(cudaFree(m_bufs[i]));
				m_bufs[i] = NULL;
			}
		}
	}

	// allocate and clear buffer on device
	// FIXME TODO CUDA
	virtual size_t device_alloc(size_t elems) {
		size_t bufmem = elems*sizeof(T);
		for (int i = 0; i < N; ++i) {
			CUDA_SAFE_CALL(cudaMalloc(m_bufs + i, bufmem));
			CUDA_SAFE_CALL(cudaMemset(m_bufs[i], m_init, bufmem));
		}
		return bufmem*N;
	}

	// return the actual array of pointers
	// needed to easily handled the TAU buffer, look for possible alternatives?
	T** get_raw_ptr()
	{ return m_bufs; }

	const T* const* get_raw_ptr() const
	{ return m_bufs; }


	// return an (untyped) pointer to the idx buffer,
	// if valid. Both const and non-const version
	virtual void *get_buffer(uint idx) {
		if (idx >= N) return NULL;
		return m_bufs[idx];
	}
	virtual const void *get_buffer(uint idx) const {
		if (idx >= N) return NULL;
		return m_bufs[idx];
	}

	// as above, plus offset
	virtual void *get_offset_buffer(uint idx, size_t offset) {
		if (idx >= N) return NULL;
		return m_bufs[idx] + offset;
	}
	virtual const void *get_offset_buffer(uint idx, size_t offset) const {
		if (idx >= N) return NULL;
		return m_bufs[idx] + offset;
	}

	virtual size_t get_element_size() const
	{ return sizeof(T); }

	virtual int get_array_count() const
	{ return N; }
};

/* We access buffers mostly by Key, and thanks to the traits scheme we can
 * produce a very cleanly accessible implementation
 */
template<flag_t Key>
class Buffer : public GenericBuffer<typename BufferTraits<Key>::type, BufferTraits<Key>::nbufs>
{
	typedef GenericBuffer<typename BufferTraits<Key>::type, BufferTraits<Key>::nbufs> baseclass;
public:
	typedef typename baseclass::element_type element_type;

	// constructor, specifying the memset initializer
	Buffer(int _init=0) : baseclass(_init) {}

	virtual ~Buffer() {
#if _DEBUG_
		printf("destroying %s\n", BufferTraits<Key>::name);
#endif
	}

	virtual const char* get_buffer_name() const
	{
		return BufferTraits<Key>::name;
	}
};

// convenience type: list of arrays.
// implemented as map instead of an actual list to allow non-consecutive keys
class BufferList : public std::map<flag_t, AbstractBuffer*>
{
	typedef std::map<flag_t, AbstractBuffer*> baseclass;
public:
	BufferList() : baseclass() {};

	void clear() {
		iterator buf = this->begin();
		const iterator done = this->end();
		while (buf != done) {
			delete buf->second;
			++buf;
		}
		baseclass::clear();
	}

	/* Gain access to the buffer Key with proper typing */
	template<flag_t Key>
	Buffer<Key> *getBuffer() {
		iterator exists = this->find(Key);
		if (exists != this->end())
			return static_cast<Buffer<Key>*>(exists->second);
		else return NULL;
	}

	/* Gain access to the num-th array in the Key buffer from the BufferList
	 * Returns a direct pointer to the array data, with the appropriate data.
	 */
	template<flag_t Key>
	typename Buffer<Key>::element_type *getBufferData(uint num=0) {
		iterator exists = this->find(Key);
		if (exists != this->end())
			return static_cast<typename Buffer<Key>::element_type*>(exists->second->get_buffer(num));
		else return NULL;
	}

	AbstractBuffer* operator[](const flag_t& Key) {
		const_iterator exists = this->find(Key);
		if (exists != this->end())
			return exists->second;
		else return NULL;
	}

	const AbstractBuffer* operator[](const flag_t& Key) const {
		const_iterator exists = this->find(Key);
		if (exists != this->end())
			return exists->second;
		else return NULL;
	}

	/* Add a new array for position Key, with the provided initalization value
	 * as initializer
	 */
	template<uint Key>
	Buffer<Key> *add(int _init = 0) {
		Buffer<Key> *ret;
		iterator exists = find(Key);
		if (exists != this->end()) {
			// TODO warn about double insertion?
			ret = static_cast<Buffer<Key>*>(exists->second);
		} else {
			ret = new Buffer<Key>(_init);
			baseclass::operator[](Key) = ret;
		}
		return ret;
	}

};

#endif

