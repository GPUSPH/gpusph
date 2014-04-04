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

/* Base class for the Buffer template class.
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
	virtual ~AbstractBuffer() {}

	// element size of the arrays
	// overloaded in subclasses
	virtual size_t get_element_size() const
	{ return 0; }

	// number of arrays
	// overloaded in subclasses
	virtual uint get_array_count() const
	{ return 0; }

	virtual const char* get_buffer_name() const
	{
		throw std::runtime_error("AbstractBuffer name queried");
	}

	// allocate buffer and return total amount of memory allocated
	virtual size_t alloc(size_t elems) {
		throw std::runtime_error("cannot allocate generic buffer");
	}

	// base method to return a specific buffer of the array
	// WARNING: this doesn't check for validity of idx.
	// We have both const and non-const version
	virtual void *get_buffer(uint idx=0) {
		return m_ptr ? m_ptr[idx] : NULL;
	}
	virtual const void *get_buffer(uint idx=0) const {
		return m_ptr ? m_ptr[idx] : NULL;
	}

	// as above, plus offset
	virtual void *get_offset_buffer(uint idx, size_t offset) {
		throw runtime_error("can't determine buffer offset in AbstractBuffer");
	}
	virtual const void *get_offset_buffer(uint idx, size_t offset) const {
		throw runtime_error("can't determine buffer offset in AbstractBuffer");
	}

	// swap elements at positions idx1, idx2 of buffer _buf
	virtual void swap_elements(uint idx1, uint idx2, uint _buf=0) {
		throw runtime_error("can't swap elements in AbstractBuffer");
	};
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

protected:
	enum { array_count = N };

	// accessors to the raw pointers. used by specializations
	// to easily handle allocs and frees
	virtual T** get_raw_ptr()
	{ return m_bufs; }
	virtual const T* const* get_raw_ptr() const
	{ return m_bufs; }

public:
	typedef T element_type;

	// constructor: ensure all buffers are NULL, set the init value
	GenericBuffer(int _init = 0) : AbstractBuffer((void**)m_bufs) {
		m_init = _init;
		for (int i = 0; i < N; ++i)
			m_bufs[i] = NULL;
	}

	virtual ~GenericBuffer() {} ;

	// return an (untyped) pointer to the idx buffer,
	// if valid. Must return void since we overload
	// AbstractBuffer methods.
	virtual void *get_buffer(uint idx=0) {
		if (idx >= N) return NULL;
		return m_bufs[idx];
	}
	virtual const void *get_buffer(uint idx=0) const {
		if (idx >= N) return NULL;
		return m_bufs[idx];
	}

	// As above, plus offset. Useful since we return void
	// and computing the offset from the result of get_buffer()
	// would require manual casting to appropiate type and/or
	// manual multiplication by the element size
	virtual void *get_offset_buffer(uint idx, size_t offset) {
		if (idx >= N || !m_bufs[idx]) return NULL;
		return m_bufs[idx] + offset;
	}
	virtual const void *get_offset_buffer(uint idx, size_t offset) const {
		if (idx >= N || !m_bufs[idx]) return NULL;
		return m_bufs[idx] + offset;
	}

	// return the initialization value set at construction time
	virtual int get_init_value() const
	{ return m_init; }

	virtual size_t get_element_size() const
	{ return sizeof(T); }

	virtual uint get_array_count() const
	{ return array_count; }
};

/* We access buffers mostly by key (BUFFER_POS etc), so our actual Buffer class is templatized on
 * that (which is a flag_t).
 * Buffer<Key> is just a thin wrapper aroung GenericBuffer<Type, Num>, that
 * exploits the BufferTraits traits scheme to know the element type and number
 * of buffers of each specific instantiation.
 * The implementation is straightforward, as we don't need to overload any method over GenericBuffer:
 * we just add a method to get the printable name associated with the given Key.
 */

// Forward-declare the BufferList class, which we want to be friend with the
// Buffer classes so that it may access the protected get_raw_ptr() methods.
class BufferList;

// since some people find this particular aspect of the C++ syntax a bit too ugly,
// let's introduce a de-uglifying macro that just returns the expected data type for
// elements in array Key (e.g. float4 for BUFFER_POS):
#define DATA_TYPE(Key) typename BufferTraits<Key>::element_type

template<flag_t Key>
class Buffer : public GenericBuffer<DATA_TYPE(Key), BufferTraits<Key>::num_buffers>
{
	// we want to use baseclass as a shortcut for the parent class of Buffer<Key>
	typedef GenericBuffer<DATA_TYPE(Key), BufferTraits<Key>::num_buffers> baseclass;

	friend class BufferList;
public:

	// constructor, specifying the memset initializer
	Buffer(int _init=0) : baseclass(_init) {}

	virtual ~Buffer() {
#if _DEBUG_
		//printf("destroying %s\n", get_buffer_name());
#endif
	}

	// get the name for buffer Key from the buffer traits
	virtual const char* get_buffer_name() const
	{
		return BufferTraits<Key>::name;
	}
};

/* Specialized list of Buffers of any type. It is implemented as a map
 * instead of an actual list to allow non-consecutive keys as well as
 * to allow limiting iterations to Buffers actually in use.
 *
 * Entries should be added to the list using the << operator:
 *
 *     BufferList bufs;
 *     bufs << new Buffer<BUFFER_POS>();
 *     bufs << new Buffer<BUFFER_VEL>();
 *
 * Three getters are defined, to work around the limitations of C++
 * overloading and subclassing:
 *
 *  * the standard [] operator, which returns an AbstractBuffer*
 *  * .get<Key>(), that returns a Buffer<Key>*
 *  * .getData<Key>(num), that returns array number _num_ of Buffer<Key>
 *    (array 0 by default), cast to the correct datatype for Buffer<Key>
 *    (e.g. it returns a float4* for Buffer<BUFFER_POS>).
 *
 *  Note that for get() and getData() the Key (e.g. BUFFER_POS) is passed
 *  as template parameter, not as argument, and _must_ be known at compile
 *  (this is necessary because the return type must be known at compile time
 *  too).
 */
class BufferList : public std::map<flag_t, AbstractBuffer*>
{
	typedef std::map<flag_t, AbstractBuffer*> baseclass;

public:
	BufferList() : baseclass() {};

	// delete all buffers before clearing the hash
	void clear() {
		iterator buf = this->begin();
		const iterator done = this->end();
		while (buf != done) {
			delete buf->second;
			++buf;
		}
		baseclass::clear();
	}

	/* We overload the [] accessor, to guarantee const correctness
	 * and to prevent insertion of buffers through assignment to
	 * specific keys. Use the << operator to fill the buffer list instead.
	 */
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

	/* Templatized getter to allow the user to access the Buffers in the list
	 * with proper typing (so e.g .get<BUFFER_POS>() will return a
	 * Buffer<BUFFER_POS>* instead of an AbstractBuffer*)
	 */
	template<flag_t Key>
	Buffer<Key> *get() {
		iterator exists = this->find(Key);
		if (exists != this->end())
			return static_cast<Buffer<Key>*>(exists->second);
		else return NULL;
	}
	// const version
	template<flag_t Key>
	const Buffer<Key> *get() const {
		const_iterator exists = this->find(Key);
		if (exists != this->end())
			return static_cast<const Buffer<Key>*>(exists->second);
		else return NULL;
	}


	/* In most cases, user wants access directly to a specific array of a given buffer
	 * possibly with correct typing, which is exactly what this method does,
	 * again exploiting the BufferTraits to deduce the element type that is
	 * expected in return.
	 * The static cast is necessary because the get_buffer() method must return a void*
	 * due to overloading rules.
	 */
	template<flag_t Key>
	DATA_TYPE(Key) *getData(uint num=0) {
		iterator exists = this->find(Key);
		if (exists != this->end())
			return static_cast<DATA_TYPE(Key)*>(exists->second->get_buffer(num));
		else return NULL;
	}
	// const version
	template<flag_t Key>
	const DATA_TYPE(Key) *getData(uint num=0) const {
		const_iterator exists = this->find(Key);
		if (exists != this->end())
			return static_cast<const DATA_TYPE(Key)*>(exists->second->get_buffer(num));
		else return NULL;
	}

	/* In a few cases, the user may want to access the base pointer to
	 * multi-buffer arrays. This could be done using get<Key>()->get_raw_ptr,
	 * but this causes a segmentation fault when the Key array isn't actually
	 * allocated, while we really want to return NULL in this case. Wrap it up
	 * for safety.
	 */
	template<flag_t Key>
	DATA_TYPE(Key) **getRawPtr() {
		Buffer<Key> *exists = this->get<Key>();
		if (exists)
			return exists->get_raw_ptr();
		else return NULL;
	}
	// const version
	template<flag_t Key>
	const DATA_TYPE(Key)* const* getRawPtr() const {
		Buffer<Key> *exists = this->get<Key>();
		if (exists)
			return exists->get_raw_ptr();
		else return NULL;
	}


	/* Add a new array for position Key, with the provided initalization value
	 * as initializer. The position is automatically deduced by the Key of the
	 * buffer.
	 */
	template<flag_t Key>
	BufferList& operator<< (Buffer<Key> * buf) {
		iterator exists = find(Key);
		if (exists != this->end()) {
			throw runtime_error("trying to add a buffer for an already-available key!");
		} else {
			baseclass::operator[](Key) = buf;
		}
		return *this;
	}

};

#undef DATA_TYPE

#endif

