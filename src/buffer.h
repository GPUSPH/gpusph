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

#include <vector>
#include <map>
#include <set>

#include <stdexcept>

#include "common_types.h"
#include "buffer_traits.h"
#include "buffer_alloc_policy.h"

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
	virtual size_t get_element_size() const = 0;

	// number of arrays
	// overloaded in subclasses
	virtual uint get_array_count() const = 0;

	virtual const char* get_buffer_name() const = 0;

	// allocate buffer and return total amount of memory allocated
	virtual size_t alloc(size_t elems) = 0;

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
	virtual void *get_offset_buffer(uint idx, size_t offset) = 0;
	virtual const void *get_offset_buffer(uint idx, size_t offset) const = 0;

	// swap elements at positions idx1, idx2 of buffer _buf
	virtual void swap_elements(uint idx1, uint idx2, uint _buf=0) = 0;
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
	// NOTE that this is an int, not a T, because initialization
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
	// would require manual casting to appropriate type and/or
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

// Forward declaration of the MultiBufferList, which we want to make friend
// with the BufferList
class MultiBufferList;

/* Specialized list of Buffers of any type. Internally it's implemented as a map
 * instead of an actual list to allow non-consecutive keys as well as
 * to allow limiting iterations to Buffers actually in use.
 *
 * Entries should be added to the list using the .addBuffer<>() template
 * function:
 *
 *     BufferList bufs;
 *     bufs.addBuffer<SomeBufferClass, BUFFER_POS>();
 *     bufs.addBuffer<SomeBufferClass, BUFFER_VEL>();
 *
 * The first template parameter is the (template) class of the Buffer
 * to add (e.g. HostBuffer, CUDABuffer, etc), the second is the key.
 * An optional initializer value for the array can be specified as
 * argument to the function (default to 0).
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
class BufferList
{
	typedef std::map<flag_t, AbstractBuffer*> map_type;

	map_type m_map;

protected:
	void addExistingBuffer(flag_t Key, AbstractBuffer* buf)
	{ m_map[Key] = buf; }

	// replace the buffer at position Key with buf, returning the
	// old one
	AbstractBuffer *replaceBuffer(flag_t Key, AbstractBuffer *buf)
	{
		AbstractBuffer *old = m_map[Key];
		m_map[Key] = buf;
		return old;
	}

	// remove a buffer without deallocating it
	// used by the MultiBufferList to remove buffers shared
	// by multiple lists
	// TODO this would be all oh so much better using C++11 shared_ptr ...
	void removeBuffer(flag_t Key)
	{ m_map.erase(Key); }


	friend class MultiBufferList;
public:
	BufferList() : m_map() {};

	~BufferList() {
		clear();
	}

	// delete all buffers before clearing the hash
	void clear() {
		map_type::iterator buf = m_map.begin();
		while (buf != m_map.end()) {
			delete buf->second;
			++buf;
		}
		m_map.clear();
	}

	/* Read-only [] accessor. Insertion of buffers should be done via the
	 * addBuffer<>() method template.
	 */
	AbstractBuffer* operator[](const flag_t& Key) {
		map_type::const_iterator exists = m_map.find(Key);
		if (exists != m_map.end())
			return exists->second;
		else return NULL;
	}
	const AbstractBuffer* operator[](const flag_t& Key) const {
		map_type::const_iterator exists = m_map.find(Key);
		if (exists != m_map.end())
			return exists->second;
		else return NULL;
	}

	/* Templatized getter to allow the user to access the Buffers in the list
	 * with proper typing (so e.g .get<BUFFER_POS>() will return a
	 * Buffer<BUFFER_POS>* instead of an AbstractBuffer*)
	 */
	template<flag_t Key>
	Buffer<Key> *get() {
		map_type::iterator exists = m_map.find(Key);
		if (exists != m_map.end())
			return static_cast<Buffer<Key>*>(exists->second);
		else return NULL;
	}
	// const version
	template<flag_t Key>
	const Buffer<Key> *get() const {
		map_type::const_iterator exists = m_map.find(Key);
		if (exists != m_map.end())
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
		map_type::iterator exists = m_map.find(Key);
		if (exists != m_map.end())
			return static_cast<DATA_TYPE(Key)*>(exists->second->get_buffer(num));
		else return NULL;
	}
	// const version
	template<flag_t Key>
	const DATA_TYPE(Key) *getData(uint num=0) const {
		map_type::const_iterator exists = m_map.find(Key);
		if (exists != m_map.end())
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
		const Buffer<Key> *exists = this->get<Key>();
		if (exists)
			return exists->get_raw_ptr();
		else return NULL;
	}


	/* Add a new buffer of the given BufferClass for position Key, with the provided
	 * initialization value as initializer. The position is automatically deduced by
	 * the Key of the buffer.
	 */
	template<template<flag_t> class BufferClass, flag_t Key>
	BufferList& addBuffer(int _init=0)
	{
		map_type::iterator exists = m_map.find(Key);
		if (exists != m_map.end()) {
			throw std::runtime_error("trying to add a buffer for an already-available key!");
		} else {
			m_map[Key] = new BufferClass<Key>(_init);
		}
		return *this;
	}


	/* map-like interface */
	// Add more methods/types here as needed

	typedef map_type::iterator iterator;
	typedef map_type::const_iterator const_iterator;
	typedef map_type::size_type size_type;

	iterator begin()
	{ return m_map.begin(); }

	const_iterator begin() const
	{ return m_map.begin(); }

	iterator end()
	{ return m_map.end(); }

	const_iterator end() const
	{ return m_map.end(); }

	size_type size() const
	{ return m_map.size(); }

};

/* A MultiBufferList takes into account that some of the buffers are needed
 * in multiple copies (double-buffered or more.) It relies on a BufferAllocPolicy
 * object to determine which ones need multiple copies and which ones
 * do not.
 *
 */
class MultiBufferList
{
public:
	// buffer allocation policy
	const BufferAllocPolicy *m_policy;

	// list of BufferLists
	std::vector<BufferList> m_lists;

	// iterators are returned by the getters
	typedef std::vector<BufferList>::iterator iterator;
	typedef std::vector<BufferList>::const_iterator const_iterator;

	// Keys of Buffers added so far
	// It's a set instead of a single flag_t to allow iteration on it
	// without bit-shuffling. Might change.
	std::set<flag_t> m_buffer_keys;

	// TODO FIXME this is for double-buffered lists only
	// In general we would have N writable list (N=1 usually)
	// and M read-only lists (M >= 1), with the number of each
	// determined by the BufferAllocPolicy.
#define READ_LIST 1
#define WRITE_LIST 0

public:

	MultiBufferList(): m_policy(NULL)
	{}

	~MultiBufferList() {
		clear();
	}

	void clear() {
		// nothing to do, if the policy was never set
		if (m_policy == NULL)
			return;

		// we cannot just clear() the lists, because that would
		// lead to a double-free of the shared pointers. In C++11
		// this could be fixed with shared_ptrs, but we can't rely
		// on C++11, so we have to do the management manually.

		// To avoid the double free, first do a manual deallocation
		// and removal of the shared buffers:
		std::set<flag_t>::const_iterator iter = m_buffer_keys.begin();
		const std::set<flag_t>::const_iterator end = m_buffer_keys.end();
		for ( ; iter != end ; ++iter) {
			const flag_t key = *iter;
			const size_t count = m_policy->get_buffer_count(key);
			if (count != 1)
				continue;
			// ok, the buffer had a count of 1, so it was not
			// double buffered, and is shared among lists:
			// we deallocated it ourselves, and remove it
			// from the lists, in order to avoid double deletions
			AbstractBuffer *buf = m_lists[0][key];

			iterator list = m_lists.begin();
			for ( ; list != m_lists.end(); ++list)
				list->removeBuffer(key);
			delete buf;
		}
		// now clear the lists
		m_lists.clear();
		// and purge the list of keys too
		m_buffer_keys.clear();
	}

	void setAllocPolicy(const BufferAllocPolicy* _policy)
	{
		if (m_policy != NULL)
			throw std::runtime_error("cannot change buffer allocation policy");
		m_policy = _policy;

		// add as many BufferLists as needed at most
		m_lists.resize(m_policy->get_max_buffer_count());
	}

	/* Add a new buffer of the given BufferClass for position Key, with the provided
	 * initialization value as initializer.
	 */
	template<template<flag_t> class BufferClass, flag_t Key>
	void addBuffer(int _init=0)
	{
		if (m_policy == NULL)
			throw std::runtime_error("trying to add buffers before setting policy");
		if (m_buffer_keys.find(Key) != m_buffer_keys.end())
			throw std::runtime_error("trying to re-add buffer");

		m_buffer_keys.insert(Key);

		// number of copies of this buffer
		const size_t count = m_policy->get_buffer_count(Key);

		if (count > 1) {
			// We currently support only two possibilities for buffers:
			// either there is a single instance, or there are as many instances
			// as the maximum (e.g. if the alloc policy is triple-buffered,
			// then a buffer has a count of either three or one)
			// TODO redesign as appropriate when the need arises
			if (count != m_lists.size())
				throw std::runtime_error("buffer count less than max but bigger than 1 not supported");
			// multi-buffered, allocate one instance in each buffer list
			iterator it(m_lists.begin());
			while (it != m_lists.end()) {
				it->addBuffer<BufferClass, Key>(_init);
				++it;
			}
		} else {
			// single-buffered, allocate once and put in all lists
			AbstractBuffer *buff = new BufferClass<Key>;

			iterator it(m_lists.begin());
			while (it != m_lists.end()) {
				it->addExistingBuffer(Key, buff);
				++it;
			}
		}
	}

	/* Swap the lists the given buffers belong to */
	// TODO make this a cyclic rotation for the case > 2
	void swapBuffers(flag_t keys) {
		std::set<flag_t>::const_iterator iter = m_buffer_keys.begin();
		const std::set<flag_t>::const_iterator end = m_buffer_keys.end();
		for (; iter != end ; ++iter) {
			const flag_t key = *iter;
			if (!(key & keys))
				continue;

			// get the old READ buffer, replace the one in WRITE
			// with it and the one in READ with the old WRITE one
			AbstractBuffer *oldread = m_lists[READ_LIST][key];
			AbstractBuffer *oldwrite = m_lists[WRITE_LIST].replaceBuffer(key, oldread);
			m_lists[READ_LIST].replaceBuffer(key, oldwrite);
		}

	}

	/* Get the set of Keys for which buffers have been added */
	const std::set<flag_t>& get_keys() const
	{ return m_buffer_keys; }

	/* Get the amount of memory that would be taken by the given buffer
	 * if it was allocated with the given number of elements */
	size_t get_memory_occupation(flag_t Key, size_t nels) const
	{
		// return 0 unless the buffer was actually added
		if (m_buffer_keys.find(Key) == m_buffer_keys.end())
			return 0;

		// get the corresponding buffer
		const AbstractBuffer *buf = m_lists[0][Key];

		size_t single = buf->get_element_size();
		single *= buf->get_array_count();
		single *= m_policy->get_buffer_count(Key);

		return single*nels;
	}

	/* Allocate all the necessary copies of the given buffer,
	 * returning the total amount of memory used */
	size_t alloc(flag_t Key, size_t nels)
	{
		// number of actual instances of the buffer
		const size_t count = m_policy->get_buffer_count(Key);
		size_t list_idx = 0;
		size_t allocated = 0;
		while (list_idx < count) {
			allocated += m_lists[list_idx][Key]->alloc(nels);
			++list_idx;
		}
		return allocated;
	}

	/* Get a specific buffer list */
	iterator getBufferList(size_t idx)
	{
		if (idx > m_lists.size())
			throw std::runtime_error("asked for non-existing buffer list");
		return m_lists.begin() + idx;
	}

	/* Get a specific buffer list (const) */
	const_iterator getBufferList(size_t idx) const
	{
		if (idx > m_lists.size())
			throw std::runtime_error("asked for non-existing buffer list");
		return m_lists.begin() + idx;
	}

	/* Get the ith read-only buffer list */
	iterator getReadBufferList(size_t i = 0)
	{ return getBufferList(READ_LIST + i); }
	const_iterator getReadBufferList(size_t i = 0) const
	{ return getBufferList(READ_LIST + i); }

	/* Get the ith read-write buffer list */
	iterator getWriteBufferList(size_t i = 0)
	{
		if (i >= READ_LIST - WRITE_LIST)
			throw std::runtime_error("no such writeable buffer");
		return getBufferList(WRITE_LIST + i);
	}
	const_iterator getWriteBufferList(size_t i = 0) const
	{
		if (i >= READ_LIST - WRITE_LIST)
			throw std::runtime_error("no such writeable buffer");
		return getBufferList(WRITE_LIST + i);
	}

#undef READ_LIST
#undef WRITE_LIST
};

#undef DATA_TYPE

#endif

