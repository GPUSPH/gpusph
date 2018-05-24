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

#include <array>
#include <map>
#include <set>

#include <stdexcept>

#include "common_types.h"
#include "buffer_traits.h"
#include "buffer_alloc_policy.h"

#define DEBUG_BUFFER_ACCESS 1

#if DEBUG_BUFFER_ACCESS
#include <iostream>
extern bool debug_inspect_buffer;
extern bool debug_clobber_invalid_buffers;
#define DEBUG_INSPECT_BUFFER(...) if (debug_inspect_buffer) std::cout << __VA_ARGS__
#define CLOBBER_INVALID if (debug_clobber_invalid_buffers) clobber()
#else
#define DEBUG_INSPECT_BUFFER(...) do {} while (0)
#define CLOBBER_INVALID do {} while (0)
#endif

#include "cpp11_missing.h"

enum BufferValidity {
	BUFFER_VALID, //<! Buffer contains valid data
	BUFFER_DIRTY, //<! Buffer contains valid data, but has been updated and needs resync in multi-device
	BUFFER_INVALID, //<! Buffer contains invalid data
};


/* Base class for the Buffer template class.
 * The base pointer is a pointer to pointer to allow easy management
 * of double-(or more)-buffered arrays.
 */
class AbstractBuffer
{
public:
private:
	void **m_ptr;

	size_t m_allocated_elements;

	BufferValidity m_validity;

	std::string m_state;

protected:
	// constructor that aliases m_ptr to some array of pointers
	AbstractBuffer(void *bufs[]) :
		m_ptr(bufs),
		m_allocated_elements(0),
		m_validity(BUFFER_INVALID),
		m_state()
	{}

	size_t set_allocated_elements(size_t allocs)
	{
		m_allocated_elements = allocs;
		return allocs;
	}

public:

	// default constructor: just ensure ptr is null
	AbstractBuffer() :
		m_ptr(NULL),
		m_validity(BUFFER_VALID),
		m_state()
	{}

	// destructor must be virtual
	virtual ~AbstractBuffer() {}

	// reset the buffer content to its initial value
	virtual void clobber() = 0;

	// access buffer validity
	inline bool is_valid() const { return m_validity == BUFFER_INVALID; }
	inline bool is_dirty() const { return m_validity == BUFFER_DIRTY; }
	inline bool is_invalid() const { return m_validity == BUFFER_INVALID; }
	inline BufferValidity validity() const  { return m_validity; }

	// modify buffer validity
	inline void mark_valid(BufferValidity validity = BUFFER_VALID) {
		m_validity = validity;
		if (m_validity == BUFFER_INVALID)
			CLOBBER_INVALID;
	}

	inline void mark_dirty() { mark_valid(BUFFER_DIRTY); }
	inline void mark_invalid() { mark_valid(BUFFER_INVALID); }

	// get buffer state
	inline std::string state() const { return m_state; }

	// change buffer state
	inline void set_state(std::string const& state) { m_state = state; }
	inline void add_state(std::string const& state) {
		if (m_state.size() > 0)
			m_state += ", ";
		m_state += state;
	}

	// element size of the arrays
	// overloaded in subclasses
	virtual size_t get_element_size() const = 0;

	// number of elements allocated
	inline size_t get_allocated_elements() const
	{ return m_allocated_elements; }

	// number of arrays
	// overloaded in subclasses
	virtual uint get_array_count() const = 0;

	virtual const char* get_buffer_name() const = 0;
	virtual const char* get_buffer_class() const
	{ return "AbstractBuffer"; }

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

	inline std::string inspect() const {
		std::string _desc;

		_desc += get_buffer_class();
		_desc += " ";
		_desc += get_buffer_name();
		_desc += ", validity ";
		_desc +=	std::to_string(m_validity);
		_desc += ", state: ";
		_desc += state();

		return _desc;
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
	GenericBuffer(int _init=-1) : AbstractBuffer((void**)m_bufs) {
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

	virtual const char* get_buffer_class() const
	{ return "GenericBuffer"; }
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
	Buffer(int _init=-1) : baseclass(_init) {}

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

	virtual const char* get_buffer_class() const
	{ return "Buffer"; }
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
 * argument to the function (default to -1).
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

	enum {
		NOT_PENDING,
		PENDING_SET,
		PENDING_ADD,
	} m_has_pending_state;

	std::string m_pending_state;
	flag_t m_updated_buffers;

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
	BufferList() :
		m_map(),
		m_has_pending_state(NOT_PENDING),
		m_pending_state(),
		m_updated_buffers(0)
	{};

	~BufferList() {
		clear_pending_state();
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

	void clear_pending_state() {
		m_has_pending_state = NOT_PENDING;
		m_pending_state.clear();
		m_updated_buffers = 0;
	}

	// modify validity of all buffers
	inline void mark_valid(BufferValidity validity = BUFFER_VALID) {
		for (auto& iter : m_map)
			iter.second->mark_valid(validity);
	}
	inline void mark_dirty() { mark_valid(BUFFER_DIRTY); }
	inline void mark_invalid() { mark_valid(BUFFER_INVALID); }

	// change state of all buffers
	inline void set_state(std::string const& state) {
		for (auto& iter : m_map)
			iter.second->set_state(state);
	}
	inline void add_state(std::string const& state) {
		for (auto& iter : m_map)
			iter.second->add_state(state);
	}
	// set state only for buffers that get accessed for writing
	void set_state_on_write(std::string const& state)
	{
		if (m_has_pending_state > NOT_PENDING)
			DEBUG_INSPECT_BUFFER("setting pending state without previous reset!" << std::endl);
		m_has_pending_state = PENDING_SET;
		m_pending_state = state;
	}
	// add state only for buffers that get accessed for writing
	void add_state_on_write(std::string const& state)
	{
		if (m_has_pending_state > NOT_PENDING)
			DEBUG_INSPECT_BUFFER("setting pending state addition without previous reset!" << std::endl);
		m_has_pending_state = PENDING_ADD;
		m_pending_state = state;
	}
	// get the list of buffers that were updated
	flag_t get_updated_buffers() const
	{ return m_updated_buffers; }


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
		if (exists == m_map.end())
			return NULL;

		auto buf(exists->second);

		DEBUG_INSPECT_BUFFER("\t" << buf->inspect() << " [");

		switch (m_has_pending_state) {
		case NOT_PENDING:
			DEBUG_INSPECT_BUFFER("no pending state");
			break;
		case PENDING_SET:
			DEBUG_INSPECT_BUFFER("state set " << m_pending_state);
			buf->set_state(m_pending_state);
			break;
		case PENDING_ADD:
			DEBUG_INSPECT_BUFFER("state add " << m_pending_state);
			buf->add_state(m_pending_state);
			break;
		}
		DEBUG_INSPECT_BUFFER("]" << std::endl);

		m_updated_buffers |= Key;
		buf->mark_dirty();

		return static_cast<DATA_TYPE(Key)*>(buf->get_buffer(num));
	}

	// const version
	template<flag_t Key>
	const DATA_TYPE(Key) *getData(uint num=0) const {
		map_type::const_iterator exists = m_map.find(Key);
		if (exists == m_map.end())
			return NULL;

		auto buf(exists->second);
		DEBUG_INSPECT_BUFFER("\t" << buf->inspect() << " [const]" << std::endl);
		if (buf->is_invalid()) {
			if (DEBUG_BUFFER_ACCESS) {
				DEBUG_INSPECT_BUFFER("\t\t(trying to read invalid data)" << std::endl);
			} else {
				throw std::invalid_argument("trying to read invalid data");
			}
		}
		return static_cast<const DATA_TYPE(Key)*>(buf->get_buffer(num));
	}

	// const access from writeable list
	template<flag_t Key>
	const DATA_TYPE(Key) *getConstData(uint num=0) {
		return as_const(*this).getData<Key>(num);
	}

	/* In a few cases, the user may want to access the base pointer to
	 * multi-buffer arrays. This could be done using get<Key>()->get_raw_ptr,
	 * but this causes a segmentation fault when the Key array isn't actually
	 * allocated, while we really want to return NULL in this case. Wrap it up
	 * for safety.
	 */
	template<flag_t Key>
	DATA_TYPE(Key) **getRawPtr() {
		Buffer<Key> *buf = this->get<Key>();
		if (!buf)
			return NULL;

		DEBUG_INSPECT_BUFFER("\t" << buf->inspect() << " [raw ptr ");
		switch (m_has_pending_state) {
		case NOT_PENDING:
			DEBUG_INSPECT_BUFFER("no pending state");
			break;
		case PENDING_SET:
			DEBUG_INSPECT_BUFFER("state set " << m_pending_state);
			buf->set_state(m_pending_state);
			break;
		case PENDING_ADD:
			DEBUG_INSPECT_BUFFER("state add " << m_pending_state);
			buf->add_state(m_pending_state);
			break;
		}

		DEBUG_INSPECT_BUFFER("]" << std::endl);

		m_updated_buffers |= Key;
		buf->mark_dirty();

		return buf->get_raw_ptr();
	}

	// const version
	template<flag_t Key>
	const DATA_TYPE(Key)* const* getRawPtr() const {
		const Buffer<Key> *buf = this->get<Key>();
		if (!buf)
			return NULL;

		DEBUG_INSPECT_BUFFER("\t" << buf->inspect() << " [const raw ptr]" << std::endl);
		if (buf->is_invalid()) {
			if (DEBUG_BUFFER_ACCESS) {
				DEBUG_INSPECT_BUFFER("\t\t(trying to read invalid data)" << std::endl);
			} else {
				throw std::invalid_argument("trying to read invalid data");
			}
		}
		return buf->get_raw_ptr();
	}

	// const accessor from writeable list
	template<flag_t Key>
	const DATA_TYPE(Key)* const* getConstRawPtr() {
		return as_const(*this).getRawPtr<Key>();
	}

	/* Add a new buffer of the given BufferClass for position Key, with the provided
	 * initialization value as initializer. The position is automatically deduced by
	 * the Key of the buffer.
	 */
	template<template<flag_t> class BufferClass, flag_t Key>
	BufferList& addBuffer(int _init=-1)
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
 * TODO FIXME the code currently assumes that buffers are _at most_ double-buffered.
 * This whole thing will soon be removed by the ParticleSystem::State mechanism.
 */
class MultiBufferList
{
public:
	// buffer allocation policy
	const BufferAllocPolicy *m_policy;

	// TODO FIXME this is for double-buffered lists only
	// In general we would have N writable list (N=1 usually)
	// and M read-only lists (M >= 1), with the number of each
	// determined by the BufferAllocPolicy.
#define READ_LIST 1
#define WRITE_LIST 0

	// list of BufferLists
	std::array<BufferList, 2> m_lists;

	// Keys of Buffers added so far
	// It's a set instead of a single flag_t to allow iteration on it
	// without bit-shuffling. Might change.
	std::set<flag_t> m_buffer_keys;

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

			for (auto& list : m_lists)
				list.removeBuffer(key);
			delete buf;
		}

		// now clear the lists
		for (auto& list : m_lists)
			list.clear();

		// and purge the list of keys too
		m_buffer_keys.clear();
	}

	void setAllocPolicy(const BufferAllocPolicy* _policy)
	{
		if (m_policy != NULL)
			throw std::runtime_error("cannot change buffer allocation policy");
		m_policy = _policy;

		// add as many BufferLists as needed at most
		// TODO would have made sense for > 2 copies,
		// in which case m_lists would have been an std::vector
		//m_lists.resize(m_policy->get_max_buffer_count());
	}

	/* Add a new buffer of the given BufferClass for position Key, with the provided
	 * initialization value as initializer.
	 */
	template<template<flag_t> class BufferClass, flag_t Key>
	void addBuffer(int _init=-1)
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
			for (auto& list : m_lists)
				list.addBuffer<BufferClass, Key>(_init);
		} else {
			// single-buffered, allocate once and put in all lists
			AbstractBuffer *buff = new BufferClass<Key>;

			for (auto& list : m_lists)
				list.addExistingBuffer(Key, buff);
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
	BufferList& getBufferList(size_t idx)
	{
		if (idx > m_lists.size())
			throw std::runtime_error("asked for non-existing buffer list");
		return m_lists[idx];
	}

	/* Get a specific buffer list (const) */
	const BufferList& getBufferList(size_t idx) const
	{
		if (idx > m_lists.size())
			throw std::runtime_error("asked for non-existing buffer list");
		return m_lists[idx];
	}

	/* Get the read-only buffer list */
	BufferList& getReadBufferList()
	{ return getBufferList(READ_LIST); }
	const BufferList& getReadBufferList() const
	{ return getBufferList(READ_LIST); }

	/* Get the read-write buffer list */
	BufferList& getWriteBufferList()
	{
		return getBufferList(WRITE_LIST);
	}
	const BufferList& getWriteBufferList() const
	{
		return getBufferList(WRITE_LIST);
	}

#undef READ_LIST
#undef WRITE_LIST
};

#undef DATA_TYPE

#endif

