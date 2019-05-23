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

/*! \file
 * Buffer class and associates
 */

#ifndef _BUFFER_H
#define _BUFFER_H

#include <algorithm>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common_types.h"
#include "buffer_traits.h"

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

// Forward-declare the BufferList and ParticleSystem classes, that need access to some Buffer protected methods
class BufferList;
class ParticleSystem;


/* Base class for the Buffer template class.
 * The base pointer is a pointer to pointer to allow easy management
 * of double-(or more)-buffered arrays.
 */
class AbstractBuffer
{
	void **m_ptr;

	size_t m_allocated_elements;

	BufferValidity m_validity;

	// The state(s) of the buffer
	std::vector<std::string> m_state;
	// the list of kernels that have manipulated the buffer since the last state change
	std::vector<std::string> m_manipulators;

	// unique buffer ID, used to track the buffer handling across states and buffer lists
	// we could use the pointer, but this isn't invariant across runs, while we can set
	// a unique ID on initialization in such a way that it only depends on initialization order
	std::string m_uid;

protected:
	// constructor that aliases m_ptr to some array of pointers
	AbstractBuffer(void *bufs[]) :
		m_ptr(bufs),
		m_allocated_elements(0),
		m_validity(BUFFER_INVALID),
		m_state(),
		m_uid("<unset>")
	{}

	void set_uid(std::string const& uid)
	{ m_uid = uid; }
	std::string const& uid() const
	{ return m_uid; }

	size_t set_allocated_elements(size_t allocs)
	{
		m_allocated_elements = allocs;
		return allocs;
	}

	friend class ParticleSystem; // needs to be able to access set_uid

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
	inline bool is_valid() const { return m_validity == BUFFER_VALID; }
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
	inline std::vector<std::string> const& state() const { return m_state; }

	// get number of states
	inline size_t num_states() const { return m_state.size(); }

	// change buffer state
	inline void clear_state()
	{
		m_state.clear();
		m_manipulators.clear();
	}
	inline void set_state(std::string const& state)
	{
		clear_state();
		m_state.push_back(state);
	}
	inline void add_state(std::string const& state)
	{
		if (m_state.size() == 0)
			throw std::runtime_error("adding state " + state +
				" to buffer " + get_buffer_name() + " without state");
		m_state.push_back(state);
	}
	// remove buffer state, return number of states still present
	inline size_t remove_state(std::string const& state) {
		auto found = std::find(m_state.begin(), m_state.end(), state);
		if (found == m_state.end())
			throw std::runtime_error("trying to remove unassigned buffer state " +
				state + " from buffer " + this->inspect());
		m_state.erase(found);
		m_manipulators.clear();
		return m_state.size();
	}
	// replace a state with  different one
	inline void replace_state(std::string const& old_s, std::string const& new_s) {
		auto found = std::find(m_state.begin(), m_state.end(), old_s);
		if (found == m_state.end())
			throw std::runtime_error("trying to replace unassigned buffer state " +
				old_s + " from buffer " + this->inspect());
		found->replace(0, found->size(), new_s);
	}

	inline void copy_state(AbstractBuffer const* other)
	{
		clear_state();
		for (auto const& s : other->state())
			m_state.push_back(s);
	}

	inline void add_manipulator(std::string const& manip)
	{
		m_manipulators.push_back(manip);
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
		_desc += ", id " + m_uid;
		_desc += ", validity ";
		_desc +=	std::to_string(m_validity);
		_desc += ", state:";
		for (auto const& s : m_state) {
			_desc += " " + s;
		}
		if (m_manipulators.size() > 0) {
			_desc += ", manipulators:";
			for (auto const& m : m_manipulators) {
				_desc += " " + m;
			}
		}

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

	// typed getters
	T *get(uint idx=0) {
		if (idx >= N) return NULL;
		return m_bufs[idx];
	}
	virtual const T *get(uint idx=0) const {
		if (idx >= N) return NULL;
		return m_bufs[idx];
	}

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
public:
	typedef std::shared_ptr<AbstractBuffer> ptr_type;
	typedef std::shared_ptr<const AbstractBuffer> const_ptr_type;

private:
	template<flag_t Key>
	using buffer_ptr_type = std::shared_ptr<Buffer<Key>>;
	template<flag_t Key>
	using const_buffer_ptr_type = std::shared_ptr<const Buffer<Key>>;

	typedef std::map<flag_t, ptr_type> map_type;

	map_type m_map;

	// An OR of all the keys present in the map
	flag_t m_keys;

	enum {
		NOT_PENDING,
		PENDING_SET,
		PENDING_ADD,
	} m_has_pending_state;

	std::string m_pending_state;
	flag_t m_updated_buffers;

protected:
	void addExistingBuffer(flag_t Key, ptr_type buf)
	{
		if (m_keys & Key) {
			std::string err = "double insertion of buffer " +
				std::to_string(Key) + " (" + buf->get_buffer_name() + ")";
			throw std::runtime_error(err);
		}
		m_map.insert(std::make_pair(Key, buf));
		m_keys |= Key;
	}

	// remove a buffer
	void removeBuffer(flag_t Key)
	{
		m_map.erase(Key);
		m_keys &= ~Key;
	}

	// replace the buffer at position Key with buf, returning the
	// old one
	ptr_type replaceBuffer(flag_t Key, ptr_type buf)
	{
		ptr_type old = m_map.at(Key);
		m_map[Key] = buf;
		return old;
	}

	friend class ParticleSystem;

	// if this boolean is true, trying to get data
	// from a non-existent key results in a throw
	mutable bool m_validate_access;
public:
	BufferList() :
		m_map(),
		m_keys(0),
		m_has_pending_state(NOT_PENDING),
		m_pending_state(),
		m_updated_buffers(0),
		m_validate_access(false)
	{};

	void validate_access() const {
		m_validate_access = true;
	}

	~BufferList() {
		clear_pending_state();
		clear();
	}

	// add the other list buffers to this one
	BufferList& operator|=(BufferList const& other)
	{
		for (auto kb : other.m_map)
			this->addExistingBuffer(kb.first, kb.second);
		return *this;
	}

	// list | list produces a new list
	friend BufferList operator|(BufferList first, BufferList const& other)
	{
		first |= other;
		return first;
	}

	// delete all buffers before clearing the hash
	void clear() {
		m_map.clear();
		m_keys = 0;
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
	inline void add_manipulator(std::string const& manip) {
		for (auto& iter : m_map)
			iter.second->add_manipulator(manip);
	}
	// set state only for buffers that get accessed for writing
	void set_state_on_write(std::string const& state)
	{
		if (m_has_pending_state > NOT_PENDING)
			DEBUG_INSPECT_BUFFER("setting pending state without previous reset!" << std::endl);
		m_has_pending_state = PENDING_SET;
		m_pending_state = state;
	}
	// add manipulator for buffers that get accessed for writing
	void add_manipulator_on_write(std::string const& state)
	{
		if (m_has_pending_state > NOT_PENDING)
			DEBUG_INSPECT_BUFFER("setting pending state addition without previous reset!" << std::endl);
		m_has_pending_state = PENDING_ADD;
		m_pending_state = state;
	}
	// get the list of buffers that were updated
	flag_t get_updated_buffers() const
	{ return m_updated_buffers; }

	//! Get the list of available keys
	flag_t get_keys() const
	{ return m_keys; }

	//! Check if the BufferList has a given buffer
	bool has(const flag_t Key)
	{ return !!(m_keys & Key); }

	/* Read-only [] accessor. Insertion of buffers should be done via the
	 * addBuffer<>() method template.
	 */
	ptr_type operator[](const flag_t& Key) {
		map_type::const_iterator exists = m_map.find(Key);
		if (exists != m_map.end())
			return exists->second;
		else return NULL;
	}
	const_ptr_type operator[](const flag_t& Key) const {
		map_type::const_iterator exists = m_map.find(Key);
		if (exists != m_map.end())
			return exists->second;
		else return NULL;
	}

	/* Templatized getter to allow the user to access the Buffers in the list
	 * with proper typing (so e.g .get<BUFFER_POS>() will return a
	 * shared pointer to Buffer<BUFFER_POS> instead of an AbstractBuffer)
	 */
	template<flag_t Key>
	buffer_ptr_type<Key> get() {
		map_type::iterator exists = m_map.find(Key);
		if (exists != m_map.end())
			return std::static_pointer_cast<Buffer<Key>>(exists->second);
		else return NULL;
	}
	// const version
	template<flag_t Key>
	const_buffer_ptr_type<Key> get() const {
		map_type::const_iterator exists = m_map.find(Key);
		if (exists != m_map.end())
			return std::static_pointer_cast<const Buffer<Key>>(exists->second);
		else return NULL;
	}

	/*! In some circumstances, we might want to access a buffer under conditions
	 * that should normally raise an error. This can be overridden by allowing
	 * manually specifying that we know we are violating some constratins, so the
	 * error should not be flagged
	 */
	enum AccessSafety {
		NO_SAFETY, ///< No special safety consideration
		MULTISTATE_SAFE, ///< Safe to access multi-state buffers for writing
	};


	/* In most cases, user wants access directly to a specific array of a given buffer
	 * possibly with correct typing, which is exactly what this method does,
	 * again exploiting the BufferTraits to deduce the element type that is
	 * expected in return.
	 * The static cast is necessary because the get_buffer() method must return a void*
	 * due to overloading rules.
	 */
	template<flag_t Key, ///< key of the buffer to retrieve
		AccessSafety safety = NO_SAFETY> ///< 
	DATA_TYPE(Key) *getData(uint num=0) {
		map_type::iterator exists = m_map.find(Key);
		if (exists == m_map.end()) {
			if (m_validate_access)
				throw std::runtime_error(std::to_string(Key) + " not found");
			return NULL;
		}

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
			DEBUG_INSPECT_BUFFER("manipulator add " << m_pending_state);
			buf->add_manipulator(m_pending_state);
			break;
		}
		DEBUG_INSPECT_BUFFER("]" << std::endl);

		// Multi-state buffers shouldn't be accessed for writing,
		// under normal conditions.
		if (buf->state().size() > 1) {
			std::string errmsg = "access multi-state buffer " +
				buf->inspect() + " for writing";
			if (safety & MULTISTATE_SAFE) {
				DEBUG_INSPECT_BUFFER(errmsg);
			} else {
				throw std::invalid_argument(errmsg);
			}
		}


		m_updated_buffers |= Key;
		buf->mark_dirty();

		return static_cast<DATA_TYPE(Key)*>(buf->get_buffer(num));
	}

	// const version
	template<flag_t Key>
	const DATA_TYPE(Key) *getData(uint num=0) const {
		map_type::const_iterator exists = m_map.find(Key);
		if (exists == m_map.end()) {
			if (m_validate_access)
				throw std::runtime_error(std::to_string(Key) + " not found");
			return NULL;
		}

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
		buffer_ptr_type<Key> buf = this->get<Key>();
		if (!buf) {
			if (m_validate_access)
				throw std::runtime_error(std::to_string(Key) + " not found");
			return NULL;
		}

		if (buf->num_states() > 1)
			throw std::invalid_argument("trying to access multi-state buffer " +
				buf->inspect() + " for writing");

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
			DEBUG_INSPECT_BUFFER("manipulator add " << m_pending_state);
			buf->add_manipulator(m_pending_state);
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
		const_buffer_ptr_type<Key> buf = this->get<Key>();
		if (!buf) {
			if (m_validate_access)
				throw std::runtime_error(std::to_string(Key) + " not found");
			return NULL;
		}

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
			m_map.insert(std::make_pair(Key, std::make_shared<BufferClass<Key>>(_init)));
			m_keys |= Key;
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

#undef DATA_TYPE

#endif

