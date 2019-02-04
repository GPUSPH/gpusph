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
#include <memory>
#include <array>
#include <map>
#include <set>
#include <vector>

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

// Forward-declare the BufferList and MultiBufferList classes, that need access to some Buffer protected methods
class BufferList;
class MultiBufferList;


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

	friend class MultiBufferList; // needs to be able to access set_uid

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
		// map.insert returns a pair<iterator, bool>
		// where the bool tells if the insertion was successful
		// (it fails if the key is present already)
		auto ret = m_map.insert(std::make_pair(Key, buf));
		if (!ret.second) {
			std::string err = "double insertion of buffer " +
				std::to_string(Key) + " (" + buf->get_buffer_name() + ")";
			throw std::runtime_error(err);
		}

	}

	// remove a buffer
	void removeBuffer(flag_t Key)
	{ m_map.erase(Key); }

	// replace the buffer at position Key with buf, returning the
	// old one
	ptr_type replaceBuffer(flag_t Key, ptr_type buf)
	{
		ptr_type old = m_map.at(Key);
		m_map[Key] = buf;
		return old;
	}

	friend class MultiBufferList;

	// if this boolean is true, trying to get data
	// from a non-existent key results in a throw
	mutable bool m_validate_access;
public:
	BufferList() :
		m_map(),
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
 *
 * \note The hardest parts during the transition to the new approach is making the two
 * forms coexist, meaning that buffers should remain accessible both via the old
 * READ/WRITE “fixed” lists and via the new “state selection” mechanism, even though
 * any specific kernel should only be able to access them only through one of the schemes,
 * not both: doing this allows us to migrate the kernels gradually, verifying each
 * single step, and deprecating the old interface only after all kernels have been
 * successfully migrated.
 *
 * This will inevitably require some duplication of information and slower execution,
 * due to the need to keep the two views of the particle system in sync. Some
 * additional commands will also be introduced, to manage the parts of this synchronization
 * that cannot be modelled automatically.
 */
class MultiBufferList
{
public:
	typedef BufferList::ptr_type ptr_type;
	typedef BufferList::const_ptr_type const_ptr_type;

private:
	// buffer allocation policy
	std::shared_ptr<const BufferAllocPolicy> m_policy;

	// Pool of buffers: this holds all the available copy of each
	// allocated buffer, when not in use in a specific state
	std::map<flag_t, std::vector<ptr_type>> m_pool;

	// Particle system states: indexed by the state name (currently a string),
	// they map to a BufferList of the buffers in that state
	std::map<std::string, BufferList> m_state;

	// Keys of Buffers added so far
	// It's a set instead of a single flag_t to allow iteration on it
	// without bit-shuffling. Might change.
	// TODO check if we still need it, we could simply take the list
	// of keys available in m_pool.
	std::set<flag_t> m_buffer_keys;

	//! Put a buffer back into the pool
	void pool_buffer(flag_t key, ptr_type buf)
	{
		m_pool[key].push_back(buf);

		/* Reset the buffer state */
		buf->mark_invalid();
		buf->clear_state();
	}
	inline void pool_buffer(std::pair<flag_t, ptr_type> const& key_buf)
	{ pool_buffer(key_buf.first, key_buf.second); }

	//! Add a buffer with a given key from the pool to the given BufferList
	/*! This is used to un-pool individual buffers, when needed
	 * (e.g. when initializing states or when the need to add a state
	 * presents itself).
	 * Both the state name and the buffer list representing it are needed.
	 */
	ptr_type add_buffer_to_state(BufferList &dst, std::string const& state, flag_t key)
	{
			auto& bufvec = m_pool.at(key);
			if (bufvec.empty()) {
				std::string errmsg = "no buffers with key "
					+ std::to_string(key) + " available for state "
					+ state;
				throw std::runtime_error(errmsg);
			}

			auto buf = bufvec.back();
			dst.addExistingBuffer(key, buf);
			buf->set_state(state);
			bufvec.pop_back();
			return buf;
	}

public:

	~MultiBufferList() {
		clear();
	}

	std::string inspect() const;

	void clear() {
		// nothing to do, if the policy was never set
		if (!m_policy)
			return;

		// clear the states and pool
		m_state.clear();
		m_pool.clear();

		// and purge the list of keys too
		m_buffer_keys.clear();
	}

	void setAllocPolicy(std::shared_ptr<const BufferAllocPolicy> _policy)
	{
		if (m_policy)
			throw std::runtime_error("cannot change buffer allocation policy");
		m_policy = _policy;
	}

	/* Add a new buffer of the given BufferClass for position Key, with the provided
	 * initialization value as initializer.
	 */
	template<template<flag_t> class BufferClass, flag_t Key>
	void addBuffer(int _init=-1)
	{
		if (!m_policy)
			throw std::runtime_error("trying to add buffers before setting policy");
		if (m_buffer_keys.find(Key) != m_buffer_keys.end())
			throw std::runtime_error("trying to re-add buffer");

		m_buffer_keys.insert(Key);

		// number of copies of this buffer
		const size_t count = m_policy->get_buffer_count(Key);

		for (size_t c = 0; c < count; ++c) {
			std::shared_ptr<AbstractBuffer> buff = std::make_shared<BufferClass<Key>>(_init);
			if (count == 1)
				buff->set_uid("0U" + std::to_string(Key));
			else
				buff->set_uid( std::string(1, 'A' + c) + std::to_string(Key) );
			m_pool[Key].push_back(buff);
		}
	}

	//! Add buffers from the pool to the given state
	/*! The state is assumed to exist already */
	void add_state_buffers(std::string const& state, flag_t req_keys)
	{
		BufferList& dst = m_state.at(state);

		// loop over all keys available in the pool, and only operate
		// on the ones that have been requested too
		// TODO we do it this way at the moment to allow a general
		// req_keys specification for all models (e.g. SA buffers
		// in non-SA mode), but in the future it should be fine tuned
		// to ensure that only the needed buffers are in req_keys
		// (no extra buffers), and all of them can be loaded
		// e.g. by ORing all operated buffers and checking that req_keys
		// is equal rather than a superset.
		for (auto key : m_buffer_keys) {
			// skip unrequested keys
			if ( !(key & req_keys) )
				continue;
			auto buf = add_buffer_to_state(dst, state, key);
		}
	}

	//! Move buffers from one state to another
	void change_buffers_state(flag_t keys, std::string const& src_state, std::string const& dst_state)
	{
		BufferList& src = m_state.at(src_state);
		BufferList& dst = m_state.at(dst_state);

		std::vector<flag_t> processed;

		for (auto pair : src) {
			flag_t key = pair.first;
			auto buf = pair.second;
			if ( !(key & keys) )
				continue;
			ptr_type old = dst[key];
			if (old)
				throw std::runtime_error("trying to replace buffer "
					+ std::string(old->get_buffer_name()) + " in state " +
					dst_state + " with buffer moved from state " +
					src_state);
			buf->remove_state(src_state);
			buf->set_state(dst_state);
			dst.addExistingBuffer(key, buf);
			processed.push_back(key);
		}

		// We remove all buffers from the src list at the end
		for (auto& key : processed) {
			src.removeBuffer(key);
		}
	}

	//! Swap buffers between states, invalidating the destination ones
	void swap_state_buffers(std::string const& src_state, std::string const& dst_state, flag_t keys)
	{
		BufferList& src = m_state.at(src_state);
		BufferList& dst = m_state.at(dst_state);

		for (auto const key : m_buffer_keys) {
			if ( !(key & keys) )
				continue;
			auto src_buf = src[key];
			auto dst_buf = dst[key];

			// if none is present, skip the key
			if (!src_buf && !dst_buf)
				continue;
			// at least one is present, but do we have both?
			if (!src_buf)
				throw std::runtime_error("trying to swap asymmetric buffer " +
					std::string(dst_buf->get_buffer_name()));
			if (!dst_buf)
				throw std::runtime_error("trying to swap asymmetric buffer " +
					std::string(src_buf->get_buffer_name()));

			src.replaceBuffer(key, dst_buf);
			dst.replaceBuffer(key, src_buf);
			dst_buf->replace_state(dst_state, src_state);
			src_buf->replace_state(src_state, dst_state);
			src_buf->mark_invalid();
		}
	}

	//! Remove buffers from one state
	void remove_state_buffers(std::string const& state, flag_t req_keys)
	{
		BufferList& list = m_state.at(state);

		std::vector<flag_t> present;

		for (auto kb : list) {
			flag_t key = kb.first;
			if (!(key & req_keys))
				continue;
			present.push_back(key);
			/* remove the state from the buffer states, and
			 * pool the buffer if there are no more states */
			if (kb.second->remove_state(state) == 0)
				pool_buffer(kb);
		}

		// We remove all buffers from the src list at the end
		for (auto& key : present) {
			list.removeBuffer(key);
		}
	}

	//! Create a new State
	/*! The State will hold an unitialized copy of each of the buffers
	 * specified in keys. A buffer not being available will result in
	 * failure.
	 */
	void initialize_state(std::string const& state, flag_t req_keys)
	{
		auto ret = m_state.insert(std::make_pair(state, BufferList()));
		if (!ret.second)
			throw std::runtime_error("state " + state + " already exists");

		add_state_buffers(state, req_keys);
	}

	//! Create a new State from a legacy list (read or write)
	/*! Similar to initialize_state(state, req_keys), but fetch buffers from
	 * the given bufferlist instead of the pool. If any of the buffers is already in use
	 * by a different state, abort.
	 */
	void initialize_state(std::string const& state, BufferList& src, flag_t req_keys)
	{
		auto ret = m_state.insert(std::make_pair(state, BufferList()));
		if (!ret.second)
			throw std::runtime_error("state " + state + " already exists");

		BufferList& dst = ret.first->second;
		for (auto pair : src) {
			// skip unrequested keys
			flag_t key = pair.first;
			if ( !(key & req_keys) )
				continue;
			auto buf = pair.second;
			if (buf->state().empty()) {
				dst.addExistingBuffer(key, buf);
				buf->set_state(state);
				/* We still need to remove the buffer from the pool */
				auto& pooled = m_pool.at(key);
				auto found = std::find(pooled.begin(), pooled.end(), buf);
				if (found == pooled.end())
					throw std::runtime_error(buf->get_buffer_name()
						+ std::string(" not found in the pool"));
				pooled.erase(found);
			} else {
				throw std::runtime_error(buf->get_buffer_name()
					+ std::string(" already in state")
					+ buf->state()[0]);
			}
		}

	}

	//! Release all the buffers in a particular state back to the free pool
	void release_state(std::string const& state)
	{
		// TODO throw a more explicative error than the default
		// if the state is not found
		BufferList& list = m_state.at(state);
		for (auto kb : list) {
			/* remove the state from the buffer states, and
			 * pool the buffer if there are no more states */
			if (kb.second->remove_state(state) == 0)
				pool_buffer(kb);
		}
		m_state.erase(state);
	}

	//! Rename the state of a BufferList representing a ParticleSystem state
	/*! \note assumes that the new state does not exist yet, throws if found
	 * \note requires all buffer in the old state to be valid
	 */
	void rename_state(std::string const& old_state, std::string const& new_state)
	{
		// TODO throw a more explicative error than the default
		// if the state is not found
		BufferList& list = m_state.at(old_state);

		auto ret = m_state.insert(std::make_pair(new_state, list));
		if (!ret.second)
			throw std::runtime_error("state " + new_state + " exists already");

		/* Reset the state for all buffers, and check if they are valid */
		for (auto kb : list) {
			ptr_type buf = kb.second;
			// TODO should throw, when we're done with the migration
			if (buf->is_invalid())
				throw std::runtime_error("trying to rename state " + old_state
					+ " with invalid buffer " + buf->inspect());

			buf->replace_state(old_state, new_state);
		}

		m_state.erase(old_state);
	}

	//! Share a buffer between states
	/*! In some circumstances, a buffer will consistently have the same value
	 * across states, either always (e.g. the INFO buffer is the same for step n and step n+1)
	 * or under particular conditions (e.g. no moving objects).
	 * In these cases, however, we will prevent write access to the buffer.
	 * This method fails if the destination state has the buffers already
	 * in a non-invalid state.
	 */
	void share_buffers(std::string const& src_state, std::string const& dst_state,
		flag_t shared_buffers)
	{
		BufferList& src = m_state.at(src_state);
		BufferList& dst = m_state.at(dst_state);

		for (auto pair : src) {
			flag_t key = pair.first;
			if ( !(key & shared_buffers) )
				continue;
			ptr_type old = dst[key];
			pair.second->add_state(dst_state);
			if (!old) {
				dst.addExistingBuffer(key, pair.second);
				continue;
			}
			/* there was a buffer already: we pool it if it was invalid and not
			 * shared already */
			if (!old->is_invalid())
				throw std::runtime_error("trying to replace valid buffer "
					+ std::string(old->get_buffer_name()) + " in state " +
					dst_state + " with shared buffer from state " +
					src_state);
			if (!old->num_states() == 1)
				throw std::runtime_error("trying to replace shared buffer "
					+ std::string(old->get_buffer_name()) + " in state " +
					dst_state + " with shared buffer from state " +
					src_state);
			dst.replaceBuffer(key, pair.second);
			pool_buffer(key, old);
		}
	}

	//! Extract a buffer list holding a subset of the buffers in a given state
	/*! If the state does not already hold the buffer, it will be added
	 * from the pool if available
	 */
	BufferList state_subset(std::string const& state, flag_t selection)
	{
		BufferList& src = m_state.at(state);
		BufferList ret;
		for (auto key : m_buffer_keys)
		{
			if (!(key & selection))
				continue;

			auto buf = src[key];
			if (!buf)
				buf = add_buffer_to_state(src, state, key);
			ret.addExistingBuffer(key, buf);
		}
		return ret;
	}

	//! Return a shared pointer to the given buffer in the given state
	const_ptr_type get_state_buffer(std::string const& state, flag_t key) const
	{
		return m_state.at(state)[key];
	}

	//! Return a shared pointer to the given buffer in the given state
	/*! If the state doesn't hold the buffer yet, it will be added */
	ptr_type get_state_buffer(std::string const& state, flag_t key)
	{
		BufferList &src = m_state.at(state);
		ptr_type buf = src[key];
		if (!buf)
			buf = add_buffer_to_state(src, state, key);
		return buf;
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
		const auto& vec = m_pool.at(Key);
		const auto buf = vec.front();

		size_t single = buf->get_element_size();
		single *= buf->get_array_count();
		single *= vec.size();

		return single*nels;
	}

	/* Allocate all the necessary copies of the given buffer,
	 * returning the total amount of memory used */
	size_t alloc(flag_t Key, size_t nels)
	{
		// number of actual instances of the buffer
		const size_t count = m_policy->get_buffer_count(Key);
		auto& vec = m_pool.at(Key);
		size_t allocated = 0;
		for (auto buf : vec)
			allocated += buf->alloc(nels);
		return allocated;
	}

	/* Get the buffer list of a specific state */
	BufferList& getState(std::string const& str)
	{ return m_state.at(str); }
	/* Get the buffer list of a specific state (const) */
	const BufferList& getState(std::string const& str) const
	{ return m_state.at(str); }
};

#undef DATA_TYPE

#endif

