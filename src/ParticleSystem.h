/*  Copyright 2019 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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
 * ParticleSystem class
 */

#ifndef PARTICLESYSTEM_H
#define PARTICLESYSTEM_H

#include <set>

#include "buffer.h"
#include "buffer_alloc_policy.h"

/* A ParticleSystem takes into account that some of the buffers are needed
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
class ParticleSystem
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

	~ParticleSystem() {
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

#endif

