/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

	/* A ParticleSystem::State is a BufferList with an associated
	 * particle system
	 */
	class State : public BufferList
	{
		ParticleSystem& m_ps;
		std::string m_name;

	public:
		State(ParticleSystem& ps, std::string const& name) :
			m_ps(ps), m_name(name)
		{}

		inline std::string const& name() const
		{ return m_name; }

		inline std::string const& set_name(std::string const& name)
		{ return m_name = name; }

		/* The non-const [] accessor for the ParticleSystem::State is read-write:
		 * if access to a buffer that is not present is requested,
		 * a buffer with the given key will be fetched from the pool,
		 * if possible
		 */
		ptr_type operator[](const flag_t key);

		/* The const [] accessor for the ParticleSystem::State
		 * throws if the buffer is not found
		 */
		const_ptr_type operator[](const flag_t key) const;

		//! Get the buffer if present, return null otherwise
		/*! This works like BufferList::operator[], i.e. it doesn't
		 * try to add the buffer or throw if the buffer is missing
		 */
		ptr_type at(const flag_t key);
		const_ptr_type at(const flag_t key) const;

	};

	typedef std::map<std::string, State> StateList;

private:
	// buffer allocation policy
	std::shared_ptr<const BufferAllocPolicy> m_policy;

	// Pool of buffers: this holds all the available copy of each
	// allocated buffer, when not in use in a specific state
	std::map<flag_t, std::vector<ptr_type>> m_pool;

	// Particle system states: indexed by the state name (currently a string),
	// they map to a State BufferList holding the buffers in that state
	StateList m_state;

	// Keys of Buffers added so far
	// It's a set instead of a single flag_t to allow iteration on it
	// without bit-shuffling. Might change.
	// TODO check if we still need it, we could simply take the list
	// of keys available in m_pool.
	std::set<flag_t> m_buffer_keys;

	//! Put a buffer back into the pool
	void pool_buffer(flag_t key, ptr_type buf);

	inline void pool_buffer(std::pair<flag_t, ptr_type> const& key_buf)
	{ pool_buffer(key_buf.first, key_buf.second); }

	//! Add a buffer with a given key from the pool to the given BufferList
	/*! This is used to un-pool individual buffers, when needed
	 * (e.g. when initializing states or when the need to add a state
	 * presents itself).
	 * Both the state name and the buffer list representing it are needed.
	 */
	ptr_type add_buffer_to_state(State &dst, flag_t key);

protected:
	friend class GPUSPH;

	//! Get list of available states
	StateList const& states() const
	{ return m_state; }

public:

	//! Remove all buffers and states from the system
	void clear();

	~ParticleSystem() {
		clear();
	}

	std::string inspect() const;

	inline void setAllocPolicy(std::shared_ptr<const BufferAllocPolicy> _policy)
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
	void add_state_buffers(std::string const& state, flag_t req_keys);

	//! Move buffers from one state to another
	void change_buffers_state(flag_t keys,
		std::string const& src_state,
		std::string const& dst_state);

	//! Swap buffers between states, invalidating the destination ones
	void swap_state_buffers(
		std::string const& src_state,
		std::string const& dst_state,
		flag_t keys);

	//! Remove buffers from one state
	void remove_state_buffers(std::string const& state, flag_t req_keys);

	//! Create a new State
	/*! The State will hold an unitialized copy of each of the buffers
	 * specified in keys. A buffer not being available will result in
	 * failure.
	 */
	State& initialize_state(std::string const& state);

	//! Release all the buffers in a particular state back to the free pool
	void release_state(std::string const& state);

	//! Change the name of a ParticleSystem state
	/*! \note assumes that the new state does not exist yet, throws if found
	 * \note requires all buffer in the old state to be valid
	 */
	void rename_state(std::string const& old_state, std::string const& new_state);

	//! Share a buffer between states
	/*! In some circumstances, a buffer will consistently have the same value
	 * across states, either always (e.g. the INFO buffer is the same for step n and step n+1)
	 * or under particular conditions (e.g. no moving objects).
	 * In these cases, however, we will prevent write access to the buffer.
	 * This method fails if the destination state has the buffers already
	 * in a non-invalid state.
	 */
	void share_buffers(std::string const& src_state, std::string const& dst_state,
		flag_t shared_buffers);

	//! Extract a buffer list holding a subset of the buffers in a given state
	/*! If the state does not already hold the buffer, it will be added
	 * from the pool if available
	 */
	BufferList state_subset(std::string const& state, flag_t selection);

	//! A version of state_subset that only picks existing buffers
	const BufferList
		state_subset_existing(std::string const& state, flag_t selection) const;

	//! Return a shared pointer to the given buffer in the given state
	const_ptr_type get_state_buffer(std::string const& state, flag_t key) const
	{ return m_state.at(state)[key]; }

	//! Return a shared pointer to the given buffer in the given state
	/*! If the state doesn't hold the buffer yet, it will be added */
	ptr_type get_state_buffer(std::string const& state, flag_t key);

	/* Get the set of Keys for which buffers have been added */
	const std::set<flag_t>& get_keys() const
	{ return m_buffer_keys; }

	/* Get the amount of memory that would be taken by the given buffer
	 * if it was allocated with the given number of elements */
	size_t get_memory_occupation(flag_t Key, size_t nels) const;

	/* Allocate all the necessary copies of the given buffer,
	 * returning the total amount of memory used */
	size_t alloc(flag_t Key, size_t nels)
	{
		// number of actual instances of the buffer
		auto& vec = m_pool.at(Key);
		size_t allocated = 0;
		for (auto buf : vec)
			allocated += buf->alloc(nels);
		return allocated;
	}

	/* Get the buffer list of a specific state */
	State& getState(std::string const& str)
	{ return m_state.at(str); }
	/* Get the buffer list of a specific state (const) */
	const State& getState(std::string const& str) const
	{ return m_state.at(str); }
};

#endif

