/*  Copyright 2018 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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
 * Implementation of methods for the ParticleSystem (and related) classes
 */

#include <sstream>
#include <iomanip>

#include "ParticleSystem.h"

using namespace std;

using ptr_type = ParticleSystem::ptr_type;
using const_ptr_type = ParticleSystem::const_ptr_type;

//! Put a buffer back into the pool
void ParticleSystem::pool_buffer(flag_t key, ptr_type buf)
{
	m_pool[key].push_back(buf);

	/* Reset the buffer state */
	buf->mark_invalid();
	buf->clear_state();
}

ptr_type ParticleSystem::add_buffer_to_state(
	BufferList &dst, string const& state, flag_t key)
{
	auto& bufvec = m_pool.at(key);
	if (bufvec.empty()) {
		string errmsg = "no buffers with key "
			+ to_string(key) + " available for state "
			+ state;
		throw runtime_error(errmsg);
	}

	auto buf = bufvec.back();
	dst.addExistingBuffer(key, buf);
	buf->set_state(state);
	bufvec.pop_back();
	return buf;
}

void ParticleSystem::clear()
{
	// nothing to do, if the policy was never set
	if (!m_policy)
		return;

	// clear the states and pool
	m_state.clear();
	m_pool.clear();

	// and purge the list of keys too
	m_buffer_keys.clear();
}

string ParticleSystem::inspect() const
{
	stringstream _desc;
	size_t count;
	size_t key_width;
	flag_t max_key = 0;

	_desc << "ParticleSystem. ";
	_desc << "Registered buffer keys: ";

	count = 0;
	for (flag_t k : m_buffer_keys) {
		if (count > 0) _desc << ", ";
		if (k > max_key) max_key = k;
		_desc << k;
		++count;
	}
	_desc << "\n";

	key_width = to_string(max_key).size();

	for (auto const& sv : m_state) {
		_desc << "\tState " << sv.first << "\n";
		count = 0;
		for (auto const& pair : sv.second) {
			_desc << "\t\t\t";
			_desc << setw(key_width) << right << pair.first;
			_desc << "\t" + pair.second->inspect() << "\n";
			++count;
		}
	}

	_desc << "\tPool:\n";

	for (auto const& sv : m_pool) {
		_desc << "\t\t" << setw(key_width) << right << sv.first << ":\n";
		for (auto const& b : sv.second) {
			_desc << "\t\t\t" + b->inspect() << "\n";
			++count;
		}
	}

	return _desc.str();
}

void ParticleSystem::add_state_buffers(string const& state, flag_t req_keys)
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

void ParticleSystem::change_buffers_state(flag_t keys,
	string const& src_state,
	string const& dst_state)
{
	BufferList& src = m_state.at(src_state);
	BufferList& dst = m_state.at(dst_state);

	vector<flag_t> processed;

	for (auto pair : src) {
		flag_t key = pair.first;
		auto buf = pair.second;
		if ( !(key & keys) )
			continue;
		ptr_type old = dst[key];
		if (old)
			throw runtime_error("trying to replace buffer "
				+ string(old->get_buffer_name()) + " in state "
				+ dst_state + " with buffer moved from state "
				+ src_state);
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

void ParticleSystem::swap_state_buffers(
	string const& src_state,
	string const& dst_state,
	flag_t keys)
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
			throw runtime_error("trying to swap asymmetric buffer " +
				string(dst_buf->get_buffer_name()));
		if (!dst_buf)
			throw runtime_error("trying to swap asymmetric buffer " +
				string(src_buf->get_buffer_name()));

		src.replaceBuffer(key, dst_buf);
		dst.replaceBuffer(key, src_buf);
		dst_buf->replace_state(dst_state, src_state);
		src_buf->replace_state(src_state, dst_state);
		src_buf->mark_invalid();
	}
}

void ParticleSystem::remove_state_buffers(string const& state, flag_t req_keys)
{
	BufferList& list = m_state.at(state);

	vector<flag_t> present;

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

void ParticleSystem::initialize_state(string const& state, flag_t req_keys)
{
	auto ret = m_state.insert(make_pair(state, BufferList()));
	if (!ret.second)
		throw runtime_error("state " + state + " already exists");

	add_state_buffers(state, req_keys);
}

void ParticleSystem::release_state(string const& state)
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

void ParticleSystem::rename_state(string const& old_state, string const& new_state)
{
	// TODO throw a more explicative error than the default
	// if the state is not found
	BufferList& list = m_state.at(old_state);

	auto ret = m_state.insert(make_pair(new_state, list));
	if (!ret.second)
		throw runtime_error("state " + new_state + " exists already");

	/* Reset the state for all buffers, and check if they are valid */
	for (auto kb : list) {
		ptr_type buf = kb.second;
		// TODO should throw, when we're done with the migration
		if (buf->is_invalid())
			throw runtime_error("trying to rename state " + old_state
				+ " with invalid buffer " + buf->inspect());

		buf->replace_state(old_state, new_state);
	}

	m_state.erase(old_state);
}

void ParticleSystem::share_buffers(string const& src_state, string const& dst_state,
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
			throw runtime_error("trying to replace valid buffer "
				+ string(old->get_buffer_name()) + " in state " +
				dst_state + " with shared buffer from state " +
				src_state);
		if (!old->num_states() == 1)
			throw runtime_error("trying to replace shared buffer "
				+ string(old->get_buffer_name()) + " in state " +
				dst_state + " with shared buffer from state " +
				src_state);
		dst.replaceBuffer(key, pair.second);
		pool_buffer(key, old);
	}
}

BufferList ParticleSystem::state_subset(string const& state, flag_t selection)
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

ptr_type ParticleSystem::get_state_buffer(string const& state, flag_t key)
{
	BufferList &src = m_state.at(state);
	ptr_type buf = src[key];
	if (!buf)
		buf = add_buffer_to_state(src, state, key);
	return buf;
}

size_t ParticleSystem::get_memory_occupation(flag_t Key, size_t nels) const
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
