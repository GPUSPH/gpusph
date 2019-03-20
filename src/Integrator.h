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
 * Interface for the integrator
 */

#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <memory> // shared_ptr
#include "command_type.h"

enum IntegratorType
{
	PREDITOR_CORRECTOR
};

struct GlobalData;

class Integrator;

//! A sequence of commands, modelling a phase of the integrator
/*! This is essentially an std::vector<CommandStruct>, with minor changes:
 * it only exposes reserve(), constant begin() and end() methods, and
 * a push_back() method that returns a reference to back()
 */
class CommandSequence
{
	using base = std::vector<CommandStruct>;
	using size_type = base::size_type;
	base m_seq;

	friend class Integrator;

protected:
	CommandStruct& at(size_type pos)
	{ return m_seq.at(pos); }

public:
	CommandSequence() : m_seq() {}

	void reserve(size_t sz)
	{ m_seq.reserve(sz); }

	base::const_iterator begin() const
	{ return m_seq.begin(); }
	base::const_iterator end() const
	{ return m_seq.end(); }

	size_type size() const
	{ return m_seq.size(); }

	// is this command sequence empty?
	bool empty() const
	{ return m_seq.empty(); }

	const CommandStruct& at(size_type pos) const
	{ return m_seq.at(pos); }

	CommandStruct& push_back(CommandStruct const& cmd)
	{
		m_seq.push_back(cmd);
		return m_seq.back();
	}
};


/*! An integrator is a sequence of phases, where each phase is a sequence of commands.
 * Phases can be both simple (once the sequence of commands is over, we move on to the next phase)
 * and iterative (e.g. in implicit or semi-implicit schemes, the sequence of commands needed for
 * implicit solving are repeated until a specific condition is met.
 * Most integrators share at least the phases for the neighbors list construction, filtering,
 * post-processing and some transitions.
 */

class Integrator
{
public:
	class Phase
	{
		Integrator const* m_owner; ///< Integrator owning this phase
		std::string m_name; ///< name of this phase
		CommandSequence m_command; ///< sequence of commands to execute for this phase
		int m_cmd_idx; ///< current command

		///< type of the functions that determine if a phase should run
		typedef bool (*should_run_t)(Phase const*, GlobalData const*);

		///< type of the functions called on reset()
		typedef void (*reset_t)(Phase*, Integrator const*);

		///< the function that determines if this phase should run
		should_run_t m_should_run;

		///< the function called on reset
		reset_t m_reset_func;

		/* The next commands should actually be only accessible to Integrator and its
		 * derived class, but there is no way to achieve that. We probably should look into
		 * providing a different interface
		 */
	public:
		void reserve(size_t num_cmds)
		{ return m_command.reserve(num_cmds); }

		CommandStruct& add_command(CommandName cmd)
		{ return m_command.push_back(cmd); }

		CommandStruct& edit_command(size_t idx)
		{ return m_command.at(idx); }

		// Reset the phase on enter.
		void reset_index()
		{ m_cmd_idx = 0; }

		void reset()
		{ m_reset_func(this, m_owner); }

		//! Change the condition under which the phase should run
		void should_run_if(should_run_t new_should_run_cond)
		{ m_should_run = new_should_run_cond; }

		//! Change the reset function
		void set_reset_function(reset_t reset_func)
		{ m_reset_func = reset_func; }

	public:

		// is this phase empty?
		bool empty() const
		{ return m_command.empty(); }

		// is this phase not empty?
		bool not_empty() const
		{ return !empty(); }

		bool should_run(GlobalData const* gdata) const
		{ return m_should_run(this, gdata); }

		// by default the phase runs if it's not empty
		static bool default_should_run(Phase const* p, GlobalData const*)
		{ return p->not_empty(); }

		// by default the reset simply resets the index to the default
		static void default_reset(Phase *p, Integrator const*)
		{ p->reset_index(); }

		Phase(Integrator const* owner, std::string && name) :
			m_owner(owner),
			m_name(name),
			m_command(),
			m_cmd_idx(0),
			m_should_run(default_should_run),
			m_reset_func(default_reset)
		{}

		std::string const& name() const
		{ return m_name; }

		// Is this phase done? Simple phases will be done when the last step is reached,
		// iterative phases will override this method with their more sophisticated checks
		virtual bool done() const
		{ return m_cmd_idx == m_command.size(); }

		CommandStruct const* current_command() const
		{ return &m_command.at(m_cmd_idx); }

		CommandStruct const* next_command()
		{
			CommandStruct const* cmd = current_command();
			++m_cmd_idx;
			return cmd;
		}


	};

protected:
	GlobalData const* gdata;

	std::string m_name; ///< name of the integrator
	std::vector<Phase *> m_phase; ///< phases of the integrator
	size_t m_phase_idx; ///< current phase

public:

	// the Integrator name
	std::string const& name() const
	{ return m_name; }

	// a pointer to the current integrator phase
	Phase const* current_phase() const
	{ return m_phase.at(m_phase_idx); }

protected:
	// a pointer to the current integrator phase
	Phase* current_phase()
	{ return m_phase.at(m_phase_idx); }

	Phase* enter_phase(size_t phase_idx);

	//! Move on to the next phase
	//! Derived classes should override this to properly implement
	//! transition to different phases
	virtual Phase* next_phase()
	{ return enter_phase(m_phase_idx + 1); }

	//! Define the standard neighbors list construction phase.
	//! It's then up to the individual integrators to put it in the
	//! correct place of the sequence
	Phase * buildNeibsPhase();

	// TODO we should move here phase generators that are common between (most)
	// integrators

public:

	// Instance the integrator defined by the given IntegratorType, constructing it
	// from the given gdata
	static
	std::shared_ptr<Integrator> instance(IntegratorType, GlobalData const* _gdata);

	Integrator(GlobalData const* _gdata, std::string && name) :
		gdata(_gdata),
		m_name(name),
		m_phase(),
		m_phase_idx(0)
	{}

	virtual ~Integrator()
	{
		for (Phase* phase : m_phase)
			delete phase;
	}

	// Start the integrator
	virtual void start()
	{ enter_phase(0); }

	// Fetch the next command
	CommandStruct const* next_command()
	{
		Phase* phase = current_phase();
		if (phase->done())
			phase = next_phase();
		return phase->next_command();
	}
};

#endif
