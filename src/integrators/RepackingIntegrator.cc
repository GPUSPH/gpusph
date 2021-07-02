/*  Copyright (c) 2019 INGV, EDF, UniCT, JHU

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

#include <string>

#include "GlobalData.h"

#include "RepackingIntegrator.h"

/*! \file
 * Repacking integrator implementation.
 */

using namespace std;

//! Function that returns the full current time-step
static float full_timestep(GlobalData const* gdata)
{ return gdata->dt; }

string
RepackingIntegrator::getCurrentStateForStep(int step_num)
{
	return "step n";
}

string
RepackingIntegrator::getNextStateForStep(int step_num)
{
	switch (step_num) {
	case 0: // repacking initialization step
	case -1: // end-of-repacking prepare simulation step
		return "step n";
	case 1:
		return "step n+1";
	default:
		throw runtime_error("cannot determine next state for step #" + to_string(step_num));
	}
}

//! Fill in the StepInfo for a given step number
static StepInfo step_info(int step_num)
{
	StepInfo step(step_num);
	if (step_num == 1)
		step.last = true;

	return step;
}

template<BoundaryType boundarytype>
void RepackingIntegrator::initializeBoundaryConditionsSequence
	(Integrator::Phase *this_phase, StepInfo const& step)
{ /* for most boundary models, there's nothing to do */ }

// formerly saBoundaryConditions()
//! Initialize the sequence of commands needed to apply SA_BOUNDARY boundary conditions.
/*! This will _not_ be called for the initialization step if we resumed,
 * since the arrays were already correctly initialized before writing
 */
template<>
void RepackingIntegrator::initializeBoundaryConditionsSequence<SA_BOUNDARY>
	(Integrator::Phase *this_phase, StepInfo const& step)
{
	// end-of-repacking, prepare simulation step?
	const bool reinit_step = (step.number == -1);
	const SimParams *sp = gdata->problem->simparams();
	const bool has_io = HAS_INLET_OUTLET(sp->simflags);

	const dt_operator_t dt_op = full_timestep;

	// We only run IO init stuff if we have IO and this is the
	// simulation preparation step
	const bool prepare_io_step = has_io && reinit_step;

	/* Boundary conditions are applied to step n during initialization step,
	 * to step n* after the integrator, and to step n+1 after the corrector
	 */
	const string state = getNextStateForStep(step.number);

	if (gdata->simframework->getBCEngine() == NULL)
		throw runtime_error("no boundary conditions engine loaded");

	// compute normal for vertices, updating BUFFER_BOUNDELEMENTS in-place
	// (reads from boundaries, writes on vertices)
	this_phase->add_command(SA_COMPUTE_VERTEX_NORMAL)
		.reading(state, BUFFER_VERTICES |
			BUFFER_INFO | BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST)
		.updating(state, BUFFER_BOUNDELEMENTS);
	if (MULTI_DEVICE)
		this_phase->add_command(UPDATE_EXTERNAL)
			.updating(state, BUFFER_BOUNDELEMENTS);

	// compute initial value of gamma for fluid and vertices
	this_phase->add_command(SA_INIT_GAMMA)
		.reading(state, BUFFER_BOUNDELEMENTS | BUFFER_VERTPOS |
			BUFFER_POS | BUFFER_INFO | BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST)
		.writing(state, BUFFER_GRADGAMMA);
	if (MULTI_DEVICE)
		this_phase->add_command(UPDATE_EXTERNAL)
			.updating(state, BUFFER_GRADGAMMA);

	// modify particle mass on open boundaries, after repacking
	// and impose open boundary conditions, after repacking is finished
	if (prepare_io_step) {
		this_phase->add_command(IDENTIFY_CORNER_VERTICES)
			.reading(state, BUFFER_VERTICES | BUFFER_BOUNDELEMENTS |
				BUFFER_POS | BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST)
			.updating(state, BUFFER_INFO);
		if (MULTI_DEVICE)
			this_phase->add_command(UPDATE_EXTERNAL)
				.updating(state, BUFFER_INFO);

		this_phase->add_command(INIT_STATE).set_src("iomass");

		// first step: count the vertices that belong to IO and the same
		// segment as each IO vertex
		// we use BUFFER_FORCES as scratch buffer to store the computed
		// IO masses
		this_phase->add_command(INIT_IO_MASS_VERTEX_COUNT)
			.reading(state, BUFFER_VERTICES | BUFFER_INFO |
				BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST)
			.writing(state, BUFFER_FORCES);
		if (MULTI_DEVICE)
			this_phase->add_command(UPDATE_EXTERNAL)
				.updating(state, BUFFER_FORCES);

		// second step: modify the mass of the IO vertices
		this_phase->add_command(INIT_IO_MASS)
			.reading(state, BUFFER_VERTICES | BUFFER_INFO | BUFFER_POS |
				BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
				BUFFER_FORCES)
			.writing("iomass", BUFFER_POS);
		if (MULTI_DEVICE)
			this_phase->add_command(UPDATE_EXTERNAL)
				.updating("iomass", BUFFER_POS);
		this_phase->add_command(SWAP_STATE_BUFFERS)
			.set_src("step n")
			.set_dst("iomass")
			.set_flags(BUFFER_POS);
		this_phase->add_command(REMOVE_STATE_BUFFERS)
			.set_src("step n")
			.set_flags(BUFFER_FORCES);
		this_phase->add_command(RELEASE_STATE)
			.set_src("iomass");

		// reduce the water depth at pressure outlets if required
		// if we have multiple devices then we need to run a global max on the different gpus / nodes
		if (MULTI_DEVICE && HAS_WATER_DEPTH(sp->simflags)) {
			// each device gets his waterdepth array from the gpu
			this_phase->add_command(DOWNLOAD_IOWATERDEPTH);
			// reduction across devices and if necessary across nodes
			this_phase->add_command(FIND_MAX_IOWATERDEPTH);
			// upload the global max value to the devices
			this_phase->add_command(UPLOAD_IOWATERDEPTH);
		}
		// impose open boundary conditions, calling the problem-specific kernel
		// TODO see if more buffers are needed in the general case;
		// the current SA example implementations only use these
		// might possibly need some way to get this information from the problem itself
		this_phase->add_command(IMPOSE_OPEN_BOUNDARY_CONDITION)
			.set_step(step)
			.reading(state, BUFFER_POS | BUFFER_HASH | BUFFER_INFO)
			.updating(state, BUFFER_VEL | BUFFER_EULERVEL | BUFFER_TKE | BUFFER_EPSILON);
	}

	// compute boundary conditions on segments and, during the last step of the integrator,
	// also detect outgoing particles at open boundaries (the relevant information
	// is stored in the BUFFER_VERTICES array, only gets swapped in these case)
	this_phase->add_command(SA_CALC_SEGMENT_BOUNDARY_CONDITIONS)
		.set_step(step)
		.reading(state,
			BUFFER_POS | BUFFER_INFO | BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
			BUFFER_VERTPOS | BUFFER_BOUNDELEMENTS | BUFFER_VERTICES)
		.updating(state,
			BUFFER_VEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_EULERVEL | BUFFER_GRADGAMMA);
	if (MULTI_DEVICE)
		this_phase->add_command(UPDATE_EXTERNAL)
			.updating(state,
				BUFFER_VEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_EULERVEL | BUFFER_GRADGAMMA);

	// compute boundary conditions on vertices including mass variation and
	// create new particles at open boundaries.
	// TODO FIXME considering splitting new particle creation/particle property reset
	// into its own kernel, in order to provide cleaner interfaces and finer-grained
	// buffer handling
	this_phase->add_command(SA_CALC_VERTEX_BOUNDARY_CONDITIONS)
		.set_step(step)
		.set_dt(dt_op)
		.reading(state,
			BUFFER_POS | BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST | BUFFER_INFO |
			BUFFER_VERTPOS | BUFFER_VERTICES |
			BUFFER_BOUNDELEMENTS)
		.updating(state,
			BUFFER_VEL | BUFFER_EULERVEL |
			BUFFER_TKE | BUFFER_EPSILON |
			/* TODO FIXME this needs to be R/W only during init,
			 * for open boundaries and for moving objects */
			BUFFER_GRADGAMMA);

	/* Note that we don't update the cloned particles buffers, because they'll be
	 * refreshed at the next buildneibs anyway.
	 * TODO consider not doing this update altogether when there are cloned particles.
	 */
	if (MULTI_DEVICE)
		this_phase->add_command(UPDATE_EXTERNAL)
			.updating(state,
				BUFFER_VEL | BUFFER_EULERVEL |
				BUFFER_TKE | BUFFER_EPSILON |
				BUFFER_GRADGAMMA);
}

Integrator::Phase *
RepackingIntegrator::initializeInitializationSequence(StepInfo const& step)
{
	const SimParams *sp = gdata->problem->simparams();

	// did we resume? (to skip applying boundary conditions)
	const bool resumed = gdata->resume;

	Phase *this_phase = new Phase(this, "initialization preparations");

	// TODO when support for Grenier's formulation is added to models
	// with boundary conditions, the computation of the new sigma and
	// smoothed density should be moved here from the beginning of
	// runIntegratorStep.
	// TODO FIXME: the issue with moving steps such as COMPUTE_DENSITY
	// and CALC_VISC here is that then we'll need to either reorder
	// the SIGMA and SPS arrays, or recompute them anyway after a neighbors
	// list rebuilt

	// boundary-model specific boundary conditions
	if (!resumed) switch (sp->boundarytype) {
	case LJ_BOUNDARY:
	case MK_BOUNDARY:
	case DYN_BOUNDARY:
		/* nothing to do for LJ, MK and dynamic boundaries */
		break;
	case SA_BOUNDARY:
		initializeBoundaryConditionsSequence<SA_BOUNDARY>(this_phase, step);
		break;
	}

	if (step.number == 0)
		this_phase->add_command(END_OF_INIT);
	else
		this_phase->add_command(END_OF_REPACKING);

	return this_phase;
}


Integrator::Phase *
RepackingIntegrator::initializeRepackingSequence(StepInfo const& step)
{
	SimParams const* sp = gdata->problem->simparams();

	const dt_operator_t dt_op = full_timestep;

	Phase *this_phase = new Phase(this, "repacking");

	/* In the scheme we use, there are four buffers that
	 * need special treatment:
	 * * the INFO buffer is always representative of both states —in fact, because of this
	 *   and because it's part of the sorting key, it's the only property buffer
	 *   which is not double buffered;
	 * * the VERTICES buffer is always representative of both states, even though it is used
	 *   as an ephemeral buffer by FLUID particles in the open boundary case, where it's updated
	 *   in-place;
	 * * the BOUNDELEMENTS buffer is representative of both states
	 */

	static const flag_t shared_buffers =
		BUFFER_INFO |
		BUFFER_VERTICES |
		BUFFER_NEXTID |
		(MULTI_DEVICE ? BUFFER_COMPACT_DEV_MAP : BUFFER_NONE) |
		BUFFER_BOUNDELEMENTS;

	static const bool striping = gdata->clOptions->striping && MULTI_DEVICE;

	const string current_state = getCurrentStateForStep(step.number);
	const string next_state = getNextStateForStep(step.number);

	if (g_debug.inspect_preforce)
		this_phase->add_command(DEBUG_DUMP)
			.set_step(step)
			.set_src(current_state);

	// compute forces only on internal particles
	CommandStruct& forces_cmd = this_phase->add_command(striping ? FORCES_ENQUEUE : FORCES_SYNC)
		.set_step(step)
		.set_dt(dt_op)
		.reading(current_state,
			BUFFER_POS | BUFFER_HASH | BUFFER_INFO | BUFFER_CELLSTART | BUFFER_NEIBSLIST | BUFFER_VEL |
			BUFFER_RB_KEYS |
			BUFFER_VOLUME | BUFFER_SIGMA |
			BUFFER_VERTPOS | BUFFER_GRADGAMMA | BUFFER_BOUNDELEMENTS | BUFFER_EULERVEL |
			BUFFER_TKE | BUFFER_EPSILON | BUFFER_TURBVISC | BUFFER_EFFVISC)
		.writing(current_state,
			BUFFER_FORCES | BUFFER_CFL | BUFFER_CFL_TEMP |
			BUFFER_CFL_GAMMA | BUFFER_CFL_KEPS |
			BUFFER_RB_FORCES | BUFFER_RB_TORQUES |
			BUFFER_XSPH |
			/* TODO BUFFER_TAU is written by forces only in the k-epsilon case,
			 * and it is not updated across devices, is this correct?
			 */
			BUFFER_DKDE | BUFFER_TAU |
			BUFFER_INTERNAL_ENERGY_UPD);

	/* When calling FORCES_SYNC, the CFL buffers must also be read in the final
	 * post_forces() call (with FORCES_ENQUEUE, this is done by FORCES_COMPLETE below
	 * instead */
	if (!striping && gdata->dtadapt)
		forces_cmd.reading(current_state, BUFFERS_CFL & ~BUFFER_CFL_TEMP)
			.set_dt(dt_op);

	if (MULTI_DEVICE)
		this_phase->add_command(UPDATE_EXTERNAL)
			.updating(current_state,
				BUFFER_FORCES | BUFFER_XSPH | BUFFER_DKDE |
				BUFFER_INTERNAL_ENERGY_UPD);

	if (striping) {
		CommandStruct& complete_cmd = this_phase->add_command(FORCES_COMPLETE)
			.set_step(step);
		if (gdata->dtadapt)
			complete_cmd
				.reading(current_state, BUFFERS_CFL & ~BUFFER_CFL_TEMP)
				.writing(current_state, BUFFER_CFL_TEMP);
	}

	// (re)init the new status (n*),
	this_phase->add_command(INIT_STATE)
		.set_src(next_state);
	/* The buffers (re)initialized during the neighbors list construction
	 * and the INFO and HASH buffers are shared between states
	 */
	this_phase->add_command(SHARE_BUFFERS)
		.set_src(current_state)
		.set_dst(next_state)
		.set_flags(shared_buffers | SUPPORT_BUFFERS);

	this_phase->add_command(EULER)
		.set_step(step)
		.set_dt(dt_op)
		.reading(current_state,
			REPACKING_PROPS_BUFFERS | BUFFER_HASH |
			BUFFER_FORCES | BUFFER_XSPH |
			BUFFER_INTERNAL_ENERGY_UPD |
			BUFFER_DKDE)
		.writing(next_state, REPACKING_PROPS_BUFFERS);

	if (g_debug.inspect_preforce)
		this_phase->add_command(DEBUG_DUMP)
			.set_step(step)
			.set_src(next_state);

	if (sp->boundarytype == SA_BOUNDARY) {
		// with SA_BOUNDARY, if not using DENSITY_SUM, rho is integrated in EULER,
		// but we still need to integrate gamma, which needs the new position and thus
		// needs to be done after EULER
		this_phase->add_command(INTEGRATE_GAMMA)
			.set_step(step)
			.set_dt(dt_op)
			.reading(current_state,
				BUFFER_POS | BUFFER_HASH | BUFFER_INFO | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
				BUFFER_VEL |
				BUFFER_VERTPOS | BUFFER_GRADGAMMA | BUFFER_BOUNDELEMENTS)
			.updating(next_state,
				BUFFER_POS /* this is only accessed for reading */ |
				BUFFER_BOUNDELEMENTS /* this is only accessed for reading */ |
				BUFFER_VEL /* this is only accessed for reading */ |
				BUFFER_GRADGAMMA);
		if (MULTI_DEVICE)
			this_phase->add_command(UPDATE_EXTERNAL)
				.updating(next_state, BUFFER_GRADGAMMA);

	}

	// at the end of the repacking we rename step n+1 to step n, in preparation
	// for the next loop
	this_phase->add_command(RELEASE_STATE)
		.set_src("step n");
	this_phase->add_command(RENAME_STATE)
		.set_src("step n+1")
		.set_dst("step n");

	// TODO compute kinetic energy to allow stop criteria based on its decrease

	this_phase->add_command(TIME_STEP_EPILOGUE);

	return this_phase;
}

template<>
void RepackingIntegrator::initializePhase<RepackingIntegrator::POST_UPLOAD>()
{
	// Upload puts stuff in the “initial upload” state, but
	// our cycle starts from “step n”:
	Phase *post_upload = new Phase(this, "post-upload");
	post_upload->add_command(RENAME_STATE)
		.set_src("initial upload")
		.set_dst("step n");

	m_phase[POST_UPLOAD] = post_upload;
}

template<>
void RepackingIntegrator::initializePhase<RepackingIntegrator::BEGIN_TIME_STEP>()
{
	Phase *ts_begin = new Phase(this, "begin time-step");

	// Host-side prelude to a time-step
	ts_begin->add_command(TIME_STEP_PRELUDE);

	// Worker-side: simply get rid of the ephemeral buffers
	ts_begin->add_command(REMOVE_STATE_BUFFERS)
		.set_src("step n")
		.set_flags(EPHEMERAL_BUFFERS);

	m_phase[BEGIN_TIME_STEP] = ts_begin;
}

template<>
void RepackingIntegrator::initializePhase<RepackingIntegrator::NEIBS_LIST>()
{
	m_phase[NEIBS_LIST] = buildNeibsPhase(REPACKING_PROPS_BUFFERS);
}

template<>
void RepackingIntegrator::initializePhase<RepackingIntegrator::INITIALIZATION>()
{
	StepInfo step = step_info(0);
	m_phase[INITIALIZATION] = initializeInitializationSequence(step);
}

template<>
void RepackingIntegrator::initializePhase<RepackingIntegrator::REPACKING>()
{
	StepInfo step = step_info(1);
	m_phase[REPACKING] = initializeRepackingSequence(step);
}

template<>
void RepackingIntegrator::initializePhase<RepackingIntegrator::FINISH_REPACKING>()
{
	Phase *this_phase = new Phase(this, "end of repacking");

	// we only work on step n during the repacking finalization
	const string state("step n");

	this_phase->add_command(DISABLE_FREE_SURF_PARTS)
		.reading(state, BUFFER_INFO)
		.updating(state, BUFFER_POS);
	if (MULTI_DEVICE)
		this_phase->add_command(UPDATE_EXTERNAL)
			.updating(state, BUFFER_POS);

	m_phase[FINISH_REPACKING] = this_phase;
}

template<>
void RepackingIntegrator::initializePhase<RepackingIntegrator::PREPARE_SIMULATION>()
{
	StepInfo step = step_info(-1);
	m_phase[PREPARE_SIMULATION] = initializeInitializationSequence(step);
}

RepackingIntegrator::RepackingIntegrator(GlobalData const* _gdata) :
	Integrator(_gdata, "repacking"),
	m_entered_main_cycle(false),
	m_finished_main_cycle(false)
{
	// Preallocate room for all phases
	m_phase.resize(NUM_PHASES);

	initializePhase<POST_UPLOAD>();

	initializePhase<NEIBS_LIST>();

	initializePhase<INITIALIZATION>();

	initializePhase<BEGIN_TIME_STEP>();

	initializePhase<REPACKING>();

	initializePhase<FINISH_REPACKING>();

	initializePhase<PREPARE_SIMULATION>();
}

Integrator::Phase *
RepackingIntegrator::next_phase()
{
	if (m_phase_idx >= NUM_PHASES)
		throw out_of_range("phase index " + to_string(m_phase_idx) + " out of bounds");

	PhaseCode next = phase_after(PhaseCode(m_phase_idx));
	if (next == BEGIN_TIME_STEP)
		m_entered_main_cycle = true;

	Phase *phase = enter_phase(next);

	if (phase->should_run(gdata))
		return phase;

	// the phase is empty: let the user know in debug mode, and
	// tail-call ourselves
	if (g_debug.print_step) {
		cout << "\t(phase is empty)" << endl;
	}
	return next_phase();
}

RepackingIntegrator::PhaseCode
RepackingIntegrator::phase_after(RepackingIntegrator::PhaseCode cur)
{
	switch (cur) {
	case POST_UPLOAD:
		return NEIBS_LIST;
	case INITIALIZATION:
		return BEGIN_TIME_STEP;
	case BEGIN_TIME_STEP:
		return NEIBS_LIST;
	case NEIBS_LIST:
	// after the first NEIBS_LIST, run INITIALIZATION
	// after entering the main cycle, run REPACKING
	// after finishing the main cycle, run PREPARE_SIMULATION
		return
			m_finished_main_cycle ? PREPARE_SIMULATION :
			m_entered_main_cycle  ? REPACKING :
			INITIALIZATION;
	case REPACKING:
		return m_finished_main_cycle ? FINISH_REPACKING : BEGIN_TIME_STEP;

	case FINISH_REPACKING:
	// after the repacking, we rerun the neighbors list construction,
	// that filters out the disabled particles
		return NEIBS_LIST;
	default:
		throw logic_error("unknown condition after phase " + to_string(cur));
	}
}
