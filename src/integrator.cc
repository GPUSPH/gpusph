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

#include <string>

#include "GlobalData.h"

// TODO only for getCurrentStateByCommandFlags and getNextStateByCommandFlags
// remove after migration is complete
#include "GPUWorker.h"

#include "integrator.h"

/*! \file
 * Integrator implementation. For the time being this implements the predictor/corrector integration scheme only,
 * it will be refactored later to include other as well.
 */

using namespace std;

Integrator::Phase *
Integrator::enter_phase(size_t phase_idx)
{
	if (phase_idx >= m_phase.size())
		throw runtime_error("trying to enter non-existing phase #" + to_string(phase_idx));

	m_phase_idx = phase_idx;
	if (gdata->debug.print_step)
		cout << "Entering phase " << current_phase()->name() << endl;
	Phase *phase = current_phase();
	phase->reset();
	return phase;
}

//! Function that returns a time-step of 0
/** \note this is different from undefined_dt, and is used e.g.
 * for the callback dt during the initialization step
 */
float null_timestep(GlobalData const* gdata)
{ return 0.0f; }

//! Function that returns half the current time-step
float half_timestep(GlobalData const* gdata)
{ return gdata->dt/2; }

//! Function that returns the full current time-step
float full_timestep(GlobalData const* gdata)
{ return gdata->dt; }

dt_operator_t
PredictorCorrector::getDtOperatorForStep(int step_num)
{
	return
		/* prepare for the simulation, so reset to 0 */
		step_num == 0 ? null_timestep :
		/* end of predictor, prepare for corrector, where forces
		 * will be computed at t + dt/2
		 */
		step_num == 1 ? half_timestep :
		/* end of corrector, prepare for next predictor, where forces
		 * will be computed at t + dt (t and dt are still the one
		 * for the current whole step)
		 */
						full_timestep ;
}

string
PredictorCorrector::getCurrentStateForStep(int step_num)
{
	switch (step_num) {
	case 0:
	case 1:
		return "step n";
	case 2:
		return "step n*";
	default:
		throw runtime_error("cannot determine current state for step #" + to_string(step_num));
	}
}

string
PredictorCorrector::getNextStateForStep(int step_num)
{
	switch (step_num) {
	case 0:
		return "step n";
	case 1:
		return "step n*";
	case 2:
		return "step n+1";
	default:
		throw runtime_error("cannot determine next state for step #" + to_string(step_num));
	}
}

template<BoundaryType boundarytype>
void PredictorCorrector::initializeBoundaryConditionsSequence
	(Integrator::Phase *this_phase, int step_num)
{ /* for most boundary models, there's nothing to do */ }

// formerly saBoundaryConditions()
//! Initialize the sequence of commands needed to apply SA_BOUNDARY boundary conditions.
/*! This will _not_ be called for the initialization step if we resumed,
 * since the arrays were already correctly initialized before writing
 */
template<>
void PredictorCorrector::initializeBoundaryConditionsSequence<SA_BOUNDARY>
	(Integrator::Phase *this_phase, int step_num)
{
	const SimParams *sp = gdata->problem->simparams();
	const bool has_io = sp->simflags & ENABLE_INLET_OUTLET;

	const dt_operator_t dt_op = getDtOperatorForStep(step_num);

	// In the open boundary case, the last integration step is when we generate
	// and destroy particles
	const bool last_io_step = has_io && (step_num == 2);
	const flag_t integrator_step = INITIALIZATION_STEP << step_num;
	/* Boundary conditions are applied to step n during initialization step,
	 * to step n* after the integrator, and to step n+1 after the corrector
	 */
	const string state = getNextStateForStep(step_num);

	if (gdata->simframework->getBCEngine() == NULL)
		throw runtime_error("no boundary conditions engine loaded");

	// initialization, only if not resuming
	if (step_num == 0) {
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

		// modify particle mass on open boundaries
		if (has_io) {

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
		}

	}

	// impose open boundary conditions
	if (has_io) {
		// reduce the water depth at pressure outlets if required
		// if we have multiple devices then we need to run a global max on the different gpus / nodes
		if (MULTI_DEVICE && sp->simflags & ENABLE_WATER_DEPTH) {
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
			.set_flags(integrator_step)
			.reading(state, BUFFER_POS | BUFFER_HASH | BUFFER_INFO)
			.updating(state, BUFFER_VEL | BUFFER_EULERVEL | BUFFER_TKE | BUFFER_EPSILON);
	}

	// compute boundary conditions on segments and, during the last step of the integrator,
	// also detect outgoing particles at open boundaries (the relevant information
	// is stored in the BUFFER_VERTICES array, only gets swapped in these case)
	this_phase->add_command(SA_CALC_SEGMENT_BOUNDARY_CONDITIONS)
		.set_flags(integrator_step)
		.reading(state,
			BUFFER_POS | BUFFER_INFO | BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
			BUFFER_VERTPOS | BUFFER_BOUNDELEMENTS | BUFFER_VERTICES)
		.updating(state,
			BUFFER_VEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_EULERVEL | BUFFER_GRADGAMMA);
	if (last_io_step)
		this_phase->add_command(FIND_OUTGOING_SEGMENT)
			.reading(state,
				BUFFER_POS | BUFFER_INFO | BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
				BUFFER_VEL |
				BUFFER_VERTPOS | BUFFER_BOUNDELEMENTS)
			.updating(state,
				BUFFER_GRADGAMMA | BUFFER_VERTICES);
	if (MULTI_DEVICE)
		this_phase->add_command(UPDATE_EXTERNAL)
			.updating(state,
				BUFFER_VEL | BUFFER_TKE | BUFFER_EPSILON | BUFFER_EULERVEL | BUFFER_GRADGAMMA |
				(last_io_step ? BUFFER_VERTICES : BUFFER_NONE));

	// compute boundary conditions on vertices including mass variation and
	// create new particles at open boundaries.
	// TODO FIXME considering splitting new particle creation/particle property reset
	// into its own kernel, in order to provide cleaner interfaces and finer-grained
	// buffer handling
	CommandStruct& vertex_bc_cmd = this_phase->add_command(SA_CALC_VERTEX_BOUNDARY_CONDITIONS)
		.set_flags(integrator_step)
		.set_dt(dt_op)
		.reading(state,
			BUFFER_POS | BUFFER_HASH | BUFFER_CELLSTART | BUFFER_NEIBSLIST | BUFFER_INFO |
			BUFFER_VERTPOS | BUFFER_VERTICES |
			BUFFER_BOUNDELEMENTS)
		.updating(state,
			(has_io ? BUFFER_POS : BUFFER_NONE) |
			BUFFER_VEL | BUFFER_EULERVEL |
			BUFFER_TKE | BUFFER_EPSILON |
			/* TODO FIXME this needs to be R/W only during init,
			 * for open boundaries and for moving objects */
			BUFFER_GRADGAMMA);
	/* If this is the last step and open boundaries are enabled, also add the buffers
	 * for cloning in the writing set
	 */
	if (last_io_step)
		vertex_bc_cmd.writing(state,
			BUFFER_FORCES | BUFFER_INFO | BUFFER_HASH |
			BUFFER_VERTICES | BUFFER_BOUNDELEMENTS | BUFFER_NEXTID);

	/* Note that we don't update the cloned particles buffers, because they'll be
	 * refreshed at the next buildneibs anyway.
	 * TODO consider not doing this update altogether when there are cloned particles.
	 */
	if (MULTI_DEVICE)
		this_phase->add_command(UPDATE_EXTERNAL)
			.updating(state,
				(has_io ? BUFFER_POS : BUFFER_NONE) |
				BUFFER_VEL | BUFFER_EULERVEL |
				BUFFER_TKE | BUFFER_EPSILON |
				BUFFER_GRADGAMMA);

	// check if we need to delete some particles which passed through open boundaries
	if (last_io_step) {
		this_phase->add_command(DISABLE_OUTGOING_PARTS)
			.reading(state, BUFFER_INFO)
			.updating(state, BUFFER_POS | BUFFER_VERTICES);
		if (MULTI_DEVICE)
			this_phase->add_command(UPDATE_EXTERNAL)
				.updating(state, BUFFER_POS | BUFFER_VERTICES);
	}
}

Integrator::Phase *
PredictorCorrector::initializeNextStepSequence(int step_num)
{
	const flag_t integrator_step = INITIALIZATION_STEP << step_num;
	const SimParams *sp = gdata->problem->simparams();

	const dt_operator_t dt_op = getDtOperatorForStep(step_num);

	// “resumed” condition applies to the initializaiton step sequence,
	// if we resumed
	const bool resumed = (step_num == 0 && !gdata->clOptions->resume_fname.empty());

	Phase *this_phase = new Phase(
		step_num == 0 ? "initialization preparations" :
		step_num == 1 ? "post-predictor preparations" :
		step_num == 2 ? "post-corrector preparations" : "this can't happen");

	if (step_num == 2 && sp->numbodies > 0)
		this_phase->add_command(EULER_UPLOAD_OBJECTS_CG);

	// variable gravity
	if (sp->gcallback) {
		this_phase->add_command(RUN_CALLBACKS)
			.set_flags(integrator_step)
			.set_dt(dt_op);
		this_phase->add_command(UPLOAD_GRAVITY);
	}

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
		initializeBoundaryConditionsSequence<SA_BOUNDARY>(this_phase, step_num);
		break;
	}

	// open boundaries: new particle generation, only at the end of the corrector
	if (step_num == 2 && sp->simflags & ENABLE_INLET_OUTLET)
	{
		this_phase->add_command(DOWNLOAD_NEWNUMPARTS);
		this_phase->add_command(CHECK_NEWNUMPARTS);
	}

	// at the end of the corrector we rename step n+1 to step n, in preparation
	// for the next loop
	if (step_num == 2) {
		this_phase->add_command(RELEASE_STATE)
			.set_src("step n");
		this_phase->add_command(RENAME_STATE)
			.set_src("step n+1")
			.set_dst("step n");
		this_phase->add_command(TIME_STEP_EPILOGUE);
	}

	if (step_num == 0)
		this_phase->add_command(END_OF_INIT);

	return this_phase;
}

Integrator::Phase *
PredictorCorrector::initializePredCorrSequence(int step_num)
{
	const flag_t integrator_step = INITIALIZATION_STEP << step_num;
	SimParams const* sp = gdata->problem->simparams();

	const dt_operator_t dt_op = getDtOperatorForStep(step_num);

	Phase *this_phase = new Phase(
		step_num == 1 ? "predictor" :
		step_num == 2 ? "corrector" : "this can't happen");

	/* In the predictor/corrector scheme we use, there are four buffers that
	 * need special treatment:
	 * * the INFO buffer is always representative of both states —in fact, because of this
	 *   and because it's part of the sorting key, it's the only property buffer
	 *   which is not double buffered;
	 * * the VERTICES buffer is always representative of both states, even though it is used
	 *   as an ephemeral buffer by FLUID particles in the open boundary case, where it's updated
	 *   in-place;
	 * * the NEXTID buffer is only ever updated at the end of the second step, and it is
	 *   representative of the previous state only until it gets updated, when it is
	 *   representative of the next state only;
	 * * the BOUNDELEMENTS buffer is representative of both states if there are no moving
	 *   boundaries, otherwise is follows the behavior of the other buffers
	 */

	static const bool has_moving_bodies = (sp->simflags & ENABLE_MOVING_BODIES);
	static const flag_t shared_buffers =
		BUFFER_INFO |
		BUFFER_VERTICES |
		BUFFER_NEXTID |
		(MULTI_DEVICE ? BUFFER_COMPACT_DEV_MAP : BUFFER_NONE) |
		(has_moving_bodies ? BUFFER_NONE : BUFFER_BOUNDELEMENTS);

	static const bool striping = gdata->clOptions->striping && MULTI_DEVICE;

	// TODO get from integrator
	// for both steps, the “starting point” for Euler and density summation is step n
	const string base_state = "step n";
	// current state is step n for the predictor, step n* for the corrector
	const string current_state = getCurrentStateForStep(step_num);
	// next state is step n* for the predictor, step n+1 for the corrector
	const string next_state = getNextStateForStep(step_num);

	// at the beginning of the corrector, we move all ephemeral buffers from step n
	// to the new step n*
	if (step_num == 2)
		this_phase->add_command(MOVE_STATE_BUFFERS)
			.set_src("step n")
			.set_dst("step n*")
			.set_flags( EPHEMERAL_BUFFERS & ~(BUFFER_PARTINDEX | POST_PROCESS_BUFFERS) );


	// for Grenier formulation, compute sigma and smoothed density
	// TODO with boundary models requiring kernels for boundary conditions,
	// this should be moved into prepareNextStep
	if (sp->sph_formulation == SPH_GRENIER) {
		// compute density and sigma, updating WRITE vel in-place
		this_phase->add_command(COMPUTE_DENSITY)
			.set_flags(integrator_step)
			.reading(current_state,
				BUFFER_POS | BUFFER_HASH | BUFFER_INFO | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
				BUFFER_VOLUME)
			.updating(current_state, BUFFER_VEL)
			.writing(current_state, BUFFER_SIGMA);
		if (MULTI_DEVICE)
			this_phase->add_command(UPDATE_EXTERNAL)
				.updating(current_state, BUFFER_SIGMA | BUFFER_VEL);
	}

	// for SPS viscosity, compute first array of tau and exchange with neighbors
	if (sp->turbmodel == SPS) {
		this_phase->add_command(CALC_VISC)
			.set_flags(integrator_step)
			.reading(current_state,
				BUFFER_POS | BUFFER_HASH | BUFFER_INFO | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
				BUFFER_VEL)
			.writing(current_state,
				BUFFER_TAU | BUFFER_SPS_TURBVISC);

		if (MULTI_DEVICE)
			this_phase->add_command(UPDATE_EXTERNAL)
				.updating(current_state, BUFFER_TAU);
	}

	if (gdata->debug.inspect_preforce)
		this_phase->add_command(DEBUG_DUMP)
			.set_src(current_state)
			.set_flags(integrator_step);

	// compute forces only on internal particles
	CommandStruct& forces_cmd = this_phase->add_command(striping ? FORCES_ENQUEUE : FORCES_SYNC)
		.set_flags(integrator_step)
		.set_dt(dt_op)
		.reading(current_state,
			BUFFER_POS | BUFFER_HASH | BUFFER_INFO | BUFFER_CELLSTART | BUFFER_NEIBSLIST | BUFFER_VEL |
			BUFFER_RB_KEYS |
			BUFFER_VOLUME | BUFFER_SIGMA |
			BUFFER_VERTPOS | BUFFER_GRADGAMMA | BUFFER_BOUNDELEMENTS | BUFFER_EULERVEL |
			BUFFER_TKE | BUFFER_EPSILON | BUFFER_TURBVISC)
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
			.set_flags(integrator_step);
		if (gdata->dtadapt)
			complete_cmd
				.reading(current_state, BUFFERS_CFL & ~BUFFER_CFL_TEMP)
				.writing(current_state, BUFFER_CFL_TEMP);
	}

	// Take care of moving bodies
	this_phase->add_command(MOVE_BODIES)
		.set_flags(integrator_step)
		.set_dt(dt_op)
		.set_src(current_state);

	// On the predictor, we need to (re)init the predicted status (n*),
	// on the corrector this will be updated (in place) to the corrected status (n+1)
	if (step_num == 1) {
		this_phase->add_command(INIT_STATE)
			.set_src("step n*");
		/* The buffers (re)initialized during the neighbors list construction
		 * and the INFO and HASH buffers are shared between states
		 */
		this_phase->add_command(SHARE_BUFFERS)
			.set_src("step n")
			.set_dst("step n*")
			.set_flags(shared_buffers | SUPPORT_BUFFERS);
	}

	CommandStruct& euler_cmd = this_phase->add_command(EULER)
		.set_flags(integrator_step)
		.set_dt(dt_op)
		// these are always taken from step n
		.reading(base_state, PARTICLE_PROPS_BUFFERS | BUFFER_HASH)
		// these are always taken from the current step
		.reading(current_state,
			BUFFER_FORCES | BUFFER_XSPH |
			BUFFER_INTERNAL_ENERGY_UPD |
			BUFFER_DKDE);

	// now, the difference:
	if (step_num == 1) {
		// predictor: the next state is empty, so we mark all the props buffer as writing:
		euler_cmd.writing(next_state, PARTICLE_PROPS_BUFFERS);
	} else {
		// corrector: we update the “current” state (step n*)
		euler_cmd.updating(current_state, PARTICLE_PROPS_BUFFERS)
		// and then rename it to step n+1; for another usage of this syntax,
		// see also the enqueue of the SORT command
			.set_src(current_state)
			.set_dst(next_state);
	}

	if (gdata->debug.inspect_preforce)
		this_phase->add_command(DEBUG_DUMP)
			.set_src(next_state)
			.set_flags(integrator_step);

	if (sp->simflags & ENABLE_DENSITY_SUM) {
		// the forces were computed in the base state for the predictor,
		// on the next state for the corrector
		// or as an alternative we could free BUFFER_FORCES from whatever state it's in
		const string forces_state = (step_num == 1 ? base_state : next_state);

		this_phase->add_command(DENSITY_SUM)
			.set_flags(integrator_step)
			.set_dt(dt_op)
			.reading(base_state, /* always read the base state, like EULER */
				BUFFER_POS | BUFFER_HASH | BUFFER_INFO | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
				BUFFER_VEL |
				BUFFER_VERTPOS | BUFFER_EULERVEL | BUFFER_GRADGAMMA | BUFFER_BOUNDELEMENTS)
			.updating(next_state,
				BUFFER_POS /* this is only accessed for reading */ |
				BUFFER_EULERVEL /* this is only accessed for reading */ |
				BUFFER_BOUNDELEMENTS /* this is only accessed for reading */ |
				BUFFER_VEL | BUFFER_GRADGAMMA)
			.writing(forces_state, BUFFER_FORCES);

		if (MULTI_DEVICE)
			this_phase->add_command(UPDATE_EXTERNAL)
				.updating(next_state, BUFFER_VEL | BUFFER_GRADGAMMA);

		// when using density sum, density diffusion is applied _after_ the density sum
		if (sp->densitydiffusiontype != DENSITY_DIFFUSION_NONE) {
			this_phase->add_command(CALC_DENSITY_DIFFUSION)
				.set_flags(integrator_step)
				.set_dt(dt_op)
				.reading(next_state,
					BUFFER_POS | BUFFER_HASH | BUFFER_INFO | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
					BUFFER_VEL |
					BUFFER_VERTPOS | BUFFER_GRADGAMMA | BUFFER_BOUNDELEMENTS)
				.writing(forces_state, BUFFER_FORCES);
			this_phase->add_command(APPLY_DENSITY_DIFFUSION)
				.set_flags(integrator_step)
				.set_dt(dt_op)
				.reading(next_state, BUFFER_INFO)
				.reading(forces_state, BUFFER_FORCES)
				.updating(next_state, BUFFER_VEL);

			if (MULTI_DEVICE)
				this_phase->add_command(UPDATE_EXTERNAL)
					.updating(next_state, BUFFER_VEL);
		}
	} else if (sp->boundarytype == SA_BOUNDARY) {
		// with SA_BOUNDARY, if not using DENSITY_SUM, rho is integrated in EULER,
		// but we still need to integrate gamma, which needs the new position and thus
		// needs to be done after EULER
		this_phase->add_command(INTEGRATE_GAMMA)
			.set_flags(integrator_step)
			.set_dt(dt_op)
			.reading(base_state, /* as in the EULER case, we always read from step n */
				BUFFER_POS | BUFFER_HASH | BUFFER_INFO | BUFFER_CELLSTART | BUFFER_NEIBSLIST |
				BUFFER_VEL |
				BUFFER_VERTPOS | BUFFER_EULERVEL | BUFFER_GRADGAMMA | BUFFER_BOUNDELEMENTS)
			.updating(next_state,
				BUFFER_POS /* this is only accessed for reading */ |
				BUFFER_EULERVEL /* this is only accessed for reading */ |
				BUFFER_BOUNDELEMENTS /* this is only accessed for reading */ |
				BUFFER_VEL /* this is only accessed for reading */ |
				BUFFER_GRADGAMMA);
		if (MULTI_DEVICE)
			this_phase->add_command(UPDATE_EXTERNAL)
				.updating(next_state, BUFFER_GRADGAMMA);

	}

	return this_phase;
}

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::POST_UPLOAD>()
{
	// Upload puts stuff in the “initial upload” state, but
	// our cycle starts from “step n”:
	Phase *post_upload = new Phase("post-upload");
	post_upload->add_command(RENAME_STATE)
		.set_src("initial upload")
		.set_dst("step n");

	m_phase[POST_UPLOAD] = post_upload;
}

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::BEGIN_TIME_STEP>()
{
	Phase *ts_begin = new Phase("begin time-step");

	// Host-side prelude to a time-step
	ts_begin->add_command(TIME_STEP_PRELUDE);

	// Worker-side: simply get rid of the ephemeral buffers
	ts_begin->add_command(REMOVE_STATE_BUFFERS)
		.set_src("step n")
		.set_flags(EPHEMERAL_BUFFERS);

	m_phase[BEGIN_TIME_STEP] = ts_begin;
}

//! A function that determines if we should build the neighbors list
/**! This is only done every buildneibsfreq or if particles got created,
 * but only if we didn't do it already in this iteration
 */
bool needs_new_neibs(Integrator::Phase const*, GlobalData const* gdata)
{
	const unsigned long iterations = gdata->iterations;
	const SimParams* sp = gdata->problem->simparams();

	return (iterations != gdata->last_buildneibs_iteration) &&
		((iterations % sp->buildneibsfreq == 0) || gdata->particlesCreated);
}

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::NEIBS_LIST>()
{
	const SimParams* sp = gdata->problem->simparams();

	Phase *neibs_phase = new Phase("build neighbors list");

	neibs_phase->should_run_if(needs_new_neibs);

	/* Initialize the neibsList Commands */
	neibs_phase->reserve(20);

	// We want to sort the particles starting from the state “step n”.
	// We remove the cell, neibslist and vertex position buffers, invalidating them.
	// They will be added to the sorted state, to be reinitialized during hash computation
	// and neighbors list construction
	neibs_phase->add_command(REMOVE_STATE_BUFFERS)
		.set_src("step n")
		.set_flags(NEIBS_SEQUENCE_REFRESH_BUFFERS);

	// Rename the state to “unsorted”
	neibs_phase->add_command(RENAME_STATE)
		.set_src("step n")
		.set_dst("unsorted");

	// Initialize the new particle system state (“sorted”) with all particle properties
	// (except for BUFFER_INFO, which will be sorted in-place), plus the auxiliary buffers
	// that get rebuilt during the sort and neighbors list construction
	// (cell start/end, vertex relative positions and the neiblists itself)
	neibs_phase->add_command(INIT_STATE)
		.set_src("sorted");

	// Some buffers can be shared between the sorted and unsorted state, because
	// they are not directly tied to the particles themselves, but the particle system
	// as a whole. The buffers that need to get shared depend on a number of conditions:

	static const flag_t has_forces_bodies = (sp->numforcesbodies > 0);

	static const flag_t sorting_shared_buffers =
	// The compact device map (when present) carries over to the other state, unchanged
		(MULTI_DEVICE ? BUFFER_COMPACT_DEV_MAP : BUFFER_NONE) |
	// The object particle key buffer is static (precomputed on host, never changes),
	// so we bring it across all particle states
		(has_forces_bodies ? BUFFER_RB_KEYS : BUFFER_NONE);

	if (sorting_shared_buffers != BUFFER_NONE)
		neibs_phase->add_command(SHARE_BUFFERS)
			.set_src("unsorted")
			.set_dst("sorted")
			.set_flags(sorting_shared_buffers);

	neibs_phase->add_command(CALCHASH)
		.reading("unsorted", BUFFER_INFO | BUFFER_COMPACT_DEV_MAP)
		.updating("unsorted", BUFFER_POS | BUFFER_HASH)
		.writing("unsorted", BUFFER_PARTINDEX);

	// reorder PARTINDEX by HASH and INFO (also sorts HASH and INFO)
	// reordering is done in-place, so we also rename the state of these buffers
	// from unsorted to sorted
	neibs_phase->add_command(SORT)
		.set_src("unsorted")
		.set_dst("sorted")
		.updating("unsorted", BUFFER_INFO | BUFFER_HASH | BUFFER_PARTINDEX);

	// reorder everything else
	// note that, as a command, REORDER is a special case: one of the buffer specifications
	// for the writing list is empty because it can only be determined at the runCommand<>
	// level, by taking the buffers that were take from the reading list
	neibs_phase->add_command(REORDER)
		// for the unsorted list, pick whatever is left of the IMPORT_BUFFERS
		.reading("unsorted", IMPORT_BUFFERS)
		// the buffers sorted in SORT are marked “updating”, but will actually be read-only
		.updating("sorted", BUFFER_INFO | BUFFER_HASH | BUFFER_PARTINDEX)
		// no buffer specification, meaning “take the reading buffer list"
		.writing("sorted", BUFFER_NONE)
		// and we also want these
		.writing("sorted", BUFFERS_CELL);

	// we don't need the unsorted state anymore
	neibs_phase->add_command(RELEASE_STATE)
		.set_src("unsorted");

	// we don't need the PARTINDEX buffer anymore
	// TODO since we only need PARTINDEX during sorting, and other ephemeral buffers
	// such as FORCES only outside of sorting, we could spare some memory recycling
	// one such ephemeral buffer in place of PARTINDEX
	neibs_phase->add_command(REMOVE_STATE_BUFFERS)
		.set_src("sorted")
		.set_flags(BUFFER_PARTINDEX);

	// get the new number of particles: with inlet/outlets, they
	// may have changed because of incoming/outgoing particle, otherwise
	// some particles might have been disabled (and discarded) for flying
	// out of the domain
	neibs_phase->add_command(DOWNLOAD_NEWNUMPARTS);

	// if running on multiple GPUs, update the external cells
	if (MULTI_DEVICE) {
		// copy cellStarts, cellEnds and segments on host
		neibs_phase->add_command(DUMP_CELLS)
			.reading("sorted", BUFFER_CELLSTART | BUFFER_CELLEND);

		neibs_phase->add_command(UPDATE_SEGMENTS);

		// here or later, before update indices: MPI_Allgather (&sendbuf,sendcount,sendtype,&recvbuf, recvcount,recvtype,comm)
		// maybe overlapping with dumping cells (run async before dumping the cells)
	}

	// update particle offsets —this is a host command, and
	// doesn't affect the device buffers directly. we do it in both the single- and multi-device case
	neibs_phase->add_command(UPDATE_ARRAY_INDICES);

	// if running on multiple GPUs, rebuild the external copies
	if (MULTI_DEVICE) {
		// crop external cells
		neibs_phase->add_command(CROP);
		// append fresh copies of the externals
		// NOTE: this imports also particle hashes without resetting the high bits, which are wrong
		// until next calchash; however, they are filtered out when using the particle hashes.
		neibs_phase->add_command(APPEND_EXTERNAL)
			.updating("sorted", IMPORT_BUFFERS);
		// update the newNumParticles device counter
		if (sp->simflags & ENABLE_INLET_OUTLET)
			neibs_phase->add_command(UPLOAD_NEWNUMPARTS);
	}

	// run the actual neighbors list construction
	neibs_phase->add_command(BUILDNEIBS)
		.reading("sorted",
			BUFFER_POS | BUFFER_INFO | BUFFER_HASH |
			BUFFER_VERTICES | BUFFER_BOUNDELEMENTS |
			BUFFER_CELLSTART | BUFFER_CELLEND)
		.writing("sorted",
			BUFFER_NEIBSLIST | BUFFER_VERTPOS);

	// BUFFER_VERTPOS needs to be synchronized with the adjacent devices
	if (MULTI_DEVICE && sp->boundarytype == SA_BOUNDARY)
		neibs_phase->add_command(UPDATE_EXTERNAL).updating("sorted", BUFFER_VERTPOS);

	// we're done, rename the state to “step n” for what follows
	neibs_phase->add_command(RENAME_STATE)
		.set_src("sorted")
		.set_dst("step n");

	// host command: check we don't have too many neighbors
	neibs_phase->add_command(CHECK_NEIBSNUM);

	neibs_phase->add_command(HANDLE_HOTWRITE);

	m_phase[NEIBS_LIST] = neibs_phase;
}

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::INITIALIZATION>()
{ m_phase[INITIALIZATION] = initializeNextStepSequence(0); }

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::PREDICTOR>()
{ m_phase[PREDICTOR] = initializePredCorrSequence(1); }

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::PREDICTOR_END>()
{ m_phase[PREDICTOR_END] = initializeNextStepSequence(1); }

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::CORRECTOR>()
{ m_phase[CORRECTOR] = initializePredCorrSequence(2); }

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::CORRECTOR_END>()
{ m_phase[CORRECTOR_END] = initializeNextStepSequence(2); }

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::FILTER_INTRO>()
{
	Phase *filter_intro = new Phase("filter intro");

	/* Things to do before looping over any filter */
	filter_intro->add_command(RENAME_STATE)
		.set_src("step n")
		.set_dst("unfiltered");
	filter_intro->add_command(INIT_STATE)
		.set_src("filtered");

	m_phase[FILTER_INTRO] = filter_intro;
}

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::FILTER_CALL>()
{
	Phase *filter_call = new Phase("filter call");

	/* Command sequence for a single filter
	 * The caller should set the command flag with the filter name
	 */
	filter_call->add_command(FILTER)
		// TODO currently runCommand<FILTER> knows that it needs to get the whole state
		// and there will be a single reading specification
		.reading("unfiltered", BUFFER_NONE)
		.writing("filtered", BUFFER_VEL);
	if (MULTI_DEVICE)
		filter_call->add_command(UPDATE_EXTERNAL)
			.updating("filtered", BUFFER_VEL);

	// swap buffers between filtered and unfiltered: this moves
	// the filtered buffer into the unfiltered state (as in: unfiltered for the
	// next filter, if any), and moves what was in the unfiltered state to filtered:
	// the latter is marked invalid, in preparation for the next filter (if any)
	filter_call->add_command(SWAP_STATE_BUFFERS)
		.set_src("unfiltered")
		.set_dst("filtered")
		.set_flags(BUFFER_VEL);


	m_phase[FILTER_CALL] = filter_call;
}

template<>
void PredictorCorrector::initializePhase<PredictorCorrector::FILTER_OUTRO>()
{
	Phase *filter_outro = new Phase("filter outro");

	/* Things to do after looping over filters */
	// a bit of a paradox: the state rename is done for the unfiltered state,
	// since this is where the previously filtered velocity has been moved with the last swap
	filter_outro->add_command(RENAME_STATE)
		.set_src("unfiltered")
		.set_dst("step n");
	filter_outro->add_command(RELEASE_STATE)
		.set_src("filtered");

	m_phase[FILTER_OUTRO] = filter_outro;
}


PredictorCorrector::PredictorCorrector(GlobalData const* _gdata) :
	Integrator(_gdata, "predictor/corrector"),
	m_needs_preparation(false),
	m_entered_main_cycle(false),
	m_enabled_filters(_gdata->simframework->getFilterFreqList()),
	m_current_filter(m_enabled_filters.cend())
{
	const SimParams *sp = gdata->problem->simparams();

	// Some formulations require stuff to be done before the beginning of the
	// main loop (partially, this is stuff that is also done at the end of each
	// time-step, but since there is no time-step preceding the first one,
	// we do it in the INITIALIZATION phase
	m_needs_preparation = sp->gcallback || sp->boundarytype == SA_BOUNDARY;

	// Preallocate room for all phases
	m_phase.resize(NUM_PHASES);

	initializePhase<POST_UPLOAD>();

	initializePhase<NEIBS_LIST>();

	initializePhase<INITIALIZATION>();

	initializePhase<BEGIN_TIME_STEP>();

	initializePhase<PREDICTOR>();
	initializePhase<PREDICTOR_END>();
	initializePhase<CORRECTOR>();
	initializePhase<CORRECTOR_END>();

	initializePhase<FILTER_INTRO>();
	initializePhase<FILTER_CALL>();
	initializePhase<FILTER_OUTRO>();

}

Integrator::Phase *
PredictorCorrector::next_phase()
{
	if (m_phase_idx >= NUM_PHASES)
		throw out_of_range("phase index " + to_string(m_phase_idx) + " out of bounds");

	PhaseCode next = phase_after(PhaseCode(m_phase_idx));
	if (next == BEGIN_TIME_STEP)
		m_entered_main_cycle = true;

	// set the flag of of the FILTER command
	// TODO FIXME this is sub-optimal, but I cannot think of a better way to handle this,
	// at least presently.
	// A possible option for the future would be to make each FILTER_CALL a separate command,
	// with its own properly-set flag, and use conditional invokation.
	if (next == FILTER_CALL) {
		CommandStruct& filter_cmd = m_phase[FILTER_CALL]->edit_command(0);
		if (filter_cmd.command != FILTER)
			throw std::logic_error("mismatch FILTER_CALL command");
		filter_cmd.clear_flags(ALL_FLAGS).set_flags(m_current_filter->first);
	}

	Phase *phase = enter_phase(next);

	if (phase->should_run(gdata))
		return phase;

	// the phase is empty: let the user know in debug mode, and
	// tail-call ourselves
	if (gdata->debug.print_step) {
		cout << "\t(phase is empty)" << endl;
	}
	return next_phase();
}

PredictorCorrector::PhaseCode
PredictorCorrector::phase_after(PredictorCorrector::PhaseCode cur)
{
	switch (cur) {
	case POST_UPLOAD:
		return NEIBS_LIST;
	case INITIALIZATION:
		return BEGIN_TIME_STEP;
	case BEGIN_TIME_STEP:
		return NEIBS_LIST;
	case PREDICTOR:
		return PREDICTOR_END;
	case PREDICTOR_END:
		return CORRECTOR;
	case CORRECTOR:
		return CORRECTOR_END;
	case CORRECTOR_END:
		return BEGIN_TIME_STEP;
	case FILTER_OUTRO:
		return PREDICTOR;
	default:
		break;
		// the other need special handling
	}

	const SimParams *sp = gdata->problem->simparams();
	const unsigned long iterations = gdata->iterations;
	static const FilterFreqList::const_iterator filters_end = m_enabled_filters.cend();

	// after the first NEIBS_LIST, run INITIALIZATION
	// otherwise, run FILTER_INTRO (if appropriate)
	// otherwise, run the PREDICTOR
	if (cur == NEIBS_LIST) {
		if (!m_entered_main_cycle)
			return INITIALIZATION;
		if (iterations > 0 && m_enabled_filters.size() > 0)
			return FILTER_INTRO;
		return PREDICTOR;
	}

	// following a FILTER_INTRO or a FILTER phase,
	// we go to the next FILTER phase or to FILTER_OUTRO
	if (cur == FILTER_INTRO) {
		m_current_filter = m_enabled_filters.cbegin();
	} else if (cur == FILTER_CALL) {
		++m_current_filter;
	}

	if (cur == FILTER_INTRO || cur == FILTER_CALL) {
		// Find a filter that should run at this iteration
		while (m_current_filter != filters_end) {
			uint freq = m_current_filter->second;
			if (iterations % freq == 0)
				break;
			++m_current_filter;
		}

		if (m_current_filter == filters_end)
			return FILTER_OUTRO;
		return FILTER_CALL;
	}

	throw logic_error("unknown condition after phase " + to_string(cur));
}
