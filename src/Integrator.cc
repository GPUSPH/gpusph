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
#include <stdexcept>

#include "GlobalData.h"

#include "Integrator.h"

// Include all known integrator type headers, for the make_integrator switch
#include "RepackingIntegrator.h"
#include "PredictorCorrectorIntegrator.h"

/*! \file
 * Integrator implementation. For the time being this implements the predictor/corrector integration scheme only,
 * it will be refactored later to include other as well.
 */

using namespace std;

shared_ptr<Integrator>
Integrator::instance(IntegratorType type, GlobalData const* gdata)
{
	shared_ptr<Integrator> ret;
	/* During the repack phase, override the caller's choice of integrator
	 * and run the REPACK integrator instead
	 */
	if (gdata->run_mode == REPACK)
		ret = make_shared<RepackingIntegrator>(gdata);
	else switch (type)
	{
	case PREDITOR_CORRECTOR:
		ret = make_shared<PredictorCorrector>(gdata);
		break;
	default:
		throw out_of_range("no known interator #" + to_string(type));
	}
	cout << "Integrator " << ret->name() << " instantiated." << endl;
	return ret;
}

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

Integrator::Phase *
Integrator::buildNeibsPhase(flag_t import_buffers)
{
	const SimParams* sp = gdata->problem->simparams();

	import_buffers |= PARTICLE_SUPPORT_BUFFERS;

	// Some buffers can be shared between the sorted and unsorted state, because
	// they are not directly tied to the particles themselves, but the particle system
	// as a whole. The buffers that need to get shared depend on a number of conditions:
	static const bool has_forces_bodies = (sp->numforcesbodies > 0);

	// Determine if we're using planes, and thus need the BUFFER_NEIBPLANES buffer
	static const bool has_planes = QUERY_ANY_FLAGS(sp->simflags, ENABLE_PLANES | ENABLE_DEM);

	static const flag_t sorting_shared_buffers =
	// The compact device map (when present) carries over to the other state, unchanged
		(MULTI_DEVICE ? BUFFER_COMPACT_DEV_MAP : BUFFER_NONE) |
	// The object particle key buffer is static (precomputed on host, never changes),
	// so we bring it across all particle states
		(has_forces_bodies ? BUFFER_RB_KEYS : BUFFER_NONE);


	Phase *neibs_phase = new Phase(this, "build neighbors list");

	neibs_phase->should_run_if(needs_new_neibs);

	/* Initialize the neibsList Commands */
	neibs_phase->reserve(20);

	// We want to sort the particles starting from the state “step n”.
	// We remove the cell, neibslist and vertex position buffers, invalidating them.
	// They will be added to the sorted state, to be reinitialized during hash computation
	// and neighbors list construction.
	// We also drop everything which isn't an import buffer or a shared buffer.
	// (Note that we need both specifications because of some buffers (such as BUFFER_VERTPOS)
	// that would be not dropped otherwise.)
	neibs_phase->add_command(REMOVE_STATE_BUFFERS)
		.set_src("step n")
		.set_flags(NEIBS_SEQUENCE_REFRESH_BUFFERS |
			~(import_buffers | sorting_shared_buffers));

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
		// for the unsorted list, pick whatever is left of the import_buffers
		.reading("unsorted", import_buffers)
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
			.updating("sorted", import_buffers);
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
			BUFFER_NEIBSLIST | BUFFER_VERTPOS |
			(has_planes ? BUFFER_NEIBPLANES : BUFFER_NONE)
			);

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

	return neibs_phase;
}
