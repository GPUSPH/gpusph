/*  Copyright (c) 2012-2019 INGV, EDF, UniCT, JHU

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
 * Implementation of the CUDA-based GPU worker
 *
 * \todo The CUDA-independent part should be split in a separate, generic
 * Worker class.
 */

// ostringstream
#include <sstream>
// FLT_MAX
#include <cfloat>

#include "GPUWorker.h"
#include "cudautil.h"

#include "cudabuffer.h"
#include "hostbuffer.h"

// round_up
#include "utils.h"

// UINT_MAX
#include "limits.h"

using namespace std;

GPUWorker::GPUWorker(GlobalData* _gdata, devcount_t _deviceIndex) :
	gdata(_gdata),

	m_simframework(gdata->simframework),
	m_physparams(gdata->problem->physparams()),
	m_simparams(gdata->problem->simparams()),

	neibsEngine(gdata->simframework->getNeibsEngine()),
	viscEngine(gdata->simframework->getViscEngine()),
	forcesEngine(gdata->simframework->getForcesEngine()),
	integrationEngine(gdata->simframework->getIntegrationEngine()),
	bcEngine(gdata->simframework->getBCEngine()),
	filterEngines(gdata->simframework->getFilterEngines()),
	postProcEngines(gdata->simframework->getPostProcEngines()),

	m_max_kinvisc(NAN),
	m_max_sound_speed(NAN),

	m_deviceIndex(_deviceIndex),
	m_globalDeviceIdx(GlobalData::GLOBAL_DEVICE_ID(gdata->mpi_rank, _deviceIndex)),
	m_cudaDeviceNumber(gdata->device[_deviceIndex]),

	// Problem::fillparts() has already been called
	m_numParticles(gdata->s_hPartsPerDevice[_deviceIndex]),
	m_nGridCells(gdata->nGridCells),
	m_numAllocatedParticles(0),
	m_numInternalParticles(m_numParticles),
	m_numForcesBodiesParticles(gdata->problem->get_forces_bodies_numparts()),

	m_particleRangeBegin(0),
	m_particleRangeEnd(m_numInternalParticles),

	m_hostMemory(0),
	m_deviceMemory(0),

	// set to true to force host staging even if peer access is set successfully
	m_disableP2Ptranfers(false),
	m_hPeerTransferBuffer(NULL),
	m_hPeerTransferBufferSize(0),

	// used if GPUDirect is disabled
	m_hNetworkTransferBuffer(NULL),
	m_hNetworkTransferBufferSize(0),

	m_dSegmentStart(NULL),
	m_dIOwaterdepth(NULL),
	m_dNewNumParticles(NULL),

	m_forcesKernelTotalNumBlocks(),

	m_asyncH2DCopiesStream(0),
	m_asyncD2HCopiesStream(0),
	m_asyncPeerCopiesStream(0),
	m_halfForcesEvent(0)
{
	printf("number of forces rigid bodies particles = %d\n", m_numForcesBodiesParticles);

	m_dBuffers.setAllocPolicy(gdata->simframework->getAllocPolicy());

	m_dBuffers.addBuffer<CUDABuffer, BUFFER_POS>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_VEL>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_INFO>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_FORCES>(0);

	if (m_simparams->numforcesbodies) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_RB_FORCES>(0);
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_RB_TORQUES>(0);
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_RB_KEYS>();
	}

	m_dBuffers.addBuffer<CUDABuffer, BUFFER_CELLSTART>(-1);
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_CELLEND>(-1);
	if (MULTI_DEVICE) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_COMPACT_DEV_MAP>();
		m_hBuffers.addBuffer<HostBuffer, BUFFER_COMPACT_DEV_MAP>();
	}

	m_dBuffers.addBuffer<CUDABuffer, BUFFER_HASH>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_PARTINDEX>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_NEIBSLIST>(-1); // neib list is initialized to all bits set

	if (m_simparams->simflags & ENABLE_XSPH)
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_XSPH>(0);

	// If the user enabled a(n actual) turbulence model, enable BUFFER_TAU, to
	// store the shear stress tensor.
	// TODO FIXME temporary: k-eps needs TAU only for temporary storage
	// across the split kernel calls in forces
	if (m_simparams->turbmodel > ARTIFICIAL)
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_TAU>(0);

	if (m_simframework->hasPostProcessOption(SURFACE_DETECTION, BUFFER_NORMALS))
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_NORMALS>();
	if (m_simframework->hasPostProcessOption(INTERFACE_DETECTION, BUFFER_NORMALS))
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_NORMALS>();

	if (m_simframework->hasPostProcessEngine(VORTICITY))
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_VORTICITY>();

	if (m_simparams->simflags & ENABLE_DTADAPT) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_CFL>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_CFL_TEMP>(0);
		if (m_simparams->boundarytype == SA_BOUNDARY && USING_DYNAMIC_GAMMA(m_simparams->simflags))
			m_dBuffers.addBuffer<CUDABuffer, BUFFER_CFL_GAMMA>();
		if (m_simparams->turbmodel == KEPSILON)
			m_dBuffers.addBuffer<CUDABuffer, BUFFER_CFL_KEPS>();
	}

	if (m_simparams->boundarytype == SA_BOUNDARY) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_GRADGAMMA>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_BOUNDELEMENTS>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_VERTICES>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_VERTPOS>();
	}

	if (m_simparams->turbmodel == KEPSILON) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_TKE>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_EPSILON>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_TURBVISC>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_DKDE>(0);
	}

	if (m_simparams->turbmodel == SPS) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_SPS_TURBVISC>();
	}

	if (NEEDS_EFFECTIVE_VISC(m_simparams->rheologytype))
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_EFFVISC>();

	if (m_simparams->rheologytype == GRANULAR) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_EFFPRES>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_JACOBI>();
	}

	if (m_simparams->boundarytype == SA_BOUNDARY &&
		(m_simparams->simflags & ENABLE_INLET_OUTLET || m_simparams->turbmodel == KEPSILON))
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_EULERVEL>();

	if (m_simparams->simflags & ENABLE_INLET_OUTLET)
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_NEXTID>();

	if (m_simparams->sph_formulation == SPH_GRENIER) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_VOLUME>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_SIGMA>();
	}

	if (m_simframework->hasPostProcessEngine(CALC_PRIVATE)) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_PRIVATE>();
		if (m_simframework->hasPostProcessOption(CALC_PRIVATE, BUFFER_PRIVATE2))
			m_dBuffers.addBuffer<CUDABuffer, BUFFER_PRIVATE2>();
		if (m_simframework->hasPostProcessOption(CALC_PRIVATE, BUFFER_PRIVATE4))
			m_dBuffers.addBuffer<CUDABuffer, BUFFER_PRIVATE4>();
	}

	if (m_simparams->simflags & ENABLE_INTERNAL_ENERGY) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_INTERNAL_ENERGY>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_INTERNAL_ENERGY_UPD>(0);
	}

	if (gdata->run_mode == REPACK) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_REPACK>(0);
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_CFL_REPACK>();
	}
	
	// all workers begin with an "initial upload” state in their particle system,
	// to hold all the buffers that will be initialized from host
	m_dBuffers.initialize_state("initial upload");

}

GPUWorker::~GPUWorker() {
	// Free everything and pthread terminate
	// should check whether the pthread is still running and force its termination?
}

// Return the number of particles currently being handled (internal and r.o.)
uint GPUWorker::getNumParticles() const
{
	return m_numParticles;
}

// Return the number of allocated particles
uint GPUWorker::getNumAllocatedParticles() const
{
	return m_numAllocatedParticles;
}

uint GPUWorker::getNumInternalParticles() const
{
	return m_numInternalParticles;
}

// Return the maximum number of particles the worker can handled (allocated)
uint GPUWorker::getMaxParticles() const
{
	return m_numAllocatedParticles;
}

// Compute the bytes required for each particle.
size_t GPUWorker::computeMemoryPerParticle()
{
	size_t tot = 0;

	set<flag_t>::const_iterator it = m_dBuffers.get_keys().begin();
	const set<flag_t>::const_iterator stop = m_dBuffers.get_keys().end();
	while (it != stop) {
		flag_t key = *it;
		size_t contrib = m_dBuffers.get_memory_occupation(key, 1);
		if (key == BUFFER_NEIBSLIST)
			contrib *= m_simparams->neiblistsize;
		// TODO compute a sensible estimate for the CFL contribution,
		// which is currently heavily overestimated
		else if (key == BUFFERS_CFL)
			contrib /= 4;
		// particle index occupancy is double to account for memory allocated
		// by thrust::sort TODO refine
		else if (key == BUFFER_PARTINDEX)
			contrib *= 2;

		tot += contrib;
#if _DEBUG_
		//printf("with %s: %zu\n", buf->second->get_buffer_name(), tot);
		// TODO: FIXME buf not declared
#endif
		++it;
	}

	// TODO
	//float4*		m_dRbForces;
	//float4*		m_dRbTorques;
	//uint*		m_dRbNum;

	// round up to next multiple of 4
	tot = round_up<size_t>(tot, 4);
	if (m_deviceIndex == 0)
		printf("Estimated memory consumption: %zuB/particle\n", tot);
	return tot;
}

// Compute the bytes required for each cell.
// NOTE: this should be update for each new device array!
size_t GPUWorker::computeMemoryPerCell()
{
	size_t tot = 0;
	tot += sizeof(BufferTraits<BUFFER_CELLSTART>::element_type);
	tot += sizeof(BufferTraits<BUFFER_CELLEND>::element_type);
	if (MULTI_DEVICE)
		tot += sizeof(BufferTraits<BUFFER_COMPACT_DEV_MAP>::element_type);
	return tot;
}

// Compute the maximum number of particles we can allocate according to the available device memory
void GPUWorker::computeAndSetAllocableParticles()
{
	size_t totMemory, memPerCells, freeMemory, safetyMargin;
	cudaMemGetInfo(&freeMemory, &totMemory);
	// TODO configurable
	#define TWOTO32 (float) (1<<20)
	printf("Device idx %u: free memory %u MiB, total memory %u MiB\n", m_cudaDeviceNumber,
			(uint)(((float)freeMemory)/TWOTO32), (uint)(((float)totMemory)/TWOTO32));
	safetyMargin = totMemory/32; // 16MB on a 512MB GPU, 64MB on a 2GB GPU
	// compute how much memory is required for the cells array
	memPerCells = (size_t)gdata->nGridCells * computeMemoryPerCell();

	if (freeMemory < 16 + safetyMargin){
		fprintf(stderr, "FATAL: not enough free device memory for safety margin (%u MiB) \n", (uint)((float) (16 + safetyMargin)/TWOTO32));
		exit(1);
	}
	#undef TWOTO32
	// TODO what are segments ?
	// Why subtract 16B of mem when we are taking MiB od safety margin ?
	freeMemory -= 16; // segments
	freeMemory -= safetyMargin;

	if (memPerCells > freeMemory) {
		fprintf(stderr, "FATAL: not enough free device memory to allocate %s cells\n", gdata->addSeparators(gdata->nGridCells).c_str());
		exit(1);
	}

	freeMemory -= memPerCells;

	// keep num allocable particles rounded to the next multiple of 4, to improve reductions' performances
	uint numAllocableParticles = round_up<uint>(freeMemory / computeMemoryPerParticle(), 4);

	if (numAllocableParticles < gdata->allocatedParticles)
		printf("NOTE: device %u can allocate %u particles, while the whole simulation might require %u\n",
			m_deviceIndex, numAllocableParticles, gdata->allocatedParticles);

	// allocate at most the number of particles required for the whole simulation
	m_numAllocatedParticles = min( numAllocableParticles, gdata->allocatedParticles );

	if (m_numAllocatedParticles < m_numParticles) {
		fprintf(stderr, "FATAL: thread %u needs %u (and up to %u) particles, but we can only store %u in %s available of %s total with %s safety margin\n",
			m_deviceIndex, m_numParticles, gdata->allocatedParticles, m_numAllocatedParticles,
			gdata->memString(freeMemory).c_str(), gdata->memString(totMemory).c_str(),
			gdata->memString(safetyMargin).c_str());
#if 1
		exit(1);
#else
		fputs("expect failures\n", stderr);
#endif
	}
}

// Cut all particles that are not internal.
// Assuming segments have already been filled and downloaded to the shared array.
// NOTE: here it would be logical to reset the cellStarts of the cells being cropped
// out. However, this would be quite inefficient. We leave them inconsistent for a
// few time and we will update them when importing peer cells.
template<>
void GPUWorker::runCommand<CROP>(CommandStruct const&)
// void GPUWorker::dropExternalParticles()
{
	m_particleRangeEnd =  m_numParticles = m_numInternalParticles;
	gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] = EMPTY_SEGMENT;
	gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_CELL] = EMPTY_SEGMENT;
}

/// compare UPDATE_EXTERNAL arguments against list of updated buffers
void GPUWorker::checkBufferUpdate(CommandStruct const& cmd)
{
	// TODO support multiple StateBuffers
	{
		string const cmd_name = getCommandName(cmd);
		if (cmd.updates.size() == 0)
			throw invalid_argument(cmd_name + " without updates");
		if (cmd.updates.size() > 1)
			throw invalid_argument(cmd_name + " with multiple updates not implemented yet");
	}
	StateBuffers const& sb = cmd.updates[0];
	auto const& buflist = m_dBuffers.getState(sb.state);
	for (auto const& iter : buflist) {
		auto const key = iter.first;
		auto const buf = iter.second;
		const bool need_update = buf->is_dirty();
		const bool listed = !!(key & sb.buffers);

		if (need_update && !listed)
			cout <<  buf->get_buffer_name() << " needs update, but is NOT listed" << endl;
		else if (listed && !need_update)
			cout <<  buf->get_buffer_name() << " is listed, but is NOT dirty" << endl;

	}
}

// Start an async inter-device transfer. This will be actually P2P if device can access peer memory
// (actually, since it is currently used only to import data from other devices, the dstDevice could be omitted or implicit)
void GPUWorker::peerAsyncTransfer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count)
{
	if (m_disableP2Ptranfers) {
		// reallocate if necessary
		if (count > m_hPeerTransferBufferSize)
			resizePeerTransferBuffer(count);
		// transfer Dsrc -> H -> Ddst
		CUDA_SAFE_CALL_NOSYNC( cudaMemcpyAsync(m_hPeerTransferBuffer, src, count, cudaMemcpyDeviceToHost, m_asyncPeerCopiesStream) );
		CUDA_SAFE_CALL_NOSYNC( cudaMemcpyAsync(dst, m_hPeerTransferBuffer, count, cudaMemcpyHostToDevice, m_asyncPeerCopiesStream) );
	} else
		CUDA_SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	dst, dstDevice, src, srcDevice, count, m_asyncPeerCopiesStream ) );
}

// Uploads cellStart and cellEnd from the shared arrays to the device memory.
// Parameters: fromCell is inclusive, toCell is exclusive
void GPUWorker::asyncCellIndicesUpload(uint fromCell, uint toCell)
{
	const uint numCells = toCell - fromCell;
	const uint transferSize = sizeof(uint)*numCells;

	// TODO migrate s_dCellStarts to the device mechanism and provide an API
	// to copy offset data between buffers (even of different types)

	BufferList sorted = m_dBuffers.state_subset("sorted",
		BUFFER_CELLSTART | BUFFER_CELLEND);

	const uint *src;
	uint *dst;

	dst = sorted.getData<BUFFER_CELLSTART>() + fromCell;
	src = gdata->s_dCellStarts[m_deviceIndex] + fromCell;
	CUDA_SAFE_CALL_NOSYNC(cudaMemcpyAsync(dst, src, transferSize, cudaMemcpyHostToDevice, m_asyncH2DCopiesStream));

	dst = sorted.getData<BUFFER_CELLEND>() + fromCell;
	src = gdata->s_dCellEnds[m_deviceIndex] + fromCell;
	CUDA_SAFE_CALL_NOSYNC(cudaMemcpyAsync(dst, src, transferSize, cudaMemcpyHostToDevice, m_asyncH2DCopiesStream));
}

// wrapper for NetworkManage send/receive methods
void GPUWorker::networkTransfer(uchar peer_gdix, TransferDirection direction, void* _ptr, size_t _size, uint bid)
{
	// reallocate host buffer if necessary
	if (!gdata->clOptions->gpudirect && _size > m_hNetworkTransferBufferSize)
		resizeNetworkTransferBuffer(_size);

	if (direction == SND) {
		if (!gdata->clOptions->gpudirect) {
			// device -> host buffer, possibly async with forces kernel
			CUDA_SAFE_CALL_NOSYNC( cudaMemcpyAsync(m_hNetworkTransferBuffer, _ptr, _size,
				cudaMemcpyDeviceToHost, m_asyncD2HCopiesStream) );
			// wait for the data transfer to complete
			cudaStreamSynchronize(m_asyncD2HCopiesStream);
			// host buffer -> network
			gdata->networkManager->sendBuffer(m_globalDeviceIdx, peer_gdix, _size, m_hNetworkTransferBuffer);
		} else {
			// GPUDirect: device -> network
			if (gdata->clOptions->asyncNetworkTransfers)
				gdata->networkManager->sendBufferAsync(m_globalDeviceIdx, peer_gdix, _size, _ptr, bid);
			else
				gdata->networkManager->sendBuffer(m_globalDeviceIdx, peer_gdix, _size, _ptr);
		}
	} else {
		if (!gdata->clOptions->gpudirect) {
			// network -> host buffer
			gdata->networkManager->receiveBuffer(peer_gdix, m_globalDeviceIdx, _size, m_hNetworkTransferBuffer);
			// host buffer -> device, possibly async with forces kernel
			CUDA_SAFE_CALL_NOSYNC( cudaMemcpyAsync(_ptr, m_hNetworkTransferBuffer, _size,
				cudaMemcpyHostToDevice, m_asyncH2DCopiesStream) );
			// wait for the data transfer to complete (actually next iteration could requre no sync, but safer to do)
			cudaStreamSynchronize(m_asyncH2DCopiesStream);
		} else {
			// GPUDirect: network -> device
			if (gdata->clOptions->asyncNetworkTransfers)
				gdata->networkManager->receiveBufferAsync(peer_gdix, m_globalDeviceIdx, _size, _ptr, bid);
			else
				gdata->networkManager->receiveBuffer(peer_gdix, m_globalDeviceIdx, _size, _ptr);
		}
	}
}

// Compute list of bursts. Currently computes both scopes
void GPUWorker::computeCellBursts()
{
	// Unlike importing from other devices in the same process, here we need one burst for each potential neighbor device
	// and for each direction. The following can be considered a list of pointers to open bursts in the m_bursts vector.
	// When a pointer is negative, there is no open bursts with the specified peer:direction pair.
	int burst_vector_index[MAX_DEVICES_PER_CLUSTER][2];

	uint network_bursts = 0;
	uint node_bursts = 0;

	// Auxiliary macros. Use with parentheses when possible
#define BURST_IS_EMPTY(peer, direction) \
	(burst_vector_index[peer][direction] == -1)
	// closing a burst means dropping the associated pointer index
#define CLOSE_BURST(peer, direction) \
	burst_vector_index[peer][direction] = -1;

	// initialize bursts pointers
	for (uint n = 0; n < MAX_DEVICES_PER_CLUSTER; n++)
		for (uint direction = SND; direction <= RCV; direction++)
			burst_vector_index[n][direction] = -1;

	// empty list of bursts
	m_bursts.clear();

	// iterate on all cells
	for (uint lin_curr_cell = 0; lin_curr_cell < m_nGridCells; lin_curr_cell++) {

		// We want to send the current cell to the neighbor processes only once, but multiple neib cells could
		// belong the the same process. Therefore we keep a list of recipient gidx who already received the
		// current cell. We will also use this list as a "recipient list", esp. to check which bursts need to
		// be closed. The list is reset for every cell, before iterating the neighbors.
		bool neighboring_device[MAX_DEVICES_PER_CLUSTER];

		// reset the lists of recipient neighbors
		for (uint d = 0; d < MAX_DEVICES_PER_CLUSTER; d++)
			neighboring_device[d] = false;

		// NOTE: we must not skip cells that are non-edge for self
		//if (m_hCompactDeviceMap[cell] == CELLTYPE_INNER_CELL_SHIFTED) return;
		//if (m_hCompactDeviceMap[cell] == CELLTYPE_OUTER_CELL_SHIFTED) return;

		// we need the 3D coords as well
		const int3 coords_curr_cell = gdata->reverseGridHashHost(lin_curr_cell);

		// find the owner
		const devcount_t curr_cell_gidx = gdata->s_hDeviceMap[lin_curr_cell];
		const devcount_t curr_cell_rank = gdata->RANK( curr_cell_gidx );

		// redundant correctness check
		if ( curr_cell_rank >= gdata->mpi_nodes ) {
			printf("FATAL: cell %u seems to belong to rank %u, but max is %u; probable memory corruption\n", lin_curr_cell, curr_cell_rank, gdata->mpi_nodes - 1);
			gdata->quit_request = true;
			return;
		}

		// is it mine?
		const bool curr_mine = (curr_cell_gidx == m_globalDeviceIdx);

		// if cell is not edging at all, it should be skipped without breaking any burst
		// (at all = between any pair of devices, unrelated to any_mine)
		bool edging = false;

		// iterate on neighbors
		for (int dz = -1; dz <= 1; dz++)
			for (int dy = -1; dy <= 1; dy++)
				for (int dx = -1; dx <= 1; dx++) {

					// skip self (also implicit with dev id check, later)
					if (dx == 0 && dy == 0 && dz == 0) continue;

					// neighbor cell coords
					int ncx = coords_curr_cell.x + dx;
					int ncy = coords_curr_cell.y + dy;
					int ncz = coords_curr_cell.z + dz;

					// warp cell coordinates if any periodicity is enabled
					periodicityWarp(ncx, ncy, ncz);

					// ensure we are inside the domain
					if ( !isCellInsideProblemDomain(ncx, ncy, ncz) ) continue;

					// NOTE: we could skip empty cells if all the nodes in the network knew the content of all the cells.
					// Instead, each process only knows the empty cells of its workers, so empty cells still break bursts
					// as if they weren't empty. One could check the performances with broadcasting all-to-all the empty
					// cells (possibly in bursts).

					// now compute the linearized hash of the neib cell and other properties
					const uint lin_neib_cell = gdata->calcGridHashHost(ncx, ncy, ncz);
					const uchar neib_cell_gidx = gdata->s_hDeviceMap[lin_neib_cell];
					const uchar neib_cell_rank = gdata->RANK( neib_cell_gidx );

					// is this neib mine?
					const bool neib_mine = (neib_cell_gidx == m_globalDeviceIdx);
					// is any of the two mine? if not, I will only manage closed bursts
					const bool any_mine = (curr_mine || neib_mine);

					// skip pairs belonging to the same device
					if (curr_cell_gidx == neib_cell_gidx) continue;

					// if we are here, at least one neib cell belongs to a different device
					edging = true;

					// did we already treat the pair current_cell:neib_node? (i.e. previously, due to another neib cell)
					if (neighboring_device[ neib_cell_gidx ]) continue;

					// mark the pair current_cell:neib_node as treated (aka: include the device in the recipient "list")
					neighboring_device[ neib_cell_gidx ] = true;

					// sending or receiving?
					const TransferDirection transfer_direction = ( curr_mine ? SND : RCV );

					// simple peer copy or mpi transfer?
					const TransferScope transfer_scope = (curr_cell_rank == neib_cell_rank ? NODE_SCOPE : NETWORK_SCOPE);

					// devices fetch peers' memory with any intervention from the sender (aka: only RCV bursts in same node)
					if (transfer_scope == NODE_SCOPE && transfer_direction == SND)
						continue;

					// the "other" device is the device owning the cell (curr or neib) which is not mine
					const devcount_t other_device_gidx = (curr_cell_gidx == m_globalDeviceIdx ? neib_cell_gidx : curr_cell_gidx);

					if (any_mine) {

						// if existing burst is non-empty, was not closed till now, so it is compatible: extend it
						if (! BURST_IS_EMPTY(other_device_gidx,transfer_direction)) {

							// cell index is higher than the last enqueued; it is edging as well; no other cell
							// interrupted the burst until now. So cell is consecutive with previous in both
							// the sending the the receiving device
							m_bursts[ burst_vector_index[other_device_gidx][transfer_direction] ].cells.push_back(lin_curr_cell);

						} else {
							// if we are here, either the burst was empty or not compatible. In both cases, create a new one
							CellList list;
							list.push_back(lin_curr_cell);

							CellBurst burst = {
								list,
								other_device_gidx,
								transfer_direction,
								transfer_scope,
								0, 0, 0
							};

							// store (overwrite, if was non-empty) its forthcoming index
							burst_vector_index[other_device_gidx][transfer_direction] = m_bursts.size();
							// append it
							m_bursts.push_back(burst);
							// NOTE: we should not keep the structure and append it to vector later, or bursts
							// could result in a sorting which can cause a deadlock (e.g. all devices try to
							// send before receiving)

							// update counters
							if (transfer_scope == NODE_SCOPE)
								node_bursts++;
							else
								network_bursts++;

							// to disable bursts, we close every burst as soon as it was created
							//CLOSE_BURST(other_device_gidx, transfer_direction)
						}
					}

					/* NOTES on burst breaking conditions
					 *
					 * A cell which needs to be sent from a node N1, device D1 to a node N2, device D2 will break:
					 * 1. All bursts in any node with recipient D2 (except of course the current from D1): that is because
					 *    burst are imported as series of consecutive cells and would be broken by current.
					 * 2. All bursts originating from D1 to any recipient that is not among the neighbors of the cell:
					 *    any device which is not neighboring the current cell will not expect to receive it.
					 * The former will be true while cellStart and cellEnd are computed immediately upon reception of the
					 * size of the cell. One could instead compute them only after having received all the cell sizes, thus
					 * compacting more bursts and also optimizing out empty cells.
					 * Condition nr. 1 is checked here while nr. 2 is checked immediately after the iteration on neighbor
					 * cells.
					 */

					// Checking condition nr. 1 (see comment before)
					if (!any_mine) {
						// I am not the sender nor the recipient; close all bursts SND to the receiver
						if (!BURST_IS_EMPTY(neib_cell_gidx,SND)) {
							CLOSE_BURST(neib_cell_gidx,SND)
						}
					} else
					if (neib_mine) {
						// I am the recipient device: close all other RCV bursts
						for (uint n = 0; n < MAX_DEVICES_PER_CLUSTER; n++)
							if (n != curr_cell_gidx && !BURST_IS_EMPTY(n,RCV)) {
								CLOSE_BURST(n,RCV)
							}
					}

				} // iterate on neibs of current cells

		// There was no neib cell (i.e. it was an internal cell for every device), so skip burst-breaking conditionals.
		// NOTE: comment the following line to allow bursts only along linearization (e.g. with Y-split and XYZ linearization,
		// only one burst will be used with the following line active; several, aka one per Y line, will be used with the
		// following commented). This can useful only for debugging or profiling purposes
		if (!edging) continue;

		// Checking condition nr. 2 (see comment before)
		for (uint n = 0; n < MAX_DEVICES_PER_CLUSTER; n++) {
			// I am the sender; let's close all bursts directed to devices which are not recipients of curr cell
			if (curr_mine && !neighboring_device[n] && !BURST_IS_EMPTY(n,SND)) {
				CLOSE_BURST(n,SND)
			}
			// I am not among the recipients and I have an open burst from curr; let's close it
			if (!neighboring_device[m_globalDeviceIdx] && !BURST_IS_EMPTY(curr_cell_gidx,RCV)) {
				CLOSE_BURST(curr_cell_gidx,RCV)
			}
		}

	} // iterate on cells

	// We need min (#network_bursts * 4) messages (since we send multiple buffers for
	// each burst). Multiplying by 8 is just safer
	gdata->networkManager->setNumRequests(network_bursts * 8);

	printf("D%u: data transfers compacted in %u bursts [%u node + %u network]\n",
		m_deviceIndex, (uint)m_bursts.size(), node_bursts, network_bursts);
	/*
	for (uint i = 0; i < m_bursts.size(); i++) {
		printf(" D %u Burst %u: %u cells, peer %u, dir %s, scope %s\n", m_deviceIndex,
			i, m_bursts[i].cells.size(), m_bursts[i].peer_gidx,
			(m_bursts[i].direction == SND ? "SND" : "RCV"),
			(m_bursts[i].scope == NODE_SCOPE ? "NODE" : "NETWORK") );
	}
	// */
#undef BURST_IS_EMPTY
#undef CLOSE_BURST
}

// iterate on the list and send/receive/read cell sizes
void GPUWorker::transferBurstsSizes()
{
	// first received cell marks the beginning of the cell range to upload
	bool receivedOneCell = false;
	uint minLinearCellIdx = 0;
	uint maxLinearCellIdx = 0;
	// Alternatively, we could initialize the minimum to gdata->nGridCells and the maximum to 0, and
	// check the min/max against them. However, in case we receive no cells at all, we want that
	// 1. max > min 2. 0 cells are uploaded

	// iterate on all bursts
	for (uint i = 0; i < m_bursts.size(); i++) {

		// first non-empty cell in this burst marks the beginning of its particle range
		bool receivedOneNonEmptyCellInBurst = false;

		// reset particle range
		m_bursts[i].selfFirstParticle = m_bursts[i].peerFirstParticle = 0;
		m_bursts[i].numParticles = 0;

		// iterate over the cells of the burst
		for (uint j = 0; j < m_bursts[i].cells.size(); j++) {
			uint lin_cell = m_bursts[i].cells[j];

			uint numPartsInCell = 0;
			uchar peerDeviceIndex = gdata->DEVICE(m_bursts[i].peer_gidx);

			// if direction is SND, scope can only be NETWORK
			if (m_bursts[i].direction == SND) {

				// compute cell size
				if (gdata->s_dCellStarts[m_deviceIndex][lin_cell] != EMPTY_CELL)
					numPartsInCell = gdata->s_dCellEnds[m_deviceIndex][lin_cell] - gdata->s_dCellStarts[m_deviceIndex][lin_cell];
				// send cell size
				gdata->networkManager->sendUint(m_globalDeviceIdx, m_bursts[i].peer_gidx, &numPartsInCell);

			} else {

				// If the direction is RCV, the scope can be NODE or NETWORK. In the former case, read the
				// cell content from the shared cellStarts; in the latter, receive if from the node
				if (m_bursts[i].scope == NETWORK_SCOPE)
					gdata->networkManager->receiveUint(m_bursts[i].peer_gidx, m_globalDeviceIdx, &numPartsInCell);
				else {
					if (gdata->s_dCellStarts[peerDeviceIndex][lin_cell] != EMPTY_CELL)
						numPartsInCell = gdata->s_dCellEnds[peerDeviceIndex][lin_cell] -
							gdata->s_dCellStarts[peerDeviceIndex][lin_cell];
				}

				// append the cell
				if (numPartsInCell > 0) {

					// set cell start and end
					gdata->s_dCellStarts[m_deviceIndex][lin_cell] = m_numParticles;
					gdata->s_dCellEnds[m_deviceIndex][lin_cell] = m_numParticles + numPartsInCell;

					// update outer edge segment, in case it was empty
					if (gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] == EMPTY_SEGMENT)
						gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] = m_numParticles;

					// update numParticles
					m_numParticles += numPartsInCell;

				} else
					// just set the cell as empty
					gdata->s_dCellStarts[m_deviceIndex][lin_cell] = EMPTY_CELL;

				// Update indices of cell range to be uploaded to device. We only care about RCV cells
				if (!receivedOneCell) {
					minLinearCellIdx = lin_cell;
					receivedOneCell = true;
				}
				// since lin_cell is increasing, the max is always updated
				maxLinearCellIdx = lin_cell;

			} // direction is RCV

			// Update indices of particle ranges (SND and RCV), which will be used for burst transfers
			if (numPartsInCell > 0) {
				if (!receivedOneNonEmptyCellInBurst) {
					m_bursts[i].selfFirstParticle = gdata->s_dCellStarts[m_deviceIndex][lin_cell];
					if (m_bursts[i].scope == NODE_SCOPE)
						m_bursts[i].peerFirstParticle = gdata->s_dCellStarts[peerDeviceIndex][lin_cell];
					receivedOneNonEmptyCellInBurst = true;
				}
				m_bursts[i].numParticles += numPartsInCell;
#if 0 // DBG
				printf(" BURST %u, incr. parts from %u to %u (+%u) because of cell %u\n", i,
						   m_bursts[i].numParticles - numPartsInCell, m_bursts[i].numParticles,
							numPartsInCell, lin_cell );
#endif
			}

		} // iterate on cells of the current burst
	} // iterate on bursts

	// update device cellStarts/Ends, if any cell needs update
	if (receivedOneCell)
		// maxLinearCellIdx is inclusive while asyncCellIndicesUpload() takes exclusive max
		asyncCellIndicesUpload(minLinearCellIdx, maxLinearCellIdx + 1);

	/*
	for (uint i = 0; i < m_bursts.size(); i++) {
		printf(" D %u Burst %u: %u cells, peer %u, dir %s, scope %s, range %u-%u, peer start %u, (tot %u parts)\n", m_deviceIndex,
				i, m_bursts[i].cells.size(), m_bursts[i].peer_gidx,
				(m_bursts[i].direction == SND ? "SND" : "RCV"),
				(m_bursts[i].scope == NETWORK_SCOPE ? "NETWORK" : "NODE"),
				m_bursts[i].selfFirstParticle, m_bursts[i].selfFirstParticle + m_bursts[i].numParticles,
				m_bursts[i].peerFirstParticle, m_bursts[i].numParticles
			);
	}
	// */
}

// Iterate on the list and send/receive bursts of particles across different nodes
void GPUWorker::transferBursts(CommandStruct const& cmd)
{
	if (cmd.updates.size() > 1)
		throw invalid_argument(string(getCommandName(cmd)) + " with multiple updates not implemented yet");

	// we support both the CommandBufferArgument updates syntax, and the src + flags syntax
	// during this transition
	const bool cmd_arg_syntax = (cmd.updates.size() == 1);

	string const& state = cmd_arg_syntax ? cmd.updates[0].state : cmd.src;
	const flag_t buf_spec = cmd_arg_syntax ? cmd.updates[0].buffers : cmd.flags;

	if (state.empty())
		throw runtime_error("transferBursts with empty state");
	if (buf_spec == BUFFER_NONE)
		throw runtime_error("transferBursts with no buffer specification");

	BufferList buflist = m_dBuffers.state_subset_existing(state, buf_spec);

	// burst id counter, needed to correctly pair asynchronous network messages
	uint bid[MAX_DEVICES_PER_CLUSTER];
	for (uint n = 0; n < MAX_DEVICES_PER_CLUSTER; n++)
		bid[n] = 0;

	// Iterate on scope type, so that intra-node transfers are performed first.
	// Decrement instead of incrementing to transfer MPI first.
	for (uint current_scope_i = NODE_SCOPE; current_scope_i <= NETWORK_SCOPE; current_scope_i++) {
		TransferScope current_scope = (TransferScope)current_scope_i;

		// iterate on all bursts
		for (uint i = 0; i < m_bursts.size(); i++) {

			// transfer bursts of one scope at a time
			if (m_bursts[i].scope != current_scope) continue;

			/*
			printf("IT %u D %u burst %u #parts %u dir %s (%u -> %u) scope %s\n",
				gdata->iterations, m_deviceIndex, i, m_bursts[i].numParticles,
				(m_bursts[i].direction == SND ? "SND" : "RCV"),
				(m_bursts[i].direction == SND ? m_globalDeviceIdx : m_bursts[i].peer_gidx),
				(m_bursts[i].direction == SND ? m_bursts[i].peer_gidx : m_globalDeviceIdx),
				(m_bursts[i].scope == NODE_SCOPE ? "NODE" : "NETWORK") );
			// */

			// iterate over all defined buffers and see which were requested
			// NOTE: map, from which BufferList is derived, is an _ordered_ container,
			// with the ordering set by the key, in our case the unsigned integer type flag_t,
			// so we have guarantee that the map will always be traversed in the same order
			// (unless stuff is inserted/deleted, which shouldn't happen at program runtime)
			BufferList::iterator bufset = buflist.begin();
			const BufferList::iterator stop = buflist.end();
			for ( ; bufset != stop ; ++bufset) {
				flag_t bufkey = bufset->first;

				// here we use the explicit type instead of auto to better
				// highlight the constness difference with peerbuf below
				shared_ptr<AbstractBuffer> buf = bufset->second;

				const unsigned int _size = m_bursts[i].numParticles * buf->get_element_size();

				// retrieve peer's indices, if intra-node
				shared_ptr<const AbstractBuffer> peerbuf;
				uint peerCudaDevNum = 0;
				if (m_bursts[i].scope == NODE_SCOPE) {
					uchar peerDevIdx = gdata->DEVICE(m_bursts[i].peer_gidx);
					peerCudaDevNum = gdata->device[peerDevIdx];
					peerbuf = gdata->GPUWORKERS[peerDevIdx]->getBuffer(state, bufkey);
				}


				// transfer the data if burst is not empty
				if (m_bursts[i].numParticles > 0) {
					// send all the arrays of which this buffer is composed
					for (uint ai = 0; ai < buf->get_array_count(); ++ai) {
						void *ptr = buf->get_offset_buffer(ai, m_bursts[i].selfFirstParticle);
						if (m_bursts[i].scope == NODE_SCOPE) {
							// node scope: just read it
							const void *peerptr = peerbuf->get_offset_buffer(ai, m_bursts[i].peerFirstParticle);
							peerAsyncTransfer(ptr, m_cudaDeviceNumber, peerptr, peerCudaDevNum, _size);
						} else {
							// network scope: SND or RCV
							networkTransfer(m_bursts[i].peer_gidx, m_bursts[i].direction, ptr, _size, bid[m_bursts[i].peer_gidx]++);
						}
					}
				}

				buf->mark_valid();
			} // for each buffer type

		} // iterate on bursts

	} // iterate on scopes

	// waits for network async transfers to complete
	if (MULTI_NODE)
		gdata->networkManager->waitAsyncTransfers();
}


// Import the external edge cells of other devices to the self device arrays. Can append the cells at the end of the current
// list of particles (APPEND_EXTERNAL) or just update the already appended ones (UPDATE_EXTERNAL), according to the current
// GlobalData::nextCommand. When appending, also update cellStarts (device and host), cellEnds (device and host) and segments
// (host only). The arrays to be imported must be specified in the command flags. If double buffered arrays are included, it
// is mandatory to specify also the buffer to be used (read or write). This information is ignored for non-buffered ones (e.g.
// forces).
// The data is transferred in bursts of consecutive cells when possible. Intra-node transfers are D2D if peer access is enabled,
// staged on host otherwise. Network transfers use the NetworkManager (MPI-based).
void GPUWorker::importExternalCells(CommandStruct const& cmd)
{
	if (gdata->debug.check_buffer_update) checkBufferUpdate(cmd);

	if (cmd.command == APPEND_EXTERNAL)
		transferBurstsSizes();
	if ( (cmd.command == APPEND_EXTERNAL) || (cmd.command == UPDATE_EXTERNAL) )
		transferBursts(cmd);

	// cudaMemcpyPeerAsync() is asynchronous with the host. If striping is disabled, we want to synchronize
	// for the completion of the transfers. Otherwise, FORCES_COMPLETE will synchronize everything
	if (!gdata->clOptions->striping && MULTI_GPU)
		cudaDeviceSynchronize();

	// here will sync the MPI transfers when (if) we'll switch to non-blocking calls
	// if (!gdata->striping && MULTI_NODE)...
}
template<>
void GPUWorker::runCommand<APPEND_EXTERNAL>(CommandStruct const& cmd) { importExternalCells(cmd); }
template<>
void GPUWorker::runCommand<UPDATE_EXTERNAL>(CommandStruct const& cmd) { importExternalCells(cmd); }

// All the allocators assume that gdata is updated with the number of particles (done by problem->fillparts).
// Later this will be changed since each thread does not need to allocate the global number of particles.
size_t GPUWorker::allocateHostBuffers() {
	// common sizes
	const size_t uintCellsSize = sizeof(uint) * m_nGridCells;

	size_t allocated = 0;

	if (MULTI_DEVICE) {
		allocated += m_hBuffers.get<BUFFER_COMPACT_DEV_MAP>()->alloc(m_nGridCells);

		// allocate a 1Mb transferBuffer if peer copies are disabled
		if (m_disableP2Ptranfers)
			resizePeerTransferBuffer(1024 * 1024);

		// ditto for network transfers
		if (!gdata->clOptions->gpudirect)
			resizeNetworkTransferBuffer(1024 * 1024);

		// TODO migrate these to the buffer system as well
		cudaMallocHost(&(gdata->s_dCellStarts[m_deviceIndex]), uintCellsSize);
		cudaMallocHost(&(gdata->s_dCellEnds[m_deviceIndex]), uintCellsSize);
		allocated += 2*uintCellsSize;
	}


	m_hostMemory += allocated;
	return allocated;
}

size_t GPUWorker::allocateDeviceBuffers() {
	// common sizes
	// compute common sizes (in bytes)

	const size_t segmentsSize = sizeof(uint) * 4; // 4 = types of cells

	size_t allocated = 0;

	// used to set up the number of elements in CFL arrays,
	// will only actually be used if adaptive timestepping is enabled

	const uint fmaxElements = forcesEngine->getFmaxElements(m_numAllocatedParticles);
	const uint tempCflEls = forcesEngine->getFmaxTempElements(fmaxElements);
	set<flag_t>::const_iterator iter = m_dBuffers.get_keys().begin();
	set<flag_t>::const_iterator stop = m_dBuffers.get_keys().end();
	while (iter != stop) {
		const flag_t key = *iter;
		// number of elements to allocate
		// most have m_numAllocatedParticles. Exceptions follow
		size_t nels = m_numAllocatedParticles;

		if (key == BUFFER_NEIBSLIST)
			nels *= m_simparams->neiblistsize; // number of particles times neib list size
		else if (key & BUFFERS_RB_PARTICLES)
			nels = m_numForcesBodiesParticles; // number of particles in rigid bodies
		else if (key & BUFFERS_CELL)
			nels = m_nGridCells; // cell buffers are sized by number of cells
		else if (key == BUFFER_CFL_TEMP)
			nels = tempCflEls;
		else if (key & BUFFERS_CFL) { // other CFL buffers
			// TODO FIXME BUFFER_CFL_GAMMA needs to be as large as the whole system,
			// because it's updated progressively across split forces calls. We could
			// do with sizing it just like that, but then during the finalizeforces
			// reductions with striping we would risk overwriting some of the data.
			// To solve this, we size it as the _sum_ of the two, and will use
			// the first numAllocatedParticles for the split-force-calls accumulation,
			// and the remaining fmaxElements for the finalize.
			// this should be improved
			if (key == BUFFER_CFL_GAMMA)
				nels = round_up(nels, size_t(4)) + fmaxElements;
			else
				nels = fmaxElements;
		}

		allocated += m_dBuffers.alloc(key, nels);
		++iter;
	}

	if (MULTI_DEVICE) {
		// alloc segment only if not single_device
		CUDA_SAFE_CALL(cudaMalloc(&m_dSegmentStart, segmentsSize));
		CUDA_SAFE_CALL(cudaMemset(m_dSegmentStart, 0, segmentsSize));
		allocated += segmentsSize;
	}

	// water depth at open boundaries
	if (m_simparams->simflags & (ENABLE_INLET_OUTLET | ENABLE_WATER_DEPTH)) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dIOwaterdepth, m_simparams->numOpenBoundaries*sizeof(uint)));
		allocated += m_simparams->numOpenBoundaries*sizeof(uint);
	}

	// newNumParticles for inlets
	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNewNumParticles, sizeof(uint)));
	allocated += sizeof(uint);

	if (m_simparams->numforcesbodies) {
		uint* rbnum = new uint[m_numForcesBodiesParticles];

		forcesEngine->setrbstart(gdata->s_hRbFirstIndex, m_simparams->numforcesbodies);

		int offset = 0;
		for (uint i = 0; i < m_simparams->numforcesbodies; i++) {
			// set rbnum for each object particle; it is the key for the reduction
			for (size_t j = 0; j < gdata->problem->get_body_numparts(i); j++)
				rbnum[offset + j] = i;
			offset += gdata->problem->get_body_numparts(i);
		}
		size_t  size = m_numForcesBodiesParticles*sizeof(uint);
		auto buf = m_dBuffers.get_state_buffer( "initial upload", BUFFER_RB_KEYS);
		CUDA_SAFE_CALL(cudaMemcpy(buf->get_buffer(), rbnum, size,
				cudaMemcpyHostToDevice));
		buf->mark_valid();

		delete[] rbnum;
	}

	if (m_simparams->simflags & ENABLE_DEM) {
		int nrows = gdata->problem->get_dem_nrows();
		int ncols = gdata->problem->get_dem_ncols();
		printf("Thread %d setting DEM texture\t cols = %d\trows =%d\n",
				m_deviceIndex, ncols, nrows);
		forcesEngine->setDEM(gdata->problem->get_dem(), ncols, nrows);
	}

	m_deviceMemory += allocated;
	return allocated;
}

void GPUWorker::deallocateHostBuffers() {
	if (MULTI_DEVICE) {
		cudaFreeHost(gdata->s_dCellStarts[m_deviceIndex]);
		cudaFreeHost(gdata->s_dCellEnds[m_deviceIndex]);
		free(gdata->s_dSegmentsStart[m_deviceIndex]);
	}

	if (m_hPeerTransferBuffer)
		cudaFreeHost(m_hPeerTransferBuffer);

	if (m_hNetworkTransferBuffer)
		cudaFreeHost(m_hNetworkTransferBuffer);

	// here: dem host buffers?
}

void GPUWorker::deallocateDeviceBuffers() {

	m_dBuffers.clear();

	if (MULTI_DEVICE) {
		CUDA_SAFE_CALL(cudaFree(m_dSegmentStart));
	}

	CUDA_SAFE_CALL(cudaFree(m_dNewNumParticles));

	if (m_simparams->simflags & (ENABLE_INLET_OUTLET | ENABLE_WATER_DEPTH))
		CUDA_SAFE_CALL(cudaFree(m_dIOwaterdepth));

	// here: dem device buffers?
}

void GPUWorker::createEventsAndStreams()
{
	// init streams
	cudaStreamCreateWithFlags(&m_asyncD2HCopiesStream, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&m_asyncH2DCopiesStream, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&m_asyncPeerCopiesStream, cudaStreamNonBlocking);
	// init events
	cudaEventCreate(&m_halfForcesEvent);
}

void GPUWorker::destroyEventsAndStreams()
{
	// destroy streams
	cudaStreamDestroy(m_asyncD2HCopiesStream);
	cudaStreamDestroy(m_asyncH2DCopiesStream);
	cudaStreamDestroy(m_asyncPeerCopiesStream);
	// destroy events
	cudaEventDestroy(m_halfForcesEvent);
}

void GPUWorker::printAllocatedMemory()
{
	printf("Device idx %u (CUDA: %u) allocated %s on host, %s on device\n"
			"  assigned particles: %s; allocated: %s\n", m_deviceIndex, m_cudaDeviceNumber,
			gdata->memString(getHostMemory()).c_str(),
			gdata->memString(getDeviceMemory()).c_str(),
			gdata->addSeparators(m_numParticles).c_str(), gdata->addSeparators(m_numAllocatedParticles).c_str());
}

string
GPUWorker::describeCommandFlagsBuffers(flag_t flags)
{
	string s;
	char sep[3] = { ' ', ' ', ' ' };
	for (auto key : m_dBuffers.get_keys()) {
		if (key & flags) {
			s.append(sep, 3);
			s.append(getBufferName(key));
			sep[1] = '|';
		}
	}

	return s;
}

//! Upload subdomain to an “initial upload” state
/*! Data is taken from the buffers allocated and sorted on host
 */
void GPUWorker::uploadSubdomain() {
	// buffers to skip in the upload. Rationale:
	// POS_GLOBAL is computed on host from POS and HASH
	// ephemeral buffers (including post-process results such as NORMALS and VORTICITY)
	// are produced on device and _downloaded_ to host, never uploaded
	// VERTPOS, while not ephemeral, is computed from scratch at each neighbors list construction,
	// and should should not be undumped
	// (note that HASH is _updated_ during the list, so we do need to upload it)
	static const flag_t skip_bufs = BUFFER_POS_GLOBAL | EPHEMERAL_BUFFERS | BUFFER_VERTPOS;

	// indices
	const uint firstInnerParticle	= gdata->s_hStartPerDevice[m_deviceIndex];
	const uint howManyParticles	= gdata->s_hPartsPerDevice[m_deviceIndex];

	// we upload data to the "initial upload"
	auto& buflist = m_dBuffers.getState("initial upload");

	// iterate over each array in the _host_ buffer list, and upload data
	// to the (first) read buffer
	// if it is not in the skip list
	BufferList::const_iterator onhost = gdata->s_hBuffers.begin();
	const BufferList::const_iterator stop = gdata->s_hBuffers.end();
	for ( ; onhost != stop ; ++onhost) {
		flag_t buf_to_up = onhost->first;
		shared_ptr<const AbstractBuffer> host_buf = onhost->second;

		if (buf_to_up & skip_bufs)
			continue;

		if (host_buf->is_invalid()) {
			printf("Thread %d skipping host buffer %s for device %d (invalid buffer)\n",
				m_deviceIndex, host_buf->get_buffer_name(), m_cudaDeviceNumber);
			continue;
		}

		auto buf = buflist[buf_to_up];
		if (!buf)
			throw runtime_error(string("Host buffer ") + host_buf->get_buffer_name() +
				" has no GPU counterpart");
		size_t _size = howManyParticles * buf->get_element_size();

		printf("Thread %d uploading %d %s items (%s) on device %d from position %d\n",
				m_deviceIndex, howManyParticles, buf->get_buffer_name(),
				gdata->memString(_size).c_str(), m_cudaDeviceNumber, firstInnerParticle);

		// only do the actual upload if the device is not empty
		// (unlikely but possible before LB kicks in)
		// Note that we don't do an early bail-out because we still want to
		// mark the buffer as valid and belonging to the correct state
		if (howManyParticles > 0) {
			// get all the arrays of which this buffer is composed
			// (actually currently all arrays are simple, since the only complex arrays (TAU
			// and VERTPOS) have no host counterpart)
			for (uint ai = 0; ai < buf->get_array_count(); ++ai) {
				void *dstptr = buf->get_buffer(ai);
				const void *srcptr = host_buf->get_offset_buffer(ai, firstInnerParticle);
				CUDA_SAFE_CALL(cudaMemcpy(dstptr, srcptr, _size, cudaMemcpyHostToDevice));
			}
		}

		buf->set_state("initial upload");
		buf->mark_valid();
	}
}

//! Initialize a new particle system state
template<>
void GPUWorker::runCommand<INIT_STATE>(CommandStruct const& cmd)
{
	m_dBuffers.initialize_state(cmd.src);
}

// Rename a particle state
template<>
void GPUWorker::runCommand<RENAME_STATE>(CommandStruct const& cmd)
{
	m_dBuffers.rename_state(cmd.src, cmd.dst);
}

// Release a particle system state
template<>
void GPUWorker::runCommand<RELEASE_STATE>(CommandStruct const& cmd)
{
	m_dBuffers.release_state(cmd.src);
}

// Remove buffers from a state, returning them to the pool if not shared
template<>
void GPUWorker::runCommand<REMOVE_STATE_BUFFERS>(CommandStruct const& cmd)
{
	m_dBuffers.remove_state_buffers(cmd.src, cmd.flags);
}

// Swap buffers between state, invalidating the destination one
template<>
void GPUWorker::runCommand<SWAP_STATE_BUFFERS>(CommandStruct const& cmd)
{
	m_dBuffers.swap_state_buffers(cmd.src, cmd.dst, cmd.flags);
}

// Move buffers from one state to the other, invalidating them
template<>
void GPUWorker::runCommand<MOVE_STATE_BUFFERS>(CommandStruct const& cmd)
{
	m_dBuffers.remove_state_buffers(cmd.src, cmd.flags);
	m_dBuffers.add_state_buffers(cmd.dst, cmd.flags);
}

// Share buffers between states
template<>
void GPUWorker::runCommand<SHARE_BUFFERS>(CommandStruct const& cmd)
{
	m_dBuffers.share_buffers(cmd.src, cmd.dst, cmd.flags);
}


// Download the subset of the specified buffer to the correspondent shared CPU array.
// Makes multiple transfers. Only downloads the subset relative to the internal particles.
// For double buffered arrays, uses the READ buffers unless otherwise specified. Can be
// used for either the read or the write buffers, not both.
template<>
void GPUWorker::runCommand<DUMP>(CommandStruct const& cmd)
// void GPUWorker::dumpBuffers()
{
	// indices
	uint firstInnerParticle	= gdata->s_hStartPerDevice[m_deviceIndex];
	uint howManyParticles	= gdata->s_hPartsPerDevice[m_deviceIndex];

	// is the device empty? (unlikely but possible before LB kicks in)
	if (howManyParticles == 0) return;

	auto const buflist = extractExistingBufferList(m_dBuffers, cmd.reads);
	const flag_t dev_keys = buflist.get_keys();

	// iterate over each array in the _host_ buffer list, and download data
	// if it was requested
	BufferList::iterator onhost = gdata->s_hBuffers.begin();
	const BufferList::iterator stop = gdata->s_hBuffers.end();
	for ( ; onhost != stop ; ++onhost) {
		flag_t buf_to_get = onhost->first;
		if (!(buf_to_get & dev_keys))
			continue;

		shared_ptr<const AbstractBuffer> buf = buflist[buf_to_get];
		shared_ptr<AbstractBuffer> hostbuf(onhost->second);
		size_t _size = howManyParticles * buf->get_element_size();
		if (buf_to_get == BUFFER_NEIBSLIST)
			_size *= gdata->problem->simparams()->neiblistsize;

		uint dst_index_offset = firstInnerParticle;

		// the cell-specific buffers are always dumped as a whole,
		// since this is only used to debug the neighbors list on host
		// TODO FIXME this probably doesn't work on multi-GPU
		if (buf_to_get & BUFFERS_CELL) {
			_size = buf->get_allocated_elements() * buf->get_element_size();
			dst_index_offset = 0;
		}

		// get all the arrays of which this buffer is composed
		// (actually currently all arrays are simple, since the only complex arrays (TAU
		// and VERTPOS) have no host counterpart)
		for (uint ai = 0; ai < buf->get_array_count(); ++ai) {
			const void *srcptr = buf->get_buffer(ai);
			void *dstptr = hostbuf->get_offset_buffer(ai, dst_index_offset);
			CUDA_SAFE_CALL(cudaMemcpy(dstptr, srcptr, _size, cudaMemcpyDeviceToHost));
		}
		// In multi-GPU, only one thread should update the host buffer state,
		// to avoid crashes due to multiple threads writing the string at the same time
		if (m_deviceIndex == 0) {
			hostbuf->copy_state(buf.get());
			hostbuf->mark_valid();
		}
	}
}

// if m_hPeerTransferBuffer is not big enough, reallocate it. Round up to 1Mb
void GPUWorker::resizePeerTransferBuffer(size_t required_size)
{
	// is it big enough already?
	if (required_size < m_hPeerTransferBufferSize) return;

	// will round up to...
	size_t ROUND_TO = 1024*1024;

	// store previous size, compute new
	size_t prev_size = m_hPeerTransferBufferSize;
	m_hPeerTransferBufferSize = ((required_size / ROUND_TO) + 1 ) * ROUND_TO;

	// dealloc first
	if (m_hPeerTransferBufferSize) {
		CUDA_SAFE_CALL(cudaFreeHost(m_hPeerTransferBuffer));
		m_hostMemory -= prev_size;
	}

	printf("Staging host buffer resized to %zu bytes\n", m_hPeerTransferBufferSize);

	// (re)allocate
	CUDA_SAFE_CALL(cudaMallocHost(&m_hPeerTransferBuffer, m_hPeerTransferBufferSize));
	m_hostMemory += m_hPeerTransferBufferSize;
}

// analog to resizeTransferBuffer
void GPUWorker::resizeNetworkTransferBuffer(size_t required_size)
{
	// is it big enough already?
	if (required_size < m_hNetworkTransferBufferSize) return;

	// will round up to...
	size_t ROUND_TO = 1024*1024;

	// store previous size, compute new
	size_t prev_size = m_hNetworkTransferBufferSize;
	m_hNetworkTransferBufferSize = ((required_size / ROUND_TO) + 1 ) * ROUND_TO;

	// dealloc first
	if (m_hNetworkTransferBufferSize) {
		CUDA_SAFE_CALL(cudaFreeHost(m_hNetworkTransferBuffer));
		m_hostMemory -= prev_size;
	}

	printf("Staging network host buffer resized to %zu bytes\n", m_hNetworkTransferBufferSize);

	// (re)allocate
	CUDA_SAFE_CALL(cudaMallocHost(&m_hNetworkTransferBuffer, m_hNetworkTransferBufferSize));
	m_hostMemory += m_hNetworkTransferBufferSize;
}

// download cellStart and cellEnd to the shared arrays
template<>
void GPUWorker::runCommand<DUMP_CELLS>(CommandStruct const& cmd)
// void GPUWorker::downloadCellsIndices()
{
	const size_t _size = gdata->nGridCells * sizeof(uint);

	// TODO provide an API to copy offset data between buffers (even of different types)
	const BufferList sorted = extractExistingBufferList(m_dBuffers, cmd.reads);

	const uint *src;
	uint *dst;

	src = sorted.getData<BUFFER_CELLSTART>();
	dst = gdata->s_dCellStarts[m_deviceIndex];
	CUDA_SAFE_CALL(cudaMemcpy(dst, src, _size, cudaMemcpyDeviceToHost));

	src = sorted.getData<BUFFER_CELLEND>();
	dst = gdata->s_dCellEnds[m_deviceIndex];
	CUDA_SAFE_CALL(cudaMemcpy(dst, src, _size, cudaMemcpyDeviceToHost));
}

void GPUWorker::downloadSegments()
{
	size_t _size = 4 * sizeof(uint);
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_dSegmentsStart[m_deviceIndex],
								m_dSegmentStart,
								_size, cudaMemcpyDeviceToHost));
	/* printf("  T%d downloaded segs: (I) %u (IE) %u (OE) %u (O) %u\n", m_deviceIndex,
			gdata->s_dSegmentsStart[m_deviceIndex][0], gdata->s_dSegmentsStart[m_deviceIndex][1],
			gdata->s_dSegmentsStart[m_deviceIndex][2], gdata->s_dSegmentsStart[m_deviceIndex][3]); */
}

void GPUWorker::uploadSegments()
{
	size_t _size = 4 * sizeof(uint);
	CUDA_SAFE_CALL(cudaMemcpy(	m_dSegmentStart,
								gdata->s_dSegmentsStart[m_deviceIndex],
								_size, cudaMemcpyHostToDevice));
}

// download segments and update the number of internal particles
template<>
void GPUWorker::runCommand<UPDATE_SEGMENTS>(CommandStruct const& cmd)
// void GPUWorker::updateSegments()
{
	// if the device is empty, set the host and device segments as empty
	if (m_numParticles == 0)
		resetSegments();
	else {
		downloadSegments();
		// update the number of internal particles
		uint newNumIntParts = m_numParticles;

		if (gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_CELL] != EMPTY_SEGMENT)
			newNumIntParts = gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_CELL];

		if (gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] != EMPTY_SEGMENT)
			newNumIntParts = gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL];

		// To debug particle migration between devices, change the preprocessor conditional
		// into an #if 1
#if 0
		if (newNumIntParts != m_numInternalParticles)
			printf("  Dev. index %u @ iteration %u: internal particles: %d => %d\n",
				m_deviceIndex, gdata->iterations,
				m_numInternalParticles, newNumIntParts);
#endif
		m_particleRangeEnd = m_numInternalParticles = newNumIntParts;
	}
}

// set all segments, host and device, as empty
void GPUWorker::resetSegments()
{
	for (uint s = 0; s < 4; s++)
		gdata->s_dSegmentsStart[m_deviceIndex][s] = EMPTY_SEGMENT;
	uploadSegments();
}

// download the updated number of particles (update by reorder and euler)
template<>
void GPUWorker::runCommand<DOWNLOAD_NEWNUMPARTS>(CommandStruct const& cmd)
// void GPUWorker::downloadNewNumParticles()
{
	// is the device empty? (unlikely but possible before LB kicks in)
	// if so, neither reorder nor euler did actually perform anything
	if (m_numParticles == 0) return;

	uint activeParticles;
	CUDA_SAFE_CALL(cudaMemcpy(&activeParticles, m_dNewNumParticles, sizeof(uint), cudaMemcpyDeviceToHost));
	if (activeParticles > m_numAllocatedParticles) {
		fprintf(stderr, "ERROR: Number of particles grew too much: %u > %u\n", activeParticles, m_numAllocatedParticles);
		gdata->quit_request = true;
	}

	gdata->particlesCreatedOnNode[m_deviceIndex] = false;

	if (activeParticles != m_numParticles) {
		// if for debug reasons we need to print the change in numParts for each device, uncomment the following:
		// printf("  Dev. index %u: particles: %d => %d\n", m_deviceIndex, m_numParticles, activeParticles);

		m_numParticles = activeParticles;
		// In multi-device simulations, m_numInternalParticles is updated in dropExternalParticles() and updateSegments();
		// it should not be updated here. Single-device simulations, instead, have it updated here.
		if (SINGLE_DEVICE)
			m_particleRangeEnd = m_numInternalParticles = activeParticles;
		// As a consequence, single-device simulations will run the forces kernel on newly cloned particles as well, while
		// multi-device simulations will not. We want to make this harmless. There are at least two possibilies:
		// 1. Reset the neighbor list buffer before building it. Doing so, the clones will always have an empty list and
		//    the forces kernel will only add gravity. Note that the clones are usually inside the vel field until next
		//    buildneibs, so the output of forces should be irrelevant; the problem, however, is that the forces kernel
		//    might find trash there and crash. This is currently implemented.
		// 2. This method is called in two phases: after the reorder and after euler. If we can distinguish between the two
		//    phases, then we can update the m_particleRangeEnd/m_numInternalParticles only after the reorder and
		//    m_numParticles in both. One way to do this is to use a command flag or to reuse cmd.only_internal. This
		//    would avoid calling forces and euler on the clones and might be undesired, since we will not apply the vel
		//    field until next bneibs.
		// Note: we would love to reset only the neibslists of the clones, but lists are interlaced and this would mean
		// multiple memory stores. TODO: check if this is more convenient than resetting the whole list
		//
		// The particlesCreatedOnNode array indicates whether or not the particle count on a device changed an causes in
		// turn a forced buildneibs
		gdata->particlesCreatedOnNode[m_deviceIndex] = true;
	}
}

// upload the value m_numParticles to "newNumParticles" on device
template<>
void GPUWorker::runCommand<UPLOAD_NEWNUMPARTS>(CommandStruct const& cmd)
// void GPUWorker::uploadNewNumParticles()
{
	// uploading even if empty (usually not, right after append)
	// TODO move this to the bcEngine too
	CUDA_SAFE_CALL(cudaMemcpy(m_dNewNumParticles, &m_numParticles, sizeof(uint), cudaMemcpyHostToDevice));
}

void GPUWorker::uploadNumOpenVertices() {
	bcEngine->uploadNumOpenVertices(gdata->numOpenVertices);
}


// upload gravity (possibily called many times)
void GPUWorker::uploadGravity()
{
	// check if variable gravity is enabled
	if (m_simparams->gcallback)
		forcesEngine->setgravity(gdata->s_varGravity);
}
template<>
void GPUWorker::runCommand<UPLOAD_GRAVITY>(CommandStruct const& cmd) { uploadGravity(); }

// upload planes (called once while planes are constant)
void GPUWorker::uploadPlanes()
{
	// check if planes > 0 (already checked before calling?)
	if (gdata->s_hPlanes.size() > 0)
		forcesEngine->setplanes(gdata->s_hPlanes);
}
template<>
void GPUWorker::runCommand<UPLOAD_PLANES>(CommandStruct const& cmd) { uploadPlanes(); }


// Create a compact device map, for this device, from the global one,
// with each cell being marked in the high bits. Correctly handles periodicity.
// Also handles the optional extra displacement for periodicity. Since the cell
// offset is truncated, we need to add more cells to the outer neighbors (the extra
// disp. vector might not be a multiple of the cell size). However, only one extra cell
// per cell is added. This means that we might miss cells if the extra displacement is
// not parallel to one cartesian axis.
void GPUWorker::createCompactDeviceMap() {
	uint *compactDeviceMap = m_hBuffers.getData<BUFFER_COMPACT_DEV_MAP>();

	// Here we have several possibilities:
	// 1. dynamic programming - visit each cell and half of its neighbors once, only write self
	//    (14*cells reads, 1*cells writes)
	// 2. visit each cell and all its neighbors, then write the output only for self
	//    (27*cells reads, 1*cells writes)
	// 3. same as 2 but reuse last read buffer; in the general case, it is possible to reuse 18 cells each time
	//    (9*cells reads, 1*cells writes)
	// 4. visit each cell; if it is assigned to self, visit its neighbors; set self and edging neighbors; if it is not self, write nothing and do not visit neighbors
	//    (initial memset, 1*extCells + 27*intCells reads, 27*intCells writes; if there are 2 devices per process, this means roughly ~14 reads and ~writes per cell)
	// 5. same as 4. but only make writes for alternate cells (3D chessboard: only white cells write on neibs)
	// 6. upload on the device, maybe recycling an existing buffer, and perform a parallel algorithm on the GPU
	//    (27xcells cached reads, cells writes)
	// 7. ...
	// Algorithms 1. and 3. may be the fastest. Number 2 is currently implemented; later will test 3.
	// One minor optimization has been implemented: iterating on neighbors stops as soon as there are enough information (e.g. cell belongs to self and there is
	// at least one neib which does not ==> inner_edge). Another optimization would be to avoid linearizing all cells but exploit burst of consecutive indices. This
	// reduces the computations but not the read/write operations.

	// iterate on all cells of the world
	for (int ix=0; ix < gdata->gridSize.x; ix++)
		for (int iy=0; iy < gdata->gridSize.y; iy++)
			for (int iz=0; iz < gdata->gridSize.z; iz++) {
				// data of current cell
				uint cell_lin_idx = gdata->calcGridHashHost(ix, iy, iz);
				uint cell_globalDevidx = gdata->s_hDeviceMap[cell_lin_idx];
				bool is_mine = (cell_globalDevidx == m_globalDeviceIdx);
				// aux vars for iterating on neibs
				bool any_foreign_neib = false; // at least one neib does not belong to me?
				bool any_mine_neib = false; // at least one neib does belong to me?
				bool enough_info = false; // when true, stop iterating on neibs
				// iterate on neighbors
				for (int dx=-1; dx <= 1 && !enough_info; dx++)
					for (int dy=-1; dy <= 1 && !enough_info; dy++)
						for (int dz=-1; dz <= 1 && !enough_info; dz++) {
							// do not iterate on self
							if (dx == 0 && dy == 0 && dz == 0) continue;
							// explicit cell coordinates for readability
							int cx = ix + dx;
							int cy = iy + dy;
							int cz = iz + dz;

							// warp cell coords if any periodicity is enabled
							periodicityWarp(cx, cy, cz);

							// if not periodic, or if still out-of-bounds after periodicity warp (which is
							// currently not possibly but might be if periodicity warps will be reduced to
							// 1-cell distance), skip it
							if ( !isCellInsideProblemDomain(cx, cy, cz) ) continue;

							// Read data of neib cell
							uint neib_lin_idx = gdata->calcGridHashHost(cx, cy, cz);
							uint neib_globalDevidx = gdata->s_hDeviceMap[neib_lin_idx];

							// does self device own any of the neib cells?
							any_mine_neib	 |= (neib_globalDevidx == m_globalDeviceIdx);
							// does a non-self device own any of the neib cells?
							any_foreign_neib |= (neib_globalDevidx != m_globalDeviceIdx);

							// do we know enough to decide for current cell?
							enough_info = (is_mine && any_foreign_neib) || (!is_mine && any_mine_neib);
						} // iterating on offsets of neighbor cells
				uint cellType;
				// assign shifted values so that they are ready to be OR'd in calchash/reorder
				if (is_mine && !any_foreign_neib)	cellType = CELLTYPE_INNER_CELL_SHIFTED;
				if (is_mine && any_foreign_neib)	cellType = CELLTYPE_INNER_EDGE_CELL_SHIFTED;
				if (!is_mine && any_mine_neib)		cellType = CELLTYPE_OUTER_EDGE_CELL_SHIFTED;
				if (!is_mine && !any_mine_neib)		cellType = CELLTYPE_OUTER_CELL_SHIFTED;
				compactDeviceMap[cell_lin_idx] = cellType;
			}
	// here it is possible to save the compact device map
	// gdata->saveCompactDeviceMapToFile("", m_deviceIndex, m_hCompactDeviceMap);
}

// self-explanatory
void GPUWorker::uploadCompactDeviceMap() {
	auto buf = m_dBuffers.get_state_buffer(
		"initial upload", BUFFER_COMPACT_DEV_MAP);
	void *dst = buf->get_buffer();
	const uint *src = m_hBuffers.getData<BUFFER_COMPACT_DEV_MAP>();

	const size_t _size = m_nGridCells * sizeof(uint);
	CUDA_SAFE_CALL(cudaMemcpy(dst, src, _size, cudaMemcpyHostToDevice));
	buf->mark_valid();
}

// this should be singleton, i.e. should check that no other thread has been started (mutex + counter or bool)
void GPUWorker::run_worker() {
	// wrapper for pthread_create()
	// NOTE: the dynamic instance of the GPUWorker is passed as parameter
	thread_id = thread(&GPUWorker::simulationThread, this);
}

// Join the simulation thread (in pthreads' terminology)
// WARNING: blocks the caller until the thread reaches pthread_exit. Be sure to call it after all barriers
// have been reached or may result in deadlock!
void GPUWorker::join_worker() {
	thread_id.join();
}

GlobalData* GPUWorker::getGlobalData() {
	return gdata;
}

unsigned int GPUWorker::getCUDADeviceNumber()
{
	return m_cudaDeviceNumber;
}

devcount_t GPUWorker::getDeviceIndex()
{
	return m_deviceIndex;
}

cudaDeviceProp GPUWorker::getDeviceProperties() {
	return m_deviceProperties;
}

size_t GPUWorker::getHostMemory() {
	return m_hostMemory;
}

size_t GPUWorker::getDeviceMemory() {
	return m_deviceMemory;
}

shared_ptr<const AbstractBuffer> GPUWorker::getBuffer(std::string const& state, flag_t key) const
{
	return m_dBuffers.get_state_buffer(state, key);
}

void GPUWorker::setDeviceProperties(cudaDeviceProp _m_deviceProperties) {
	m_deviceProperties = _m_deviceProperties;
}

// enable direct p2p memory transfers by allowing the other devices to access the current device memory
void GPUWorker::enablePeerAccess()
{
	// iterate on all devices
	for (uint d=0; d < gdata->devices; d++) {
		// skip self
		if (d == m_deviceIndex) continue;
		// read peer's CUDA device number
		uint peerCudaDevNum = gdata->device[d];
		// is peer access possible?
		int res;
		cudaDeviceCanAccessPeer(&res, m_cudaDeviceNumber, peerCudaDevNum);
		// update value in table
		gdata->s_hDeviceCanAccessPeer[m_deviceIndex][d] = (res == 1);
		if (res == 0) {
			// if this happens, peer copies will be buffered on host. We do it explicitly on a dedicated
			// host buffer instead of letting the CUDA runtime do it automatically
			m_disableP2Ptranfers = true;
			printf("WARNING: device %u (CUDA device %u) cannot enable direct peer access for device %u (CUDA device %u)\n",
				m_deviceIndex, m_cudaDeviceNumber, d, peerCudaDevNum);
		} else
			cudaDeviceEnablePeerAccess(peerCudaDevNum, 0);
	}

	if (m_disableP2Ptranfers)
		printf("Device %u (CUDA device %u) could not enable complete peer access; will stage P2P transfers on host\n",
			m_deviceIndex, m_cudaDeviceNumber);
}

void GPUWorker::initialize()
{
	// allow peers to access the device memory (for cudaMemcpyPeer[Async])
	enablePeerAccess();

	// compute #parts to allocate according to the free memory on the device
	// must be done before uploading constants since some constants
	// (e.g. those for neibslist traversal) depend on the number of particles
	// allocated
	computeAndSetAllocableParticles();

	// upload constants (PhysParames, some SimParams)
	uploadConstants();

	// upload planes, if any
	uploadPlanes();

	// allocate CPU and GPU arrays
	allocateHostBuffers();
	allocateDeviceBuffers();
	printAllocatedMemory();

	// upload centers of gravity of the bodies
	uploadEulerBodiesCentersOfGravity();
	uploadForcesBodiesCentersOfGravity();

	// create and upload the compact device map (2 bits per cell)
	if (MULTI_DEVICE) {
		createCompactDeviceMap();
		computeCellBursts();
		uploadCompactDeviceMap();

		// init streams for async memcpys
		createEventsAndStreams();
	}

	// TODO: here set_reduction_params() will be called (to be implemented in this class). These parameters can be device-specific.
}

void GPUWorker::finalize()
{
	// destroy streams
	if (MULTI_DEVICE)
		destroyEventsAndStreams();

	// deallocate buffers
	deallocateHostBuffers();
	deallocateDeviceBuffers();
	// ...what else?

	cudaDeviceReset();
}

template<>
void GPUWorker::runCommand<CALCHASH>(CommandStruct const& cmd)
// void GPUWorker::kernel_calcHash()
{
	// is the device empty? (unlikely but possible before LB kicks in)
	if (m_numParticles == 0) return;
	const BufferList bufread =
		extractExistingBufferList(m_dBuffers, cmd.reads);

	BufferList bufwrite =
		extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);

	// calcHashDevice() should use CPU-computed hashes at iteration 0, or some particles
	// might be lost (if a GPU computes a different hash and does not recognize the particles
	// as "own"). However, the high bits should be set, or edge cells won't be compacted at
	// the end and bursts will be sparse.
	// This is required only in MULTI_DEVICE simulations but it holds also on single-device
	// ones to keep numerical consistency.

	const bool run_fix = (gdata->iterations == 0);

	bufwrite.add_manipulator_on_write(run_fix ? "fixHash" : "calcHash");

	if (run_fix)
		neibsEngine->fixHash(bufread, bufwrite, m_numParticles);
	else
		neibsEngine->calcHash(bufread, bufwrite, m_numParticles);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<SORT>(CommandStruct const& cmd)
// void GPUWorker::kernel_sort()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_numInternalParticles : m_numParticles);

	BufferList bufwrite =
		extractExistingBufferList(m_dBuffers, cmd.updates);
	bufwrite.add_manipulator_on_write("sort");

	neibsEngine->sort(
			BufferList(), /* there aren't any buffers that are only read by SORT */
			bufwrite,
			numPartsToElaborate);

	m_dBuffers.change_buffers_state(bufwrite.get_updated_buffers(), cmd.src, cmd.dst);
	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<REORDER>(CommandStruct const& cmd)
// void GPUWorker::kernel_reorderDataAndFindCellStart()
{
	const BufferList unsorted =
		extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList sorted =
		// updates holds the buffers sorted in SORT, which will only read actually
		extractExistingBufferList(m_dBuffers, cmd.updates) |
		// the writes specification includes a dynamic buffer selection,
		// because the sorted state will have the buffers that were also
		// present in unsorted
		extractGeneralBufferList(m_dBuffers, cmd.writes, unsorted);

	sorted.add_manipulator_on_write("reorder");

	// reset also if the device is empty (or we will download uninitialized values)
	sorted.get<BUFFER_CELLSTART>()->clobber();

	// if the device is not empty, do the actual sorting. Otherwise, just mark the buffers as updated
	if (m_numParticles > 0) {
		neibsEngine->reorderDataAndFindCellStart(
							m_dSegmentStart,
							// output: sorted buffers
							sorted,
							// input: unsorted buffers
							unsorted,
							m_numParticles,
							m_dNewNumParticles);
	} else {
		sorted.mark_dirty();
	}

	sorted.clear_pending_state();
}

template<>
void GPUWorker::runCommand<BUILDNEIBS>(CommandStruct const& cmd)
// void GPUWorker::kernel_buildNeibsList()
{
	neibsEngine->resetinfo();

	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread =
		extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite =
		extractGeneralBufferList(m_dBuffers, cmd.writes);

	bufwrite.add_manipulator_on_write("buildneibs");

	// reset the neighbor list
	bufwrite.get<BUFFER_NEIBSLIST>()->clobber();

	// this is the square the distance used for neighboursearching of boundaries
	// it is delta p / 2 bigger than the standard radius
	// it is used to add segments into the neighbour list even if they are outside the kernel support
	const float boundNlSqInflRad = powf(sqrt(m_simparams->nlSqInfluenceRadius) + m_simparams->slength/m_simparams->sfactor/2.0f,2.0f);

	neibsEngine->buildNeibsList(
					bufread,
					bufwrite,
					m_numParticles,
					numPartsToElaborate,
					m_nGridCells,
					m_simparams->nlSqInfluenceRadius,
					boundNlSqInflRad);

	// download the peak number of neighbors and the estimated number of interactions
	neibsEngine->getinfo( gdata->timingInfo[m_deviceIndex] );

	bufwrite.clear_pending_state();
}

// returns numBlocks as computed by forces()
uint GPUWorker::enqueueForcesOnRange(CommandStruct const& cmd,
	BufferListPair& buffer_lists, uint fromParticle, uint toParticle, uint cflOffset)
{
	const int step = cmd.step.number;

	const BufferList& bufread = buffer_lists.first;
	BufferList& bufwrite = buffer_lists.second;
	bufwrite.add_manipulator_on_write("forces" + to_string(step));

	return forcesEngine->basicstep(
		bufread,
		bufwrite,
		m_numParticles,
		fromParticle,
		toParticle,
		gdata->problem->m_deltap,
		m_simparams->slength,
		m_simparams->dtadaptfactor,
		m_simparams->influenceRadius,
		m_simparams->epsilon,
		m_dIOwaterdepth,
		cflOffset,
		gdata->run_mode,
		step,
		cmd.dt(gdata),
		(m_simparams->numforcesbodies > 0) ? true : false);
}

/// Run the steps necessary for forces execution
/** This includes things such as binding textures and clearing the CFL buffers
 */
GPUWorker::BufferListPair GPUWorker::pre_forces(CommandStruct const& cmd, uint numPartsToElaborate)
{
	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);

	BufferList bufwrite = extractGeneralBufferList(m_dBuffers, cmd.writes);

	bufwrite.add_manipulator_on_write("pre-forces" + to_string(cmd.step.number));


	// clear out the buffers computed by forces
	bufwrite.get<BUFFER_FORCES>()->clobber();
	if (gdata->run_mode == REPACK) {
		bufwrite.get<BUFFER_REPACK>()->clobber();
		bufwrite.get<BUFFER_CFL_REPACK>()->clobber();
	}
	
	if (m_simparams->simflags & ENABLE_XSPH)
		bufwrite.get<BUFFER_XSPH>()->clobber();

	if (m_simparams->turbmodel == KEPSILON) {
		bufwrite.get<BUFFER_DKDE>()->clobber();
		// TODO tau currently is reset in KEPSILON, but must NOT be reset if SPS
		// ideally tau should be computed in its own kernel in the KEPSILON case too
		bufwrite.get<BUFFER_TAU>()->clobber();
	}

	if (m_simparams->simflags & ENABLE_INTERNAL_ENERGY) {
		bufwrite.get<BUFFER_INTERNAL_ENERGY_UPD>()->clobber();
	}

	// if we have objects potentially shared across different devices, must reset their forces
	// and torques to avoid spurious contributions
	if (m_simparams->numforcesbodies > 0 && MULTI_DEVICE) {
		bufwrite.get<BUFFER_RB_FORCES>()->clobber();
		bufwrite.get<BUFFER_RB_TORQUES>()->clobber();
	}

	if (m_simparams->simflags & ENABLE_DTADAPT) {
		bufwrite.get<BUFFER_CFL>()->clobber();
		bufwrite.get<BUFFER_CFL_TEMP>()->clobber();
		if (m_simparams->boundarytype == SA_BOUNDARY && USING_DYNAMIC_GAMMA(m_simparams->simflags))
			bufwrite.get<BUFFER_CFL_GAMMA>()->clobber();
		if (m_simparams->turbmodel == KEPSILON)
			bufwrite.get<BUFFER_CFL_KEPS>()->clobber();
	}

	bufwrite.clear_pending_state();

	if (numPartsToElaborate > 0)
		forcesEngine->bind_textures(bufread, m_numParticles, gdata->run_mode);

	return make_pair(bufread, bufwrite);

}

/// Run the steps necessary to cleanup and complete forces execution
/** This includes things such as ubinding textures and getting the
 * maximum allowed time-step
 */
float GPUWorker::post_forces(CommandStruct const& cmd)
{
	forcesEngine->unbind_textures(gdata->run_mode);

	// no reduction for fixed timestep
	if (!(m_simparams->simflags & ENABLE_DTADAPT))
		return m_simparams->dt;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractGeneralBufferList(m_dBuffers, cmd.writes);
	bufwrite.add_manipulator_on_write("post-forces");

	// TODO FIXME when using the MONAGHAN viscous model,
	// there seems to be a need for stricter time-stepping conditions,
	// or the simulations turns out to be unstable. Making this
	// dependent on the additional coefficient for the viscous model
	// seems to work, but a more detailed stability analysis is needed
	// TODO FIXME with ESPANOL_REVENGA we have a similar issue.
	// Definitely need to investigate more.
	float max_kinvisc_for_dt = m_max_kinvisc;
	switch (m_simparams->viscmodel)
	{
	case MORRIS: break; // nothing extra
	case MONAGHAN:
		max_kinvisc_for_dt *= m_physparams->monaghan_visc_coeff;
		break;
	case ESPANOL_REVENGA:
		max_kinvisc_for_dt *= 5;
		break;
	}

	auto ret = forcesEngine->dtreduce(
		m_simparams->slength,
		m_simparams->dtadaptfactor,
		m_max_sound_speed,
		max_kinvisc_for_dt,
		bufread,
		bufwrite,
		m_forcesKernelTotalNumBlocks,
		m_numParticles);

	bufwrite.clear_pending_state();
	return ret;
}

float GPUWorker::post_reduce(const BufferList bufread, BufferList bufwrite)
{
	if (gdata->run_mode == REPACK) {
		float temp;
		const float *repack_data = bufread.getData<BUFFER_CFL_REPACK>();
		bufwrite.get<BUFFER_CFL_TEMP>()->clobber();
		float *repack_temp = bufwrite.getData<BUFFER_CFL_TEMP>();

		if (repack_data)
			temp = forcesEngine->reduceMax(repack_data, repack_temp, m_forcesKernelTotalNumBlocks, m_numParticles);
	
		return temp;
	}
	else
		return 0.f;
}

// Aux method to warp signed cell coordinates if periodicity is enabled.
// Cell coordinates are passed by reference; they are left unchanged if periodicity
// is not enabled.
void GPUWorker::periodicityWarp(int &cx, int &cy, int &cz)
{
	// NOTE: checking if c* is negative MUST be done before checking if it's greater than
	// the grid, otherwise it will be cast to uint and "-1" will be "greater" than the gridSize!

	// checking X periodicity
	if (m_simparams->periodicbound & PERIODIC_X) {
		if (cx < 0) {
			cx = gdata->gridSize.x - 1;
		} else
		if (cx >= gdata->gridSize.x) {
			cx = 0;
		}
	}

	// checking Y periodicity
	if (m_simparams->periodicbound & PERIODIC_Y) {
		if (cy < 0) {
			cy = gdata->gridSize.y - 1;
		} else
		if (cy >= gdata->gridSize.y) {
			cy = 0;
		}
	}

	// checking Z periodicity
	if (m_simparams->periodicbound & PERIODIC_Z) {
		if (cz < 0) {
			cz = gdata->gridSize.z - 1;
		} else
		if (cz >= gdata->gridSize.z) {
			cz = 0;
		}
	}
}

// aux method to check wether cell coords are inside the domain (does NOT take into account periodicity)
bool GPUWorker::isCellInsideProblemDomain(int cx, int cy, int cz)
{
	return ((cx >= 0 && cx <= gdata->gridSize.x) &&
			(cy >= 0 && cy <= gdata->gridSize.y) &&
			(cz >= 0 && cz <= gdata->gridSize.z));
}

template<>
void GPUWorker::runCommand<FORCES_ENQUEUE>(CommandStruct const& cmd)
// void GPUWorker::kernel_forces_async_enqueue()
{
	if (!cmd.only_internal)
		printf("WARNING: forces kernel called with only_internal == false, ignoring flag!\n");

	uint numPartsToElaborate = m_particleRangeEnd;

	m_forcesKernelTotalNumBlocks = 0;

	// NOTE: the stripe containing the internal edge particles must be run first, so that the
	// transfers can be performed in parallel with the second stripe. The size of the first
	// stripe, S1, should be:
	// A. Greater or equal to the number of internal edge particles (mandatory). If this value
	//    is zero, we use the next constraints (we might still need to receive particles)
	// B. Greater or equal to the number of particles which saturate the device (recommended)
	// C. As small as possible, or the second stripe might not be long enough to cover the
	//    transfers. For example, one half (recommended)
	// D. Multiple of block size (optional, to possibly save one block). Actually, since
	//    internal edge particles are compacted at the end, we should align its beginning
	//    (i.e. the size of S2 should be a multiple of the block size)
	// So we compute S1 = max( A, min(B,C) ) , and then we round it by excess as
	// S1 = totParticles - round_by_defect(S2)
	// TODO:
	// - Estimate saturation according to the number of MPs
	// - Improve criterion C (one half might be not optimal and leave uncovered transfers)

	// constraint A: internalEdgeParts
	const uint internalEdgeParts =
		(gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_INNER_EDGE_CELL] == EMPTY_SEGMENT ?
		0 : numPartsToElaborate - gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_INNER_EDGE_CELL]);

	// constraint B: saturatingParticles
	// 64K parts should saturate the newest hardware (y~2014), but it is safe to be "generous".
	const uint saturatingParticles = 64 * 1024;

	// constraint C
	const uint halfParts = numPartsToElaborate / 2;

	// stripe size
	uint edgingStripeSize = max( internalEdgeParts, min( saturatingParticles, halfParts) );

	// round
	uint nonEdgingStripeSize = numPartsToElaborate - edgingStripeSize;
	nonEdgingStripeSize = forcesEngine->round_particles(nonEdgingStripeSize);
	edgingStripeSize = numPartsToElaborate - nonEdgingStripeSize;

	// setup for forces execution
	BufferListPair buffer_lists = pre_forces(cmd, numPartsToElaborate);

	if (numPartsToElaborate > 0 ) {
		// enqueue the first kernel call (on the particles in edging cells)
		m_forcesKernelTotalNumBlocks += enqueueForcesOnRange(cmd, buffer_lists,
			nonEdgingStripeSize, numPartsToElaborate, m_forcesKernelTotalNumBlocks);

		// the following event will be used to wait for the first stripe to complete
		cudaEventRecord(m_halfForcesEvent, 0);

		// enqueue the second kernel call (on the rest)
		m_forcesKernelTotalNumBlocks += enqueueForcesOnRange(cmd, buffer_lists,
			0, nonEdgingStripeSize, m_forcesKernelTotalNumBlocks);

		// We could think of synchronizing in UPDATE_EXTERNAL or APPEND_EXTERNAL instead of here, so that we do not
		// cause any overhead (waiting here means waiting before next barrier, which means that devices which are
		// faster in the computation of the first stripe have to wait the others before issuing the second). However,
		// we need to ensure that the first stripe is finished in the *other* devices, before importing their cells.
		cudaEventSynchronize(m_halfForcesEvent);
	} else {
		// we didn't call forces because we didn't have particles,
		// but let's mark the write buffers as dirty to be consistent with the workers
		// that did do the work
		buffer_lists.second.mark_dirty();
	}
}

template<>
void GPUWorker::runCommand<FORCES_COMPLETE>(CommandStruct const& cmd)
// void GPUWorker::kernel_forces_async_complete()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// FLOAT_MAX is returned if kernels are not run (e.g. numPartsToElaborate == 0)
	float returned_dt = FLT_MAX;

	if (numPartsToElaborate > 0 ) {
		// wait for the completion of the kernel
		cudaDeviceSynchronize();

		// unbind the textures
		returned_dt = post_forces(cmd);
	}

	// for multi-step integrators, use the minumum of the estimations across all timesteps
	// otherwise use the currently computed one
	if (cmd.step > 1)
		gdata->dts[m_deviceIndex] = min(gdata->dts[m_deviceIndex], returned_dt);
	else
		gdata->dts[m_deviceIndex] = returned_dt;
}


template<>
void GPUWorker::runCommand<FORCES_SYNC>(CommandStruct const& cmd)
// void GPUWorker::kernel_forces()
{
	if (!cmd.only_internal)
		printf("WARNING: forces kernel called with only_internal == false, ignoring flag!\n");

	uint numPartsToElaborate = m_particleRangeEnd;

	m_forcesKernelTotalNumBlocks = 0;

	// FLOAT_MAX is returned if kernels are not run (e.g. numPartsToElaborate == 0)
	float returned_dt = FLT_MAX;

	const uint fromParticle = 0;
	const uint toParticle = numPartsToElaborate;

	// setup for forces execution
	BufferListPair buffer_lists = pre_forces(cmd, numPartsToElaborate);

	if (numPartsToElaborate > 0 ) {
		// enqueue the kernel call
		m_forcesKernelTotalNumBlocks = enqueueForcesOnRange(cmd,
			buffer_lists, fromParticle, toParticle, 0);

		// cleanup post forces and get dt
		returned_dt = post_forces(cmd);

		if (cmd.step.last)
			gdata->velmaxs[m_deviceIndex] = post_reduce(buffer_lists.first, buffer_lists.second);
	} else {
		// we didn't call forces because we didn't have particles,
		// but let's mark the write buffers as dirty to be consistent with the workers
		// that did do the work
		for (auto buf_iter : buffer_lists.second) {
			buf_iter.second->mark_dirty();
		}
	}

	// for multi-step integrators, use the minumum of the estimations across all timesteps
	// otherwise use the currently computed one
	if (cmd.step > 1)
		gdata->dts[m_deviceIndex] = min(gdata->dts[m_deviceIndex], returned_dt);
	else
		gdata->dts[m_deviceIndex] = returned_dt;
}

template<>
void GPUWorker::runCommand<EULER>(CommandStruct const& cmd)
// void GPUWorker::kernel_euler()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);
	bufwrite.add_manipulator_on_write("euler" + to_string(step));

	const float dt = cmd.dt(gdata);

	// run the kernel if the device is not empty (unlikely but possible before LB kicks in)
	// otherwise just mark the buffers
	if (numPartsToElaborate > 0) {
		integrationEngine->basicstep(
			bufread,
			bufwrite,
			m_numParticles,
			numPartsToElaborate,
			dt,
			step,
			gdata->t + dt,
			m_simparams->slength,
			m_simparams->influenceRadius,
			gdata->run_mode);
	} else {
		bufwrite.mark_dirty();
	}

	// should we rename the state?
	if (!cmd.src.empty() && !cmd.dst.empty())
		m_dBuffers.rename_state(cmd.src, cmd.dst);
	bufwrite.clear_pending_state();

}

template<>
void GPUWorker::runCommand<DENSITY_SUM>(CommandStruct const& cmd)
// void GPUWorker::kernel_density_sum()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);

	bufwrite.add_manipulator_on_write("densitySum" + to_string(step));

	const float dt = cmd.dt(gdata);

	integrationEngine->density_sum(
		bufread,
		bufwrite,
		m_numParticles,
		numPartsToElaborate,
		dt,
		step,
		gdata->t + dt,
		m_simparams->epsilon,
		gdata->problem->m_deltap,
		m_simparams->slength,
		m_simparams->influenceRadius);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<INTEGRATE_GAMMA>(CommandStruct const& cmd)
// void GPUWorker::kernel_integrate_gamma()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);

	bufwrite.add_manipulator_on_write("integrateGamma" + to_string(step));

	const float dt = cmd.dt(gdata);

	integrationEngine->integrate_gamma(
		bufread,
		bufwrite,
		m_numParticles,
		numPartsToElaborate,
		dt,
		step,
		gdata->t + dt,
		m_simparams->epsilon,
		m_simparams->slength,
		m_simparams->influenceRadius,
		gdata->run_mode);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<CALC_DENSITY_DIFFUSION>(CommandStruct const& cmd)
// void GPUWorker::kernel_calc_density_diffusion()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);

	bufwrite.add_manipulator_on_write("calcDensityDiffusion" + to_string(step));

	const float dt = cmd.dt(gdata);

	forcesEngine->compute_density_diffusion(
		bufread,
		bufwrite,
		m_numParticles,
		numPartsToElaborate,
		gdata->problem->m_deltap,
		m_simparams->slength,
		m_simparams->influenceRadius,
		dt);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<APPLY_DENSITY_DIFFUSION>(CommandStruct const& cmd)
// void GPUWorker::kernel_apply_density_diffusion()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);

	bufwrite.add_manipulator_on_write("applyDensityDiffusion" + to_string(step));

	const float dt = cmd.dt(gdata);

	integrationEngine->apply_density_diffusion(
		bufread,
		bufwrite,
		m_numParticles,
		numPartsToElaborate,
		dt);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<DOWNLOAD_IOWATERDEPTH>(CommandStruct const& cmd)
// void GPUWorker::kernel_download_iowaterdepth()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	bcEngine->downloadIOwaterdepth(
			gdata->h_IOwaterdepth[m_deviceIndex],
			m_dIOwaterdepth,
			m_simparams->numOpenBoundaries);

}

template<>
void GPUWorker::runCommand<UPLOAD_IOWATERDEPTH>(CommandStruct const& cmd)
// void GPUWorker::kernel_upload_iowaterdepth()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	bcEngine->uploadIOwaterdepth(
			gdata->h_IOwaterdepth[0],
			m_dIOwaterdepth,
			m_simparams->numOpenBoundaries);

}

template<>
void GPUWorker::runCommand<IMPOSE_OPEN_BOUNDARY_CONDITION>(CommandStruct const& cmd)
// void GPUWorker::kernel_imposeBoundaryCondition()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);

	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates);

	bufwrite.add_manipulator_on_write("imposeBC" + to_string(step)) ;

	gdata->problem->imposeBoundaryConditionHost(
		bufwrite,
		bufread,
		(m_simparams->simflags & ENABLE_WATER_DEPTH) ? m_dIOwaterdepth : NULL,
		gdata->t,
		m_numParticles,
		m_simparams->numOpenBoundaries,
		numPartsToElaborate);

	bufwrite.clear_pending_state();

}

template<>
void GPUWorker::runCommand<INIT_IO_MASS_VERTEX_COUNT>(CommandStruct const& cmd)
// void GPUWorker::kernel_initIOmass_vertexCount()
{
	uint numPartsToElaborate = m_numParticles;

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractGeneralBufferList(m_dBuffers, cmd.writes);
	bufwrite.add_manipulator_on_write("initIOmass vertex count");

	bcEngine->initIOmass_vertexCount(
		bufwrite,
		bufread,
		m_numParticles,
		numPartsToElaborate);

	bufwrite.clear_pending_state();

}

template<>
void GPUWorker::runCommand<INIT_IO_MASS>(CommandStruct const& cmd)
// void GPUWorker::kernel_initIOmass()
{
	uint numPartsToElaborate = m_numParticles;

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractGeneralBufferList(m_dBuffers, cmd.writes);
	bufwrite.add_manipulator_on_write("initIOmass");

	bcEngine->initIOmass(
		bufwrite,
		bufread,
		m_numParticles,
		numPartsToElaborate,
		gdata->problem->m_deltap);

	bufwrite.clear_pending_state();

}

template<>
void GPUWorker::runCommand<FILTER>(CommandStruct const& cmd)
// void GPUWorker::kernel_filter()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	FilterType filtertype = FilterType(cmd.flags);
	FilterEngineSet::const_iterator filterpair(filterEngines.find(filtertype));
	// make sure we're going to call an instantiated filter
	if (filterpair == filterEngines.end()) {
		throw invalid_argument("non-existing filter " + to_string(cmd.flags) + " invoked");
	}

	// TODO be more selective
	auto const& bufread = m_dBuffers.getState(cmd.reads[0].state);
	BufferList bufwrite = extractGeneralBufferList(m_dBuffers, cmd.writes);

	bufwrite.add_manipulator_on_write(string("filter/") + FilterName[filtertype]);

	filterpair->second->process(
		bufread, bufwrite,
		m_numParticles,
		numPartsToElaborate,
		m_simparams->slength,
		m_simparams->influenceRadius);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<POSTPROCESS>(CommandStruct const& cmd)
// void GPUWorker::kernel_postprocess()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	PostProcessType proctype = PostProcessType(cmd.flags);
	PostProcessEngineSet::const_iterator procpair(postProcEngines.find(proctype));
	// make sure we're going to call an instantiated filter
	if (procpair == postProcEngines.end()) {
		throw invalid_argument("non-existing postprocess filter invoked");
	}

	auto& processor = procpair->second;

	const flag_t updated = processor->get_updated_buffers();
	const flag_t written = processor->get_written_buffers();

	/* Add POST_PROCESS_BUFFERS, as needed */
	m_dBuffers.add_state_buffers("step n", written);

	auto const& bufread = m_dBuffers.getState("step n");
	/* TODO currently in post-processing we do not support ping-pong buffering,
	 * so we don't actually differentiate meaningfully between in-place updates
	 * and freshly written buffers */
	BufferList bufwrite = m_dBuffers.state_subset("step n",
		updated | written);

	bufwrite.add_manipulator_on_write(string("postprocess/") + PostProcessName[proctype]);

	// run the kernel if the device is not empty (unlikely but possible before LB kicks in)
	// otherwise just mark the buffers
	if (numPartsToElaborate > 0) {
		processor->process(
			bufread, bufwrite,
			m_numParticles,
			numPartsToElaborate,
			m_deviceIndex,
			gdata);
	} else {
		bufwrite.mark_dirty();
	}

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<COMPUTE_DENSITY>(CommandStruct const& cmd)
// void GPUWorker::kernel_compute_density()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);
	bufwrite.add_manipulator_on_write("compute density" + to_string(step));

	// run the kernel if the device is not empty (unlikely but possible before LB kicks in)
	// otherwise just mark the buffers
	if (numPartsToElaborate > 0) {
		forcesEngine->compute_density(bufread, bufwrite,
			numPartsToElaborate,
			m_simparams->slength,
			m_simparams->influenceRadius);
	} else {
		bufwrite.mark_dirty();
	}

	bufwrite.clear_pending_state();
}


template<>
void GPUWorker::runCommand<CALC_VISC>(CommandStruct const& cmd)
// void GPUWorker::kernel_visc()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractGeneralBufferList(m_dBuffers, cmd.writes);
	bufwrite.add_manipulator_on_write("calc visc" + to_string(step));

	// run the kernel if the device is not empty (unlikely but possible before LB kicks in)
	// otherwise just mark the buffers
	if (numPartsToElaborate > 0)  {
		float max_kinvisc = viscEngine->calc_visc(bufread, bufwrite,
			m_numParticles,
			numPartsToElaborate,
			gdata->problem->m_deltap,
			m_simparams->slength,
			m_simparams->influenceRadius);

		// was the maximum kinematic visc computed? store it
		if (!isnan(max_kinvisc))
			m_max_kinvisc = max_kinvisc;
	} else {
		bufwrite.mark_dirty();
		m_max_kinvisc = 0.0f;
	}

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<REDUCE_BODIES_FORCES>(CommandStruct const& cmd)
// void GPUWorker::kernel_reduceRBForces()
{
	const int step = cmd.step.number;
	const string current_state = cmd.src;

	const size_t numforcesbodies = m_simparams->numforcesbodies;

	// make sure this device does not add any obsolete contribute to forces acting on objects
	if (MULTI_DEVICE) {
		for (uint ob = 0; ob < numforcesbodies; ob++) {
			gdata->s_hRbDeviceTotalForce[m_deviceIndex*numforcesbodies + ob] = make_float3(0.0f);
			gdata->s_hRbDeviceTotalTorque[m_deviceIndex*numforcesbodies + ob] = make_float3(0.0f);
		}
	}

	// is the device empty? (unlikely but possible before LB kicks in)
	if (m_numParticles == 0) return;

	// if we have ODE objects but not particles on them, do not reduce
	// (possible? e.g. vector objects?)
	if (m_numForcesBodiesParticles == 0) return;

	if (numforcesbodies) {
		BufferList bufwrite = m_dBuffers.state_subset(current_state, BUFFERS_RB_PARTICLES);
		bufwrite.add_manipulator_on_write("reduceRBforces" + to_string(step));
		forcesEngine->reduceRbForces(bufwrite, gdata->s_hRbLastIndex,
				gdata->s_hRbDeviceTotalForce + m_deviceIndex*numforcesbodies,
				gdata->s_hRbDeviceTotalTorque + m_deviceIndex*numforcesbodies,
				numforcesbodies, m_numForcesBodiesParticles);
	}

}

template<>
void GPUWorker::runCommand<SA_CALC_SEGMENT_BOUNDARY_CONDITIONS>(CommandStruct const& cmd)
// void GPUWorker::kernel_saSegmentBoundaryConditions()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates);

	bufwrite.add_manipulator_on_write("saSegmentBoundaryConditions" + to_string(step));

	bcEngine->saSegmentBoundaryConditions(
		bufwrite, bufread,
		m_numParticles,
		numPartsToElaborate,
		gdata->problem->m_deltap,
		m_simparams->slength,
		m_simparams->influenceRadius,
		step,
		gdata->run_mode);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<FIND_OUTGOING_SEGMENT>(CommandStruct const& cmd)
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates);

	bufwrite.add_manipulator_on_write("findOutgoingSegment");
	bcEngine->findOutgoingSegment(
		bufwrite, bufread,
		m_numParticles,
		numPartsToElaborate,
		gdata->problem->m_deltap,
		m_simparams->slength,
		m_simparams->influenceRadius);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<SA_CALC_VERTEX_BOUNDARY_CONDITIONS>(CommandStruct const& cmd)
// void GPUWorker::kernel_saVertexBoundaryConditions()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const int step = cmd.step.number;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		/* if this is the last step and we have open boundaries, we will also
		 * be given the buffers needed for cloning in the writes list.
		 * Otherwise this contribution will be empty.
		 */
		extractGeneralBufferList(m_dBuffers, cmd.writes);
	bufwrite.add_manipulator_on_write("saVertexBoundaryConditions" + to_string(step));

	bcEngine->saVertexBoundaryConditions(
		bufwrite, bufread,
				m_numParticles,
				numPartsToElaborate,
				gdata->problem->m_deltap,
				m_simparams->slength,
				m_simparams->influenceRadius,
				step,
				!gdata->clOptions->resume_fname.empty(),
				cmd.dt(gdata),
				m_dNewNumParticles,
				m_globalDeviceIdx,
				gdata->totDevices,
				gdata->totParticles,
				gdata->run_mode);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<SA_COMPUTE_VERTEX_NORMAL>(CommandStruct const& cmd)
// void GPUWorker::kernel_saComputeVertexNormal()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);

	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates);

	bufwrite.add_manipulator_on_write("saComputeVertexNormal");

	bcEngine->computeVertexNormal(
				bufread,
				bufwrite,
				m_numParticles,
				numPartsToElaborate);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<SA_INIT_GAMMA>(CommandStruct const& cmd)
// void GPUWorker::kernel_saInitGamma()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);

	BufferList bufwrite = extractGeneralBufferList(m_dBuffers, cmd.writes);

	bufwrite.add_manipulator_on_write("saInitGamma");

	bcEngine->saInitGamma(
				bufread,
				bufwrite,
				m_simparams->slength,
				m_simparams->influenceRadius,
				gdata->problem->m_deltap,
				m_simparams->epsilon,
				m_numParticles,
				numPartsToElaborate);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<IDENTIFY_CORNER_VERTICES>(CommandStruct const& cmd)
// void GPUWorker::kernel_saIdentifyCornerVertices()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates);
	bufwrite.add_manipulator_on_write("saIdentifyCornerVertices");

	bcEngine->saIdentifyCornerVertices(
		bufread, bufwrite,
				m_numParticles,
				numPartsToElaborate,
				gdata->problem->m_deltap,
				m_simparams->epsilon);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<DISABLE_OUTGOING_PARTS>(CommandStruct const& cmd)
// void GPUWorker::kernel_disableOutgoingParts()
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates);
	bufwrite.add_manipulator_on_write("disableOutgoingParts");

	bcEngine->disableOutgoingParts(
		bufread, bufwrite,
				m_numParticles,
				numPartsToElaborate);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<DISABLE_FREE_SURF_PARTS>(CommandStruct const& cmd)
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates);
	bufwrite.add_manipulator_on_write("disableFreeSurfParts");

	integrationEngine->disableFreeSurfParts(
			bufwrite.getData<BUFFER_POS>(),
			bufread.getData<BUFFER_INFO>(),
			m_numParticles,
			numPartsToElaborate);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<JACOBI_FS_BOUNDARY_CONDITIONS>(CommandStruct const& cmd)
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates);
	bufwrite.add_manipulator_on_write("enforce_jacobi_fs_boundary_conditions");

	viscEngine->enforce_jacobi_fs_boundary_conditions(
		bufread, bufwrite,
		m_numParticles,
		numPartsToElaborate,
		gdata->problem->m_deltap,
		m_simparams->slength,
		m_simparams->influenceRadius);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<JACOBI_WALL_BOUNDARY_CONDITIONS>(CommandStruct const& cmd)
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);
	bufwrite.add_manipulator_on_write("enforce_jacobi_wall_boundary_conditions");

	gdata->h_jacobiBackwardError[m_deviceIndex] = viscEngine->enforce_jacobi_wall_boundary_conditions(
		bufread, bufwrite,
		m_numParticles,
		numPartsToElaborate,
		gdata->problem->m_deltap,
		m_simparams->slength,
		m_simparams->influenceRadius);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<JACOBI_BUILD_VECTORS>(CommandStruct const& cmd)
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);
	bufwrite.add_manipulator_on_write("build_jacobi_vectors");

	viscEngine->build_jacobi_vectors(bufread, bufwrite,
		m_numParticles,
		numPartsToElaborate,
		gdata->problem->m_deltap,
		m_simparams->slength,
		m_simparams->influenceRadius);

	bufwrite.clear_pending_state();
}

template<>
void GPUWorker::runCommand<JACOBI_UPDATE_EFFPRES>(CommandStruct const& cmd)
{
	uint numPartsToElaborate = (cmd.only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	const BufferList bufread = extractExistingBufferList(m_dBuffers, cmd.reads);
	BufferList bufwrite = extractExistingBufferList(m_dBuffers, cmd.updates) |
		extractGeneralBufferList(m_dBuffers, cmd.writes);

	bufwrite.add_manipulator_on_write("update_jacobi_effpres");

	gdata->h_jacobiResidual[m_deviceIndex] = viscEngine->update_jacobi_effpres(
		bufread, bufwrite,
		m_numParticles,
		numPartsToElaborate,
		gdata->problem->m_deltap,
		m_simparams->slength,
		m_simparams->influenceRadius);

	bufwrite.clear_pending_state();
}

void GPUWorker::uploadConstants()
{
	// NOTE: visccoeff must be set before uploading the constants. This is done in GPUSPH main cycle

	// Setting kernels and kernels derivative factors
	forcesEngine->setconstants(m_simparams, m_physparams, gdata->problem->m_deltap, gdata->worldOrigin, gdata->gridSize, gdata->cellSize,
		m_numAllocatedParticles);
	integrationEngine->setconstants(m_physparams, gdata->worldOrigin, gdata->gridSize, gdata->cellSize,
		m_numAllocatedParticles, m_simparams->neiblistsize, m_simparams->slength);
	neibsEngine->setconstants(m_simparams, m_physparams, gdata->worldOrigin, gdata->gridSize, gdata->cellSize,
		m_numAllocatedParticles);
	if(!postProcEngines.empty())
		postProcEngines.begin()->second->setconstants(m_simparams, m_physparams, m_numAllocatedParticles);

	// Compute maximum viscosity in Newtonian case
	if (m_simparams->rheologytype == NEWTONIAN)
		for (uint f = 0; f < m_physparams->numFluids(); ++f)
			m_max_kinvisc = fmaxf(m_max_kinvisc, m_physparams->kinematicvisc[f]);

	// Compute maximum sound speed, for CFL condition
	for (uint f = 0; f < m_physparams->numFluids(); ++f)
		m_max_sound_speed = fmaxf(m_max_sound_speed, m_physparams->sscoeff[f]);
	m_max_sound_speed *= 1.1;

}

// Auxiliary method for debugging purposes: downloads on the host one or multiple field values of
// a single particle of given INDEX. It should be considered a canvas for writing more complex,
// context-dependent checks. It replaces a minimal subset of capabilities of a proper debugger
// (like cuda-dbg) when that is not available or too slow.
// Parameters:
// - printID is just a constant string to distinguish method calls in different parts of the code;
// - pindex is the current index of the particle being investigated (to be found with BUFFER_VERTIDINDEX
//   (when available) or in doWrite() after saving (if a save was performed after last reorder).
// Possible improvement: make it accept buffer flags. But is it worth the time?
void GPUWorker::checkPartValByIndex(CommandStruct const& cmd,
	const char* printID, const uint pindex)
{
	// here it is possible to set a condition on the simulation state, device number, e.g.:
	// if (gdata->iterations <= 900 || gdata->iterations >= 1000) return;
	// if (m_deviceIndex == 1) return;

	const string current_state = cmd.src;
	const string next_state = cmd.dst;

	BufferList const& bufread = m_dBuffers.getState(current_state);
	BufferList const& bufwrite = m_dBuffers.getState(next_state);

	// get particle info
	particleinfo pinfo;
	CUDA_SAFE_CALL(cudaMemcpy(&pinfo, bufread.getData<BUFFER_INFO>() + pindex, sizeof(particleinfo),
		cudaMemcpyDeviceToHost));

	// this is the right place to filter for particle type, e.g.:
	// if (!FLUID(pinfo)) return;

	/*
	// get hash
	hashKey phash;
	CUDA_SAFE_CALL(cudaMemcpy(&phash, m_dBuffers.getData<BUFFER_HASH>() + pindex,
		sizeof(hashKey), cudaMemcpyDeviceToHost));
	uint3 gridpos = gdata->calcGridPosFromCellHash(cellHashFromParticleHash(phash));
	printf("HHd%u_%s: id %u (%s) idx %u IT %u, phash %lx, cell %u (%d,%d,%d)\n",
		m_deviceIndex, printID, id(pinfo),
		(FLUID(pinfo) ? "F" : (BOUNDARY(pinfo) ? "B" : (VERTEX(pinfo) ? "V" : "-"))),
		pindex, gdata->iterations, phash, cellHashFromParticleHash(phash),
		gridpos.x, gridpos.y, gridpos.z);
	*/

	// get vel(s)
	float4 rVel, wVel;
	CUDA_SAFE_CALL(cudaMemcpy(&rVel, bufread.getData<BUFFER_VEL>() + pindex,
		sizeof(float4), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&wVel, bufwrite.getData<BUFFER_VEL>() + pindex,
		sizeof(float4), cudaMemcpyDeviceToHost));
	printf("XXd%u_%s: id %u (%s) idx %u IT %lu, readVel (%g,%g,%g %g) writeVel  (%g,%g,%g %g)\n",
		m_deviceIndex, printID, id(pinfo),
		(FLUID(pinfo) ? "F" : (BOUNDARY(pinfo) ? "B" : (VERTEX(pinfo) ? "V" : "-"))),
		pindex, gdata->iterations,
		rVel.x, rVel.y, rVel.z, rVel.w, wVel.x, wVel.y, wVel.z, wVel.w );

	/*
	// get pos(s)
	// WARNING: it is a *local* pos! It is only useful if we are checking for relative distances of clearly wrong values
	float4 rPos, wPos;
	CUDA_SAFE_CALL(cudaMemcpy(&rPos, bufread.getData<BUFFER_POS>() + pindex,
		sizeof(float4), cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(&wPos, bufwrite.getData<BUFFER_POS>() + pindex,
		sizeof(float4), cudaMemcpyDeviceToHost));
	printf("XXd%u_%s: id %u (%s) idx %u IT %u, readPos (%g,%g,%g %g) writePos (%g,%g,%g %g)\n",
		m_deviceIndex, printID, id(pinfo),
		(FLUID(pinfo) ? "F" : (BOUNDARY(pinfo) ? "B" : (VERTEX(pinfo) ? "V" : "-"))),
		pindex, gdata->iterations,
		rPos.x, rPos.y, rPos.z, rPos.w, wPos.x, wPos.y, wPos.z, wPos.w );
	*/

	/*
	// get force
	float4 force;
	CUDA_SAFE_CALL(cudaMemcpy(&force, bufread.getData<BUFFER_FORCES>() + pindex,
		sizeof(float4), cudaMemcpyDeviceToHost));
	printf("XXd%u_%s: id %u (%s) idx %u IT %u, force (%g,%g,%g %g)\n",
		m_deviceIndex, printID, id(pinfo),
		(FLUID(pinfo) ? "F" : (BOUNDARY(pinfo) ? "B" : (VERTEX(pinfo) ? "V" : "-"))),
		pindex, gdata->iterations,
		force.x, force.y, force.z, force.w);
	*/

	/*
	// example of additional values
	if (m_simparams->boundarytype == SA_BOUNDARY) {
		// get grad_gammas
		float4 rGgam, wGgam;
		CUDA_SAFE_CALL(cudaMemcpy(&rGgam, m_dBuffers.getData<BUFFER_GRADGAMMA>(gdata->currentRead[BUFFER_GRADGAMMA]) + pindex,
			sizeof(float4), cudaMemcpyDeviceToHost));
		CUDA_SAFE_CALL(cudaMemcpy(&wGgam, m_dBuffers.getData<BUFFER_GRADGAMMA>(gdata->currentWrite[BUFFER_GRADGAMMA]) + pindex,
			sizeof(float4), cudaMemcpyDeviceToHost));
		printf("XXd%u_%s: id %u (%s) idx %u IT %u, rGGam (%g,%g,%g %g) wGGam (%g,%g,%g %g)\n",
			m_deviceIndex, printID, id(pinfo),
			(FLUID(pinfo) ? "F" : (BOUNDARY(pinfo) ? "B" : (VERTEX(pinfo) ? "V" : "-"))),
			pindex, gdata->iterations,
			rGgam.x, rGgam.y, rGgam.z, rGgam.w, wGgam.x, wGgam.y, wGgam.z, wGgam.w );

		if (BOUNDARY(pinfo)) {
			// get vert pos
			float2 vPos0, vPos1, vPos2;
			CUDA_SAFE_CALL(cudaMemcpy(&vPos0, m_dBuffers.getData<BUFFER_VERTPOS>(0) + pindex,
				sizeof(float2), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(&vPos1, m_dBuffers.getData<BUFFER_VERTPOS>(1) + pindex,
				sizeof(float2), cudaMemcpyDeviceToHost));
			CUDA_SAFE_CALL(cudaMemcpy(&vPos2, m_dBuffers.getData<BUFFER_VERTPOS>(2) + pindex,
				sizeof(float2), cudaMemcpyDeviceToHost));
			// printf...
		}
	}
	*/
}


void GPUWorker::uploadEulerBodiesCentersOfGravity()
{
	integrationEngine->setrbcg(gdata->s_hRbCgGridPos, gdata->s_hRbCgPos, m_simparams->numbodies);
}
template<>
void GPUWorker::runCommand<EULER_UPLOAD_OBJECTS_CG>(CommandStruct const& cmd)
{ uploadEulerBodiesCentersOfGravity(); }

void GPUWorker::uploadForcesBodiesCentersOfGravity()
{
	forcesEngine->setrbcg(gdata->s_hRbCgGridPos, gdata->s_hRbCgPos, m_simparams->numbodies);
}
template<>
void GPUWorker::runCommand<FORCES_UPLOAD_OBJECTS_CG>(CommandStruct const& cmd)
{ uploadForcesBodiesCentersOfGravity(); }


template<>
void GPUWorker::runCommand<UPLOAD_OBJECTS_MATRICES>(CommandStruct const& cmd)
// void GPUWorker::uploadBodiesTransRotMatrices()
{
	integrationEngine->setrbtrans(gdata->s_hRbTranslations, m_simparams->numbodies);
	integrationEngine->setrbsteprot(gdata->s_hRbRotationMatrices, m_simparams->numbodies);
}

template<>
void GPUWorker::runCommand<UPLOAD_OBJECTS_VELOCITIES>(CommandStruct const& cmd)
// void GPUWorker::uploadBodiesVelocities()
{
	integrationEngine->setrblinearvel(gdata->s_hRbLinearVelocities, m_simparams->numbodies);
	integrationEngine->setrbangularvel(gdata->s_hRbAngularVelocities, m_simparams->numbodies);
}

template<>
void GPUWorker::runCommand<IDLE>(CommandStruct const& cmd)
{ /* do nothing */ }

template<>
void GPUWorker::runCommand<QUIT>(CommandStruct const& cmd)
{
	/* TODO: this currently does nothing, but it should probably check
	 * that gdata->keep_going has been set false
	 */
}

template<>
void GPUWorker::runCommand<NUM_WORKER_COMMANDS>(CommandStruct const& cmd)
{
	unknownCommand(NUM_WORKER_COMMANDS);
}

void GPUWorker::unknownCommand(CommandName cmd)
{
	string err = "FATAL: command " + to_string(cmd)
		+ " (" + getCommandName(cmd) + ") issued on device " + to_string(m_deviceIndex)
		+ " is not implemented";
	throw std::runtime_error(err);
}


template<CommandName Cmd>
void GPUWorker::describeCommand(CommandStruct const& cmd)
{
	if (Cmd >= NUM_WORKER_COMMANDS) {
		/* nothing to describe, this is an error condition;
		 * it will be handled separately in the command dispatch switch
		 */
		return;
	}

	string desc = " T " + to_string(m_deviceIndex) +
		" issuing " + getCommandName(Cmd);

	// Add buffer specification, if needed
	if (CommandTraits<Cmd>::buffer_usage == DYNAMIC_BUFFER_USAGE)
		desc += describeCommandFlagsBuffers(cmd.flags);

	// Add extra information, if needed
	switch (Cmd) {
	case FILTER:
		desc += " " + string(FilterName[cmd.flags]);
		break;
	case POSTPROCESS:
		desc += " " + string(PostProcessName[cmd.flags]);
		break;
	case INIT_STATE:
	case RELEASE_STATE:
		desc += " " + cmd.src;
		break;
	case RENAME_STATE:
		desc += " " + cmd.src + " -> " + cmd.dst;
		break;
	case REMOVE_STATE_BUFFERS:
		desc += " < " + cmd.src;
		break;
	case SHARE_BUFFERS:
		desc += " : " + cmd.src + " <> " + cmd.dst;
		break;
	default:
		/* no other special cases */
		break;
	}

	cout << desc << endl;

}

// Actual thread calling GPU-methods
// Note that this has to be defined last because it needs to know about all
// the specializations of runCommand
void GPUWorker::simulationThread() {
	// INITIALIZATION PHASE

	CommandStruct cmd(IDLE);

	try {

		setDeviceProperties( checkCUDA(gdata, m_deviceIndex) );

		initialize();

		gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***

		// here GPUSPH::initialize is over and GPUSPH::runSimulation() is called

		gdata->threadSynchronizer->barrier(); // begins UPLOAD ***

		// if anything else failed (e.g. another worker was assigned an
		// non-existent device number and failed to complete initialize()
		// correctly), we shouldn't do anything. So check that keep_going is still true
		if (gdata->keep_going)
			uploadSubdomain();

		if (gdata->problem->simparams()->simflags & ENABLE_INLET_OUTLET)
			uploadNumOpenVertices();

		gdata->threadSynchronizer->barrier();  // end of UPLOAD, begins SIMULATION ***

		const bool dbg_step_printf = gdata->debug.print_step;
		const bool dbg_buffer_lists = gdata->debug.inspect_buffer_lists;

		// TODO automate the dbg_step_printf output
		// Here is a copy-paste from the CPU thread worker of branch cpusph, as a canvas
		while (gdata->keep_going) {

			cmd = gdata->nextCommand;

			switch (cmd.command) {
#define DEFINE_COMMAND(code, ...) \
			case code: \
				if (dbg_step_printf) describeCommand<code>(cmd); \
				runCommand<code>(cmd); \
				break;
#include "define_worker_commands.h"
#undef DEFINE_COMMAND
			default:
				unknownCommand(cmd.command);
			}
			if (dbg_buffer_lists) {
				string desc = " T " + to_string(m_deviceIndex) + " " + m_dBuffers.inspect();
				cout << desc << endl;
			}
			if (gdata->keep_going) {
				/*
				// example usage of checkPartValBy*()
				// alternatively, can be used in the previous switch construct, to check who changes what
				if (gdata->iterations >= 10) {
				dbg_step_printf = true; // optional
				checkPartValByIndex(cmd, "test", 0);
				}
				*/
				// the first barrier waits for the main thread to set the next command; the second is to unlock
				gdata->threadSynchronizer->barrier();  // CYCLE BARRIER 1
				gdata->threadSynchronizer->barrier();  // CYCLE BARRIER 2
			}
		}
	} catch (exception const& e) {
		cerr << "Device " << (int)m_deviceIndex << " thread " << hex << this_thread::get_id() << dec
			<< " iteration " << gdata->iterations
			<< " last command: " << cmd.command << " (" << getCommandName(cmd)
			<< "). Exception: " << e.what() << endl;
		// TODO FIXME cleaner way to handle this
		const_cast<GlobalData*>(gdata)->keep_going = false;
		const_cast<GlobalData*>(gdata)->ret |= 1;
		if (MULTI_NODE)
			gdata->networkManager->sendKillRequest();
	}

	gdata->threadSynchronizer->barrier();  // end of SIMULATION, begins FINALIZATION ***

	try {
		finalize();
	} catch (exception const& e) {
		// if anything goes wrong here, there isn't much we can do,
		// so just show the error and carry on
		cerr << e.what() << endl;
		const_cast<GlobalData*>(gdata)->ret |= 1;
		if (MULTI_NODE)
			gdata->networkManager->sendKillRequest();
	}

	gdata->threadSynchronizer->barrier();  // end of FINALIZATION ***

}

