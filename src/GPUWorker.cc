/*  Copyright 2012-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

// ostringstream
#include <sstream>
// FLT_MAX
#include <float.h>

#include "GPUWorker.h"
#include "Problem.h"

#include "cudabuffer.h"

// round_up
#include "utils.h"

// UINT_MAX
#include "limits.h"

GPUWorker::GPUWorker(GlobalData* _gdata, devcount_t _deviceIndex) :
	gdata(_gdata),
	m_simframework(gdata->simframework),
	neibsEngine(gdata->simframework->getNeibsEngine()),
	viscEngine(gdata->simframework->getViscEngine()),
	forcesEngine(gdata->simframework->getForcesEngine()),
	integrationEngine(gdata->simframework->getIntegrationEngine()),
	bcEngine(gdata->simframework->getBCEngine()),
	filterEngines(gdata->simframework->getFilterEngines()),
	postProcEngines(gdata->simframework->getPostProcEngines())
{
	m_deviceIndex = _deviceIndex;
	m_cudaDeviceNumber = gdata->device[m_deviceIndex];

	m_globalDeviceIdx = GlobalData::GLOBAL_DEVICE_ID(gdata->mpi_rank, _deviceIndex);

	printf("Global device id: %d (%d)\n", m_globalDeviceIdx, gdata->totDevices);

	// we know that GPUWorker is initialized when Problem was already
	m_simparams = gdata->problem->get_simparams();
	m_physparams = gdata->problem->get_physparams();

	// we also know Problem::fillparts() has already been called
	m_numInternalParticles = m_numParticles = gdata->s_hPartsPerDevice[m_deviceIndex];

	m_particleRangeBegin = 0;
	m_particleRangeEnd = m_numInternalParticles;

	m_numAllocatedParticles = 0;
	m_nGridCells = gdata->nGridCells;

	m_hostMemory = m_deviceMemory = 0;

	// set to true to force host staging even if peer access is set successfully
	m_disableP2Ptranfers = false;
	m_hPeerTransferBuffer = NULL;
	m_hPeerTransferBufferSize = 0;

	// used if GPUDirect is disabled
	m_hNetworkTransferBuffer = NULL;
	m_hNetworkTransferBufferSize = 0;

	m_dCompactDeviceMap = NULL;
	m_hCompactDeviceMap = NULL;
	m_dSegmentStart = NULL;

	m_forcesKernelTotalNumBlocks = 0;

	m_dBuffers.setAllocPolicy(gdata->simframework->getAllocPolicy());

	m_dBuffers.addBuffer<CUDABuffer, BUFFER_POS>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_VEL>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_INFO>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_FORCES>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_CONTUPD>();

	m_dBuffers.addBuffer<CUDABuffer, BUFFER_HASH>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_PARTINDEX>();
	m_dBuffers.addBuffer<CUDABuffer, BUFFER_NEIBSLIST>(-1); // neib list is initialized to all bits set

	if (m_simparams->simflags & ENABLE_XSPH)
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_XSPH>();

	if (m_simparams->visctype == SPSVISC)
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_TAU>();

	if (m_simframework->hasPostProcessOption(SURFACE_DETECTION, BUFFER_NORMALS))
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_NORMALS>();
	if (m_simframework->hasPostProcessEngine(VORTICITY))
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_VORTICITY>();

	if (m_simparams->simflags & ENABLE_DTADAPT) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_CFL>();
		if (m_simparams->simflags & ENABLE_DENSITY_SUM)
			m_dBuffers.addBuffer<CUDABuffer, BUFFER_CFL_DS>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_CFL_TEMP>();
		if (m_simparams->visctype == KEPSVISC)
			m_dBuffers.addBuffer<CUDABuffer, BUFFER_CFL_KEPS>();
	}

	if (m_simparams->boundarytype == SA_BOUNDARY) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_VERTIDINDEX>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_GRADGAMMA>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_BOUNDELEMENTS>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_VERTICES>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_VERTPOS>();
	}

	if (m_simparams->visctype == KEPSVISC) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_TKE>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_EPSILON>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_TURBVISC>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_DKDE>();
	}

	if (m_simparams->visctype == SPSVISC) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_SPS_TURBVISC>();
	}

	if (m_simparams->simflags & ENABLE_INLET_OUTLET || m_simparams->visctype == KEPSVISC)
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_EULERVEL>();

	if (m_simparams->sph_formulation == SPH_GRENIER) {
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_VOLUME>();
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_SIGMA>();
	}

	if (m_simframework->hasPostProcessEngine(CALC_PRIVATE))
		m_dBuffers.addBuffer<CUDABuffer, BUFFER_PRIVATE>();
}

GPUWorker::~GPUWorker() {
	// Free everything and pthread terminate
	// should check whether the pthread is still running and force its termination?
}

// Return the number of particles currently being handled (internal and r.o.)
uint GPUWorker::getNumParticles()
{
	return m_numParticles;
}

// Return the number of allocated particles
uint GPUWorker::getNumAllocatedParticles()
{
	return m_numAllocatedParticles;
}

uint GPUWorker::getNumInternalParticles() {
	return m_numInternalParticles;
}

// Return the maximum number of particles the worker can handled (allocated)
uint GPUWorker::getMaxParticles()
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
			contrib *= m_simparams->maxneibsnum;
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
	tot += sizeof(m_dCellStart[0]);
	tot += sizeof(m_dCellEnd[0]);
	if (MULTI_DEVICE)
		tot += sizeof(m_dCompactDeviceMap[0]);
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

	uint numAllocableParticles = (freeMemory / computeMemoryPerParticle());

	if (numAllocableParticles < gdata->allocatedParticles)
		printf("NOTE: device %u can allocate %u particles, while the whole simulation might require %u\n",
			m_deviceIndex, numAllocableParticles, gdata->allocatedParticles);

	// allocate at most the number of particles required for the whole simulation
	m_numAllocatedParticles = min( numAllocableParticles, gdata->allocatedParticles );

	if (m_numAllocatedParticles < m_numParticles) {
		fprintf(stderr, "FATAL: thread %u needs %u particles, but we can only store %u in %s available of %s total with %s safety margin\n",
			m_deviceIndex, m_numParticles, m_numAllocatedParticles,
			gdata->memString(freeMemory).c_str(), gdata->memString(totMemory).c_str(),
			gdata->memString(safetyMargin).c_str());
		exit(1);
	}
}

// Cut all particles that are not internal.
// Assuming segments have already been filled and downloaded to the shared array.
// NOTE: here it would be logical to reset the cellStarts of the cells being cropped
// out. However, this would be quite inefficient. We leave them inconsistent for a
// few time and we will update them when importing peer cells.
void GPUWorker::dropExternalParticles()
{
	m_particleRangeEnd =  m_numParticles = m_numInternalParticles;
	gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] = EMPTY_SEGMENT;
	gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_CELL] = EMPTY_SEGMENT;
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
// NOTE/TODO: using async copies although gdata->s_dCellStarts[][] is not pinned yet
void GPUWorker::asyncCellIndicesUpload(uint fromCell, uint toCell)
{
	uint numCells = toCell - fromCell;
	CUDA_SAFE_CALL_NOSYNC(cudaMemcpyAsync(	(m_dCellStart + fromCell),
										(gdata->s_dCellStarts[m_deviceIndex] + fromCell),
										sizeof(uint) * numCells, cudaMemcpyHostToDevice, m_asyncH2DCopiesStream));
	CUDA_SAFE_CALL_NOSYNC(cudaMemcpyAsync(	(m_dCellEnd + fromCell),
										(gdata->s_dCellEnds[m_deviceIndex] + fromCell),
										sizeof(uint) * numCells, cudaMemcpyHostToDevice, m_asyncH2DCopiesStream));
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
				if (m_deviceIndex == 1 && gdata->iterations == 30 && false)
					printf(" BURST %u, incr. parts from %u to %u (+%u) because of cell %u\n", i,
						   m_bursts[i].numParticles - numPartsInCell, m_bursts[i].numParticles,
							numPartsInCell, lin_cell );
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
void GPUWorker::transferBursts()
{
	// The buffer list that we want to access depends on the double-buffer selection.
	// The MultiBufferList::iterator works like a BufferList* , with the
	// advantage that we can get the index of the BufferList by subtracting the
	// iterator returned by getting the first BufferList
	MultiBufferList::iterator buflist = getBufferListByCommandFlags(gdata->commandFlags);

	// actual index of the buffer list in the multibufferlist (used to get the same
	// buffer list from the peer)
	const size_t buflist_idx = buflist - m_dBuffers.getBufferList(0);

	// Sanity check: if any of the buffers to transfer is double-buffered, then
	// which of the copies needs to be transferred _must_ have been specified
	const flag_t need_dbl_buffer_specified = gdata->allocPolicy->get_multi_buffered(gdata->commandFlags);
	// was it specified?
	const bool dbl_buffer_specified = ( (gdata->commandFlags & DBLBUFFER_READ ) || (gdata->commandFlags & DBLBUFFER_WRITE) );

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

			// transfer the data if burst is not empty
			if (m_bursts[i].numParticles == 0) continue;

			/*
			printf("IT %u D %u burst %u #parts %u dir %s (%u -> %u) scope %s\n",
				gdata->iterations, m_deviceIndex, i, m_bursts[i].numParticles,
				(m_bursts[i].direction == SND ? "SND" : "RCV"),
				(m_bursts[i].direction == SND ? m_globalDeviceIdx : m_bursts[i].peer_gidx),
				(m_bursts[i].direction == SND ? m_bursts[i].peer_gidx : m_globalDeviceIdx),
				(m_bursts[i].scope == NODE_SCOPE ? "NODE" : "NETWORK") );
			// */

			// iterate over all defined buffers and see which were requested
			// NOTE: std::map, from which BufferList is derived, is an _ordered_ container,
			// with the ordering set by the key, in our case the unsigned integer type flag_t,
			// so we have guarantee that the map will always be traversed in the same order
			// (unless stuff is inserted/deleted, which shouldn't happen at program runtime)
			BufferList::iterator bufset = buflist->begin();
			const BufferList::iterator stop = buflist->end();
			for ( ; bufset != stop ; ++bufset) {
				flag_t bufkey = bufset->first;
				if (!(gdata->commandFlags & bufkey))
					continue; // skip unwanted buffers

				AbstractBuffer *buf = bufset->second;

				// TODO it would be better to have this check done in a doCommand() sanitizer
				if ((bufkey & need_dbl_buffer_specified) && !dbl_buffer_specified) {
					std::stringstream err_msg;
					err_msg << "Import request for double-buffered " << buf->get_buffer_name()
						<< " array without a specification of which buffer to use.";
						throw runtime_error(err_msg.str());
				}

				const unsigned int _size = m_bursts[i].numParticles * buf->get_element_size();

				// retrieve peer's indices, if intra-node
				const AbstractBuffer *peerbuf = NULL;
				uint peerCudaDevNum = 0;
				if (m_bursts[i].scope == NODE_SCOPE) {
					uchar peerDevIdx = gdata->DEVICE(m_bursts[i].peer_gidx);
					peerCudaDevNum = gdata->device[peerDevIdx];
					peerbuf = gdata->GPUWORKERS[peerDevIdx]->getBuffer(buflist_idx, bufkey);
				}

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
void GPUWorker::importExternalCells()
{
	if (gdata->nextCommand == APPEND_EXTERNAL)
		transferBurstsSizes();
	if ( (gdata->nextCommand == APPEND_EXTERNAL) || (gdata->nextCommand == UPDATE_EXTERNAL) )
		transferBursts();

	// cudaMemcpyPeerAsync() is asynchronous with the host. If striping is disabled, we want to synchronize
	// for the completion of the transfers. Otherwise, FORCES_COMPLETE will synchronize everything
	if (!gdata->clOptions->striping && MULTI_GPU)
		cudaDeviceSynchronize();

	// here will sync the MPI transfers when (if) we'll switch to non-blocking calls
	// if (!gdata->striping && MULTI_NODE)...
}

// All the allocators assume that gdata is updated with the number of particles (done by problem->fillparts).
// Later this will be changed since each thread does not need to allocate the global number of particles.
size_t GPUWorker::allocateHostBuffers() {
	// common sizes
	const size_t uintCellsSize = sizeof(uint) * m_nGridCells;

	size_t allocated = 0;

	if (MULTI_DEVICE) {
		m_hCompactDeviceMap = new uint[m_nGridCells];
		memset(m_hCompactDeviceMap, 0, uintCellsSize);
		allocated += uintCellsSize;

		// allocate a 1Mb transferBuffer if peer copies are disabled
		if (m_disableP2Ptranfers)
			resizePeerTransferBuffer(1024 * 1024);

		// ditto for network transfers
		if (!gdata->clOptions->gpudirect)
			resizeNetworkTransferBuffer(1024 * 1024);
	}

	m_hostMemory += allocated;
	return allocated;
}

size_t GPUWorker::allocateDeviceBuffers() {
	// common sizes
	// compute common sizes (in bytes)

	const size_t uintCellsSize = sizeof(uint) * m_nGridCells;
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
			nels *= m_simparams->maxneibsnum; // number of particles times max neibs num
		else if (key == BUFFER_CFL_TEMP)
			nels = tempCflEls;
		else if (key == BUFFERS_CFL) // other CFL buffers
			nels = fmaxElements;

		allocated += m_dBuffers.alloc(key, nels);
		++iter;
	}

	CUDA_SAFE_CALL(cudaMalloc(&m_dCellStart, uintCellsSize));
	allocated += uintCellsSize;

	CUDA_SAFE_CALL(cudaMalloc(&m_dCellEnd, uintCellsSize));
	allocated += uintCellsSize;

	if (MULTI_DEVICE) {
		// TODO: an array of uchar would suffice
		CUDA_SAFE_CALL(cudaMalloc(&m_dCompactDeviceMap, uintCellsSize));
		// initialize anyway for single-GPU simulations
		CUDA_SAFE_CALL(cudaMemset(m_dCompactDeviceMap, 0, uintCellsSize));
		allocated += uintCellsSize;

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
		m_numForcesBodiesParticles = gdata->problem->get_forces_bodies_numparts();
		printf("number of forces rigid bodies particles = %d\n", m_numForcesBodiesParticles);

		int objParticlesFloat4Size = m_numForcesBodiesParticles*sizeof(float4);
		int objParticlesUintSize = m_numForcesBodiesParticles*sizeof(uint);

		CUDA_SAFE_CALL(cudaMalloc(&m_dRbTorques, objParticlesFloat4Size));
		CUDA_SAFE_CALL(cudaMalloc(&m_dRbForces, objParticlesFloat4Size));
		CUDA_SAFE_CALL(cudaMalloc(&m_dRbNum, objParticlesUintSize));

		allocated += 2 * objParticlesFloat4Size + objParticlesUintSize;

		uint* rbnum = new uint[m_numForcesBodiesParticles];

		forcesEngine->setrbstart(gdata->s_hRbFirstIndex, m_simparams->numforcesbodies);

		int offset = 0;
		for (uint i = 0; i < m_simparams->numforcesbodies; i++) {
			// set rbnum for each object particle; it is the key for the reduction
			for (int j = 0; j < gdata->problem->get_body_numparts(i); j++)
				rbnum[offset + j] = i;
			offset += gdata->problem->get_body_numparts(i);
		}
		size_t  size = m_numForcesBodiesParticles*sizeof(uint);
		CUDA_SAFE_CALL(cudaMemcpy((void *) m_dRbNum, (void*) rbnum, size, cudaMemcpyHostToDevice));

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
	if (MULTI_DEVICE)
		delete [] m_hCompactDeviceMap;

	if (m_hPeerTransferBuffer)
		cudaFreeHost(m_hPeerTransferBuffer);

	if (m_hNetworkTransferBuffer)
		cudaFreeHost(m_hNetworkTransferBuffer);

	// here: dem host buffers?
}

void GPUWorker::deallocateDeviceBuffers() {

	m_dBuffers.clear();

	CUDA_SAFE_CALL(cudaFree(m_dCellStart));
	CUDA_SAFE_CALL(cudaFree(m_dCellEnd));

	if (MULTI_DEVICE) {
		CUDA_SAFE_CALL(cudaFree(m_dCompactDeviceMap));
		CUDA_SAFE_CALL(cudaFree(m_dSegmentStart));
	}

	CUDA_SAFE_CALL(cudaFree(m_dNewNumParticles));

	if (m_simparams->simflags & (ENABLE_INLET_OUTLET | ENABLE_WATER_DEPTH))
		CUDA_SAFE_CALL(cudaFree(m_dIOwaterdepth));

	if (m_simparams->numforcesbodies) {
		CUDA_SAFE_CALL(cudaFree(m_dRbTorques));
		CUDA_SAFE_CALL(cudaFree(m_dRbForces));
		CUDA_SAFE_CALL(cudaFree(m_dRbNum));

		// DEBUG
		// delete [] m_hRbForces;
		// delete [] m_hRbTorques;
	}


	// here: dem device buffers?
}

void GPUWorker::createEventsAndStreams()
{
	// init streams
#if CUDA_VERSION < 5000
	cudaStreamCreate(&m_asyncD2HCopiesStream);
	cudaStreamCreate(&m_asyncH2DCopiesStream);
	cudaStreamCreate(&m_asyncPeerCopiesStream);
#else
	// init streams
	cudaStreamCreateWithFlags(&m_asyncD2HCopiesStream, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&m_asyncH2DCopiesStream, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&m_asyncPeerCopiesStream, cudaStreamNonBlocking);
#endif
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

MultiBufferList::iterator
GPUWorker::getBufferListByCommandFlags(flag_t flags)
{
	return (flags & DBLBUFFER_READ ?
		m_dBuffers.getReadBufferList() : flags & DBLBUFFER_WRITE ?
		m_dBuffers.getWriteBufferList() : m_dBuffers.getBufferList(0));
}

// upload subdomain, just allocated and sorted by main thread
void GPUWorker::uploadSubdomain() {
	// indices
	const uint firstInnerParticle	= gdata->s_hStartPerDevice[m_deviceIndex];
	const uint howManyParticles	= gdata->s_hPartsPerDevice[m_deviceIndex];

	// is the device empty? (unlikely but possible before LB kicks in)
	if (howManyParticles == 0) return;

	// buffers to skip in the upload. Rationale:
	// POS_GLOBAL is computed on host from POS and HASH
	// NORMALS and VORTICITY are post-processing, so always produced on device
	// and _downloaded_ to host, never uploaded
	static const flag_t skip_bufs = BUFFER_POS_GLOBAL |
		BUFFER_NORMALS | BUFFER_VORTICITY;

	// we upload data to the READ buffers
	BufferList& buflist = *m_dBuffers.getReadBufferList();

	// iterate over each array in the _host_ buffer list, and upload data
	// to the (first) read buffer
	// if it is not in the skip list
	BufferList::const_iterator onhost = gdata->s_hBuffers.begin();
	const BufferList::const_iterator stop = gdata->s_hBuffers.end();
	for ( ; onhost != stop ; ++onhost) {
		flag_t buf_to_up = onhost->first;
		if (buf_to_up & skip_bufs)
			continue;

		AbstractBuffer *buf = buflist[buf_to_up];
		size_t _size = howManyParticles * buf->get_element_size();

		printf("Thread %d uploading %d %s items (%s) on device %d from position %d\n",
				m_deviceIndex, howManyParticles, buf->get_buffer_name(),
				gdata->memString(_size).c_str(), m_cudaDeviceNumber, firstInnerParticle);

		// get all the arrays of which this buffer is composed
		// (actually currently all arrays are simple, since the only complex arrays (TAU
		// and VERTPOS) have no host counterpart)
		for (uint ai = 0; ai < buf->get_array_count(); ++ai) {
			void *dstptr = buf->get_buffer(ai);
			const void *srcptr = onhost->second->get_offset_buffer(ai, firstInnerParticle);
			CUDA_SAFE_CALL(cudaMemcpy(dstptr, srcptr, _size, cudaMemcpyHostToDevice));
		}
	}
}

// Download the subset of the specified buffer to the correspondent shared CPU array.
// Makes multiple transfers. Only downloads the subset relative to the internal particles.
// For double buffered arrays, uses the READ buffers unless otherwise specified. Can be
// used for either the read or the write buffers, not both.
void GPUWorker::dumpBuffers() {
	// indices
	uint firstInnerParticle	= gdata->s_hStartPerDevice[m_deviceIndex];
	uint howManyParticles	= gdata->s_hPartsPerDevice[m_deviceIndex];

	// is the device empty? (unlikely but possible before LB kicks in)
	if (howManyParticles == 0) return;

	const flag_t flags = gdata->commandFlags;

	// get the bufferlist to download data from
	const BufferList& buflist = *getBufferListByCommandFlags(flags);

	// iterate over each array in the _host_ buffer list, and download data
	// if it was requested
	BufferList::iterator onhost = gdata->s_hBuffers.begin();
	const BufferList::iterator stop = gdata->s_hBuffers.end();
	for ( ; onhost != stop ; ++onhost) {
		flag_t buf_to_get = onhost->first;
		if (!(buf_to_get & flags))
			continue;

		const AbstractBuffer *buf = buflist[buf_to_get];
		size_t _size = howManyParticles * buf->get_element_size();

		// get all the arrays of which this buffer is composed
		// (actually currently all arrays are simple, since the only complex arrays (TAU
		// and VERTPOS) have no host counterpart)
		for (uint ai = 0; ai < buf->get_array_count(); ++ai) {
			const void *srcptr = buf->get_buffer(ai);
			void *dstptr = onhost->second->get_offset_buffer(ai, firstInnerParticle);
			CUDA_SAFE_CALL(cudaMemcpy(dstptr, srcptr, _size, cudaMemcpyDeviceToHost));
		}
	}
}

// Swap the given double-buffered buffers
void GPUWorker::swapBuffers()
{
	const flag_t flags = gdata->commandFlags;

	m_dBuffers.swapBuffers(flags);
}

// Sets all cells as empty in device memory. Used before reorder
void GPUWorker::setDeviceCellsAsEmpty()
{
	CUDA_SAFE_CALL(cudaMemset(m_dCellStart, UINT_MAX, gdata->nGridCells  * sizeof(uint)));
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
void GPUWorker::downloadCellsIndices()
{
	size_t _size = gdata->nGridCells * sizeof(uint);
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_dCellStarts[m_deviceIndex],
								m_dCellStart,
								_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_dCellEnds[m_deviceIndex],
								m_dCellEnd,
								_size, cudaMemcpyDeviceToHost));
	/*_size = 4 * sizeof(uint);
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_dSegmentsStart[m_deviceIndex],
								m_dSegmentStart,
								_size, cudaMemcpyDeviceToHost));*/
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
void GPUWorker::updateSegments()
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
void GPUWorker::downloadNewNumParticles()
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
		if (activeParticles > m_numParticles)
			gdata->highestDevId[m_deviceIndex] += (activeParticles-m_numParticles)*gdata->totDevices;
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
		//    m_numParticles in both. One way to do this is to use a command flag or to reuse gdata->only_internal. This
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
void GPUWorker::uploadNewNumParticles()
{
	// uploading even if empty (usually not, right after append)
	CUDA_SAFE_CALL(cudaMemcpy(m_dNewNumParticles, &m_numParticles, sizeof(uint), cudaMemcpyHostToDevice));
}


// upload gravity (possibily called many times)
void GPUWorker::uploadGravity()
{
	// check if variable gravity is enabled
	if (m_simparams->gcallback)
		forcesEngine->setgravity(gdata->s_varGravity);
}

// upload planes (called once while planes are constant)
void GPUWorker::uploadPlanes()
{
	// check if planes > 0 (already checked before calling?)
	if (gdata->numPlanes > 0)
		forcesEngine->setplanes(gdata->numPlanes, gdata->s_hPlaneNormal,
			gdata->s_hPlanePointGridPos, gdata->s_hPlanePointLocalPos);
}


// Create a compact device map, for this device, from the global one,
// with each cell being marked in the high bits. Correctly handles periodicity.
// Also handles the optional extra displacement for periodicity. Since the cell
// offset is truncated, we need to add more cells to the outer neighbors (the extra
// disp. vector might not be a multiple of the cell size). However, only one extra cell
// per cell is added. This means that we might miss cells if the extra displacement is
// not parallel to one cartesian axis.
void GPUWorker::createCompactDeviceMap() {
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
				m_hCompactDeviceMap[cell_lin_idx] = cellType;
			}
	// here it is possible to save the compact device map
	// gdata->saveCompactDeviceMapToFile("", m_deviceIndex, m_hCompactDeviceMap);
}

// self-explanatory
void GPUWorker::uploadCompactDeviceMap() {
	size_t _size = m_nGridCells * sizeof(uint);
	CUDA_SAFE_CALL(cudaMemcpy(m_dCompactDeviceMap, m_hCompactDeviceMap, _size, cudaMemcpyHostToDevice));
}

// this should be singleton, i.e. should check that no other thread has been started (mutex + counter or bool)
void GPUWorker::run_worker() {
	// wrapper for pthread_create()
	// NOTE: the dynamic instance of the GPUWorker is passed as parameter
	pthread_create(&pthread_id, NULL, simulationThread, (void*)this);
}

// Join the simulation thread (in pthreads' terminology)
// WARNING: blocks the caller until the thread reaches pthread_exit. Be sure to call it after all barriers
// have been reached or may result in deadlock!
void GPUWorker::join_worker() {
	pthread_join(pthread_id, NULL);
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

const AbstractBuffer* GPUWorker::getBuffer(size_t list_idx, flag_t key) const
{
	return (*m_dBuffers.getBufferList(list_idx))[key];
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

// Actual thread calling GPU-methods
void* GPUWorker::simulationThread(void *ptr) {
	// INITIALIZATION PHASE

	// take the pointer of the instance starting this thread
	GPUWorker* instance = (GPUWorker*) ptr;

	// retrieve GlobalData and device number (index in process array)
	const GlobalData* gdata = instance->getGlobalData();
	const unsigned int cudaDeviceNumber = instance->getCUDADeviceNumber();
	const unsigned int deviceIndex = instance->getDeviceIndex();

	instance->setDeviceProperties( checkCUDA(gdata, deviceIndex) );

	// allow peers to access the device memory (for cudaMemcpyPeer[Async])
	instance->enablePeerAccess();

	// compute #parts to allocate according to the free memory on the device
	// must be done before uploading constants since some constants
	// (e.g. those for neibslist traversal) depend on the number of particles
	// allocated
	instance->computeAndSetAllocableParticles();

	// upload constants (PhysParames, some SimParams)
	instance->uploadConstants();

	// upload planes, if any
	instance->uploadPlanes();

	// allocate CPU and GPU arrays
	instance->allocateHostBuffers();
	instance->allocateDeviceBuffers();
	instance->printAllocatedMemory();

	// upload centers of gravity of the bodies
	instance->uploadEulerBodiesCentersOfGravity();
	instance->uploadForcesBodiesCentersOfGravity();

	// create and upload the compact device map (2 bits per cell)
	if (MULTI_DEVICE) {
		instance->createCompactDeviceMap();
		instance->computeCellBursts();
		instance->uploadCompactDeviceMap();
	}

	// TODO: here set_reduction_params() will be called (to be implemented in this class). These parameters can be device-specific.

	// init streams for async memcpys (only useful for multigpu?)
	instance->createEventsAndStreams();

	gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***

	// here GPUSPH::initialize is over and GPUSPH::runSimulation() is called

	gdata->threadSynchronizer->barrier(); // begins UPLOAD ***

	instance->uploadSubdomain();

	gdata->threadSynchronizer->barrier();  // end of UPLOAD, begins SIMULATION ***

	bool dbg_step_printf = false;

	// TODO
	// Here is a copy-paste from the CPU thread worker of branch cpusph, as a canvas
	while (gdata->keep_going) {
		switch (gdata->nextCommand) {
			// logging here?
			case IDLE:
				break;
			case SWAP_BUFFERS:
				if (dbg_step_printf) printf(" T %d issuing SWAP_BUFFERS\n", deviceIndex);
				instance->swapBuffers();
				break;
			case CALCHASH:
				if (dbg_step_printf) printf(" T %d issuing HASH\n", deviceIndex);
				instance->kernel_calcHash();
				break;
			case SORT:
				if (dbg_step_printf) printf(" T %d issuing SORT\n", deviceIndex);
				instance->kernel_sort();
				break;
			case CROP:
				if (dbg_step_printf) printf(" T %d issuing CROP\n", deviceIndex);
				instance->dropExternalParticles();
				break;
			case REORDER:
				if (dbg_step_printf) printf(" T %d issuing REORDER\n", deviceIndex);
				instance->kernel_reorderDataAndFindCellStart();
				break;
			case BUILDNEIBS:
				if (dbg_step_printf) printf(" T %d issuing BUILDNEIBS\n", deviceIndex);
				instance->kernel_buildNeibsList();
				break;
			case FORCES_SYNC:
				if (dbg_step_printf) printf(" T %d issuing FORCES_SYNC\n", deviceIndex);
				instance->kernel_forces();
				break;
			case FORCES_ENQUEUE:
				if (dbg_step_printf) printf(" T %d issuing FORCES_ENQUEUE\n", deviceIndex);
				instance->kernel_forces_async_enqueue();
				break;
			case FORCES_COMPLETE:
				if (dbg_step_printf) printf(" T %d issuing FORCES_COMPLETE\n", deviceIndex);
				instance->kernel_forces_async_complete();
				break;
			case EULER:
				if (dbg_step_printf) printf(" T %d issuing EULER\n", deviceIndex);
				instance->kernel_euler();
				break;
			case DUMP:
				if (dbg_step_printf) printf(" T %d issuing DUMP\n", deviceIndex);
				instance->dumpBuffers();
				break;
			case DUMP_CELLS:
				if (dbg_step_printf) printf(" T %d issuing DUMP_CELLS\n", deviceIndex);
				instance->downloadCellsIndices();
				break;
			case UPDATE_SEGMENTS:
				if (dbg_step_printf) printf(" T %d issuing UPDATE_SEGMENTS\n", deviceIndex);
				instance->updateSegments();
				break;
			case DOWNLOAD_IOWATERDEPTH:
				if (dbg_step_printf) printf(" T %d issuing DOWNLOAD_IOWATERDEPTH\n", deviceIndex);
				instance->kernel_download_iowaterdepth();
				break;
			case UPLOAD_IOWATERDEPTH:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_IOWATERDEPTH\n", deviceIndex);
				instance->kernel_upload_iowaterdepth();
				break;
			case DOWNLOAD_NEWNUMPARTS:
				if (dbg_step_printf) printf(" T %d issuing DOWNLOAD_NEWNUMPARTS\n", deviceIndex);
				instance->downloadNewNumParticles();
				break;
			case UPLOAD_NEWNUMPARTS:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_NEWNUMPARTS\n", deviceIndex);
				instance->uploadNewNumParticles();
				break;
			case APPEND_EXTERNAL:
				if (dbg_step_printf) printf(" T %d issuing APPEND_EXTERNAL\n", deviceIndex);
				instance->importExternalCells();
				break;
			case UPDATE_EXTERNAL:
				if (dbg_step_printf) printf(" T %d issuing UPDATE_EXTERNAL\n", deviceIndex);
				instance->importExternalCells();
				break;
			case FILTER:
				if (dbg_step_printf) printf(" T %d issuing FILTER\n", deviceIndex);
				instance->kernel_filter();
				break;
			case POSTPROCESS:
				if (dbg_step_printf) printf(" T %d issuing POSTPROCESS\n", deviceIndex);
				instance->kernel_postprocess();
				break;
			case DISABLE_OUTGOING_PARTS:
				if (dbg_step_printf) printf(" T %d issuing DISABLE_OUTGOING_PARTS:\n", deviceIndex);
				instance->kernel_disableOutgoingParts();
				break;
			case SA_CALC_SEGMENT_BOUNDARY_CONDITIONS:
				if (dbg_step_printf) printf(" T %d issuing SA_CALC_SEGMENT_BOUNDARY_CONDITIONS\n", deviceIndex);
				instance->kernel_saSegmentBoundaryConditions();
				break;
			case SA_CALC_VERTEX_BOUNDARY_CONDITIONS:
				if (dbg_step_printf) printf(" T %d issuing SA_CALC_VERTEX_BOUNDARY_CONDITIONS\n", deviceIndex);
				instance->kernel_saVertexBoundaryConditions();
				break;
			case SA_UPDATE_VERTIDINDEX:
				if (dbg_step_printf) printf(" T %d issuing SA_UPDATE_VERTIDINDEX\n", deviceIndex);
				instance->kernel_updateVertIdIndexBuffer();
				break;
			case IDENTIFY_CORNER_VERTICES:
				if (dbg_step_printf) printf(" T %d issuing IDENTIFY_CORNER_VERTICES\n", deviceIndex);
				instance->kernel_saIdentifyCornerVertices();
				break;
			case FIND_CLOSEST_VERTEX:
				if (dbg_step_printf) printf(" T %d issuing FIND_CLOSEST_VERTEX\n", deviceIndex);
				instance->kernel_saFindClosestVertex();
				break;
			case COMPUTE_DENSITY:
				if (dbg_step_printf) printf(" T %d issuing COMPUTE_DENSITY\n", deviceIndex);
				instance->kernel_compute_density();
				break;
			case SPS:
				if (dbg_step_printf) printf(" T %d issuing SPS\n", deviceIndex);
				instance->kernel_sps();
				break;
			case REDUCE_BODIES_FORCES:
				if (dbg_step_printf) printf(" T %d issuing REDUCE_BODIES_FORCES\n", deviceIndex);
				instance->kernel_reduceRBForces();
				break;
			case UPLOAD_GRAVITY:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_GRAVITY\n", deviceIndex);
				instance->uploadGravity();
				break;
			case UPLOAD_PLANES:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_PLANES\n", deviceIndex);
				instance->uploadPlanes();
				break;
			case EULER_UPLOAD_OBJECTS_CG:
				if (dbg_step_printf) printf(" T %d issuing EULER_UPLOAD_OBJECTS_CG\n", deviceIndex);
				instance->uploadEulerBodiesCentersOfGravity();
				break;
			case FORCES_UPLOAD_OBJECTS_CG:
				if (dbg_step_printf) printf(" T %d issuing FORCES_UPLOAD_OBJECTS_CG\n", deviceIndex);
				instance->uploadForcesBodiesCentersOfGravity();
				break;
			case UPLOAD_OBJECTS_MATRICES:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_OBJECTS_MATRICES\n", deviceIndex);
				instance->uploadBodiesTransRotMatrices();
				break;
			case UPLOAD_OBJECTS_VELOCITIES:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_OBJECTS_VELOCITIES\n", deviceIndex);
				instance->uploadBodiesVelocities();
				break;
			case IMPOSE_OPEN_BOUNDARY_CONDITION:
				if (dbg_step_printf) printf(" T %d issuing IMPOSE_OPEN_BOUNDARY_CONDITION\n", deviceIndex);
				instance->kernel_imposeBoundaryCondition();
				break;
			case QUIT:
				if (dbg_step_printf) printf(" T %d issuing QUIT\n", deviceIndex);
				// actually, setting keep_going to false and unlocking the barrier should be enough to quit the cycle
				break;
			default:
				fprintf(stderr, "FATAL: command (%d) issued on device %d is not implemented\n", gdata->nextCommand, deviceIndex);
				exit(1);
		}
		if (gdata->keep_going) {
			/*
			// example usage of checkPartValBy*()
			// alternatively, can be used in the previous switch construct, to check who changes what
			if (gdata->iterations >= 10) {
				dbg_step_printf = true; // optional
				instance->checkPartValByIndex("test", 0);
			}
			*/
			// the first barrier waits for the main thread to set the next command; the second is to unlock
			gdata->threadSynchronizer->barrier();  // CYCLE BARRIER 1
			gdata->threadSynchronizer->barrier();  // CYCLE BARRIER 2
		}
	}

	gdata->threadSynchronizer->barrier();  // end of SIMULATION, begins FINALIZATION ***

	// destroy streams
	instance->destroyEventsAndStreams();

	// deallocate buffers
	instance->deallocateHostBuffers();
	instance->deallocateDeviceBuffers();
	// ...what else?

	gdata->threadSynchronizer->barrier();  // end of FINALIZATION ***

	pthread_exit(NULL);
}

void GPUWorker::kernel_calcHash()
{
	// is the device empty? (unlikely but possible before LB kicks in)
	if (m_numParticles == 0) return;

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	// calcHashDevice() should use CPU-computed hashes at iteration 0, or some particles
	// might be lost (if a GPU computes a different hash and does not recognize the particles
	// as "own"). However, the high bits should be set, or edge cells won't be compacted at
	// the end and bursts will be sparse.
	// This is required only in MULTI_DEVICE simulations but it holds also on single-device
	// ones to keep numerical consistency.

	if (gdata->iterations == 0)
		neibsEngine->fixHash(
					bufwrite.getData<BUFFER_HASH>(),
					bufwrite.getData<BUFFER_PARTINDEX>(),
					bufread.getData<BUFFER_INFO>(),
					m_dCompactDeviceMap,
					m_numParticles);
	else
		neibsEngine->calcHash(
			// TODO FIXME POS is in/out, but it's taken on the READ position
					(float4*)bufread.getData<BUFFER_POS>(),
					bufwrite.getData<BUFFER_HASH>(),
					bufwrite.getData<BUFFER_PARTINDEX>(),
					bufread.getData<BUFFER_INFO>(),
					m_dCompactDeviceMap,
					m_numParticles);
}

void GPUWorker::kernel_sort()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_numInternalParticles : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	neibsEngine->sort(
			bufwrite.getData<BUFFER_HASH>(),
			bufwrite.getData<BUFFER_PARTINDEX>(),
			numPartsToElaborate);
}

void GPUWorker::kernel_reorderDataAndFindCellStart()
{
	// reset also if the device is empty (or we will download uninitialized values)
	setDeviceCellsAsEmpty();

	// is the device empty? (unlikely but possible before LB kicks in)
	if (m_numParticles == 0) return;

	MultiBufferList::const_iterator unsorted = m_dBuffers.getReadBufferList();
	MultiBufferList::iterator sorted = m_dBuffers.getWriteBufferList();

	// TODO this kernel needs a thorough reworking to only pass the needed buffers
	neibsEngine->reorderDataAndFindCellStart(
							m_dCellStart,	  // output: cell start index
							m_dCellEnd,		// output: cell end index
							m_dSegmentStart,

							// hash
							sorted->getData<BUFFER_HASH>(),
							// sorted particle indices
							sorted->getData<BUFFER_PARTINDEX>(),

							// output: sorted buffers
							sorted,
							// input: unsorted buffers
							unsorted,
							m_numParticles,
							m_dNewNumParticles);
}

void GPUWorker::kernel_buildNeibsList()
{
	neibsEngine->resetinfo();

	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	// reset the neighbor list
	CUDA_SAFE_CALL(cudaMemset(bufwrite.getData<BUFFER_NEIBSLIST>(),
		0xff, numPartsToElaborate * sizeof(neibdata) * m_simparams->maxneibsnum));

	// this is the square the distance used for neighboursearching of boundaries
	// it is delta p / 2 bigger than the standard radius
	// it is used to add segments into the neighbour list even if they are outside the kernel support
	const float boundNlSqInflRad = powf(sqrt(m_simparams->nlSqInfluenceRadius) + m_simparams->slength/m_simparams->sfactor/2.0f,2.0f);

	neibsEngine->buildNeibsList(
					bufwrite.getData<BUFFER_NEIBSLIST>(),
					bufread.getData<BUFFER_POS>(),
					bufread.getData<BUFFER_INFO>(),
			// TODO FIXME VERTICES is in/out, but it's taken on the READ position
					(vertexinfo*)bufread.getData<BUFFER_VERTICES>(),
					bufread.getData<BUFFER_BOUNDELEMENTS>(),
					bufwrite.getRawPtr<BUFFER_VERTPOS>(),
					bufwrite.getData<BUFFER_VERTIDINDEX>(),
					bufwrite.getData<BUFFER_HASH>(),
					m_dCellStart,
					m_dCellEnd,
					m_numParticles,
					numPartsToElaborate,
					m_nGridCells,
					m_simparams->nlSqInfluenceRadius,
					boundNlSqInflRad);

	// download the peak number of neighbors and the estimated number of interactions
	neibsEngine->getinfo( gdata->timingInfo[m_deviceIndex] );
}

// returns numBlocks as computed by forces()
uint GPUWorker::enqueueForcesOnRange(uint fromParticle, uint toParticle, uint cflOffset)
{
	bool firstStep = (gdata->commandFlags & INTEGRATOR_STEP_1);

	return forcesEngine->basicstep(
		m_dBuffers.getReadBufferList(),
		m_dBuffers.getWriteBufferList(),
		m_dRbForces,
		m_dRbTorques,
		m_dCellStart,
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
		firstStep ? 1 : 2);
}

// Bind the textures needed by forces kernel
void GPUWorker::bind_textures_forces()
{
	forcesEngine->bind_textures(m_dBuffers.getReadBufferList(),
		m_numParticles);
}

// Unbind the textures needed by forces kernel
void GPUWorker::unbind_textures_forces()
{
	forcesEngine->unbind_textures();
}

// Reduce array of maximum dt after forces, but only for adaptive timesteps
// Otherwise, just return the current (fixed) timestep
float GPUWorker::forces_dt_reduce()
{
	// no reduction for fixed timestep
	if (!(m_simparams->simflags & ENABLE_DTADAPT))
		return m_simparams->dt;

	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	// TODO multifluid: dtreduce needs the maximum viscosity. We compute it
	// here and pass it over. This is inefficient as we compute it every time,
	// while it should be done, while it could be done once only. OTOH, for
	// non-constant viscosities this should actually be done in-kernel to determine
	// the _actual_ maximum viscosity

	float max_kinematic = NAN;
	if (m_simparams->visctype != ARTVISC)
		for (uint f = 0; f < m_physparams->numFluids(); ++f)
			max_kinematic = fmaxf(max_kinematic, m_physparams->kinematicvisc[f]);

	return forcesEngine->dtreduce(
		m_simparams->slength,
		m_simparams->dtadaptfactor,
		max_kinematic,
		bufwrite.getData<BUFFER_CFL>(),
		bufwrite.getData<BUFFER_CFL_DS>(),
		bufwrite.getData<BUFFER_CFL_KEPS>(),
		bufwrite.getData<BUFFER_CFL_TEMP>(),
		m_forcesKernelTotalNumBlocks);
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

void GPUWorker::kernel_forces_async_enqueue()
{
	if (!gdata->only_internal)
		printf("WARNING: forces kernel called with only_internal == true, ignoring flag!\n");

	uint numPartsToElaborate = m_particleRangeEnd;

	m_forcesKernelTotalNumBlocks = 0;

	// if we have objects potentially shared across different devices, must reset their forces
	// and torques to avoid spurious contributions
	if (m_simparams->numforcesbodies > 0 && MULTI_DEVICE) {
		uint bodiesPartsSize = m_numForcesBodiesParticles * sizeof(float4);
		CUDA_SAFE_CALL(cudaMemset(m_dRbForces, 0.0f, bodiesPartsSize));
		CUDA_SAFE_CALL(cudaMemset(m_dRbTorques, 0.0f, bodiesPartsSize));
	}

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

	if (numPartsToElaborate > 0 ) {

		// bind textures
		bind_textures_forces();

		// enqueue the first kernel call (on the particles in edging cells)
		m_forcesKernelTotalNumBlocks += enqueueForcesOnRange(nonEdgingStripeSize, numPartsToElaborate, m_forcesKernelTotalNumBlocks);

		// the following event will be used to wait for the first stripe to complete
		cudaEventRecord(m_halfForcesEvent, 0);

		// enqueue the second kernel call (on the rest)
		m_forcesKernelTotalNumBlocks += enqueueForcesOnRange(0, nonEdgingStripeSize, m_forcesKernelTotalNumBlocks);

		// We could think of synchronizing in UPDATE_EXTERNAL or APPEND_EXTERNAL instead of here, so that we do not
		// cause any overhead (waiting here means waiting before next barrier, which means that devices which are
		// faster in the computation of the first stripe have to wait the others before issuing the second). However,
		// we need to ensure that the first stripe is finished in the *other* devices, before importing their cells.
		cudaEventSynchronize(m_halfForcesEvent);
	}
}

void GPUWorker::kernel_forces_async_complete()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// FLOAT_MAX is returned if kernels are not run (e.g. numPartsToElaborate == 0)
	float returned_dt = FLT_MAX;

	bool firstStep = (gdata->commandFlags & INTEGRATOR_STEP_1);

	if (numPartsToElaborate > 0 ) {
		// wait for the completion of the kernel
		cudaDeviceSynchronize();

		// unbind the textures
		unbind_textures_forces();

		// reduce dt
		returned_dt = forces_dt_reduce();
	}

	// gdata->dts is directly used instead of handling dt1 and dt2
	//printf(" Step %d, bool %d, returned %g, current %g, ",
	//	gdata->step, firstStep, returned_dt, gdata->dts[devnum]);
	if (firstStep)
		gdata->dts[m_deviceIndex] = returned_dt;
	else
		gdata->dts[m_deviceIndex] = min(gdata->dts[m_deviceIndex], returned_dt);
}


void GPUWorker::kernel_forces()
{
	if (!gdata->only_internal)
		printf("WARNING: forces kernel called with only_internal == true, ignoring flag!\n");

	uint numPartsToElaborate = m_particleRangeEnd;

	m_forcesKernelTotalNumBlocks = 0;

	// FLOAT_MAX is returned if kernels are not run (e.g. numPartsToElaborate == 0)
	float returned_dt = FLT_MAX;

	bool firstStep = (gdata->commandFlags & INTEGRATOR_STEP_1);

	// if we have objects potentially shared across different devices, must reset their forces
	// and torques to avoid spurious contributions
	if (m_simparams->numforcesbodies > 0 && MULTI_DEVICE) {
		uint bodiesPartsSize = m_numForcesBodiesParticles * sizeof(float4);
		CUDA_SAFE_CALL(cudaMemset(m_dRbForces, 0.0f, bodiesPartsSize));
		CUDA_SAFE_CALL(cudaMemset(m_dRbTorques, 0.0f, bodiesPartsSize));
	}

	const uint fromParticle = 0;
	const uint toParticle = numPartsToElaborate;

	if (numPartsToElaborate > 0 ) {

		// bind textures
		bind_textures_forces();

		// enqueue the kernel call
		m_forcesKernelTotalNumBlocks = enqueueForcesOnRange(fromParticle, toParticle, 0);

		// unbind the textures
		unbind_textures_forces();

		// reduce dt
		returned_dt = forces_dt_reduce();
	}

	// gdata->dts is directly used instead of handling dt1 and dt2
	//printf(" Step %d, bool %d, returned %g, current %g, ",
	//	gdata->step, firstStep, returned_dt, gdata->dts[devnum]);
	if (firstStep)
		gdata->dts[m_deviceIndex] = returned_dt;
	else
		gdata->dts[m_deviceIndex] = min(gdata->dts[m_deviceIndex], returned_dt);
	//printf("set to %g\n",gdata->dts[m_deviceIndex]);
}

void GPUWorker::kernel_euler()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	bool firstStep = (gdata->commandFlags & INTEGRATOR_STEP_1);

	integrationEngine->basicstep(
		m_dBuffers.getReadBufferList(),	// this is the read only arrays
		m_dBuffers.getReadBufferList(),	// the read array but it will be written to in certain cases (densitySum)
		m_dBuffers.getWriteBufferList(),
		m_dCellStart,
		m_numParticles,
		numPartsToElaborate,
		gdata->dt, // m_dt,
		gdata->dt/2.0f, // m_dt/2.0,
		firstStep ? 1 : 2,
		gdata->t + (firstStep ? gdata->dt / 2.0f : gdata->dt),
		m_simparams->slength,
		m_simparams->influenceRadius);
}

void GPUWorker::kernel_download_iowaterdepth()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	bcEngine->downloadIOwaterdepth(
			gdata->h_IOwaterdepth[m_deviceIndex],
			m_dIOwaterdepth,
			m_simparams->numOpenBoundaries);

}

void GPUWorker::kernel_upload_iowaterdepth()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	bcEngine->uploadIOwaterdepth(
			gdata->h_IOwaterdepth[0],
			m_dIOwaterdepth,
			m_simparams->numOpenBoundaries);

}

void GPUWorker::kernel_imposeBoundaryCondition()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	gdata->problem->imposeBoundaryConditionHost(
			bufwrite.getData<BUFFER_VEL>(),
			bufwrite.getData<BUFFER_EULERVEL>(),
			bufwrite.getData<BUFFER_TKE>(),
			bufwrite.getData<BUFFER_EPSILON>(),
			bufread.getData<BUFFER_INFO>(),
			bufread.getData<BUFFER_POS>(),
			(m_simparams->simflags & ENABLE_WATER_DEPTH) ? m_dIOwaterdepth : NULL,
			gdata->t,
			m_numParticles,
			m_simparams->numOpenBoundaries,
			numPartsToElaborate,
			bufread.getData<BUFFER_HASH>());

}

void GPUWorker::kernel_filter()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	FilterType filtertype = FilterType(gdata->extraCommandArg);
	FilterEngineSet::const_iterator filterpair(filterEngines.find(filtertype));
	// make sure we're going to call an instantiated filter
	if (filterpair == filterEngines.end()) {
		throw invalid_argument("non-existing filter invoked");
	}

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	filterpair->second->process(
		bufread.getData<BUFFER_POS>(),
		bufread.getData<BUFFER_VEL>(),
		bufwrite.getData<BUFFER_VEL>(),
		bufread.getData<BUFFER_INFO>(),
		bufread.getData<BUFFER_HASH>(),
		m_dCellStart,
		bufread.getData<BUFFER_NEIBSLIST>(),
		m_numParticles,
		numPartsToElaborate,
		m_simparams->slength,
		m_simparams->influenceRadius);
}

void GPUWorker::kernel_postprocess()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	PostProcessType proctype = PostProcessType(gdata->extraCommandArg);
	PostProcessEngineSet::const_iterator procpair(postProcEngines.find(proctype));
	// make sure we're going to call an instantiated filter
	if (procpair == postProcEngines.end()) {
		throw invalid_argument("non-existing postprocess filter invoked");
	}


	procpair->second->process(
		m_dBuffers.getReadBufferList(),
		m_dBuffers.getWriteBufferList(),
		m_dCellStart,
		m_numParticles,
		numPartsToElaborate,
		m_simparams->slength,
		m_simparams->influenceRadius);
}

void GPUWorker::kernel_compute_density()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	MultiBufferList::const_iterator bufread = m_dBuffers.getReadBufferList();
	MultiBufferList::iterator bufwrite = m_dBuffers.getWriteBufferList();

	forcesEngine->compute_density(bufread, bufwrite,
		m_dCellStart,
		numPartsToElaborate,
		m_simparams->slength,
		m_simparams->influenceRadius);
}


// TODO FIXME RENAME METHOD
void GPUWorker::kernel_sps()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	viscEngine->process(bufwrite.getRawPtr<BUFFER_TAU>(),
		bufwrite.getData<BUFFER_SPS_TURBVISC>(),
		bufread.getData<BUFFER_POS>(),
		bufread.getData<BUFFER_VEL>(),
		bufread.getData<BUFFER_INFO>(),
		bufread.getData<BUFFER_HASH>(),
		m_dCellStart,
		bufread.getData<BUFFER_NEIBSLIST>(),
		m_numParticles,
		numPartsToElaborate,
		m_simparams->slength,
		m_simparams->influenceRadius);
}

void GPUWorker::kernel_reduceRBForces()
{
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

	if (numforcesbodies)
		forcesEngine->reduceRbForces(m_dRbForces, m_dRbTorques, m_dRbNum, gdata->s_hRbLastIndex,
				gdata->s_hRbDeviceTotalForce + m_deviceIndex*numforcesbodies,
				gdata->s_hRbDeviceTotalTorque + m_deviceIndex*numforcesbodies,
				numforcesbodies, m_numForcesBodiesParticles);

}

void GPUWorker::kernel_saSegmentBoundaryConditions()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	bool initStep = (gdata->commandFlags & INITIALIZATION_STEP);
	bool firstStep = (gdata->commandFlags & INTEGRATOR_STEP_1);

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	bcEngine->saSegmentBoundaryConditions(
				bufwrite.getData<BUFFER_POS>(),
				bufwrite.getData<BUFFER_VEL>(),
				bufwrite.getData<BUFFER_TKE>(),
				bufwrite.getData<BUFFER_EPSILON>(),
				bufwrite.getData<BUFFER_EULERVEL>(),
				bufwrite.getData<BUFFER_GRADGAMMA>(),
				bufwrite.getData<BUFFER_VERTICES>(),
				bufread.getData<BUFFER_VERTIDINDEX>(),
				bufread.getRawPtr<BUFFER_VERTPOS>(),
				bufread.getData<BUFFER_BOUNDELEMENTS>(),
				bufread.getData<BUFFER_INFO>(),
				bufread.getData<BUFFER_HASH>(),
				m_dCellStart,
				bufread.getData<BUFFER_NEIBSLIST>(),
				m_numParticles,
				numPartsToElaborate,
				gdata->problem->m_deltap,
				m_simparams->slength,
				m_simparams->influenceRadius,
				initStep,
				firstStep ? 1 : 2);
}

void GPUWorker::kernel_updateVertIdIndexBuffer()
{
	// it is possible to run on internal particles only, although current design makes it meaningful only on all particles
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	neibsEngine->updateVertIDToIndex(
						bufread.getData<BUFFER_INFO>(),
						bufwrite.getData<BUFFER_VERTIDINDEX>(),
						numPartsToElaborate);
}

void GPUWorker::kernel_saVertexBoundaryConditions()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	// pos, vel, tke, eps are read from current*Read, except
	// on the second step, whe they are read from current*Write
	bool initStep = (gdata->commandFlags & INITIALIZATION_STEP);
	bool firstStep = (gdata->commandFlags & INTEGRATOR_STEP_1);

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	bcEngine->saVertexBoundaryConditions(
				bufwrite.getData<BUFFER_POS>(),
				bufwrite.getData<BUFFER_VEL>(),
				bufwrite.getData<BUFFER_TKE>(),
				bufwrite.getData<BUFFER_EPSILON>(),
				bufwrite.getData<BUFFER_GRADGAMMA>(),
				bufwrite.getData<BUFFER_EULERVEL>(),
				bufwrite.getData<BUFFER_FORCES>(),
				bufwrite.getData<BUFFER_CONTUPD>(),
				bufread.getData<BUFFER_BOUNDELEMENTS>(),
				bufwrite.getData<BUFFER_VERTICES>(),
				bufread.getData<BUFFER_VERTIDINDEX>(),

				// TODO FIXME INFO and HASH are in/out, but it's taken on the READ position
				// (updated in-place for generated particles)
				(particleinfo*)bufread.getData<BUFFER_INFO>(),
				(hashKey*)bufread.getData<BUFFER_HASH>(),

				m_dCellStart,
				bufread.getData<BUFFER_NEIBSLIST>(),
				m_numParticles,
				(firstStep ? NULL : m_dNewNumParticles),	// no m_dNewNumParticles at first step
				numPartsToElaborate,
				firstStep ? gdata->dt / 2.0f : gdata->dt,
				initStep ? 0 : (firstStep ? 1 : 2),
				gdata->problem->m_deltap,
				m_simparams->slength,
				m_simparams->influenceRadius,
				gdata->highestDevId[m_deviceIndex],
				initStep,
				m_globalDeviceIdx,
				gdata->totDevices);
}

void GPUWorker::kernel_saIdentifyCornerVertices()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	bcEngine->saIdentifyCornerVertices(
				bufread.getData<BUFFER_POS>(),
				bufread.getData<BUFFER_BOUNDELEMENTS>(),
				bufwrite.getData<BUFFER_INFO>(),
				bufread.getData<BUFFER_HASH>(),
				bufread.getData<BUFFER_VERTICES>(),
				m_dCellStart,
				bufread.getData<BUFFER_NEIBSLIST>(),
				m_numParticles,
				numPartsToElaborate,
				gdata->problem->m_deltap,
				m_simparams->epsilon);
}

void GPUWorker::kernel_saFindClosestVertex()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	bcEngine->saFindClosestVertex(
				bufread.getData<BUFFER_POS>(),
				bufwrite.getData<BUFFER_INFO>(),
				bufwrite.getData<BUFFER_VERTICES>(),
				bufread.getData<BUFFER_VERTIDINDEX>(),
				bufread.getData<BUFFER_HASH>(),
				m_dCellStart,
				bufread.getData<BUFFER_NEIBSLIST>(),
				m_numParticles,
				numPartsToElaborate);
}

void GPUWorker::kernel_disableOutgoingParts()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

	bcEngine->disableOutgoingParts(
				bufwrite.getData<BUFFER_POS>(),
				bufwrite.getData<BUFFER_VERTICES>(),
				bufread.getData<BUFFER_INFO>(),
				m_numParticles,
				numPartsToElaborate);
}

void GPUWorker::uploadConstants()
{
	// NOTE: visccoeff must be set before uploading the constants. This is done in GPUSPH main cycle

	// Setting kernels and kernels derivative factors
	forcesEngine->setconstants(m_simparams, m_physparams, gdata->worldOrigin, gdata->gridSize, gdata->cellSize,
		m_numAllocatedParticles);
	integrationEngine->setconstants(m_physparams, gdata->worldOrigin, gdata->gridSize, gdata->cellSize,
		m_numAllocatedParticles, m_simparams->maxneibsnum, m_simparams->slength);
	neibsEngine->setconstants(m_simparams, m_physparams, gdata->worldOrigin, gdata->gridSize, gdata->cellSize,
		m_numAllocatedParticles);
	if (m_simparams->simflags & ENABLE_INLET_OUTLET)
		gdata->problem->setboundconstants(m_physparams, gdata->worldOrigin, gdata->gridSize, gdata->cellSize);
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
void GPUWorker::checkPartValByIndex(const char* printID, const uint pindex)
{
	// here it is possible to set a condition on the simulation state, device number, e.g.:
	// if (gdata->iterations <= 900 || gdata->iterations >= 1000) return;
	// if (m_deviceIndex == 1) return;

	BufferList const& bufread = *m_dBuffers.getReadBufferList();
	BufferList &bufwrite = *m_dBuffers.getWriteBufferList();

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

// Analogous to checkPartValByIndex(), but locates the particle by ID through the BUFFER_VERTIDINDEX
// hash table. This is available only when SA_BOUNDARY is being used and only for VERTEX particles,
// unless updateVertIDToIndexDevice() is changed to update also non-vertex particles.
// WARNING: fixing updateVertIDToIndexDevice() for fluid particles is dangerous if there is an inlet,
// since the ID of generate parts easily overflows the number of allocated particles!
void GPUWorker::checkPartValById(const char* printID, const uint pid)
{
	// here it is possible to set a condition, e.g.:
	// if (gdata->iterations <= 900 || gdata->iterations >= 1000) return;

	uint pidx = 0;

	// retrieve part index, if BUFFER_VERTIDINDEX was set also for this particle
	CUDA_SAFE_CALL(cudaMemcpy(&pidx, m_dBuffers.getReadBufferList()->getData<BUFFER_VERTIDINDEX>() + pid, sizeof(uint), cudaMemcpyDeviceToHost));

	checkPartValByIndex(printID, pidx);
}


void GPUWorker::uploadEulerBodiesCentersOfGravity()
{
	integrationEngine->setrbcg(gdata->s_hRbCgGridPos, gdata->s_hRbCgPos, m_simparams->numbodies);
}


void GPUWorker::uploadForcesBodiesCentersOfGravity()
{
	forcesEngine->setrbcg(gdata->s_hRbCgGridPos, gdata->s_hRbCgPos, m_simparams->numbodies);
}


void GPUWorker::uploadBodiesTransRotMatrices()
{
	integrationEngine->setrbtrans(gdata->s_hRbTranslations, m_simparams->numbodies);
	integrationEngine->setrbsteprot(gdata->s_hRbRotationMatrices, m_simparams->numbodies);
}

void GPUWorker::uploadBodiesVelocities()
{
	integrationEngine->setrblinearvel(gdata->s_hRbLinearVelocities, m_simparams->numbodies);
	integrationEngine->setrbangularvel(gdata->s_hRbAngularVelocities, m_simparams->numbodies);
}
