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
#include "buildneibs.cuh"
#include "forces.cuh"
#include "euler.cuh"

#include "cudabuffer.h"

// round_up
#include "utils.h"

// UINT_MAX
#include "limits.h"

GPUWorker::GPUWorker(GlobalData* _gdata, unsigned int _deviceIndex) {
	gdata = _gdata;
	m_deviceIndex = _deviceIndex;
	m_cudaDeviceNumber = gdata->device[m_deviceIndex];

	m_globalDeviceIdx = GlobalData::GLOBAL_DEVICE_ID(gdata->mpi_rank, _deviceIndex);

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
	m_hTransferBuffer = NULL;
	m_hTransferBufferSize = 0;

	m_dCompactDeviceMap = NULL;
	m_hCompactDeviceMap = NULL;
	m_dSegmentStart = NULL;

	m_forcesKernelTotalNumBlocks = 0;

	m_dBuffers << new CUDABuffer<BUFFER_POS>();
	m_dBuffers << new CUDABuffer<BUFFER_VEL>();
	m_dBuffers << new CUDABuffer<BUFFER_INFO>();
	m_dBuffers << new CUDABuffer<BUFFER_FORCES>();

	m_dBuffers << new CUDABuffer<BUFFER_HASH>();
	m_dBuffers << new CUDABuffer<BUFFER_PARTINDEX>();
	m_dBuffers << new CUDABuffer<BUFFER_NEIBSLIST>(-1); // neib list is initialized to all bits set

	if (m_simparams->xsph)
		m_dBuffers << new CUDABuffer<BUFFER_XSPH>();

	if (m_simparams->visctype == SPSVISC)
		m_dBuffers << new CUDABuffer<BUFFER_TAU>();

	if (m_simparams->savenormals)
		m_dBuffers << new CUDABuffer<BUFFER_NORMALS>();
	if (m_simparams->vorticity)
		m_dBuffers << new CUDABuffer<BUFFER_VORTICITY>();

	if (m_simparams->dtadapt) {
		m_dBuffers << new CUDABuffer<BUFFER_CFL>();
		m_dBuffers << new CUDABuffer<BUFFER_CFL_TEMP>();
		if (m_simparams->visctype == KEPSVISC)
			m_dBuffers << new CUDABuffer<BUFFER_CFL_KEPS>();
	}

	if (m_simparams->boundarytype == SA_BOUNDARY) {
		m_dBuffers << new CUDABuffer<BUFFER_INVINDEX>();
		m_dBuffers << new CUDABuffer<BUFFER_GRADGAMMA>();
		m_dBuffers << new CUDABuffer<BUFFER_BOUNDELEMENTS>();
		m_dBuffers << new CUDABuffer<BUFFER_VERTICES>();
		m_dBuffers << new CUDABuffer<BUFFER_VERTPOS>();
	}

	if (m_simparams->visctype == KEPSVISC) {
		m_dBuffers << new CUDABuffer<BUFFER_TKE>();
		m_dBuffers << new CUDABuffer<BUFFER_EPSILON>();
		m_dBuffers << new CUDABuffer<BUFFER_TURBVISC>();
		m_dBuffers << new CUDABuffer<BUFFER_DKDE>();
	}

	if (m_simparams->calcPrivate)
		m_dBuffers << new CUDABuffer<BUFFER_PRIVATE>();
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

	BufferList::iterator buf = m_dBuffers.begin();
	const BufferList::iterator stop = m_dBuffers.end();
	while (buf != stop) {
		size_t contrib = buf->second->get_element_size()*buf->second->get_array_count();
		if (buf->first & BUFFER_NEIBSLIST)
			contrib *= m_simparams->maxneibsnum;
		// TODO compute a sensible estimate for the CFL contribution,
		// which is currently heavily overestimated
		else if (buf->first & BUFFERS_CFL)
			contrib /= 4;
		// particle index occupancy is double to account for memory allocated
		// by thrust::sort TODO refine
		else if (buf->first & BUFFER_PARTINDEX)
			contrib *= 2;

		tot += contrib;
#if _DEBUG_
		printf("with %s: %zu\n", buf->second->get_buffer_name(), tot);
#endif
		++buf;
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
	safetyMargin = totMemory/32; // 16MB on a 512MB GPU, 64MB on a 2GB GPU
	// compute how much memory is required for the cells array
	memPerCells = (size_t)gdata->nGridCells * computeMemoryPerCell();

	freeMemory -= 16; // segments
	freeMemory -= safetyMargin;

	if (memPerCells > freeMemory) {
		fprintf(stderr, "FATAL: not enough free device memory to allocate %s cells\n", gdata->addSeparators(gdata->nGridCells).c_str());
		exit(1);
	}

	freeMemory -= memPerCells;

	uint numAllocableParticles = (freeMemory / computeMemoryPerParticle());

	if (numAllocableParticles < gdata->totParticles)
		printf("NOTE: device %u can allocate %u particles, while the whole simulation might require %u\n",
			m_deviceIndex, numAllocableParticles, gdata->totParticles);

	// allocate at most the number of particles required for the whole simulation
	m_numAllocatedParticles = min( numAllocableParticles, gdata->totParticles );

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
		if (count > m_hTransferBufferSize)
			resizeTransferBuffer(count);
		// transfer Dsrc -> H -> Ddst
		CUDA_SAFE_CALL_NOSYNC( cudaMemcpyAsync(m_hTransferBuffer, src, count, cudaMemcpyDeviceToHost, m_asyncD2HCopiesStream) );
		CUDA_SAFE_CALL_NOSYNC( cudaMemcpyAsync(dst, m_hTransferBuffer, count, cudaMemcpyHostToDevice, m_asyncH2DCopiesStream) );
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

// Compute list of bursts. Currently computes both scopes, but only network scope is used
void GPUWorker::computeCellBursts()
{
	// Unlike importing from other devices in the same process, here we need one burst for each potential neighbor device
	// and for each direction. The following can be considered a list of pointers to open bursts in the m_bursts vector.
	// When a pointer is negative, there is no open bursts with the specified peer:direction pair.
	int burst_vector_index[MAX_DEVICES_PER_CLUSTER][2];

	uint network_bursts = 0;
	uint node_bursts = 0;

	// Auxiliary macros. Use with parentheses when possibile
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

		// We want to send the current cell to the neigbor processes only once, but multiple neib cells could
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
		const uchar curr_cell_gidx = gdata->s_hDeviceMap[lin_curr_cell];
		const uchar curr_cell_rank = gdata->RANK( curr_cell_gidx );

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

					// ensure we are inside the grid
					// TODO: fix for periodicity. ImportPeer* as well?
					if (coords_curr_cell.x + dx < 0 || coords_curr_cell.x + dx >= gdata->gridSize.x) continue;
					if (coords_curr_cell.y + dy < 0 || coords_curr_cell.y + dy >= gdata->gridSize.y) continue;
					if (coords_curr_cell.z + dz < 0 || coords_curr_cell.z + dz >= gdata->gridSize.z) continue;

					// NOTE: we could skip empty cells if all the nodes in the network knew the content of all the cells.
					// Instead, each process only knows the empty cells of its workers, so empty cells still break bursts
					// as if they weren't empty. One could check the performances with broadcasting all-to-all the empty
					// cells (possibly in bursts).

					// now compute the linearized hash of the neib cell and other properties
					const uint lin_neib_cell = gdata->calcGridHashHost(coords_curr_cell.x + dx, coords_curr_cell.y + dy, coords_curr_cell.z + dz);
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

					// devices fecth peers' memory with any intervention from the sender (aka: only RCV bursts in same node)
					if (transfer_scope == NODE_SCOPE && transfer_direction == SND)
						continue;

					// the "other" device is the device owning the cell (curr or neib) which is not mine
					const uint other_device_gidx = (curr_cell_gidx == m_globalDeviceIdx ? neib_cell_gidx : curr_cell_gidx);

					if (any_mine) {

						// if existing burst is non-empty, was not closed till now, so it is compatible: extend it
						if (! BURST_IS_EMPTY(other_device_gidx,transfer_direction)) {

							// cell index is higher than the last enqueued; it is edging as well; no other cell
							// interrrupted the burst until now. So cell is consecutive with previous in both
							// the sending the the receiving device
							m_bursts[ burst_vector_index[other_device_gidx][transfer_direction] ].cells.push_back(lin_curr_cell);

						} else {
							// if we are here, either the burst was empty or not compatabile. In both cases, create a new one
							CellList list;
							list.push_back(lin_curr_cell);

							CellBurst burst = {
								list,
								other_device_gidx,
								transfer_direction,
								transfer_scope,
								0, 0, 0
							};

							// store (ovewrite, if was non-empty) its forthcoming index
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

		// There was no neib cell (i.e. it was an internal cell for every device), so skip burst-breaking conditinals.
		// NOTE: comment the following line to allow bursts only along linearization (e.g. with Y-split and XYZ linearizazion,
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

	printf("D%u: data transfers compacted in %u bursts [%u node + %u network]\n",
		m_deviceIndex, (uint)m_bursts.size(), node_bursts, network_bursts);
	/*
	for (uint i = 0; i < m_bursts.size(); i++) {
		printf(" D %u Burst %u: %u cells, peer %u, dir %s, scope %u\n", m_deviceIndex,
			i, m_bursts[i].cells.size(), m_bursts[i].peer_gidx,
			(m_bursts[i].direction == SND ? "SND" : "RCV"), m_bursts[i].scope);
	}
	*/
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

	/* for (uint i = 0; i < m_bursts.size(); i++) {
		printf(" D %u Burst %u: %u cells, peer %u, dir %s, scope %u, range %u-%u, peer %u, (tot %u parts)\n", m_deviceIndex,
				i, m_bursts[i].cells.size(), m_bursts[i].peer_gidx,
				(m_bursts[i].direction == SND ? "SND" : "RCV"), m_bursts[i].scope,
				m_bursts[i].selfFirstParticle, m_bursts[i].selfFirstParticle + m_bursts[i].numParticles,
				m_bursts[i].peerFirstParticle, m_bursts[i].numParticles
			);
	} */
}

// Iterate on the list and send/receive bursts of particles across different nodes
void GPUWorker::transferBursts()
{
	bool dbl_buffer_specified = ( (gdata->commandFlags & DBLBUFFER_READ ) || (gdata->commandFlags & DBLBUFFER_WRITE) );
	uint dbl_buf_idx;

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

			// abstract from self / other
			const uint sender_gidx = (m_bursts[i].direction == SND ? m_globalDeviceIdx : m_bursts[i].peer_gidx);
			const uint recipient_gidx = (m_bursts[i].direction == SND ? m_bursts[i].peer_gidx : m_globalDeviceIdx);

			// iterate over all defined buffers and see which were requested
			// NOTE: std::map, from which BufferList is derived, is an _ordered_ container,
			// with the ordering set by the key, in our case the unsigned integer type flag_t,
			// so we have guarantee that the map will always be traversed in the same order
			// (unless stuff is inserted/deleted, which shouldn't happen at program runtime)
			BufferList::iterator bufset = m_dBuffers.begin();
			const BufferList::iterator stop = m_dBuffers.end();
			for ( ; bufset != stop ; ++bufset) {
				flag_t bufkey = bufset->first;
				if (!(gdata->commandFlags & bufkey))
					continue; // skip unwanted buffers

				AbstractBuffer *buf = bufset->second;

				// handling of double-buffered arrays
				// note that TAU is not considered here
				if (buf->get_array_count() == 2) {
					// for buffers with more than one array the caller should have specified which buffer
					// is to be imported. complain
					if (!dbl_buffer_specified) {
						std::stringstream err_msg;
						err_msg << "Import request for double-buffered " << buf->get_buffer_name()
						<< " array without a specification of which buffer to use.";
						throw runtime_error(err_msg.str());
					}

					if (gdata->commandFlags & DBLBUFFER_READ)
						dbl_buf_idx = gdata->currentRead[bufkey];
					else
						dbl_buf_idx = gdata->currentWrite[bufkey];
				} else {
					dbl_buf_idx = 0;
				}

				const unsigned int _size = m_bursts[i].numParticles * buf->get_element_size();

				// retrieve peer's indices, if intra-node
				const AbstractBuffer *peerbuf = NULL;
				uchar peerCudaDevNum = 0;
				if (m_bursts[i].scope == NODE_SCOPE) {
					uchar peerDevIdx = gdata->DEVICE(m_bursts[i].peer_gidx);
					peerbuf = gdata->GPUWORKERS[peerDevIdx]->getBuffer(bufkey);
					peerCudaDevNum = gdata->device[peerDevIdx];
				}

				// special treatment for big buffers (like TAU), since in that case we need to transfers all 3 arrays
				if (bufkey != BUFFER_BIG) {
					void *ptr = buf->get_offset_buffer(dbl_buf_idx, m_bursts[i].selfFirstParticle);
					if (m_bursts[i].scope == NODE_SCOPE) {
						// node scope: just read it
						const void *peerptr = peerbuf->get_offset_buffer(dbl_buf_idx, m_bursts[i].peerFirstParticle);
						peerAsyncTransfer(ptr, m_cudaDeviceNumber, peerptr, peerCudaDevNum, _size);
					} else {
						// network scope: SND/RCV
						if (m_bursts[i].direction == SND)
							gdata->networkManager->sendBuffer(sender_gidx, recipient_gidx, _size, ptr);
						else
							gdata->networkManager->receiveBuffer(sender_gidx, recipient_gidx, _size, ptr);
					}
				} else {
					// generic, so that it can work for other buffers like TAU, if they are ever
					// introduced; just fix the conditional
					for (uint ai = 0; ai < buf->get_array_count(); ++ai) {
						void *ptr = buf->get_offset_buffer(ai, m_bursts[i].selfFirstParticle);
						if (m_bursts[i].scope == NODE_SCOPE) {
							// node scope: just read it
							const void *peerptr = peerbuf->get_offset_buffer(ai, m_bursts[i].peerFirstParticle);
							peerAsyncTransfer(ptr, m_cudaDeviceNumber, peerptr, peerCudaDevNum, _size);
						} else {
							// network scope: SND/RCV
							if (m_bursts[i].direction == SND)
								gdata->networkManager->sendBuffer(sender_gidx, recipient_gidx, _size, ptr);
							else
								gdata->networkManager->receiveBuffer(sender_gidx, recipient_gidx, _size, ptr);
						}
					}
				} // buf is BUFFER_BIG

			} // for each buffer type

		} // iterate on bursts

	} // iterate on scopes
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
	// Waiting for the first stripe to complete is mandatory for correct edge-cells exchange
	// (any transfer scope).
	if (gdata->striping && MULTI_DEVICE)
		cudaEventSynchronize(m_halfForcesEvent);

	if (gdata->nextCommand == APPEND_EXTERNAL)
		transferBurstsSizes();
	if ( (gdata->nextCommand == APPEND_EXTERNAL) || (gdata->nextCommand == UPDATE_EXTERNAL) )
		transferBursts();

	// cudaMemcpyPeerAsync() is asynchronous with the host. If striping is disabled, we want to synchronize
	// for the completion of the transfers. Otherwise, FORCES_COMPLETE will synchronize everything
	if (!gdata->striping && MULTI_GPU)
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
			resizeTransferBuffer(1024 * 1024);
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

	uint fmaxElements = getFmaxElements(m_numAllocatedParticles);
	uint tempCflEls = getFmaxTempElements(fmaxElements);
	BufferList::iterator iter = m_dBuffers.begin();
	while (iter != m_dBuffers.end()) {
		// number of elements to allocate
		// most have m_numAllocatedParticles. Exceptions follow
		size_t nels = m_numAllocatedParticles;

		// iter->first: the key
		// iter->second: the Buffer
		if (iter->first & BUFFER_NEIBSLIST)
			nels *= m_simparams->maxneibsnum; // number of particles times max neibs num
		else if (iter->first & BUFFER_CFL_TEMP)
			nels = tempCflEls;
		else if (iter->first & BUFFERS_CFL) // other CFL buffers
			nels = fmaxElements;

		allocated += iter->second->alloc(nels);
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

	if (m_simparams->numODEbodies) {
		m_numBodiesParticles = gdata->problem->get_ODE_bodies_numparts();
		printf("number of rigid bodies particles = %d\n", m_numBodiesParticles);

		int objParticlesFloat4Size = m_numBodiesParticles*sizeof(float4);
		int objParticlesUintSize = m_numBodiesParticles*sizeof(uint);

		CUDA_SAFE_CALL(cudaMalloc(&m_dRbTorques, objParticlesFloat4Size));
		CUDA_SAFE_CALL(cudaMalloc(&m_dRbForces, objParticlesFloat4Size));
		CUDA_SAFE_CALL(cudaMalloc(&m_dRbNum, objParticlesUintSize));

		allocated += 2 * objParticlesFloat4Size + objParticlesUintSize;

		// DEBUG
		// m_hRbForces = new float4[m_numBodiesParticles];
		// m_hRbTorques = new float4[m_numBodiesParticles];

		uint rbfirstindex[MAXBODIES];
		uint* rbnum = new uint[m_numBodiesParticles];

		rbfirstindex[0] = 0;
		for (uint i = 1; i < m_simparams->numODEbodies; i++) {
			rbfirstindex[i] = rbfirstindex[i - 1] + gdata->problem->get_ODE_body_numparts(i - 1);
		}
		setforcesrbstart(rbfirstindex, m_simparams->numODEbodies);

		int offset = 0;
		for (uint i = 0; i < m_simparams->numODEbodies; i++) {
			gdata->s_hRbLastIndex[i] = gdata->problem->get_ODE_body_numparts(i) - 1 + offset;

			for (int j = 0; j < gdata->problem->get_ODE_body_numparts(i); j++) {
				rbnum[offset + j] = i;
			}
			offset += gdata->problem->get_ODE_body_numparts(i);
		}
		size_t  size = m_numBodiesParticles*sizeof(uint);
		CUDA_SAFE_CALL(cudaMemcpy((void *) m_dRbNum, (void*) rbnum, size, cudaMemcpyHostToDevice));

		delete[] rbnum;
	}

	// TODO: call setDemTexture(), which allocates and reads the DEM
	//if (m_simparams->usedem) {
	//	//printf("Using DEM\n");
	//	//printf("cols = %d\trows =% d\n", m_problem->m_ncols, m_problem->m_nrows);
	//	setDemTexture(m_problem->m_dem, m_problem->m_ncols, m_problem->m_nrows);
	//}

	m_deviceMemory += allocated;
	return allocated;
}

void GPUWorker::deallocateHostBuffers() {
	if (MULTI_DEVICE)
		delete [] m_hCompactDeviceMap;

	if (m_hTransferBuffer)
		cudaFreeHost(m_hTransferBuffer);

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

	if (m_simparams->numODEbodies) {
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
	cudaStreamCreate(&m_asyncD2HCopiesStream);
	cudaStreamCreate(&m_asyncH2DCopiesStream);
	cudaStreamCreate(&m_asyncPeerCopiesStream);
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

	// iterate over each array in the _host_ buffer list, and upload data
	// if it is not in the skip list
	BufferList::iterator onhost = gdata->s_hBuffers.begin();
	const BufferList::iterator stop = gdata->s_hBuffers.end();
	for ( ; onhost != stop ; ++onhost) {
		flag_t buf_to_up = onhost->first;
		if (buf_to_up & skip_bufs)
			continue;

		AbstractBuffer *buf = m_dBuffers[buf_to_up];
		size_t _size = howManyParticles * buf->get_element_size();

		printf("Thread %d uploading %d %s items (%s) on device %d from position %d\n",
				m_deviceIndex, howManyParticles, buf->get_buffer_name(),
				gdata->memString(_size).c_str(), m_cudaDeviceNumber, firstInnerParticle);

		void *dstptr = buf->get_buffer(gdata->currentRead[buf_to_up]);
		const void *srcptr = onhost->second->get_offset_buffer(0, firstInnerParticle);
		CUDA_SAFE_CALL(cudaMemcpy(dstptr, srcptr, _size, cudaMemcpyHostToDevice));
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

	// iterate over each array in the _host_ buffer list, and download data
	// if it was requested
	BufferList::iterator onhost = gdata->s_hBuffers.begin();
	const BufferList::iterator stop = gdata->s_hBuffers.end();
	for ( ; onhost != stop ; ++onhost) {
		flag_t buf_to_get = onhost->first;
		if (!(buf_to_get & flags))
			continue;

		const AbstractBuffer *buf = m_dBuffers[buf_to_get];
		size_t _size = howManyParticles * buf->get_element_size();

		uint which_buffer = 0;
		if (flags & DBLBUFFER_READ) which_buffer = gdata->currentRead[buf_to_get];
		if (flags & DBLBUFFER_WRITE) which_buffer = gdata->currentWrite[buf_to_get];

		const void *srcptr = buf->get_buffer(which_buffer);
		void *dstptr = onhost->second->get_offset_buffer(0, firstInnerParticle);
		CUDA_SAFE_CALL(cudaMemcpy(dstptr, srcptr, _size, cudaMemcpyDeviceToHost));
	}
}

// Sets all cells as empty in device memory. Used before reorder
void GPUWorker::setDeviceCellsAsEmpty()
{
	CUDA_SAFE_CALL(cudaMemset(m_dCellStart, UINT_MAX, gdata->nGridCells  * sizeof(uint)));
}

// if m_hTransferBuffer is not big enough, reallocate it. Round up to 1Mb
void GPUWorker::resizeTransferBuffer(size_t required_size)
{
	// is it big enough already?
	if (required_size < m_hTransferBufferSize) return;

	// will round up to...
	size_t ROUND_TO = 1024*1024;

	// store previous size, compute new
	size_t prev_size = m_hTransferBufferSize;
	m_hTransferBufferSize = ((required_size / ROUND_TO) + 1 ) * ROUND_TO;

	// dealloc first
	if (m_hTransferBufferSize) {
		CUDA_SAFE_CALL(cudaFreeHost(m_hTransferBuffer));
		m_hostMemory -= prev_size;
	}

	// (re)allocate
	CUDA_SAFE_CALL(cudaMallocHost(&m_hTransferBuffer, m_hTransferBufferSize));
	m_hostMemory += m_hTransferBufferSize;
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

// upload mbData for moving boundaries (possibily called many times)
void GPUWorker::uploadMBData()
{
	// check if MB are active and if gdata->s_mbData is not NULL
	if (m_simparams->mbcallback && gdata->s_mbData)
		setmbdata(gdata->s_mbData, gdata->mbDataSize);
}

// upload gravity (possibily called many times)
void GPUWorker::uploadGravity()
{
	// check if variable gravity is enabled
	if (m_simparams->gcallback)
		setgravity(gdata->s_varGravity);
}

// upload planes (called once until planes arae constant)
void GPUWorker::uploadPlanes()
{
	// check if planes > 0 (already checked before calling?)
	if (gdata->numPlanes > 0)
		setplaneconstants(gdata->numPlanes, gdata->s_hPlanesDiv, gdata->s_hPlanes);
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
							// warp periodic boundaries
							if (m_simparams->periodicbound) {
								// periodicity along X
								if (m_physparams->dispvect.x) {
									// WARNING: checking if c* is negative MUST be done before checking if it's greater than
									// the grid, otherwise it will be cast to uint and "-1" will be "greater" than the gridSize
									if (cx < 0) {
										cx = gdata->gridSize.x - 1;
									} else
									if (cx >= gdata->gridSize.x) {
										cx = 0;
									}
								} // if dispvect.x
								// periodicity along Y
								if (m_physparams->dispvect.y) {
									if (cy < 0) {
										cy = gdata->gridSize.y - 1;
									} else
									if (cy >= gdata->gridSize.y) {
										cy = 0;
									}
								} // if dispvect.y
								// periodicity along Z
								if (m_physparams->dispvect.z) {
									if (cz < 0) {
										cz = gdata->gridSize.z - 1;
									} else
									if (cz >= gdata->gridSize.z) {
										cz = 0;
									}
								} // if dispvect.z
							}
							// if not periodic, or if still out-of-bounds after periodicity warp, skip it
							if (cx < 0 || cx >= gdata->gridSize.x ||
								cy < 0 || cy >= gdata->gridSize.y ||
								cz < 0 || cz >= gdata->gridSize.z) continue;

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

unsigned int GPUWorker::getDeviceIndex()
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

const AbstractBuffer* GPUWorker::getBuffer(flag_t key) const
{
	return m_dBuffers[key];
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

	// upload centers of gravity of the bodies
	instance->uploadBodiesCentersOfGravity();

	// allocate CPU and GPU arrays
	instance->allocateHostBuffers();
	instance->allocateDeviceBuffers();
	instance->printAllocatedMemory();

	// create and upload the compact device map (2 bits per cell)
	if (MULTI_DEVICE) {
		instance->createCompactDeviceMap();
		instance->computeCellBursts();
		instance->uploadCompactDeviceMap();
	}

	// TODO: here set_reduction_params() will be called (to be implemented in this class). These parameters can be device-specific.

	// TODO: here setDemTexture() will be called. It is device-wide, but reading the DEM file is process wide and will be in GPUSPH class

	// init streams for async memcpys (only useful for multigpu?)
	instance->createEventsAndStreams();

	gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***

	// here GPUSPH::initialize is over and GPUSPH::runSimulation() is called

	gdata->threadSynchronizer->barrier(); // begins UPLOAD ***

	instance->uploadSubdomain();

	gdata->threadSynchronizer->barrier();  // end of UPLOAD, begins SIMULATION ***

	const bool dbg_step_printf = false;

	// TODO
	// Here is a copy-paste from the CPU thread worker of branch cpusph, as a canvas
	while (gdata->keep_going) {
		switch (gdata->nextCommand) {
			// logging here?
			case IDLE:
				break;
			case CALCHASH:
				if (dbg_step_printf) printf(" T %d issuing HASH\n", deviceIndex);
				instance->kernel_calcHash();
				break;
			case SORT:
				if (dbg_step_printf) printf(" T %d issuing SORT\n", deviceIndex);
				instance->kernel_sort();
				break;
			case INVINDEX:
				if (dbg_step_printf) printf(" T %d issuing INVINDEX\n", deviceIndex);
				instance->kernel_inverseParticleIndex();
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
			case APPEND_EXTERNAL:
				if (dbg_step_printf) printf(" T %d issuing APPEND_EXTERNAL\n", deviceIndex);
				instance->importExternalCells();
				break;
			case UPDATE_EXTERNAL:
				if (dbg_step_printf) printf(" T %d issuing UPDATE_EXTERNAL\n", deviceIndex);
				instance->importExternalCells();
				break;
			case MLS:
				if (dbg_step_printf) printf(" T %d issuing MLS\n", deviceIndex);
				instance->kernel_mls();
				break;
			case SHEPARD:
				if (dbg_step_printf) printf(" T %d issuing SHEPARD\n", deviceIndex);
				instance->kernel_shepard();
				break;
			case VORTICITY:
				if (dbg_step_printf) printf(" T %d issuing VORTICITY\n", deviceIndex);
				instance->kernel_vorticity();
				break;
			case SURFACE_PARTICLES:
				if (dbg_step_printf) printf(" T %d issuing SURFACE_PARTICLES\n", deviceIndex);
				instance->kernel_surfaceParticles();
				break;
			case SA_CALC_BOUND_CONDITIONS:
				if (dbg_step_printf) printf(" T %d issuing SA_CALC_BOUND_CONDITIONS\n", deviceIndex);
				instance->kernel_dynamicBoundaryConditions();
				break;
			case SA_UPDATE_BOUND_VALUES:
				if (dbg_step_printf) printf(" T %d issuing SA_UPDATE_BOUND_VALUES\n", deviceIndex);
				instance->kernel_updateValuesAtBoundaryElements();
				break;
			case SPS:
				if (dbg_step_printf) printf(" T %d issuing SPS\n", deviceIndex);
				instance->kernel_sps();
				break;
			case REDUCE_BODIES_FORCES:
				if (dbg_step_printf) printf(" T %d issuing REDUCE_BODIES_FORCES\n", deviceIndex);
				instance->kernel_reduceRBForces();
				break;
			case UPLOAD_MBDATA:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_MBDATA\n", deviceIndex);
				instance->uploadMBData();
				break;
			case UPLOAD_GRAVITY:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_GRAVITY\n", deviceIndex);
				instance->uploadGravity();
				break;
			case UPLOAD_PLANES:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_PLANES\n", deviceIndex);
				instance->uploadPlanes();
				break;
			case UPLOAD_OBJECTS_CG:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_OBJECTS_CG\n", deviceIndex);
				instance->uploadBodiesCentersOfGravity();
				break;
			case UPLOAD_OBJECTS_MATRICES:
				if (dbg_step_printf) printf(" T %d issuing UPLOAD_OBJECTS_CG\n", deviceIndex);
				instance->uploadBodiesTransRotMatrices();
				break;
			case CALC_PRIVATE:
				if (dbg_step_printf) printf(" T %d issuing CALC_PRIVATE\n", deviceIndex);
				instance->kernel_calcPrivate();
				break;
			case COMPUTE_TESTPOINTS:
				if (dbg_step_printf) printf(" T %d issuing COMPUTE_TESTPOINTS\n", deviceIndex);
				instance->kernel_testpoints();
				break;
			case QUIT:
				if (dbg_step_printf) printf(" T %d issuing QUIT\n", deviceIndex);
				// actually, setting keep_going to false and unlocking the barrier should be enough to quit the cycle
				break;
		}
		if (gdata->keep_going) {
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

	// calcHashDevice() should use CPU-computed hashes at iteration 0, or some particles
	// might be lost (if a GPU computes a different hash and does not recognize the particles
	// as "own"). However, the high bits should be set, or edge cells won't be compacted at
	// the end and bursts will be sparse.
	// This is required only in MULTI_DEVICE simulations but it holds also on single-device
	// ones to keep numerical consistency.

	if (gdata->iterations == 0)
		fixHash(	m_dBuffers.getData<BUFFER_HASH>(),
					m_dBuffers.getData<BUFFER_PARTINDEX>(),
					m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
					m_dCompactDeviceMap,
					m_numParticles);
	else
		calcHash(	m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
					m_dBuffers.getData<BUFFER_HASH>(),
					m_dBuffers.getData<BUFFER_PARTINDEX>(),
					m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
					m_dCompactDeviceMap,
					m_numParticles,
					m_simparams->periodicbound);
}

void GPUWorker::kernel_sort()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_numInternalParticles : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	sort(	m_dBuffers.getData<BUFFER_HASH>(),
			m_dBuffers.getData<BUFFER_PARTINDEX>(),
			numPartsToElaborate);
}

void GPUWorker::kernel_inverseParticleIndex()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_numInternalParticles : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	inverseParticleIndex (	m_dBuffers.getData<BUFFER_PARTINDEX>(),
							m_dBuffers.getData<BUFFER_INVINDEX>(),
							numPartsToElaborate);
}

void GPUWorker::kernel_reorderDataAndFindCellStart()
{
	// reset also if the device is empty (or we will download uninitialized values)
	setDeviceCellsAsEmpty();

	// is the device empty? (unlikely but possible before LB kicks in)
	if (m_numParticles == 0) return;

	// TODO this kernel needs a thorough reworking to only pass the needed buffers
	reorderDataAndFindCellStart(m_dCellStart,	  // output: cell start index
							m_dCellEnd,		// output: cell end index
							m_dSegmentStart,
							// output: sorted arrays
							m_dBuffers.getData<BUFFER_POS>(gdata->currentWrite[BUFFER_POS]),
							m_dBuffers.getData<BUFFER_VEL>(gdata->currentWrite[BUFFER_VEL]),
							m_dBuffers.getData<BUFFER_INFO>(gdata->currentWrite[BUFFER_INFO]),
							m_dBuffers.getData<BUFFER_BOUNDELEMENTS>(gdata->currentWrite[BUFFER_BOUNDELEMENTS]),
							m_dBuffers.getData<BUFFER_GRADGAMMA>(gdata->currentWrite[BUFFER_GRADGAMMA]),
							m_dBuffers.getData<BUFFER_VERTICES>(gdata->currentWrite[BUFFER_VERTICES]),
							m_dBuffers.getData<BUFFER_TKE>(gdata->currentWrite[BUFFER_TKE]),
							m_dBuffers.getData<BUFFER_EPSILON>(gdata->currentWrite[BUFFER_EPSILON]),
							m_dBuffers.getData<BUFFER_TURBVISC>(gdata->currentWrite[BUFFER_TURBVISC]),

							// hash
							m_dBuffers.getData<BUFFER_HASH>(),
							// sorted particle indices
							m_dBuffers.getData<BUFFER_PARTINDEX>(),

							// input: arrays to sort
							m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
							m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),
							m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
							m_dBuffers.getData<BUFFER_BOUNDELEMENTS>(gdata->currentRead[BUFFER_BOUNDELEMENTS]),
							m_dBuffers.getData<BUFFER_GRADGAMMA>(gdata->currentRead[BUFFER_GRADGAMMA]),
							m_dBuffers.getData<BUFFER_VERTICES>(gdata->currentRead[BUFFER_VERTICES]),
							m_dBuffers.getData<BUFFER_TKE>(gdata->currentRead[BUFFER_TKE]),
							m_dBuffers.getData<BUFFER_EPSILON>(gdata->currentRead[BUFFER_EPSILON]),
							m_dBuffers.getData<BUFFER_TURBVISC>(gdata->currentRead[BUFFER_TURBVISC]),

							m_numParticles,
							m_nGridCells,
							m_dBuffers.getData<BUFFER_INVINDEX>());
}

void GPUWorker::kernel_buildNeibsList()
{
	resetneibsinfo();

	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	// this is the square of delta p / 2
	// it is used to add segments into the neighbour list even if they are outside the kernel support
	const float sqdpo2 = powf(m_simparams->slength/m_simparams->sfactor/2.0f,2.0f);

	buildNeibsList(	m_dBuffers.getData<BUFFER_NEIBSLIST>(),
					m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
					m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
					m_dBuffers.getData<BUFFER_VERTICES>(gdata->currentRead[BUFFER_VERTICES]),
					m_dBuffers.getData<BUFFER_BOUNDELEMENTS>(gdata->currentRead[BUFFER_BOUNDELEMENTS]),
					m_dBuffers.getRawPtr<BUFFER_VERTPOS>(),
					m_dBuffers.getData<BUFFER_HASH>(),
					m_dCellStart,
					m_dCellEnd,
					m_numParticles,
					numPartsToElaborate,
					m_nGridCells,
					m_simparams->nlSqInfluenceRadius,
					sqdpo2,
					m_simparams->periodicbound);

	// download the peak number of neighbors and the estimated number of interactions
	getneibsinfo( gdata->timingInfo[m_deviceIndex] );
}

// returns numBlocks as computed by forces()
uint GPUWorker::enqueueForcesOnRange(uint fromParticle, uint toParticle, uint cflOffset)
{
	return forces(
			m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),   // pos(n)
			m_dBuffers.getRawPtr<BUFFER_VERTPOS>(),
			m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),   // vel(n)
			m_dBuffers.getData<BUFFER_FORCES>(),					// f(n
			m_dBuffers.getData<BUFFER_GRADGAMMA>(gdata->currentRead[BUFFER_GRADGAMMA]),
			m_dBuffers.getData<BUFFER_GRADGAMMA>(gdata->currentWrite[BUFFER_GRADGAMMA]),
			m_dBuffers.getData<BUFFER_BOUNDELEMENTS>(gdata->currentRead[BUFFER_BOUNDELEMENTS]),
			m_dRbForces,
			m_dRbTorques,
			m_dBuffers.getData<BUFFER_XSPH>(),
			m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
			m_dBuffers.getData<BUFFER_HASH>(),
			m_dCellStart,
			m_dBuffers.getData<BUFFER_NEIBSLIST>(),
			m_numParticles,
			fromParticle,
			toParticle,
			gdata->problem->m_deltap,
			m_simparams->slength,
			gdata->dt, // m_dt,
			m_simparams->dtadapt,
			m_simparams->dtadaptfactor,
			m_simparams->xsph,
			m_simparams->kerneltype,
			m_simparams->influenceRadius,
			m_simparams->epsilon,
			m_simparams->movingBoundaries,
			m_simparams->visctype,
			m_physparams->visccoeff,
			m_dBuffers.getData<BUFFER_TURBVISC>(gdata->currentRead[BUFFER_TURBVISC]),	// nu_t(n)
			m_dBuffers.getData<BUFFER_TKE>(gdata->currentRead[BUFFER_TKE]),	// k(n)
			m_dBuffers.getData<BUFFER_EPSILON>(gdata->currentRead[BUFFER_EPSILON]),	// e(n)
			m_dBuffers.getData<BUFFER_DKDE>(),
			m_dBuffers.getData<BUFFER_CFL>(),
			m_dBuffers.getData<BUFFER_CFL_KEPS>(),
			m_dBuffers.getData<BUFFER_CFL_TEMP>(),
			cflOffset,
			m_simparams->sph_formulation,
			m_simparams->boundarytype,
			m_simparams->usedem);
}

// Bind the textures needed by forces kernel
void GPUWorker::bind_textures_forces()
{
	forces_bind_textures(
		m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),   // pos(n)
		m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),   // vel(n)
		m_dBuffers.getData<BUFFER_GRADGAMMA>(gdata->currentRead[BUFFER_GRADGAMMA]),
		m_dBuffers.getData<BUFFER_BOUNDELEMENTS>(gdata->currentRead[BUFFER_BOUNDELEMENTS]),
		m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
		m_numParticles,
		m_simparams->visctype,
		m_dBuffers.getData<BUFFER_TKE>(gdata->currentRead[BUFFER_TKE]),	// k(n)
		m_dBuffers.getData<BUFFER_EPSILON>(gdata->currentRead[BUFFER_EPSILON]),	// e(n)
		m_simparams->boundarytype
	);
}

// Unbind the textures needed by forces kernel
void GPUWorker::unbind_textures_forces()
{
	forces_unbind_textures(m_simparams->visctype, m_simparams->boundarytype);
}

// Dt reduction after forces kernel
float GPUWorker::forces_dt_reduce()
{
	return forces_dtreduce(
		m_simparams->slength,
		m_simparams->dtadaptfactor,
		m_simparams->visctype,
		m_physparams->visccoeff,
		m_dBuffers.getData<BUFFER_CFL>(),
		m_dBuffers.getData<BUFFER_CFL_KEPS>(),
		m_dBuffers.getData<BUFFER_CFL_TEMP>(),
		m_forcesKernelTotalNumBlocks);
}

void GPUWorker::kernel_forces_async_enqueue()
{
	if (!gdata->only_internal)
		printf("WARNING: forces kernel called with only_internal == true, ignoring flag!\n");

	uint numPartsToElaborate = m_particleRangeEnd;

	m_forcesKernelTotalNumBlocks = 0;

	// if we have objects potentially shared across different devices, must reset their forces
	// and torques to avoid spurious contributions
	if (m_simparams->numODEbodies > 0 && MULTI_DEVICE) {
		uint bodiesPartsSize = m_numBodiesParticles * sizeof(float4);
		CUDA_SAFE_CALL(cudaMemset(m_dRbForces, 0.0F, bodiesPartsSize));
		CUDA_SAFE_CALL(cudaMemset(m_dRbTorques, 0.0F, bodiesPartsSize));
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
	nonEdgingStripeSize = (nonEdgingStripeSize / BLOCK_SIZE_FORCES) * BLOCK_SIZE_FORCES;
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

		// UPDATE_EXTERNAL or APPEND_EXTERNAL will wait for the first stripe to be complete; FORCES_COMPLETE will
		// completely synchronize the device.
		// We could synchronize here instead but this would bring some overhead for those devices which are faster
		// in the computation of the first stripe; waiting for the event after doing all the bursts stuff makes more
		// CPU/GPU overlap.
	}
}

void GPUWorker::kernel_forces_async_complete()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// FLOAT_MAX is returned if kernels are not run (e.g. numPartsToElaborate == 0)
	float returned_dt = FLT_MAX;

	bool firstStep = (gdata->commandFlags == INTEGRATOR_STEP_1);

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

	bool firstStep = (gdata->commandFlags == INTEGRATOR_STEP_1);

	// if we have objects potentially shared across different devices, must reset their forces
	// and torques to avoid spurious contributions
	if (m_simparams->numODEbodies > 0 && MULTI_DEVICE) {
		uint bodiesPartsSize = m_numBodiesParticles * sizeof(float4);
		CUDA_SAFE_CALL(cudaMemset(m_dRbForces, 0.0F, bodiesPartsSize));
		CUDA_SAFE_CALL(cudaMemset(m_dRbTorques, 0.0F, bodiesPartsSize));
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

	bool firstStep = (gdata->commandFlags == INTEGRATOR_STEP_1);

		euler(
			// previous pos, vel, k, e, info
			m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
			m_dBuffers.getData<BUFFER_HASH>(),
			m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),
			m_dBuffers.getData<BUFFER_TKE>(gdata->currentRead[BUFFER_TKE]),
			m_dBuffers.getData<BUFFER_EPSILON>(gdata->currentRead[BUFFER_EPSILON]),
			m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
			// f(n+1/2)
			m_dBuffers.getData<BUFFER_FORCES>(),
			// dkde(n)
			m_dBuffers.getData<BUFFER_DKDE>(),
			m_dBuffers.getData<BUFFER_XSPH>(),
			// integrated pos vel, k, e
			m_dBuffers.getData<BUFFER_POS>(gdata->currentWrite[BUFFER_POS]),
			m_dBuffers.getData<BUFFER_VEL>(gdata->currentWrite[BUFFER_VEL]),
			m_dBuffers.getData<BUFFER_TKE>(gdata->currentWrite[BUFFER_TKE]),
			m_dBuffers.getData<BUFFER_EPSILON>(gdata->currentWrite[BUFFER_EPSILON]),
			m_numParticles,
			numPartsToElaborate,
			gdata->dt, // m_dt,
			gdata->dt/2.0f, // m_dt/2.0,
			firstStep ? 1 : 2,
			gdata->t + (firstStep ? gdata->dt / 2.0f : gdata->dt),
			m_simparams->xsph);
}

void GPUWorker::kernel_mls()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	mls(m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
		m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),
		m_dBuffers.getData<BUFFER_VEL>(gdata->currentWrite[BUFFER_VEL]),
		m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
		m_dBuffers.getData<BUFFER_HASH>(),
		m_dCellStart,
		m_dBuffers.getData<BUFFER_NEIBSLIST>(),
		m_numParticles,
		numPartsToElaborate,
		m_simparams->slength,
		m_simparams->kerneltype,
		m_simparams->influenceRadius);
}

void GPUWorker::kernel_shepard()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	shepard(m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
			m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),
			m_dBuffers.getData<BUFFER_VEL>(gdata->currentWrite[BUFFER_VEL]),
			m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
			m_dBuffers.getData<BUFFER_HASH>(),
			m_dCellStart,
			m_dBuffers.getData<BUFFER_NEIBSLIST>(),
			m_numParticles,
			numPartsToElaborate,
			m_simparams->slength,
			m_simparams->kerneltype,
			m_simparams->influenceRadius);
}

void GPUWorker::kernel_vorticity()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	// Calling vorticity computation kernel
	vorticity(	m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
				m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),
				m_dBuffers.getData<BUFFER_VORTICITY>(),
				m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
				m_dBuffers.getData<BUFFER_HASH>(),
				m_dCellStart,
				m_dBuffers.getData<BUFFER_NEIBSLIST>(),
				m_numParticles,
				numPartsToElaborate,
				m_simparams->slength,
				m_simparams->kerneltype,
				m_simparams->influenceRadius);
}

void GPUWorker::kernel_surfaceParticles()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	surfaceparticle(m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
					m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),
					m_dBuffers.getData<BUFFER_NORMALS>(),
					m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
					m_dBuffers.getData<BUFFER_INFO>(gdata->currentWrite[BUFFER_INFO]),
					m_dBuffers.getData<BUFFER_HASH>(),
					m_dCellStart,
					m_dBuffers.getData<BUFFER_NEIBSLIST>(),
					m_numParticles,
					numPartsToElaborate,
					m_simparams->slength,
					m_simparams->kerneltype,
					m_simparams->influenceRadius,
					m_simparams->savenormals);
}

void GPUWorker::kernel_sps()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	sps(m_dBuffers.getRawPtr<BUFFER_TAU>(),
		m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
		m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),
		m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
		m_dBuffers.getData<BUFFER_HASH>(),
		m_dCellStart,
		m_dBuffers.getData<BUFFER_NEIBSLIST>(),
		m_numParticles,
		numPartsToElaborate,
		m_simparams->slength,
		m_simparams->kerneltype,
		m_simparams->influenceRadius);
}

void GPUWorker::kernel_reduceRBForces()
{
	// is the device empty? (unlikely but possible before LB kicks in)
	if (m_numParticles == 0) {

		// make sure this device does not add any obsolete contribute to forces acting on objects
		for (uint ob = 0; ob < m_simparams->numODEbodies; ob++) {
			gdata->s_hRbTotalForce[m_deviceIndex][ob] = make_float3(0.0F);
			gdata->s_hRbTotalTorque[m_deviceIndex][ob] = make_float3(0.0F);
		}

		return;
	}

	reduceRbForces(m_dRbForces, m_dRbTorques, m_dRbNum, gdata->s_hRbLastIndex, gdata->s_hRbTotalForce[m_deviceIndex],
					gdata->s_hRbTotalTorque[m_deviceIndex], m_simparams->numODEbodies, m_numBodiesParticles);
}

void GPUWorker::kernel_updateValuesAtBoundaryElements()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	// vel, tke, eps are read from current*Read, except
	// on the second step, whe they are read from current*Write
	bool initStep = (gdata->commandFlags & INITIALIZATION_STEP);

	updateBoundValues(
				m_dBuffers.getData<BUFFER_VEL>(gdata->currentWrite[BUFFER_VEL]),
				m_dBuffers.getData<BUFFER_TKE>(gdata->currentWrite[BUFFER_TKE]),
				m_dBuffers.getData<BUFFER_EPSILON>(gdata->currentWrite[BUFFER_EPSILON]),
				m_dBuffers.getData<BUFFER_VERTICES>(gdata->currentRead[BUFFER_VERTICES]),
				m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
				m_numParticles,
				numPartsToElaborate,
				initStep);
}

void GPUWorker::kernel_dynamicBoundaryConditions()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	// pos, vel, tke, eps are read from current*Read, except
	// on the second step, whe they are read from current*Write
	bool initStep = (gdata->commandFlags & INITIALIZATION_STEP);

	dynamicBoundConditions(
				m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
				m_dBuffers.getData<BUFFER_VEL>(gdata->currentWrite[BUFFER_VEL]),
				m_dBuffers.getData<BUFFER_TKE>(gdata->currentWrite[BUFFER_TKE]),
				m_dBuffers.getData<BUFFER_EPSILON>(gdata->currentWrite[BUFFER_EPSILON]),
				m_dBuffers.getData<BUFFER_GRADGAMMA>(gdata->currentWrite[BUFFER_GRADGAMMA]),
				m_dBuffers.getData<BUFFER_BOUNDELEMENTS>(gdata->currentRead[BUFFER_BOUNDELEMENTS]),
				m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
				m_dBuffers.getData<BUFFER_HASH>(),
				m_dCellStart,
				m_dBuffers.getData<BUFFER_NEIBSLIST>(),
				m_numParticles,
				numPartsToElaborate,
				gdata->problem->m_deltap,
				m_simparams->slength,
				m_simparams->kerneltype,
				m_simparams->influenceRadius,
				initStep);
}

void GPUWorker::kernel_calcPrivate()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	calcPrivate(m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
				m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),
				m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
				m_dBuffers.getData<BUFFER_PRIVATE>(),
				m_dBuffers.getData<BUFFER_HASH>(),
				m_dCellStart,
				m_dBuffers.getData<BUFFER_NEIBSLIST>(),
				m_simparams->slength,
				m_simparams->influenceRadius,
				m_numParticles,
				numPartsToElaborate);
}

void GPUWorker::kernel_testpoints()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	testpoints(m_dBuffers.getData<BUFFER_POS>(gdata->currentRead[BUFFER_POS]),
				m_dBuffers.getData<BUFFER_VEL>(gdata->currentRead[BUFFER_VEL]),
				m_dBuffers.getData<BUFFER_INFO>(gdata->currentRead[BUFFER_INFO]),
				m_dBuffers.getData<BUFFER_HASH>(),
				m_dCellStart,
				m_dBuffers.getData<BUFFER_NEIBSLIST>(),
				m_numParticles,
				numPartsToElaborate,
				m_simparams->slength,
				m_simparams->kerneltype,
				m_simparams->influenceRadius);
}

void GPUWorker::uploadConstants()
{
	// NOTE: visccoeff must be set before uploading the constants. This is done in GPUSPH main cycle

	// Setting kernels and kernels derivative factors
	setforcesconstants(m_simparams, m_physparams, gdata->worldOrigin, gdata->gridSize, gdata->cellSize,
		m_numAllocatedParticles);
	seteulerconstants(m_physparams, gdata->worldOrigin, gdata->gridSize, gdata->cellSize);
	setneibsconstants(m_simparams, m_physparams, gdata->worldOrigin, gdata->gridSize, gdata->cellSize,
		m_numAllocatedParticles);
}

void GPUWorker::uploadBodiesCentersOfGravity()
{
	setforcesrbcg(gdata->s_hRbGravityCenters, m_simparams->numODEbodies);
	seteulerrbcg(gdata->s_hRbGravityCenters, m_simparams->numODEbodies);
}

void GPUWorker::uploadBodiesTransRotMatrices()
{
	seteulerrbtrans(gdata->s_hRbTranslations, m_simparams->numODEbodies);
	seteulerrbsteprot(gdata->s_hRbRotationMatrices, m_simparams->numODEbodies);
}

