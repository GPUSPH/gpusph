/*
 * GPUWorker.cpp
 *
 *  Created on: Dec 19, 2012
 *      Author: rustico
 */

#include "GPUWorker.h"
#include "buildneibs.cuh"
#include "forces.cuh"
#include "euler.cuh"
// ostringstream
#include <sstream>
// FLT_MAX
#include "float.h"

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
// NOTE: this should be updated for each new device array!
size_t GPUWorker::computeMemoryPerParticle()
{
	size_t tot = 0;

	tot += sizeof(m_dPos[0][0]) * 2; // double buffered
	tot += sizeof(m_dVel[0][0]) * 2; // double buffered
	tot += sizeof(m_dInfo[0][0]) * 2; // double buffered
	tot += sizeof(m_dForces[0]);
	tot += sizeof(m_dParticleHash[0]);
	tot += sizeof(m_dParticleIndex[0]);
	// Now we estimate the memory consumption of thrust::sort() by simply assuming another m_dParticleIndex is allocated. This might be slighly different
	// and should be fixed if/when detailed information are found. If sorting goes out of memory, a bigger safety buffer than 100Mb might be used.
	// To include also the size of the hash in the estimation, just add "tot += sizeof(m_dParticleHash[0]);"
	tot += sizeof(m_dParticleIndex[0]);
	tot += sizeof(m_dNeibsList[0]) * m_simparams->maxneibsnum; // underestimated: the total is rounded up to next multiple of NEIBINDEX_INTERLEAVE
	// Memory for the cfl reduction is widely overestimated: there are actually 2 arrays of float but one should then divide by BLOCK_SIZE_FORCES.
	// We consider instead a fourth of one array only, so roughly 1/8 instead of 1/128 or 1/256. It's still about 8 times more.
	tot += sizeof(m_dCfl[0])/4;
	// conditional here?
	tot += sizeof(m_dXsph[0]);

	// optional arrays
	if (m_simparams->savenormals) tot += sizeof(m_dNormals[0]);
	if (m_simparams->vorticity) tot += sizeof(m_dVort[0]);
	if (m_simparams->visctype == SPSVISC) tot += sizeof(m_dTau[0][0]) * 3; // 6 tau per part

	// TODO
	//float4*		m_dRbForces;
	//float4*		m_dRbTorques;
	//uint*		m_dRbNum;

	// round up to next multiple of 4

	return (tot/4 + 1) * 4;
}

// Compute the bytes required for each cell.
// NOTE: this should be update for each new device array!
size_t GPUWorker::computeMemoryPerCell()
{
	size_t tot = 0;
	tot += sizeof(m_dCellStart[0]);
	tot += sizeof(m_dCellEnd[0]);
	tot += sizeof(m_dCompactDeviceMap[0]);
	return tot;
}

// Compute the maximum number of particles we can allocate according to the available device memory
void GPUWorker::computeAndSetAllocableParticles()
{
	size_t totMemory, freeMemory;
	cudaMemGetInfo(&freeMemory, &totMemory);
	freeMemory -= gdata->nGridCells * computeMemoryPerCell();
	freeMemory -= 16; // segments
	freeMemory -= 100*1024*1024; // leave 100Mb as safety margin
	m_numAllocatedParticles = (freeMemory / computeMemoryPerParticle());

	if (m_numAllocatedParticles < m_numParticles) {
		printf("FATAL: thread %u needs %lu particles, but there is memory for %lu (plus safety margin)\n", m_deviceIndex, m_numParticles, m_numAllocatedParticles);
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
	gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] == EMPTY_SEGMENT;
	gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_CELL] == EMPTY_SEGMENT;
}

// append a copy of the external edge cells of other devices to the self device arrays
// and update cellStarts, cellEnds and segments
void GPUWorker::importPeerEdgeCells()
{
	// at the moment, the cells are imported in the same order they are encountered iterating
	// on the device map. We wonder if iterating per-device would lead more optimized transfers.
	// Keeping a list of external cells to be updated could be useful to this aim.
	// For sure one optimization should be to compact (when possible) burst of cells coming from
	// the same peer device.

	// iterate on all cells
	for (uint cell=0; cell < m_nGridCells; cell++)
		// if the current is an external edge cell and it belongs to a device of the same node...
		if (m_hCompactDeviceMap[cell] == CELLTYPE_OUTER_EDGE_CELL_SHIFTED && gdata->RANK(gdata->s_hDeviceMap[cell]) == gdata->mpi_rank) {
			// check in which device it is
			uchar peerDevIndex = gdata->DEVICE( gdata->s_hDeviceMap[cell] );
			if (peerDevIndex == m_deviceIndex)
				printf("WARNING: cell %u is outer edge for thread %u, but SELF in the device map!\n", cell, m_deviceIndex);
			if (peerDevIndex >= gdata->devices) {
				printf("FATAL: cell %u has peer index %u, probable memory corruption\n", cell, peerDevIndex);
				gdata->quit_request = true;
				return;
			}
			uint peerCudaDevNum = gdata->device[peerDevIndex];
			// find its cellStart and cellEnd
			uint peerCellStart = gdata->s_dCellStarts[peerDevIndex][cell];
			uint peerCellEnd = gdata->s_dCellEnds[peerDevIndex][cell];
			// if it is empty, update cellStarts; otherwise...
			if (peerCellStart == 0xFFFFFFFF) {
				// set the cell as empty
				gdata->s_dCellStarts[m_deviceIndex][cell] = 0xFFFFFFFF;
				// update device array
				CUDA_SAFE_CALL_NOSYNC(cudaMemcpy(	(m_dCellStart + cell),
													(gdata->s_dCellStarts[m_deviceIndex] + cell),
													sizeof(uint), cudaMemcpyHostToDevice));
			} else {
				// cellEnd is exclusive
				uint numPartsInPeerCell = peerCellEnd - peerCellStart;
				// retrieve device pointers of peer device
				const float4** peer_dPos = gdata->GPUWORKERS[peerDevIndex]->getDPosBuffers();
				const float4** peer_dVel = gdata->GPUWORKERS[peerDevIndex]->getDVelBuffers();
				const particleinfo** peer_dInfo = gdata->GPUWORKERS[peerDevIndex]->getDInfoBuffers();
				// append pos, vel and info data
				size_t _size = numPartsInPeerCell * sizeof(float4);
				CUDA_SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	m_dPos[ gdata->currentPosRead ] + m_numParticles,
												m_cudaDeviceNumber,
												peer_dPos[ gdata->currentPosRead ] + peerCellStart,
												peerCudaDevNum,
												_size));
				// _size = numPartsInPeerCell * sizeof(float4);
				CUDA_SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	m_dVel[ gdata->currentVelRead ] + m_numParticles,
												m_cudaDeviceNumber,
												peer_dVel[ gdata->currentVelRead ] + peerCellStart,
												peerCudaDevNum,
												_size));
				_size = numPartsInPeerCell * sizeof(particleinfo);
				CUDA_SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	m_dInfo[ gdata->currentInfoRead ] + m_numParticles,
												m_cudaDeviceNumber,
												peer_dInfo[ gdata->currentInfoRead ] + peerCellStart,
												peerCudaDevNum,
												_size));
				// now we should write
				// cellStart[cell] = m_numParticles
				// cellEnd[cell] = m_numParticles + numPartsInPeerCell
				// in both device memory and host buffers (although will not be used)

				// Update host copy of cellStart and cellEnd. Since it is an external cell,
				// it is unlikely that the host copy will be used, but it is always good to keep
				// indices consistent
				gdata->s_dCellStarts[m_deviceIndex][cell] = m_numParticles;
				gdata->s_dCellEnds[m_deviceIndex][cell] = m_numParticles + numPartsInPeerCell;

				// Update device copy of cellStart (later cellEnd). This allows for building the
				// neighbor list directly, without the need of running again calchash, sort and reorder
				CUDA_SAFE_CALL_NOSYNC(cudaMemcpy(	(m_dCellStart + cell),
													(gdata->s_dCellStarts[m_deviceIndex] + cell),
													sizeof(uint), cudaMemcpyHostToDevice));
				CUDA_SAFE_CALL_NOSYNC(cudaMemcpy(	(m_dCellEnd + cell),
													(gdata->s_dCellEnds[m_deviceIndex] + cell),
													sizeof(uint), cudaMemcpyHostToDevice));

				// update outer edge segment
				// NOTE: keeping correctness only if there are no OUTER particles (which we assume)
				if (gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] == EMPTY_SEGMENT)
					gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] = m_numParticles;

				// update the total number of particles
				m_numParticles += numPartsInPeerCell;
			} // if cell is not empty
		} // if cell is external edge and in the same node

	// cudaMemcpyPeerAsync() is asynchronous with the host. We synchronize at the end to wait for the
	// transfers to be complete.
	cudaDeviceSynchronize();
}

void GPUWorker::importNetworkPeerEdgeCells()
{
	// We need to import every cell of the neigbor processes only once. To this aim, we keep a list of recipient
	// ranks who already received the current cell. The list, in form of a bitmap, is reset before iterating on
	// all the neighbor cells
	bool recipient_devices[MAX_DEVICES_PER_CLUSTER];

	// iterate on all cells
	for (int cx = 0; cx < gdata->gridSize.x; cx++)
		for (int cy = 0; cy < gdata->gridSize.y; cy++)
			for (int cz = 0; cz < gdata->gridSize.z; cz++) {

				uint lin_curr_cell = gdata->calcGridHashHost(cx, cy, cz);

				// optimization: if not edging, continue
				if (m_hCompactDeviceMap[lin_curr_cell] == CELLTYPE_OUTER_CELL_SHIFTED ||
					m_hCompactDeviceMap[lin_curr_cell] == CELLTYPE_INNER_CELL_SHIFTED ) continue;

				// reset the list fo recipient neib processes
				for (uint d = 0; d < MAX_DEVICES_PER_CLUSTER; d++)
					recipient_devices[d] = false;

				uint curr_cell_globalDevIdx = gdata->s_hDeviceMap[lin_curr_cell];
				uchar curr_cell_rank = gdata->RANK( curr_cell_globalDevIdx );
				bool curr_mine = (curr_cell_globalDevIdx == m_globalDeviceIdx);

				// iterate on neighbors
				for (int dx = -1; dx <= 1; dx++)
					for (int dy = -1; dy <= 1; dy++)
						for (int dz = -1; dz <= 1; dz++) {

							// check that we are inside the grid
							if (cx + dx < 0 || cx + dx >= gdata->gridSize.x) continue;
							if (cy + dy < 0 || cy + dy >= gdata->gridSize.y) continue;
							if (cz + dz < 0 || cz + dz >= gdata->gridSize.z) continue;

							// linearized hash of neib cell
							uint lin_neib_cell = gdata->calcGridHashHost(cx + dx, cy + dy, cz + dz);
							uint neib_cell_globalDevIdx = gdata->s_hDeviceMap[lin_neib_cell];

							// will be set in different way depending on the rank (mine, then local cellStarts, or not, then receive size via network)
							// note: it is important that it is initialized to 0 in case the cell is empty
							uint partsInCurrCell = 0;

							bool neib_mine = (neib_cell_globalDevIdx == m_globalDeviceIdx);
							uchar neib_cell_rank = gdata->RANK( neib_cell_globalDevIdx );

							// did we already treat the pair (curr_rank <-> neib_rank) for this cell?
							if (recipient_devices[ gdata->GLOBAL_DEVICE_NUM(neib_cell_globalDevIdx) ]) continue;

							// do they belong to different nodes?
							if (curr_cell_rank != neib_cell_rank) {

								// current is mine: send the cell to the process holding the neighbor cell
								if (curr_mine) {

									uint curr_cell_start = gdata->s_dCellStarts[m_deviceIndex][lin_curr_cell];

									// send the size of the cell...
									if (curr_cell_start != EMPTY_CELL)
										partsInCurrCell = gdata->s_dCellEnds[m_deviceIndex][lin_curr_cell] - curr_cell_start;
									gdata->networkManager->sendUint(curr_cell_globalDevIdx, neib_cell_globalDevIdx, &partsInCurrCell);

									if (curr_cell_start != EMPTY_CELL) {

										// ... then the data (pos, vel, info):
										gdata->networkManager->sendFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dPos[ gdata->currentPosRead ] + curr_cell_start) );
										gdata->networkManager->sendFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dVel[ gdata->currentVelRead ] + curr_cell_start) );
										gdata->networkManager->sendShorts(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (ushort*)(m_dInfo[ gdata->currentInfoRead ] + curr_cell_start) );
									}

								} else
								// neighbor is mine: receive the cell from the process holding the current cell
								if (neib_mine) {

									// receive the size of the cell...
									gdata->networkManager->receiveUint(curr_cell_globalDevIdx, neib_cell_globalDevIdx, &partsInCurrCell);

									if (partsInCurrCell > 0) {

										// ... then the data (pos, vel, info):
										gdata->networkManager->receiveFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dPos[ gdata->currentPosRead ] + m_numParticles) );
										gdata->networkManager->receiveFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dVel[ gdata->currentVelRead ] + m_numParticles) );
										gdata->networkManager->receiveShorts(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (ushort*)(m_dInfo[ gdata->currentInfoRead ] + m_numParticles) );

										// update local cellStarts/Ends
										gdata->s_dCellStarts[m_deviceIndex][lin_curr_cell] = m_numParticles;
										gdata->s_dCellEnds[m_deviceIndex][lin_curr_cell] = m_numParticles + partsInCurrCell;

									} else
										gdata->s_dCellStarts[m_deviceIndex][lin_curr_cell] = EMPTY_CELL;

									// update metadata et al. (see importPeerEdgeCells())
									CUDA_SAFE_CALL_NOSYNC(cudaMemcpy( (m_dCellStart + lin_curr_cell), (gdata->s_dCellStarts[m_deviceIndex] + lin_curr_cell),
										sizeof(uint), cudaMemcpyHostToDevice));
									if (partsInCurrCell > 0)
										CUDA_SAFE_CALL_NOSYNC(cudaMemcpy( (m_dCellEnd + lin_curr_cell), (gdata->s_dCellEnds[m_deviceIndex] + lin_curr_cell),
											sizeof(uint), cudaMemcpyHostToDevice));

									// update outer edge segment
									if (gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] == EMPTY_SEGMENT && partsInCurrCell > 0)
										gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] = m_numParticles;

									// update the total number of particles
									m_numParticles += partsInCurrCell;
								} // curr or neib are mine

								// mark the current pair of cell as treated
								if (curr_mine || neib_mine)
									recipient_devices[ gdata->GLOBAL_DEVICE_NUM(neib_cell_globalDevIdx) ] = true;

							} // curr and neib belong to different processes

						} // iterate on neighbor cells
			} // iterate on cells
}

// overwrite the external edge cells with an updated copy
// NOTE: for this method and importPeerEdgeCells() could be encapsulated somehow, since their
// algorithms have a strong overlap
// TODO: make double-buffers safe by checking the commandFlags
void GPUWorker::updatePeerEdgeCells()
{
	// at the moment, the cells are imported in the same order they are encountered iterating
	// on the device map. We wonder if iterating per-device would lead more optimized transfers.
	// Keeping a list of external cells to be updated could be useful to this aim.
	// For sure one optimization should be to compact (when possible) burst of cells coming from
	// the same peer device.

	// iterate on all cells
	for (uint cell=0; cell < m_nGridCells; cell++)
		// if the current is an external edge cell and it belongs to a device of the same node...
		if (m_hCompactDeviceMap[cell] == CELLTYPE_OUTER_EDGE_CELL_SHIFTED && gdata->RANK(gdata->s_hDeviceMap[cell]) == gdata->mpi_rank) {
			// check in which device it is
			uchar peerDevIndex = gdata->DEVICE( gdata->s_hDeviceMap[cell] );
			uint peerCudaDevNum = gdata->device[peerDevIndex];
			// find its cellStart and cellEnd on the peer device
			uint peerCellStart = gdata->s_dCellStarts[peerDevIndex][cell];
			uint peerCellEnd = gdata->s_dCellEnds[peerDevIndex][cell];
			// find its cellStart and cellEnd on self
			uint selfCellStart = gdata->s_dCellStarts[m_deviceIndex][cell];
			uint selfCellEnd = gdata->s_dCellEnds[m_deviceIndex][cell];
			// if it is not empty...
			// (it is redundant to check also self but could used as a correctness check)
			if (peerCellStart != 0xFFFFFFFF) {
				// cellEnd is exclusive
				uint numPartsInPeerCell = peerCellEnd - peerCellStart;

				// update the requested buffers of the current cell
				size_t _size;

				if (gdata->commandFlags & BUFFER_POS) {
					const float4** peer_dPos = gdata->GPUWORKERS[peerDevIndex]->getDPosBuffers();
					_size = numPartsInPeerCell * sizeof(float4);
					CUDA_SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	m_dPos[ gdata->currentPosWrite ] + selfCellStart,
													m_cudaDeviceNumber,
													peer_dPos[ gdata->currentPosWrite ] + peerCellStart,
													peerCudaDevNum,
													_size));
				}
				if (gdata->commandFlags & BUFFER_VEL) {
					const float4** peer_dVel = gdata->GPUWORKERS[peerDevIndex]->getDVelBuffers();
					_size = numPartsInPeerCell * sizeof(float4);
					CUDA_SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	m_dVel[ gdata->currentVelWrite ] + selfCellStart,
													m_cudaDeviceNumber,
													peer_dVel[ gdata->currentVelWrite ] + peerCellStart,
													peerCudaDevNum,
													_size));
				}
				if (gdata->commandFlags & BUFFER_INFO) {
					const particleinfo** peer_dInfo = gdata->GPUWORKERS[peerDevIndex]->getDInfoBuffers();
					_size = numPartsInPeerCell * sizeof(particleinfo);
					CUDA_SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	m_dInfo[ gdata->currentInfoWrite ] + selfCellStart,
													m_cudaDeviceNumber,
													peer_dInfo[ gdata->currentInfoWrite ] + peerCellStart,
													peerCudaDevNum,
													_size));
				}
				if (gdata->commandFlags & BUFFER_FORCES) {
					const float4* peer_dForces = gdata->GPUWORKERS[peerDevIndex]->getDForceBuffer();
					_size = numPartsInPeerCell * sizeof(float4);
					CUDA_SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	m_dForces + selfCellStart,
													m_cudaDeviceNumber,
													peer_dForces + peerCellStart,
													peerCudaDevNum,
													_size));
				}
			} // if cell is not empty
		} // if cell is external edge and same node

	// cudaMemcpyPeerAsync() is asynchronous with the host. We synchronize at the end to wait for the
	// transfers to be complete.
	cudaDeviceSynchronize();
}

void GPUWorker::updateNetworkPeerEdgeCells()
{
	// Same list technique as in importNetrokPeerEdgeCells()
	bool recipient_devices[MAX_DEVICES_PER_CLUSTER];

	// iterate on all cells
	for (int cx = 0; cx < gdata->gridSize.x; cx++)
		for (int cy = 0; cy < gdata->gridSize.y; cy++)
			for (int cz = 0; cz < gdata->gridSize.z; cz++) {

				uint lin_curr_cell = gdata->calcGridHashHost(cx, cy, cz);

				// optimization: if not edging, continue
				if (m_hCompactDeviceMap[lin_curr_cell] == CELLTYPE_OUTER_CELL_SHIFTED ||
					m_hCompactDeviceMap[lin_curr_cell] == CELLTYPE_INNER_CELL_SHIFTED) continue;

				uint curr_cell_globalDevIdx = gdata->s_hDeviceMap[lin_curr_cell];

				// reset the list fo recipient neib processes
				for (uint d = 0; d < MAX_DEVICES_PER_CLUSTER; d++)
					recipient_devices[d] = false;

				bool curr_mine = (curr_cell_globalDevIdx == m_globalDeviceIdx);
				uchar curr_cell_rank = gdata->RANK( curr_cell_globalDevIdx );

				// iterate on neighbors
				for (int dx = -1; dx <= 1; dx++)
					for (int dy = -1; dy <= 1; dy++)
						for (int dz = -1; dz <= 1; dz++) {

							// check that we are inside the grid
							if (cx + dx < 0 || cx + dx >= gdata->gridSize.x) continue;
							if (cy + dy < 0 || cy + dy >= gdata->gridSize.y) continue;
							if (cz + dz < 0 || cz + dz >= gdata->gridSize.z) continue;

							// linearized hash of neib cell
							uint lin_neib_cell = gdata->calcGridHashHost(cx + dx, cy + dy, cz + dz);
							uint neib_cell_globalDevIdx = gdata->s_hDeviceMap[lin_neib_cell];

							// if needed, will be read from local cellStarts/Ends arrays
							uint curr_cell_start = 0;
							// note: it is important that it is initialized to 0 in case the cell is empty
							uint partsInCurrCell = 0;

							bool neib_mine = (neib_cell_globalDevIdx == m_globalDeviceIdx);
							uchar neib_cell_rank = gdata->RANK( neib_cell_globalDevIdx );

							// did we already treat the pair (curr_rank <-> neib_rank) for this cell?
							if (recipient_devices[ gdata->GLOBAL_DEVICE_NUM(neib_cell_globalDevIdx) ]) continue;

							// do they belong to different nodes?
							if (curr_cell_rank != neib_cell_rank) {

								// if either one is mine, prepare sizes for an exchange
								if (curr_mine || neib_mine) {
									curr_cell_start = gdata->s_dCellStarts[m_deviceIndex][lin_curr_cell];

									if (curr_cell_start != EMPTY_CELL)
										partsInCurrCell = gdata->s_dCellEnds[m_deviceIndex][lin_curr_cell] - curr_cell_start;
								}

								// current is mine: send the cell to the process holding the neighbor cell
								if (curr_mine && partsInCurrCell > 0) {

									// sen pos, vel, info
									if (gdata->commandFlags & BUFFER_POS)
										gdata->networkManager->sendFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dPos[ gdata->currentPosWrite ] + curr_cell_start) );
									if (gdata->commandFlags & BUFFER_VEL)
										gdata->networkManager->sendFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dVel[ gdata->currentVelWrite ] + curr_cell_start) );
									if (gdata->commandFlags & BUFFER_INFO)
										gdata->networkManager->sendShorts(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (ushort*)(m_dInfo[ gdata->currentInfoWrite ] + curr_cell_start) );
									if (gdata->commandFlags & BUFFER_FORCES)
										gdata->networkManager->sendFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dForces + curr_cell_start) );

								} else
								// neighbor is mine: receive the cell from the process holding the current cell
								if (neib_mine && partsInCurrCell > 0) {

									// receive pos, vel, info
									if (gdata->commandFlags & BUFFER_POS)
										gdata->networkManager->receiveFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dPos[ gdata->currentPosWrite ] + curr_cell_start) );
									if (gdata->commandFlags & BUFFER_VEL)
										gdata->networkManager->receiveFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dVel[ gdata->currentVelWrite ] + curr_cell_start) );
									if (gdata->commandFlags & BUFFER_INFO)
										gdata->networkManager->receiveShorts(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (ushort*)(m_dInfo[ gdata->currentInfoWrite ] + curr_cell_start) );
									if (gdata->commandFlags & BUFFER_FORCES)
										gdata->networkManager->receiveFloats(curr_cell_globalDevIdx, neib_cell_globalDevIdx, partsInCurrCell * 4, (float*)(m_dForces + curr_cell_start) );

								} // curr or neib are mine

								// mark the current pair of cell as treated
								if (curr_mine || neib_mine)
									recipient_devices[ gdata->GLOBAL_DEVICE_NUM(neib_cell_globalDevIdx) ] = true;

							} // curr and neib belong to different processes

						} // iterate on neighbor cells
			} // iterate on cells
}

// All the allocators assume that gdata is updated with the number of particles (done by problem->fillparts).
// Later this will be changed since each thread does not need to allocate the global number of particles.
size_t GPUWorker::allocateHostBuffers() {
	// common sizes
	const uint float3Size = sizeof(float3) * m_numAllocatedParticles;
	const uint float4Size = sizeof(float4) * m_numAllocatedParticles;
	const uint infoSize = sizeof(particleinfo) * m_numAllocatedParticles;
	const uint uintCellsSize = sizeof(uint) * m_nGridCells;

	size_t allocated = 0;

	/*m_hPos = new float4[m_numAllocatedParticles];
	memset(m_hPos, 0, float4Size);
	allocated += float4Size;

	m_hVel = new float4[m_numAllocatedParticles];
	memset(m_hVel, 0, float4Size);
	allocated += float4Size;

	m_hInfo = new particleinfo[m_numAllocatedParticles];
	memset(m_hInfo, 0, infoSize);
	allocated += infoSize; */

	if (MULTI_DEVICE) {
		m_hCompactDeviceMap = new uint[m_nGridCells];
		memset(m_hCompactDeviceMap, 0, uintCellsSize);
		allocated += uintCellsSize;
	}

	/*if (m_simparams->vorticity) {
		m_hVort = new float3[m_numAllocatedParticles];
		allocated += float3Size;
		// NOTE: *not* memsetting, as in master branch
	}*/

	m_hostMemory += allocated;
	return allocated;
}

size_t GPUWorker::allocateDeviceBuffers() {
	// common sizes
	// compute common sizes (in bytes)
	//const uint floatSize = sizeof(float) * m_numAlocatedParticles;
	const uint float2Size = sizeof(float2) * m_numAllocatedParticles;
	const uint float3Size = sizeof(float3) * m_numAllocatedParticles;
	const uint float4Size = sizeof(float4) * m_numAllocatedParticles;
	const uint infoSize = sizeof(particleinfo) * m_numAllocatedParticles;
	const uint intSize = sizeof(uint) * m_numAllocatedParticles;
	const uint uintCellsSize = sizeof(uint) * m_nGridCells;
	const uint neibslistSize = sizeof(uint) * m_simparams->maxneibsnum*(m_numAllocatedParticles/NEIBINDEX_INTERLEAVE + 1)*NEIBINDEX_INTERLEAVE;
	const uint hashSize = sizeof(hashKey) * m_numAllocatedParticles;
	const uint segmentsSize = sizeof(uint) * 4; // 4 = types of cells
	//const uint neibslistSize = sizeof(uint) * 128 * m_numAlocatedParticles;
	//const uint sliceArraySize = sizeof(uint) * m_gridSize.PSA;

	size_t allocated = 0;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dForces, float4Size));
	CUDA_SAFE_CALL(cudaMemset(m_dForces, 0, float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dXsph, float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dPos[0], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dPos[1], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVel[0], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVel[1], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dInfo[0], infoSize));
	allocated += infoSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dInfo[1], infoSize));
	allocated += infoSize;

	// Free surface detection
	if (m_simparams->savenormals) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNormals, float4Size));
		allocated += float4Size;
	}

	if (m_simparams->vorticity) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVort, float3Size));
		allocated += float3Size;
	}

	if (m_simparams->visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[0], float2Size));
		allocated += float2Size;

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[1], float2Size));
		allocated += float2Size;

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[2], float2Size));
		allocated += float2Size;
	}

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dParticleHash, hashSize));
	allocated += hashSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dParticleIndex, intSize));
	allocated += intSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCellStart, uintCellsSize));
	allocated += uintCellsSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCellEnd, uintCellsSize));
	allocated += uintCellsSize;

	//CUDA_SAFE_CALL(cudaMalloc((void**)&m_dSliceStart, sliceArraySize));
	//allocated += sliceArraySize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNeibsList, neibslistSize));
	CUDA_SAFE_CALL(cudaMemset(m_dNeibsList, 0xffffffff, neibslistSize));
	allocated += neibslistSize;

	// TODO: an array of uchar would suffice
	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCompactDeviceMap, uintCellsSize));
	// initialize anyway for single-GPU simulations
	CUDA_SAFE_CALL(cudaMemset(m_dCompactDeviceMap, 0, uintCellsSize));
	allocated += uintCellsSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dSegmentStart, segmentsSize));
	// ditto
	CUDA_SAFE_CALL(cudaMemset(m_dSegmentStart, 0, segmentsSize));
	allocated += segmentsSize;

	// newNumParticles for inlets
	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNewNumParticles, sizeof(uint)));
	allocated += sizeof(uint);

	// TODO: allocation for rigid bodies

	if (m_simparams->dtadapt) {
		// for the allocation we use m_numPartsFmax computed from m_numAlocatedParticles;
		// after forces we use an updated value instead (the numblocks of forces)
		uint m_numPartsFmax = getNumPartsFmax(m_numAllocatedParticles);
		const uint fmaxTableSize = m_numPartsFmax*sizeof(float);

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCfl, fmaxTableSize));
		CUDA_SAFE_CALL(cudaMemset(m_dCfl, 0, fmaxTableSize));

		const uint tempCflSize = getFmaxTempStorageSize(m_numPartsFmax);
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTempCfl, tempCflSize));
		CUDA_SAFE_CALL(cudaMemset(m_dTempCfl, 0, tempCflSize));

		allocated += fmaxTableSize + tempCflSize;
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
	//delete [] m_hPos;
	//delete [] m_hVel;
	//delete [] m_hInfo;
	if (MULTI_DEVICE)
		delete [] m_hCompactDeviceMap;
	/*if (m_simparams->vorticity)
		delete [] m_hVort;*/
	// here: dem host buffers?
}

void GPUWorker::deallocateDeviceBuffers() {
	CUDA_SAFE_CALL(cudaFree(m_dForces));
	CUDA_SAFE_CALL(cudaFree(m_dXsph));
	CUDA_SAFE_CALL(cudaFree(m_dPos[0]));
	CUDA_SAFE_CALL(cudaFree(m_dPos[1]));
	CUDA_SAFE_CALL(cudaFree(m_dVel[0]));
	CUDA_SAFE_CALL(cudaFree(m_dVel[1]));
	CUDA_SAFE_CALL(cudaFree(m_dInfo[0]));
	CUDA_SAFE_CALL(cudaFree(m_dInfo[1]));

	if (m_simparams->savenormals)
		CUDA_SAFE_CALL(cudaFree(m_dNormals));

	if (m_simparams->vorticity)
		CUDA_SAFE_CALL(cudaFree(m_dVort));

	if (m_simparams->visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaFree(m_dTau[0]));
		CUDA_SAFE_CALL(cudaFree(m_dTau[1]));
		CUDA_SAFE_CALL(cudaFree(m_dTau[2]));
	}

	CUDA_SAFE_CALL(cudaFree(m_dParticleHash));
	CUDA_SAFE_CALL(cudaFree(m_dParticleIndex));
	CUDA_SAFE_CALL(cudaFree(m_dCellStart));
	CUDA_SAFE_CALL(cudaFree(m_dCellEnd));
	CUDA_SAFE_CALL(cudaFree(m_dNeibsList));

	CUDA_SAFE_CALL(cudaFree(m_dCompactDeviceMap));
	CUDA_SAFE_CALL(cudaFree(m_dSegmentStart));
	CUDA_SAFE_CALL(cudaFree(m_dNewNumParticles));

	// TODO: deallocation for rigid bodies

	if (m_simparams->dtadapt) {
		CUDA_SAFE_CALL(cudaFree(m_dCfl));
		CUDA_SAFE_CALL(cudaFree(m_dTempCfl));
	}

	// here: dem device buffers?
}

void GPUWorker::printAllocatedMemory()
{
	printf("Device idx %u (CUDA: %u) allocated %.2f Gb on host, %.2f Gb on device\n"
			"  assigned particles: %s; allocated: %s\n", m_deviceIndex, m_cudaDeviceNumber,
			getHostMemory()/1000000000.0, getDeviceMemory()/1000000000.0,
			gdata->addSeparators(m_numParticles).c_str(), gdata->addSeparators(m_numAllocatedParticles).c_str());
}

// upload subdomain, just allocated and sorted by main thread
void GPUWorker::uploadSubdomain() {
	// indices
	uint firstInnerParticle	= gdata->s_hStartPerDevice[m_deviceIndex];
	uint howManyParticles	= gdata->s_hPartsPerDevice[m_deviceIndex];

	// is the device empty? (unlikely but possible before LB kicks in)
	if (howManyParticles == 0) return;

	size_t _size = 0;

	// memcpys - recalling GPU arrays are double buffered
	_size = howManyParticles * sizeof(float4);
	//printf("Thread %d uploading %d POS items (%u Kb) on device %d from position %d\n",
	//		m_deviceIndex, howManyParticles, (uint)_size/1000, m_cudaDeviceNumber, firstInnerParticle);
	CUDA_SAFE_CALL(cudaMemcpy(	m_dPos[ gdata->currentPosRead ],
								gdata->s_hPos + firstInnerParticle,
								_size, cudaMemcpyHostToDevice));

	_size = howManyParticles * sizeof(float4);
	//printf("Thread %d uploading %d VEL items (%u Kb) on device %d from position %d\n",
	//		m_deviceIndex, howManyParticles, (uint)_size/1000, m_cudaDeviceNumber, firstInnerParticle);
	CUDA_SAFE_CALL(cudaMemcpy(	m_dVel[ gdata->currentVelRead ],
								gdata->s_hVel + firstInnerParticle,
								_size, cudaMemcpyHostToDevice));

	_size = howManyParticles * sizeof(particleinfo);
	//printf("Thread %d uploading %d INFO items (%u Kb) on device %d from position %d\n",
	//		m_deviceIndex, howManyParticles, (uint)_size/1000, m_cudaDeviceNumber, firstInnerParticle);
	CUDA_SAFE_CALL(cudaMemcpy(	m_dInfo[ gdata->currentInfoRead ],
								gdata->s_hInfo + firstInnerParticle,
								_size, cudaMemcpyHostToDevice));
}

// Download the subset of the specified buffer to the correspondent shared CPU array.
// Makes multiple transfers. Only downloads the subset relative to the internal particles.
// For doulbe buffered arrays, uses the READ buffers unless otherwise specified. Can be
// used for either for th read or the write buffers, not both.
// TODO: write a macro to encapsulate all memcpys
// TODO: use sizeof(array[0]) to make it general purpose?
void GPUWorker::dumpBuffers() {
	// indices
	uint firstInnerParticle	= gdata->s_hStartPerDevice[m_deviceIndex];
	uint howManyParticles	= gdata->s_hPartsPerDevice[m_deviceIndex];

	// is the device empty? (unlikely but possible before LB kicks in)
	if (howManyParticles == 0) return;

	size_t _size = 0;
	uint flags = gdata->commandFlags;

	if (flags & BUFFER_POS) {
		_size = howManyParticles * sizeof(float4);
		uchar dbl_buffer_pointer = gdata->currentPosRead;
		if (flags & DBLBUFFER_READ) dbl_buffer_pointer = gdata->currentPosRead; else
		if (flags & DBLBUFFER_WRITE) dbl_buffer_pointer = gdata->currentPosWrite;
		CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hPos + firstInnerParticle,
									m_dPos[dbl_buffer_pointer],
									_size, cudaMemcpyDeviceToHost));
	}

	if (flags & BUFFER_VEL) {
		_size = howManyParticles * sizeof(float4);
		uchar dbl_buffer_pointer = gdata->currentVelRead;
		if (flags & DBLBUFFER_READ) dbl_buffer_pointer = gdata->currentVelRead; else
		if (flags & DBLBUFFER_WRITE) dbl_buffer_pointer = gdata->currentVelWrite;
		CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hVel + firstInnerParticle,
									m_dVel[dbl_buffer_pointer],
									_size, cudaMemcpyDeviceToHost));
	}

	if (flags & BUFFER_INFO) {
		_size = howManyParticles * sizeof(particleinfo);
		uchar dbl_buffer_pointer = gdata->currentInfoRead;
		if (flags & DBLBUFFER_READ) dbl_buffer_pointer = gdata->currentInfoRead; else
		if (flags & DBLBUFFER_WRITE) dbl_buffer_pointer = gdata->currentInfoWrite;
		CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hInfo + firstInnerParticle,
									m_dInfo[dbl_buffer_pointer],
									_size, cudaMemcpyDeviceToHost));
	}

	if (flags & BUFFER_VORTICITY) {
		_size = howManyParticles * sizeof(float3);
		CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hVorticity + firstInnerParticle,
									m_dVort,
									_size, cudaMemcpyDeviceToHost));
	}

	if (flags & BUFFER_NORMALS) {
		_size = howManyParticles * sizeof(float3);
		CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hNormals + firstInnerParticle,
									m_dNormals,
									_size, cudaMemcpyDeviceToHost));
	}
}

// Sets all cells as empty in device memory. Used before reorder
void GPUWorker::setDeviceCellsAsEmpty()
{
	CUDA_SAFE_CALL(cudaMemset(m_dCellStart, 0xffffffff, gdata->nGridCells  * sizeof(uint)));
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

	if (activeParticles != m_numParticles) {
		// if for debug reasons we need to print the change in numParts for each device, uncomment the following:
		// printf("  Dev. index %u: particles: %d => %d\n", m_deviceIndex, m_numParticles, activeParticles);
		m_numParticles = activeParticles;
	}
}

// upload the value m_numParticles to "newNumParticles" on device
void GPUWorker::uploadNewNumParticles()
{
	// uploading even if empty (usually not, right after append)
	CUDA_SAFE_CALL(cudaMemcpy(m_dNewNumParticles, &m_numParticles, sizeof(uint), cudaMemcpyHostToDevice));
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

	// compute the extra_offset vector (z-lift) in terms of cells
	int3 extra_offset = make_int3(0);
	// Direction of the extra_offset (depents on where periodicity occurs), to add extra cells if the extra displacement
	// does not multiply the size of the cells
	int3 extra_offset_unit_vector = make_int3(0);
	if (m_simparams->periodicbound) {
		extra_offset.x = (int) (m_physparams->dispOffset.x / gdata->cellSize.x);
		extra_offset.y = (int) (m_physparams->dispOffset.y / gdata->cellSize.y);
		extra_offset.z = (int) (m_physparams->dispOffset.z / gdata->cellSize.z);
	}

	// when there is an extra_offset to consider, we have to repeat the same checks twice. Let's group them
#define CHECK_CURRENT_CELL \
	/* data of neib cell */ \
	uint neib_lin_idx = gdata->calcGridHashHost(cx, cy, cz); \
	uint neib_globalDevidx = gdata->s_hDeviceMap[neib_lin_idx]; \
	any_mine_neib	 |= (neib_globalDevidx == m_globalDeviceIdx); \
	any_foreign_neib |= (neib_globalDevidx != m_globalDeviceIdx); \
	/* did we read enough to decide for current cell? */ \
	enough_info = (is_mine && any_foreign_neib) || (!is_mine && any_mine_neib);

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
							bool apply_extra_offset = false;
							// warp periodic boundaries
							if (m_simparams->periodicbound) {
								// periodicity along X
								if (m_physparams->dispvect.x) {
									// WARNING: checking if c* is negative MUST be done before checking if it's greater than
									// the grid, otherwise it will be cast to uint and "-1" will be "greater" than the gridSize
									if (cx < 0) {
										cx = gdata->gridSize.x - 1;
										if (extra_offset.y != 0 || extra_offset.z != 0) {
											extra_offset_unit_vector.y = extra_offset_unit_vector.z = 1;
											apply_extra_offset = true;
										}
									} else
									if (cx >= gdata->gridSize.x) {
										cx = 0;
										if (extra_offset.y != 0 || extra_offset.z != 0) {
											extra_offset_unit_vector.y = extra_offset_unit_vector.z = -1;
											apply_extra_offset = true;
										}
									}
								} // if dispvect.x
								// periodicity along Y
								if (m_physparams->dispvect.y) {
									if (cy < 0) {
										cy = gdata->gridSize.y - 1;
										if (extra_offset.x != 0 || extra_offset.z != 0) {
											extra_offset_unit_vector.x = extra_offset_unit_vector.z = 1;
											apply_extra_offset = true;
										}
									} else
									if (cy >= gdata->gridSize.y) {
										cy = 0;
										if (extra_offset.x != 0 || extra_offset.z != 0) {
											extra_offset_unit_vector.x = extra_offset_unit_vector.z = -1;
											apply_extra_offset = true;
										}
									}
								} // if dispvect.y
								// periodicity along Z
								if (m_physparams->dispvect.z) {
									if (cz < 0) {
										cz = gdata->gridSize.z - 1;
										if (extra_offset.x != 0 || extra_offset.y != 0) {
											extra_offset_unit_vector.x = extra_offset_unit_vector.y = 1;
											apply_extra_offset = true;
										}
									} else
									if (cz >= gdata->gridSize.z) {
										cz = 0;
										if (extra_offset.x != 0 || extra_offset.y != 0) {
											extra_offset_unit_vector.x = extra_offset_unit_vector.y = -1;
											apply_extra_offset = true;
										}
									}
								} // if dispvect.z
								// apply extra displacement (e.g. zlift), if any
								// (actually, should be safe to add without the conditional)
								if (apply_extra_offset) {
									cx += extra_offset.x * extra_offset_unit_vector.x;
									cy += extra_offset.y * extra_offset_unit_vector.y;
									cz += extra_offset.z * extra_offset_unit_vector.z;
								}
							}
							// if not periodic, or if still out-of-bounds after periodicity warp, skip it
							if (cx < 0 || cx >= gdata->gridSize.x ||
								cy < 0 || cy >= gdata->gridSize.y ||
								cz < 0 || cz >= gdata->gridSize.z) continue;
							// check current cell and if we have already enough information
							CHECK_CURRENT_CELL
							// if there is any extra offset, check one more cell
							// WARNING: might miss cells if the extra displacement is not parallel to one cartesian axis
							if (m_simparams->periodicbound && apply_extra_offset) {
								cx += extra_offset_unit_vector.x;
								cy += extra_offset_unit_vector.y;
								cz += extra_offset_unit_vector.z;
								// check extra cells if extra displacement is not a multiple of the cell size
								CHECK_CURRENT_CELL
							}
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

unsigned long GPUWorker::getHostMemory() {
	return m_hostMemory;
}

unsigned long GPUWorker::getDeviceMemory() {
	return m_deviceMemory;
}

const float4** GPUWorker::getDPosBuffers()
{
	return (const float4**)m_dPos;
}

const float4** GPUWorker::getDVelBuffers()
{
	return (const float4**)m_dVel;
}

const particleinfo** GPUWorker::getDInfoBuffers()
{
	return (const particleinfo**)m_dInfo;
}

const float4* GPUWorker::getDForceBuffer()
{
	return (const float4*)m_dForces;
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
		// is peer access possible?
		int res;
		cudaDeviceCanAccessPeer(&res, m_deviceIndex, d);
		if (res != 1)
			printf("WARNING: device %u cannot enable peer access of device %u; peer copies will be buffered on host\n", m_deviceIndex, d);
		else
			cudaDeviceEnablePeerAccess(d, 0);
	}
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

	// upload constants (PhysParames, some SimParams)
	instance->uploadConstants();

	// upload planes, if any
	instance->uploadPlanes();

	// upload inlets and outlets
	instance->uploadInlets();
	instance->uploadOutlets();

	// compute #parts to allocate according to the free memory on the device
	instance->computeAndSetAllocableParticles();

	// allocate CPU and GPU arrays
	instance->allocateHostBuffers();
	instance->allocateDeviceBuffers();
	instance->printAllocatedMemory();

	// create and upload the compact device map (2 bits per cell)
	if (MULTI_DEVICE) {
		instance->createCompactDeviceMap();
		instance->uploadCompactDeviceMap();
	}

	// TODO: here set_reduction_params() will be called (to be implemented in this class). These parameters can be device-specific.

	// TODO: here setDemTexture() will be called. It is device-wide, but reading the DEM file is process wide and will be in GPUSPH class

	gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***

	// here GPUSPH::initialize is over and GPUSPH::runSimulation() is called

	gdata->threadSynchronizer->barrier(); // begins UPLOAD ***

	instance->uploadSubdomain();

	gdata->threadSynchronizer->barrier();  // end of UPLOAD, begins SIMULATION ***

	// TODO
	// Here is a copy-paste from the CPU thread worker of branch cpusph, as a canvas
	while (gdata->keep_going) {
		switch (gdata->nextCommand) {
			// logging here?
			case IDLE:
				break;
			case CALCHASH:
				//printf(" T %d issuing HASH\n", deviceIndex);
				instance->kernel_calcHash();
				break;
			case SORT:
				//printf(" T %d issuing SORT\n", deviceIndex);
				instance->kernel_sort();
				break;
			case CROP:
				//printf(" T %d issuing CROP\n", deviceIndex);
				instance->dropExternalParticles();
				break;
			case REORDER:
				//printf(" T %d issuing REORDER\n", deviceIndex);
				instance->kernel_reorderDataAndFindCellStart();
				break;
			case BUILDNEIBS:
				//printf(" T %d issuing BUILDNEIBS\n", deviceIndex);
				instance->kernel_buildNeibsList();
				break;
			case FORCES:
				//printf(" T %d issuing FORCES\n", deviceIndex);
				instance->kernel_forces();
				break;
			case EULER:
				//printf(" T %d issuing EULER\n", deviceIndex);
				instance->kernel_euler();
				break;
			case DUMP:
				//printf(" T %d issuing DUMP\n", deviceIndex);
				instance->dumpBuffers();
				break;
			case DUMP_CELLS:
				//printf(" T %d issuing DUMP_CELLS\n", deviceIndex);
				instance->downloadCellsIndices();
				break;
			case UPDATE_SEGMENTS:
				//printf(" T %d issuing UPDATE_SEGMENTS\n", deviceIndex);
				instance->updateSegments();
				break;
			case DOWNLOAD_NEWNUMPARTS:
				//printf(" T %d issuing DOWNLOAD_NEWNUMPARTS\n", deviceIndex);
				instance->downloadNewNumParticles();
				break;
			case UPLOAD_NEWNUMPARTS:
				//printf(" T %d issuing UPLOAD_NEWNUMPARTS\n", deviceIndex);
				instance->uploadNewNumParticles();
				break;
			case APPEND_EXTERNAL:
				//printf(" T %d issuing APPEND_EXTERNAL\n", deviceIndex);
				if (MULTI_GPU)
					instance->importPeerEdgeCells();
				if (MULTI_NODE)
					instance->importNetworkPeerEdgeCells();
				break;
			case UPDATE_EXTERNAL:
				//printf(" T %d issuing UPDATE_EXTERNAL\n", deviceIndex);
				if (MULTI_GPU)
					instance->updatePeerEdgeCells();
				if (MULTI_NODE)
					instance->updateNetworkPeerEdgeCells();
				break;
			case MLS:
				//printf(" T %d issuing MLS\n", deviceIndex);
				instance->kernel_mls();
				break;
			case SHEPARD:
				//printf(" T %d issuing SHEPARD\n", deviceIndex);
				instance->kernel_shepard();
				break;
			case VORTICITY:
				//printf(" T %d issuing VORTICITY\n", deviceIndex);
				instance->kernel_vorticity();
				break;
			case SURFACE_PARTICLES:
				//printf(" T %d issuing SURFACE_PARTICLES\n", deviceIndex);
				instance->kernel_surfaceParticles();
				break;
			case UPLOAD_MBDATA:
				//printf(" T %d issuing UPLOAD_MBDATA\n", deviceIndex);
				instance->uploadMBData();
				break;
			case UPLOAD_GRAVITY:
				//printf(" T %d issuing UPLOAD_GRAVITY\n", deviceIndex);
				instance->uploadGravity();
				break;
			case UPLOAD_PLANES:
				//printf(" T %d issuing UPLOAD_PLANES\n", deviceIndex);
				instance->uploadPlanes();
				break;
			case QUIT:
				//printf(" T %d issuing QUIT\n", deviceIndex);
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

	calcHash(m_dPos[gdata->currentPosRead],
#if HASH_KEY_SIZE >= 64
					m_dInfo[gdata->currentInfoRead],
					m_dCompactDeviceMap,
#endif
					m_dParticleHash,
					m_dParticleIndex,
					gdata->gridSize,
					gdata->cellSize,
					gdata->worldOrigin,
					m_numParticles);
}

void GPUWorker::kernel_sort()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_numInternalParticles : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	sort(m_dParticleHash, m_dParticleIndex, numPartsToElaborate);
}

void GPUWorker::kernel_reorderDataAndFindCellStart()
{
	// reset also if the device is empty (or we will download uninitialized values)
	setDeviceCellsAsEmpty();

	// is the device empty? (unlikely but possible before LB kicks in)
	if (m_numParticles == 0) return;

	reorderDataAndFindCellStart(m_dCellStart,	  // output: cell start index
							m_dCellEnd,		// output: cell end index
							m_dPos[gdata->currentPosWrite],		 // output: sorted positions
							m_dVel[gdata->currentVelWrite],		 // output: sorted velocities
							m_dInfo[gdata->currentInfoWrite],		 // output: sorted info
							m_dParticleHash,
							m_dParticleIndex,  // input: sorted particle indices
							m_dPos[gdata->currentPosRead],		 // input: sorted position array
							m_dVel[gdata->currentVelRead],		 // input: sorted velocity array
							m_dInfo[gdata->currentInfoRead],		 // input: sorted info array
#if HASH_KEY_SIZE >= 64
							m_dSegmentStart,
#endif
							m_numParticles,
							m_dNewNumParticles,
							m_nGridCells);
}

void GPUWorker::kernel_buildNeibsList()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	buildNeibsList(	m_dNeibsList,
						m_dPos[gdata->currentPosRead],
						m_dInfo[gdata->currentInfoRead],
						m_dParticleHash,
						m_dCellStart,
						m_dCellEnd,
						gdata->gridSize,
						gdata->cellSize,
						gdata->worldOrigin,
						m_numParticles,
						numPartsToElaborate,
						m_nGridCells,
						m_simparams->nlSqInfluenceRadius,
						m_simparams->periodicbound);
}

void GPUWorker::kernel_forces()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// FLOAT_MAX is returned if kernels are not run (e.g. numPartsToElaborate == 0)
	float returned_dt = FLT_MAX;

	bool firstStep = (gdata->commandFlags == INTEGRATOR_STEP_1);

	// first step
	if (numPartsToElaborate > 0 && firstStep)
		returned_dt = forces(  m_dPos[gdata->currentPosRead],   // pos(n)
						m_dVel[gdata->currentVelRead],   // vel(n)
						m_dForces,					// f(n)
						0, // float* rbforces
						0, // float* rbtorques
						m_dXsph,
						m_dInfo[gdata->currentInfoRead],
						m_dNeibsList,
						m_numParticles,
						numPartsToElaborate,
						m_simparams->slength,
						gdata->dt, // m_dt,
						m_simparams->dtadapt,
						m_simparams->dtadaptfactor,
						m_simparams->xsph,
						m_simparams->kerneltype,
						m_simparams->influenceRadius,
						m_simparams->visctype,
						m_physparams->visccoeff,
						m_dCfl,
						m_dTempCfl,
						m_dTau,
						m_simparams->periodicbound,
						m_simparams->sph_formulation,
						m_simparams->boundarytype,
						m_simparams->usedem);
	else
	// second step
	if (numPartsToElaborate > 0 && !firstStep)
		returned_dt = forces(  m_dPos[gdata->currentPosWrite],  // pos(n+1/2)
						m_dVel[gdata->currentVelWrite],  // vel(n+1/2)
						m_dForces,					// f(n+1/2)
						0, // float* rbforces,
						0, // float* rbtorques,
						m_dXsph,
						m_dInfo[gdata->currentInfoRead],
						m_dNeibsList,
						m_numParticles,
						numPartsToElaborate,
						m_simparams->slength,
						gdata->dt, // m_dt,
						m_simparams->dtadapt,
						m_simparams->dtadaptfactor,
						m_simparams->xsph,
						m_simparams->kerneltype,
						m_simparams->influenceRadius,
						m_simparams->visctype,
						m_physparams->visccoeff,
						m_dCfl,
						m_dTempCfl,
						m_dTau,
						m_simparams->periodicbound,
						m_simparams->sph_formulation,
						m_simparams->boundarytype,
						m_simparams->usedem );

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

	if (firstStep)
		euler(  m_dPos[gdata->currentPosRead],	// pos(n)
				m_dVel[gdata->currentVelRead],	// vel(n)
				m_dInfo[gdata->currentInfoRead], //particleInfo
				m_dForces,						// f(n+1/2)
				m_dXsph,
				m_dPos[gdata->currentPosWrite],	// pos(n+1) = pos(n) + velc(n+1/2)*dt
				m_dVel[gdata->currentVelWrite],	// vel(n+1) = vel(n) + f(n+1/2)*dt
				m_numParticles,
				NULL,							// no m_dNewNumParticles at this step
				gdata->totParticles,
				numPartsToElaborate,
				gdata->dt, // m_dt,
				gdata->dt/2.0f, // m_dt/2.0,
				1,
				gdata->t + gdata->dt / 2.0f, // + m_dt,
				m_simparams->xsph,
				m_simparams->periodicbound);
	else
		euler(  m_dPos[gdata->currentPosRead],   // pos(n)
				m_dVel[gdata->currentVelRead],   // vel(n)
				m_dInfo[gdata->currentInfoRead], //particleInfo
				m_dForces,					// f(n+1/2)
				m_dXsph,
				m_dPos[gdata->currentPosWrite],  // pos(n+1) = pos(n) + velc(n+1/2)*dt
				m_dVel[gdata->currentVelWrite],  // vel(n+1) = vel(n) + f(n+1/2)*dt
				m_numParticles,
				m_dNewNumParticles,
				gdata->totParticles,
				numPartsToElaborate,
				gdata->dt, // m_dt,
				gdata->dt/2.0f, // m_dt/2.0,
				2,
				gdata->t + gdata->dt,// + m_dt,
				m_simparams->xsph,
				m_simparams->periodicbound);
}

void GPUWorker::kernel_mls()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	mls(	m_dPos[gdata->currentPosRead],
			m_dVel[gdata->currentVelRead],
			m_dVel[gdata->currentVelWrite],
			m_dInfo[gdata->currentInfoRead],
			m_dNeibsList,
			m_numParticles,
			numPartsToElaborate,
			m_simparams->slength,
			m_simparams->kerneltype,
			m_simparams->influenceRadius,
			m_simparams->periodicbound);
}

void GPUWorker::kernel_shepard()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	shepard(m_dPos[gdata->currentPosRead],
			m_dVel[gdata->currentVelRead],
			m_dVel[gdata->currentVelWrite],
			m_dInfo[gdata->currentInfoRead],
			m_dNeibsList,
			m_numParticles,
			numPartsToElaborate,
			m_simparams->slength,
			m_simparams->kerneltype,
			m_simparams->influenceRadius,
			m_simparams->periodicbound);
}

void GPUWorker::kernel_vorticity()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	// Calling vorticity computation kernel
	vorticity(	m_dPos[gdata->currentPosRead],
				m_dVel[gdata->currentVelRead],
				m_dVort,
				m_dInfo[gdata->currentInfoRead],
				m_dNeibsList,
				numPartsToElaborate,
				m_simparams->slength,
				m_simparams->kerneltype,
				m_simparams->influenceRadius,
				m_simparams->periodicbound);
}

void GPUWorker::kernel_surfaceParticles()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	surfaceparticle( m_dPos[gdata->currentPosRead],
					 m_dVel[gdata->currentVelRead],
					 m_dNormals,
					 m_dInfo[gdata->currentInfoRead],
					 m_dInfo[gdata->currentInfoWrite],
					 m_dNeibsList,
					 numPartsToElaborate,
					 m_simparams->slength,
					 m_simparams->kerneltype,
					 m_simparams->influenceRadius,
					 m_simparams->periodicbound,
					 m_simparams->savenormals);
}

void GPUWorker::uploadConstants()
{
	// NOTE: visccoeff must be set before uploading the constants. This is done in GPUSPH main cycle

	// Setting kernels and kernels derivative factors
	setforcesconstants(m_simparams, m_physparams);
	seteulerconstants(m_physparams);
	setneibsconstants(m_simparams, m_physparams);
}

void GPUWorker::uploadInlets()
{
	//if (m_physparams->inlets == 0) return;
	printf("Dev idx %u uploading %u intlets\n", m_deviceIndex, m_physparams->inlets);
	// no need for letting the forces kernel know about the inlets
	setinleteuler(m_physparams);
}

void GPUWorker::uploadOutlets()
{
	//if (m_physparams->outlets == 0 ) return;
	printf("Dev idx %u uploading %u outlets\n", m_deviceIndex, m_physparams->outlets);
	setoutletforces(m_physparams);
	setoutleteuler(m_physparams);
}

