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
		printf("FATAL: thread %u needs %u particles, but there is memory for %u (plus safety margin)\n", m_deviceIndex, m_numParticles, m_numAllocatedParticles);
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

// Start an async inter-device transfer. This will be actually P2P if device can access peer memory
// (actually, since it is currently used only to import data from other devices, the dstDevice could be omitted or implicit)
void GPUWorker::peerAsyncTransfer(void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count)
{
	CUDA_SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	dst, dstDevice, src, srcDevice, count, m_asyncPeerCopiesStream ) );
}

// Uploads cellStart and cellEnd from the shared arrays to the device memory.
// Parameters: fromCell is inclusive, toCell is exclusive
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

// Import the external edge cells of other devices to the self device arrays. Can append the cells at the end of the current
// list of particles (APPEND_EXTERNAL) or just update the already appended ones (UPDATE_EXTERNAL), according to the current
// command. When appending, also update cellStarts (device and host), cellEnds (device and host) and segments (host only).
// The arrays to be imported must be specified in the command flags. Currently supports pos, vel, info, forces and tau; for the
// double buffered arrays, it is mandatory to specify also the buffer to be used (read or write). This information is ignored
// for the non-buffered arrays (e.g. forces).
// The data is transferred in bursts of consecutive cells when possible. Transfers are actually D2D if peer access is enabled.
// Update: supports also BUFFER_BOUNDELEMENTS, BUFFER_GRADGAMMA, BUFFER_VERTICES, BUFFER_PRESSURE,
// BUFFER_TKE, BUFFER_EPSILON, BUFFER_TURBVISC, BUFFER_STRAIN_RATE
void GPUWorker::importPeerEdgeCells()
{
	// if next command is not an import nor an append, something wrong is going on
	if (! ( (gdata->nextCommand == APPEND_EXTERNAL) || (gdata->nextCommand == UPDATE_EXTERNAL) ) ) {
		printf("WARNING: importPeerEdgeCells() was called, but current command is not APPEND nor UPDATE!\n");
		return;
	}

	// check if at least one double buffer was specified
	bool dbl_buffer_specified = ( (gdata->commandFlags & DBLBUFFER_READ ) || (gdata->commandFlags & DBLBUFFER_WRITE) );
	uint dbl_buf_idx;

	// We aim to make the fewest possible transfers. So we keep track of each burst of consecutive
	// cells from the same peer device, to transfer it with a single memcpy.
	// To this aim, it is important that we iterate on the linearized index so that consecutive
	// cells are also consecutive in memory, regardless the linearization function

	// pointers to peer buffers
	const float4* const* peer_dPos;
	const float4* const* peer_dVel;
	const particleinfo* const* peer_dInfo;
	const float4* peer_dForces;
	const float2* const* peer_dTaus;
	const hashKey* peer_dHash;
	const uint* peer_dPartIndex;
	const float4* const* peer_dBoundElems;
	const float4* const* peer_dGradGamma;
	const vertexinfo* const* peer_dVertices;
	const float* const* peer_dPressure;
	const float* const* peer_dTKE;
	const float* const* peer_dEps;
	const float* const* peer_dTurbVisc;
	const float* const* peer_dStrainRate;

	// aux var for transfers
	size_t _size;

	// indices of current burst
	uint burst_self_index_begin = 0;
	uint burst_peer_index_begin = 0;
	uint burst_peer_index_end = 0;
	uint burst_numparts = 0; // this is redundant with burst_peer_index_end, but cleaner
	uint burst_peer_dev_index = 0;

	// utility defines to handle the bursts
#define BURST_IS_EMPTY (burst_numparts == 0)
#define BURST_SET_CURRENT_CELL \
	burst_self_index_begin = selfCellStart; \
	burst_peer_index_begin = peerCellStart; \
	burst_peer_index_end = peerCellEnd; \
	burst_numparts = numPartsInPeerCell; \
	burst_peer_dev_index = peerDevIndex;

	// Burst of cells (for cellStart / cellEnd). Please not the burst of cells is independent from the one of cells.
	// More specifically, it is a superset, since we are also interested in empty cells; for code readability reasons
	// we handle it as if it were completely independent.
	uint cellsBurst_begin = 0;
	uint cellsBurst_end = 0;
#define CELLS_BURST_IS_EMPTY (cellsBurst_end - cellsBurst_begin == 0)

	// iterate on all cells
	for (uint cell=0; cell < m_nGridCells; cell++)

		// if the current is an external edge cell and it belongs to a device of the same process...
		if (m_hCompactDeviceMap[cell] == CELLTYPE_OUTER_EDGE_CELL_SHIFTED && gdata->RANK(gdata->s_hDeviceMap[cell]) == gdata->mpi_rank) {

			// handle the cell burst: as long as it is OUTER_EDGE and we are appending, we need to copy it, also if empty
			if (gdata->nextCommand == APPEND_EXTERNAL) {
				// 1st case: burst is empty, let's initialize it
				if (CELLS_BURST_IS_EMPTY) {
					cellsBurst_begin = cell;
					cellsBurst_end = cell + 1;
				} else
				// 2nd case: burst is not emtpy but ends with current cell; let's append the cell
				if (cellsBurst_end == cell) {
					cellsBurst_end++;
				} else { // 3rd case: the burst is not emtpy and not compatible; need to flush it and make it new
					// this will also transfer cellEnds for empty cells but this is not a problem since we won't use that
					asyncCellIndicesUpload(cellsBurst_begin, cellsBurst_end);
					cellsBurst_begin = cell;
					cellsBurst_end = cell + 1;
				}
			}

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

			// cellStart and cellEnd on self
			uint selfCellStart;
			uint selfCellEnd;

			// if it is empty, we might only update cellStarts
			if (peerCellStart == EMPTY_CELL) {

				if (gdata->nextCommand == APPEND_EXTERNAL) {
					// set the cell as empty
					gdata->s_dCellStarts[m_deviceIndex][cell] = EMPTY_CELL;
				}

			} else {
				// cellEnd is exclusive
				uint numPartsInPeerCell = peerCellEnd - peerCellStart;

				if (gdata->nextCommand == UPDATE_EXTERNAL) {
					// if updating, we already have cell indices
					selfCellStart = gdata->s_dCellStarts[m_deviceIndex][cell];
					selfCellEnd = gdata->s_dCellEnds[m_deviceIndex][cell];
				} else {
					// otherwise, we are importing and cell is being appended at current numParts
					selfCellStart = m_numParticles;
					selfCellEnd = m_numParticles + numPartsInPeerCell;
				}

				if (gdata->nextCommand == APPEND_EXTERNAL) {
					// Only when appending, we also need to update cellStarts and cellEnds, both on host (for next updatee) and on device (for accessing cells).
					// Now we write
					//   cellStart[cell] = m_numParticles
					//   cellEnd[cell] = m_numParticles + numPartsInPeerCell
					// in both device memory (used in neib search) and host buffers (partially used in next update: if
					// only the receiving device reads them, they are not used).

					// Update host copy of cellStart and cellEnd. Since it is an external cell,
					// it is unlikely that the host copy will be used, but it is always good to keep
					// indices consistent. The device copy is being updated throught the burst mechanism.
					gdata->s_dCellStarts[m_deviceIndex][cell] = selfCellStart;
					gdata->s_dCellEnds[m_deviceIndex][cell] = selfCellEnd;

					// Also update outer edge segment, if it was empty.
					// NOTE: keeping correctness only if there are no OUTER particles (which we assume)
					if (gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] == EMPTY_SEGMENT)
						gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL] = selfCellStart;

					// finally, update the total number of particles
					m_numParticles += numPartsInPeerCell;
				}

				// now we deal with the actual data in the cells

				if (BURST_IS_EMPTY) {

					// no burst yet; initialize with current cell
					BURST_SET_CURRENT_CELL

				} else
				if (burst_peer_dev_index == peerDevIndex && burst_peer_index_end == peerCellStart) {

					// previous burst is compatible with current cell: extend it
					burst_peer_index_end = peerCellEnd;
					burst_numparts += numPartsInPeerCell;

				} else {
					// Previous burst is not compatible, need to flush the "buffer" and reset it to the current cell.

					// retrieve the appropriate peer memory pointer and transfer the requested data
					if ( (gdata->commandFlags & BUFFER_POS) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(float4);
						peer_dPos = gdata->GPUWORKERS[burst_peer_dev_index]->getDPosBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPosRead : gdata->currentPosWrite );
						peerAsyncTransfer( m_dPos[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dPos[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_VEL) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(float4);
						peer_dVel = gdata->GPUWORKERS[burst_peer_dev_index]->getDVelBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVelRead : gdata->currentVelWrite );
						peerAsyncTransfer( m_dVel[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dVel[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_INFO) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(particleinfo);
						peer_dInfo = gdata->GPUWORKERS[burst_peer_dev_index]->getDInfoBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentInfoRead : gdata->currentInfoWrite );
						peerAsyncTransfer( m_dInfo[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dInfo[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( gdata->commandFlags & BUFFER_FORCES) {
						_size = burst_numparts * sizeof(float4);
						peer_dForces = gdata->GPUWORKERS[burst_peer_dev_index]->getDForceBuffer();
						peerAsyncTransfer( m_dForces + burst_self_index_begin, m_cudaDeviceNumber, peer_dForces + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( gdata->commandFlags & BUFFER_TAU) {
						_size = burst_numparts * sizeof(float2);
						peer_dTaus = gdata->GPUWORKERS[burst_peer_dev_index]->getDTauBuffers();
						for (uint itau = 0; itau < 3; itau++)
							peerAsyncTransfer( m_dTau[itau] + burst_self_index_begin, m_cudaDeviceNumber, peer_dTaus[itau] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_BOUNDELEMENTS) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(float4);
						peer_dBoundElems = gdata->GPUWORKERS[burst_peer_dev_index]->getDBoundElemsBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentBoundElementRead : gdata->currentBoundElementWrite );
						peerAsyncTransfer( m_dBoundElement[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dBoundElems[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_GRADGAMMA) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(float4);
						peer_dGradGamma = gdata->GPUWORKERS[burst_peer_dev_index]->getDGradGammaBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentGradGammaRead : gdata->currentGradGammaWrite );
						peerAsyncTransfer( m_dGradGamma[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dGradGamma[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_VERTICES) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(vertexinfo);
						peer_dVertices = gdata->GPUWORKERS[burst_peer_dev_index]->getDVerticesBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVerticesRead: gdata->currentVerticesWrite );
						peerAsyncTransfer( m_dVertices[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dVertices[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_PRESSURE) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(float);
						peer_dPressure = gdata->GPUWORKERS[burst_peer_dev_index]->getDPressureBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPressureRead : gdata->currentPressureWrite );
						peerAsyncTransfer( m_dPressure[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dPressure[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_TKE) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(float);
						peer_dTKE = gdata->GPUWORKERS[burst_peer_dev_index]->getDTKEBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTKERead : gdata->currentTKEWrite );
						peerAsyncTransfer( m_dTKE[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dTKE[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_EPSILON) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(float);
						peer_dEps = gdata->GPUWORKERS[burst_peer_dev_index]->getDEpsBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentEpsRead : gdata->currentEpsWrite );
						peerAsyncTransfer( m_dEps[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dEps[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_TURBVISC) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(float);
						peer_dTurbVisc = gdata->GPUWORKERS[burst_peer_dev_index]->getDTurbViscBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTurbViscRead : gdata->currentTurbViscWrite );
						peerAsyncTransfer( m_dPos[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dTurbVisc[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}
					if ( (gdata->commandFlags & BUFFER_STRAIN_RATE) && dbl_buffer_specified) {
						_size = burst_numparts * sizeof(float);
						peer_dStrainRate = gdata->GPUWORKERS[burst_peer_dev_index]->getDStrainRateBuffers();
						dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentStrainRateRead : gdata->currentStrainRateWrite );
						peerAsyncTransfer( m_dPos[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
											peer_dStrainRate[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
					}

					// reset burst to current cell
					BURST_SET_CURRENT_CELL
				} // burst flush

			} // if cell is not empty
		} // if cell is external edge and in the same node

	// flush the burst if not empty (should always happen if at least one edge cell is not empty)
	if (!BURST_IS_EMPTY) {

		// transfer the requested data
		if ( (gdata->commandFlags & BUFFER_POS) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(float4);
			peer_dPos = gdata->GPUWORKERS[burst_peer_dev_index]->getDPosBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPosRead : gdata->currentPosWrite );
			peerAsyncTransfer( m_dPos[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dPos[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_VEL) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(float4);
			peer_dVel = gdata->GPUWORKERS[burst_peer_dev_index]->getDVelBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVelRead : gdata->currentVelWrite );
			peerAsyncTransfer( m_dVel[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dVel[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_INFO) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(particleinfo);
			peer_dInfo = gdata->GPUWORKERS[burst_peer_dev_index]->getDInfoBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentInfoRead : gdata->currentInfoWrite );
			peerAsyncTransfer( m_dInfo[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dInfo[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( gdata->commandFlags & BUFFER_FORCES) {
			_size = burst_numparts * sizeof(float4);
			peer_dForces = gdata->GPUWORKERS[burst_peer_dev_index]->getDForceBuffer();
			peerAsyncTransfer( m_dForces + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dForces + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( gdata->commandFlags & BUFFER_TAU) {
			_size = burst_numparts * sizeof(float2);
			peer_dTaus = gdata->GPUWORKERS[burst_peer_dev_index]->getDTauBuffers();
			for (uint itau = 0; itau < 3; itau++)
				peerAsyncTransfer( m_dTau[itau] + burst_self_index_begin, m_cudaDeviceNumber, peer_dTaus[itau] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_BOUNDELEMENTS) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(float4);
			peer_dBoundElems = gdata->GPUWORKERS[burst_peer_dev_index]->getDBoundElemsBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentBoundElementRead : gdata->currentBoundElementWrite );
			peerAsyncTransfer( m_dBoundElement[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dBoundElems[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_GRADGAMMA) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(float4);
			peer_dGradGamma = gdata->GPUWORKERS[burst_peer_dev_index]->getDGradGammaBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentGradGammaRead : gdata->currentGradGammaWrite );
			peerAsyncTransfer( m_dGradGamma[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dGradGamma[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_VERTICES) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(vertexinfo);
			peer_dVertices = gdata->GPUWORKERS[burst_peer_dev_index]->getDVerticesBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVerticesRead: gdata->currentVerticesWrite );
			peerAsyncTransfer( m_dVertices[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dVertices[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_PRESSURE) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(float);
			peer_dPressure = gdata->GPUWORKERS[burst_peer_dev_index]->getDPressureBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPressureRead : gdata->currentPressureWrite );
			peerAsyncTransfer( m_dPressure[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dPressure[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_TKE) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(float);
			peer_dTKE = gdata->GPUWORKERS[burst_peer_dev_index]->getDTKEBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTKERead : gdata->currentTKEWrite );
			peerAsyncTransfer( m_dTKE[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dTKE[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_EPSILON) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(float);
			peer_dEps = gdata->GPUWORKERS[burst_peer_dev_index]->getDEpsBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentEpsRead : gdata->currentEpsWrite );
			peerAsyncTransfer( m_dEps[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dEps[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_TURBVISC) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(float);
			peer_dTurbVisc = gdata->GPUWORKERS[burst_peer_dev_index]->getDTurbViscBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTurbViscRead : gdata->currentTurbViscWrite );
			peerAsyncTransfer( m_dPos[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dTurbVisc[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}
		if ( (gdata->commandFlags & BUFFER_STRAIN_RATE) && dbl_buffer_specified) {
			_size = burst_numparts * sizeof(float);
			peer_dStrainRate = gdata->GPUWORKERS[burst_peer_dev_index]->getDStrainRateBuffers();
			dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentStrainRateRead : gdata->currentStrainRateWrite );
			peerAsyncTransfer( m_dPos[ dbl_buf_idx ] + burst_self_index_begin, m_cudaDeviceNumber,
								peer_dStrainRate[ dbl_buf_idx ] + burst_peer_index_begin, burst_peer_dev_index, _size);
		}

	} // burst is empty?

	// also flush cell buffer, if any
	if (gdata->nextCommand == APPEND_EXTERNAL && !CELLS_BURST_IS_EMPTY)
		asyncCellIndicesUpload(cellsBurst_begin, cellsBurst_end);

#undef BURST_IS_EMPTY
#undef BURST_SET_CURRENT_CELL
#undef CELLS_BURST_IS_EMPTY

	// cudaMemcpyPeerAsync() is asynchronous with the host. We synchronize at the end to wait for the
	// transfers to be complete.
	cudaDeviceSynchronize();
}

// Import the external edge cells of other nodes to the self device arrays. Can append the cells at the end of the current
// list of particles (APPEND_EXTERNAL) or just update the already appended ones (UPDATE_EXTERNAL), according to the current
// command. When appending, also update cellStarts (device and host), cellEnds (device and host) and segments (host only).
// The arrays to be imported must be specified in the command flags. Currently supports pos, vel, info, forces and tau; for the
// double buffered arrays, it is mandatory to specify also the buffer to be used (read or write). This information is ignored
// for the non-buffered arrays (e.g. forces).
// The data is transferred in bursts of consecutive cells when possible.
// Update: supports also BUFFER_BOUNDELEMENTS, BUFFER_GRADGAMMA, BUFFER_VERTICES, BUFFER_PRESSURE,
// BUFFER_TKE, BUFFER_EPSILON, BUFFER_TURBVISC, BUFFER_STRAIN_RATE
void GPUWorker::importNetworkPeerEdgeCells()
{
	// if next command is not an import nor an append, something wrong is going on
	if (! ( (gdata->nextCommand == APPEND_EXTERNAL) || (gdata->nextCommand == UPDATE_EXTERNAL) ) ) {
		printf("WARNING: importNetworkPeerEdgeCells() was called, but current command is not APPEND nor UPDATE!\n");
		return;
	}

	// is a double buffer specified?
	bool dbl_buffer_specified = ( (gdata->commandFlags & DBLBUFFER_READ ) || (gdata->commandFlags & DBLBUFFER_WRITE) );
	uint dbl_buf_idx;

	// We need to import every cell of the neigbor processes only once. To this aim, we keep a list of recipient
	// ranks who already received the current cell. The list, in form of a bitmap, is reset before iterating on
	// all the neighbor cells
	bool recipient_devices[MAX_DEVICES_PER_CLUSTER];

	// Unlike importing from other devices in the same process, here we need one burst for each potential neighbor device;
	// moreover, we only need the "self" address
	uint burst_self_index_begin[MAX_DEVICES_PER_CLUSTER];
	uint burst_self_index_end[MAX_DEVICES_PER_CLUSTER];
	uint burst_numparts[MAX_DEVICES_PER_CLUSTER]; // redundant with burst_peer_index_end, but cleaner
	bool burst_is_sending[MAX_DEVICES_PER_CLUSTER]; // true for bursts of sends, false for bursts of receives

	// initialize bursts
	for (uint n = 0; n < MAX_DEVICES_PER_CLUSTER; n++) {
		burst_self_index_begin[n] = 0;
		burst_self_index_end[n] = 0;
		burst_numparts[n] = 0;
		burst_is_sending[n] = false;
	}

	// utility defines to handle the bursts
#define BURST_IS_EMPTY (burst_numparts[otherDeviceGlobalDevIdx] == 0)
#define BURST_SET_CURRENT_CELL \
	burst_self_index_begin[otherDeviceGlobalDevIdx] = curr_cell_start; \
	burst_self_index_end[otherDeviceGlobalDevIdx] = curr_cell_start + partsInCurrCell; \
	burst_numparts[otherDeviceGlobalDevIdx] = partsInCurrCell; \
	burst_is_sending[otherDeviceGlobalDevIdx] = is_sending;

	// While we iterate on the cells we refer as "curr" to the cell indexed by the outer cycles and as "neib"
	// to the neib cell indexed by the inner cycles. Either cell could belong to current process (so the bools
	// curr_mine and neib_mine); one should not think that neib_cell is necessary in a neib device or process.
	// Instead, otherDeviceGlobalDevIdx, which is used to handle the bursts, is always the global index of the
	// "other" device.

	// iterate on all cells
	for (uint lin_curr_cell = 0; lin_curr_cell < m_nGridCells; lin_curr_cell++) {

		// we will need the 3D coords as well
		int3 curr_cell = gdata->reverseGridHashHost(lin_curr_cell);

		// optimization: if not edging, continue
		if (m_hCompactDeviceMap[lin_curr_cell] == CELLTYPE_OUTER_CELL_SHIFTED ||
			m_hCompactDeviceMap[lin_curr_cell] == CELLTYPE_INNER_CELL_SHIFTED ) continue;

		// reset the list for recipient neib processes
		for (uint d = 0; d < MAX_DEVICES_PER_CLUSTER; d++)
			recipient_devices[d] = false;

		uint curr_cell_globalDevIdx = gdata->s_hDeviceMap[lin_curr_cell];
		uchar curr_cell_rank = gdata->RANK( curr_cell_globalDevIdx );
		if ( curr_cell_rank >= gdata->mpi_nodes ) {
			printf("FATAL: cell %u seems to belong to rank %u, but max is %u; probable memory corruption\n", lin_curr_cell, curr_cell_rank, gdata->mpi_nodes - 1);
			gdata->quit_request = true;
			return;
		}

		bool curr_mine = (curr_cell_globalDevIdx == m_globalDeviceIdx);

		// iterate on neighbors
		for (int dz = -1; dz <= 1; dz++)
			for (int dy = -1; dy <= 1; dy++)
				for (int dx = -1; dx <= 1; dx++) {

					// skip self (should be implicit with dev id check, later)
					if (dx == 0 && dy == 0 && dz == 0) continue;

					// ensure we are inside the grid
					if (curr_cell.x + dx < 0 || curr_cell.x + dx >= gdata->gridSize.x) continue;
					if (curr_cell.y + dy < 0 || curr_cell.y + dy >= gdata->gridSize.y) continue;
					if (curr_cell.z + dz < 0 || curr_cell.z + dz >= gdata->gridSize.z) continue;

					// linearized hash of neib cell
					uint lin_neib_cell = gdata->calcGridHashHost(curr_cell.x + dx, curr_cell.y + dy, curr_cell.z + dz);
					uint neib_cell_globalDevIdx = gdata->s_hDeviceMap[lin_neib_cell];

					// will be set in different way depending on the rank (mine, then local cellStarts, or not, then receive size via network)
					uint partsInCurrCell;
					uint curr_cell_start;

					bool neib_mine = (neib_cell_globalDevIdx == m_globalDeviceIdx);
					uchar neib_cell_rank = gdata->RANK( neib_cell_globalDevIdx );

					// global dev idx of the "other" device
					uint otherDeviceGlobalDevIdx = (curr_cell_globalDevIdx == m_globalDeviceIdx ? neib_cell_globalDevIdx : curr_cell_globalDevIdx);
					bool is_sending = curr_mine;

					// did we already treat the pair (curr_rank <-> neib_rank) for this cell?
					if (recipient_devices[ gdata->GLOBAL_DEVICE_NUM(neib_cell_globalDevIdx) ]) continue;

					// we can skip 1. pairs belonging to the same node 2. pairs where not current nor neib belong to current process
					if (curr_cell_rank == neib_cell_rank || ( !curr_mine && !neib_mine ) ) continue;

					// initialize the number of particles in the cell
					partsInCurrCell = 0;

					// if the cell belong to a neib node and we are appending, there are a few things to handle
					if (neib_mine && gdata->nextCommand == APPEND_EXTERNAL) {

						// prepare to append it to the end of the present array
						curr_cell_start = m_numParticles;

						// receive the size from the cell process
						gdata->networkManager->receiveUint(curr_cell_globalDevIdx, neib_cell_globalDevIdx, &partsInCurrCell);

						// update host cellStarts/Ends
						if (partsInCurrCell > 0) {
							gdata->s_dCellStarts[m_deviceIndex][lin_curr_cell] = curr_cell_start;
							gdata->s_dCellEnds[m_deviceIndex][lin_curr_cell] = curr_cell_start + partsInCurrCell;
						} else
							gdata->s_dCellStarts[m_deviceIndex][lin_curr_cell] = EMPTY_CELL;

						// update device cellStarts/Ends
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
					} // we could "else" here and totally delete the condition (it is opposite to the previuos, but left for clarity)

					// if current cell is mine or if we already imported it and we just need to update, read its size from cellStart/End
					if (curr_mine || gdata->nextCommand == UPDATE_EXTERNAL) {
						// read the size
						curr_cell_start = gdata->s_dCellStarts[m_deviceIndex][lin_curr_cell];

						// set partsInCurrCell
						if (curr_cell_start != EMPTY_CELL)
							partsInCurrCell = gdata->s_dCellEnds[m_deviceIndex][lin_curr_cell] - curr_cell_start;
					}

					// finally, if the cell belongs to current process and we are in appending phase, we want to send its size to the neib process
					if (curr_mine && gdata->nextCommand == APPEND_EXTERNAL)
						gdata->networkManager->sendUint(curr_cell_globalDevIdx, neib_cell_globalDevIdx, &partsInCurrCell);

					// mark the pair current_cell:neib_node as treated
					recipient_devices[ gdata->GLOBAL_DEVICE_NUM(neib_cell_globalDevIdx) ] = true;

					// if the cell is empty, just continue to next one
					if  (curr_cell_start == EMPTY_CELL) continue;

					if (BURST_IS_EMPTY) {

						// burst is empty, so create a new one and continue
						BURST_SET_CURRENT_CELL

					} else
					// condition: curr cell begins when burst end AND burst is in the same direction (send / receive)
					if (burst_self_index_end[otherDeviceGlobalDevIdx] == curr_cell_start && burst_is_sending[otherDeviceGlobalDevIdx] == is_sending) {

						// cell is compatible, extend the burst
						burst_self_index_end[otherDeviceGlobalDevIdx] += partsInCurrCell;
						burst_numparts[otherDeviceGlobalDevIdx] += partsInCurrCell;

					} else {

						// the burst for the current "other" node was non-empty and not compatible; flush it and make a new one

						// Now partsInCurrCell and curr_cell_start are ready; let's handle the actual transfers.

						if ( burst_is_sending[otherDeviceGlobalDevIdx] ) {
							// Current is mine: send the cells to the process holding the neighbor cells
							// (in the following, m_globalDeviceIdx === curr_cell_globalDevIdx)

							if ( (gdata->commandFlags & BUFFER_POS) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPosRead : gdata->currentPosWrite );
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dPos[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_VEL) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVelRead : gdata->currentVelWrite );
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dVel[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_INFO) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentInfoRead : gdata->currentInfoWrite );
								gdata->networkManager->sendShorts(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(ushort*)(m_dInfo[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( gdata->commandFlags & BUFFER_FORCES )
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dForces + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							if ( gdata->commandFlags & BUFFER_TAU)
								for (uint itau = 0; itau < 3; itau++)
									gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 2,
											(float*)(m_dTau[itau] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							if ( (gdata->commandFlags & BUFFER_BOUNDELEMENTS) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentBoundElementRead : gdata->currentBoundElementWrite );
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dBoundElement[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_GRADGAMMA) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentGradGammaRead : gdata->currentGradGammaWrite );
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dGradGamma[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_VERTICES) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVerticesRead: gdata->currentVerticesWrite );
								gdata->networkManager->sendUints(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(uint*)(m_dVertices[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_PRESSURE) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPressureRead : gdata->currentPressureWrite );
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dPressure[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_TKE) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTKERead : gdata->currentTKEWrite );
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dTKE[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_EPSILON) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentEpsRead : gdata->currentEpsWrite );
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dEps[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_TURBVISC) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTurbViscRead : gdata->currentTurbViscWrite );
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dTurbVisc[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_STRAIN_RATE) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentStrainRateRead : gdata->currentStrainRateWrite );
								gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDeviceGlobalDevIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dStrainRate[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}

						} else {
							// neighbor is mine: receive the cells from the process holding the current cell
							// (in the following, m_globalDeviceIdx === neib_cell_globalDevIdx)

							if ( (gdata->commandFlags & BUFFER_POS) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPosRead : gdata->currentPosWrite );
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dPos[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_VEL) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVelRead : gdata->currentVelWrite );
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dVel[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_INFO) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentInfoRead : gdata->currentInfoWrite );
								gdata->networkManager->receiveShorts(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(ushort*)(m_dInfo[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( gdata->commandFlags & BUFFER_FORCES )
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dForces + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							if ( gdata->commandFlags & BUFFER_TAU)
								for (uint itau = 0; itau < 3; itau++)
									gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 2,
											(float*)(m_dTau[itau] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							if ( (gdata->commandFlags & BUFFER_BOUNDELEMENTS) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentBoundElementRead : gdata->currentBoundElementWrite );
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dBoundElement[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_GRADGAMMA) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentGradGammaRead : gdata->currentGradGammaWrite );
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(float*)(m_dGradGamma[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_VERTICES) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVerticesRead: gdata->currentVerticesWrite );
								gdata->networkManager->receiveUints(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 4,
										(uint*)(m_dVertices[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_PRESSURE) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPressureRead : gdata->currentPressureWrite );
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dPressure[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_TKE) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTKERead : gdata->currentTKEWrite );
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dTKE[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_EPSILON) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentEpsRead : gdata->currentEpsWrite );
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dEps[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_TURBVISC) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTurbViscRead : gdata->currentTurbViscWrite );
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dTurbVisc[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}
							if ( (gdata->commandFlags & BUFFER_STRAIN_RATE) && dbl_buffer_specified) {
								dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentStrainRateRead : gdata->currentStrainRateWrite );
								gdata->networkManager->receiveFloats(otherDeviceGlobalDevIdx, m_globalDeviceIdx, burst_numparts[otherDeviceGlobalDevIdx] * 1,
										(float*)(m_dStrainRate[ dbl_buf_idx ] + burst_self_index_begin[otherDeviceGlobalDevIdx]) );
							}

						}

						BURST_SET_CURRENT_CELL
					} // end of flushing and resetting the burst

				} // iterate on neighbor cells
	} // iterate on cells

	// here: flush all the non-empty bursts
	for (uint otherDevGlobalIdx = 0; otherDevGlobalIdx < MAX_DEVICES_PER_CLUSTER; otherDevGlobalIdx++)
		if ( burst_numparts[otherDevGlobalIdx] > 0) {

			if ( burst_is_sending[otherDevGlobalIdx] ) {
				// Current is mine: send the cells to the process holding the neighbor cells

				if ( (gdata->commandFlags & BUFFER_POS) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPosRead : gdata->currentPosWrite );
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dPos[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_VEL) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVelRead : gdata->currentVelWrite );
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dVel[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_INFO) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentInfoRead : gdata->currentInfoWrite );
					gdata->networkManager->sendShorts(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(ushort*)(m_dInfo[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( gdata->commandFlags & BUFFER_FORCES )
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dForces + burst_self_index_begin[otherDevGlobalIdx]) );
				if ( gdata->commandFlags & BUFFER_TAU)
					for (uint itau = 0; itau < 3; itau++)
						gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 2,
								(float*)(m_dTau[itau] + burst_self_index_begin[otherDevGlobalIdx]) );
				if ( (gdata->commandFlags & BUFFER_BOUNDELEMENTS) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentBoundElementRead : gdata->currentBoundElementWrite );
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dBoundElement[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_GRADGAMMA) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentGradGammaRead : gdata->currentGradGammaWrite );
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dGradGamma[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_VERTICES) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVerticesRead: gdata->currentVerticesWrite );
					gdata->networkManager->sendUints(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(uint*)(m_dVertices[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_PRESSURE) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPressureRead : gdata->currentPressureWrite );
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dPressure[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_TKE) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTKERead : gdata->currentTKEWrite );
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dTKE[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_EPSILON) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentEpsRead : gdata->currentEpsWrite );
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dEps[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_TURBVISC) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTurbViscRead : gdata->currentTurbViscWrite );
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dTurbVisc[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_STRAIN_RATE) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentStrainRateRead : gdata->currentStrainRateWrite );
					gdata->networkManager->sendFloats(m_globalDeviceIdx, otherDevGlobalIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dStrainRate[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}

			} else {
				// neighbor is mine: receive the cells from the process holding the current cell

				if ( (gdata->commandFlags & BUFFER_POS) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPosRead : gdata->currentPosWrite );
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dPos[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_VEL) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVelRead : gdata->currentVelWrite );
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dVel[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_INFO) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentInfoRead : gdata->currentInfoWrite );
					gdata->networkManager->receiveShorts(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(ushort*)(m_dInfo[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( gdata->commandFlags & BUFFER_FORCES )
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dForces + burst_self_index_begin[otherDevGlobalIdx]) );
				if ( gdata->commandFlags & BUFFER_TAU)
						for (uint itau = 0; itau < 3; itau++)
							gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 2,
									(float*)(m_dTau[itau] + burst_self_index_begin[otherDevGlobalIdx]) );
				if ( (gdata->commandFlags & BUFFER_BOUNDELEMENTS) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentBoundElementRead : gdata->currentBoundElementWrite );
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dBoundElement[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_GRADGAMMA) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentGradGammaRead : gdata->currentGradGammaWrite );
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(float*)(m_dGradGamma[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_VERTICES) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentVerticesRead: gdata->currentVerticesWrite );
					gdata->networkManager->receiveUints(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 4,
							(uint*)(m_dVertices[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_PRESSURE) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentPressureRead : gdata->currentPressureWrite );
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dPressure[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_TKE) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTKERead : gdata->currentTKEWrite );
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dTKE[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_EPSILON) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentEpsRead : gdata->currentEpsWrite );
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dEps[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_TURBVISC) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentTurbViscRead : gdata->currentTurbViscWrite );
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dTurbVisc[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}
				if ( (gdata->commandFlags & BUFFER_STRAIN_RATE) && dbl_buffer_specified) {
					dbl_buf_idx = (gdata->commandFlags & DBLBUFFER_READ ? gdata->currentStrainRateRead : gdata->currentStrainRateWrite );
					gdata->networkManager->receiveFloats(otherDevGlobalIdx, m_globalDeviceIdx, burst_numparts[otherDevGlobalIdx] * 1,
							(float*)(m_dStrainRate[ dbl_buf_idx ] + burst_self_index_begin[otherDevGlobalIdx]) );
				}

			}
		}

	// don't need the defines anymore
#undef BURST_IS_EMPTY
#undef BURST_SET_CURRENT_CELL

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
	const size_t floatSize = sizeof(float) * m_numAllocatedParticles;
	const size_t float2Size = sizeof(float2) * m_numAllocatedParticles;
	const size_t float3Size = sizeof(float3) * m_numAllocatedParticles;
	const size_t float4Size = sizeof(float4) * m_numAllocatedParticles;
	const size_t infoSize = sizeof(particleinfo) * m_numAllocatedParticles;
	const size_t vertexSize = sizeof(vertexinfo) * m_numAllocatedParticles;
	const size_t intSize = sizeof(uint) * m_numAllocatedParticles;
	const size_t uintCellsSize = sizeof(uint) * m_nGridCells;
	const size_t neibslistSize = sizeof(neibdata)*m_simparams->maxneibsnum*m_numAllocatedParticles;
	const size_t hashSize = sizeof(hashKey) * m_numAllocatedParticles;
	const size_t segmentsSize = sizeof(uint) * 4; // 4 = types of cells
	//const size_t neibslistSize = sizeof(uint) * 128 * m_numAlocatedParticles;
	//const size_t sliceArraySize = sizeof(uint) * m_gridSize.PSA;

	size_t allocated = 0;

	CUDA_SAFE_CALL(cudaMalloc(&m_dForces, float4Size));
	CUDA_SAFE_CALL(cudaMemset(m_dForces, 0, float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc(&m_dXsph, float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc(&m_dPos[0], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc(&m_dPos[1], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc(&m_dVel[0], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc(&m_dVel[1], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc(&m_dInfo[0], infoSize));
	allocated += infoSize;

	CUDA_SAFE_CALL(cudaMalloc(&m_dInfo[1], infoSize));
	allocated += infoSize;

	// Free surface detection
	if (m_simparams->savenormals) {
		CUDA_SAFE_CALL(cudaMalloc(&m_dNormals, float4Size));
		allocated += float4Size;
	} else {
		m_dNormals = NULL;
	}

	if (m_simparams->vorticity) {
		CUDA_SAFE_CALL(cudaMalloc(&m_dVort, float3Size));
		allocated += float3Size;
	} else {
		m_dVort = NULL;
	}

	if (m_simparams->visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaMalloc(&m_dTau[0], float2Size));
		allocated += float2Size;

		CUDA_SAFE_CALL(cudaMalloc(&m_dTau[1], float2Size));
		allocated += float2Size;

		CUDA_SAFE_CALL(cudaMalloc(&m_dTau[2], float2Size));
		allocated += float2Size;
	} else {
		m_dTau[0] = m_dTau[1] = m_dTau[2];
	}

	for (uint i = 0; i < 2; ++i) {
		if (m_simparams->boundarytype == MF_BOUNDARY) {
			CUDA_SAFE_CALL(cudaMalloc(&m_dGradGamma[i], float4Size));
			allocated += float4Size;

			CUDA_SAFE_CALL(cudaMalloc(&m_dBoundElement[i], float4Size));
			allocated += float4Size;

			CUDA_SAFE_CALL(cudaMalloc(&m_dVertices[i], vertexSize));
			allocated += vertexSize;

			CUDA_SAFE_CALL(cudaMalloc(&m_dPressure[i], floatSize));
			allocated += floatSize;
		} else {
			m_dGradGamma[i] = m_dBoundElement[i] = NULL;
			m_dVertices[i] = NULL; m_dPressure[i] = NULL;
		}

		if (m_simparams->visctype == KEPSVISC) {
			CUDA_SAFE_CALL(cudaMalloc(&m_dTKE[i], floatSize));
			allocated += floatSize;

			CUDA_SAFE_CALL(cudaMalloc(&m_dEps[i], floatSize));
			allocated += floatSize;

			CUDA_SAFE_CALL(cudaMalloc(&m_dTurbVisc[i], floatSize));
			allocated += floatSize;

			CUDA_SAFE_CALL(cudaMalloc(&m_dStrainRate[i], floatSize));
			allocated += floatSize;
		} else {
			m_dTKE[i] = m_dEps[i] = m_dTurbVisc[i] = m_dStrainRate[i] = NULL;
		}
	}

	if (m_simparams->boundarytype == MF_BOUNDARY) {
		CUDA_SAFE_CALL(cudaMalloc(&m_dInversedParticleIndex, hashSize));
		allocated += hashSize;
	} else {
		m_dInversedParticleIndex = NULL;
	}


	if (m_simparams->visctype == KEPSVISC) {
		CUDA_SAFE_CALL(cudaMalloc(&m_dDkDe, float2Size));
		allocated += float2Size;
	} else {
		m_dDkDe = NULL;
	}

	CUDA_SAFE_CALL(cudaMalloc(&m_dParticleHash, hashSize));
	allocated += hashSize;

	CUDA_SAFE_CALL(cudaMalloc(&m_dParticleIndex, intSize));
	allocated += intSize;

	CUDA_SAFE_CALL(cudaMalloc(&m_dCellStart, uintCellsSize));
	allocated += uintCellsSize;

	CUDA_SAFE_CALL(cudaMalloc(&m_dCellEnd, uintCellsSize));
	allocated += uintCellsSize;

	//CUDA_SAFE_CALL(cudaMalloc(&m_dSliceStart, sliceArraySize));
	//allocated += sliceArraySize;

	CUDA_SAFE_CALL(cudaMalloc(&m_dNeibsList, neibslistSize));
	CUDA_SAFE_CALL(cudaMemset(m_dNeibsList, 0xffffffff, neibslistSize));
	allocated += neibslistSize;

	// TODO: an array of uchar would suffice
	CUDA_SAFE_CALL(cudaMalloc(&m_dCompactDeviceMap, uintCellsSize));
	// initialize anyway for single-GPU simulations
	CUDA_SAFE_CALL(cudaMemset(m_dCompactDeviceMap, 0, uintCellsSize));
	allocated += uintCellsSize;

	CUDA_SAFE_CALL(cudaMalloc(&m_dSegmentStart, segmentsSize));
	// ditto
	CUDA_SAFE_CALL(cudaMemset(m_dSegmentStart, 0, segmentsSize));
	allocated += segmentsSize;

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

	if (m_simparams->dtadapt) {
		// for the allocation we use m_numPartsFmax computed from m_numAlocatedParticles;
		// after forces we use an updated value instead (the numblocks of forces)
		uint fmaxElements = getFmaxElements(m_numAllocatedParticles);
		const uint fmaxTableSize = fmaxElements*sizeof(float);

		CUDA_SAFE_CALL(cudaMalloc(&m_dCfl, fmaxTableSize));
		CUDA_SAFE_CALL(cudaMemset(m_dCfl, 0, fmaxTableSize));
		allocated += fmaxTableSize;

		if (m_simparams->boundarytype == MF_BOUNDARY) {
			CUDA_SAFE_CALL(cudaMalloc(&m_dCflGamma, fmaxTableSize));
			CUDA_SAFE_CALL(cudaMemset(m_dCflGamma, 0, fmaxTableSize));
			allocated += fmaxTableSize;
		}

		if(m_simparams->visctype == KEPSVISC) {
			CUDA_SAFE_CALL(cudaMalloc(&m_dCflTVisc, fmaxTableSize));
			CUDA_SAFE_CALL(cudaMemset(m_dCflTVisc, 0, fmaxTableSize));
			allocated += fmaxTableSize;
		}

		const uint tempCflSize = getFmaxTempStorageSize(fmaxElements);
		CUDA_SAFE_CALL(cudaMalloc(&m_dTempCfl, tempCflSize));
		CUDA_SAFE_CALL(cudaMemset(m_dTempCfl, 0, tempCflSize));

		allocated += tempCflSize;
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

	for (uint i = 0 ; i < 2 ; ++i) {
		if (m_simparams->boundarytype == MF_BOUNDARY) {
			CUDA_SAFE_CALL(cudaFree(m_dGradGamma[i]));
			CUDA_SAFE_CALL(cudaFree(m_dBoundElement[i]));
			CUDA_SAFE_CALL(cudaFree(m_dVertices[i]));
			CUDA_SAFE_CALL(cudaFree(m_dPressure[i]));
		}

		if (m_simparams->visctype == KEPSVISC) {
			CUDA_SAFE_CALL(cudaFree(m_dTKE[i]));
			CUDA_SAFE_CALL(cudaFree(m_dEps[i]));
			CUDA_SAFE_CALL(cudaFree(m_dTurbVisc[i]));
			CUDA_SAFE_CALL(cudaFree(m_dStrainRate[i]));
		}
	}

	if (m_simparams->boundarytype == MF_BOUNDARY) {
		CUDA_SAFE_CALL(cudaFree(m_dInversedParticleIndex));
	}


	if (m_simparams->visctype == KEPSVISC) {
		CUDA_SAFE_CALL(cudaFree(m_dDkDe));
	}

	CUDA_SAFE_CALL(cudaFree(m_dParticleHash));
	CUDA_SAFE_CALL(cudaFree(m_dParticleIndex));
	CUDA_SAFE_CALL(cudaFree(m_dCellStart));
	CUDA_SAFE_CALL(cudaFree(m_dCellEnd));
	CUDA_SAFE_CALL(cudaFree(m_dNeibsList));

	CUDA_SAFE_CALL(cudaFree(m_dCompactDeviceMap));
	CUDA_SAFE_CALL(cudaFree(m_dSegmentStart));

	if (m_simparams->dtadapt) {
		CUDA_SAFE_CALL(cudaFree(m_dCfl));
		CUDA_SAFE_CALL(cudaFree(m_dTempCfl));
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

void GPUWorker::createStreams()
{
	cudaStreamCreate(&m_asyncD2HCopiesStream);
	cudaStreamCreate(&m_asyncH2DCopiesStream);
	cudaStreamCreate(&m_asyncPeerCopiesStream);
}

void GPUWorker::destroyStreams()
{
	// destroy streams
	cudaStreamDestroy(m_asyncD2HCopiesStream);
	cudaStreamDestroy(m_asyncH2DCopiesStream);
	cudaStreamDestroy(m_asyncPeerCopiesStream);
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

const float4* const* GPUWorker::getDPosBuffers() const
{
	return (const float4**)m_dPos;
}

const float4* const* GPUWorker::getDVelBuffers() const
{
	return (const float4**)m_dVel;
}

const particleinfo* const* GPUWorker::getDInfoBuffers() const
{
	return (const particleinfo**)m_dInfo;
}

const float4* GPUWorker::getDForceBuffer() const
{
	return (const float4*)m_dForces;
}

const float2* const* GPUWorker::getDTauBuffers() const
{
	return (const float2**)m_dTau;
}

const hashKey* GPUWorker::getDHashBuffer() const
{ return m_dParticleHash; }

const uint* GPUWorker::getDPartIndexBuffer() const
{ return m_dParticleIndex; }

const float4* const* GPUWorker::getDBoundElemsBuffers() const
{ return m_dBoundElement; }

const float4* const* GPUWorker::getDGradGammaBuffers() const
{ return m_dGradGamma; }

const vertexinfo* const* GPUWorker::getDVerticesBuffers() const
{ return m_dVertices; }

const float* const* GPUWorker::getDPressureBuffers() const
{ return m_dPressure; }

const float* const* GPUWorker::getDTKEBuffers() const
{ return m_dTKE; }

const float* const* GPUWorker::getDEpsBuffers() const
{ return m_dEps; }

const float* const* GPUWorker::getDTurbViscBuffers() const
{ return m_dTurbVisc; }

const float* const* GPUWorker::getDStrainRateBuffers() const
{ return m_dStrainRate; }


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

		// upload centers of gravity of the bodies
	instance->uploadBodiesCentersOfGravity();

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

	// init streams for async memcpys (only useful for multigpu?)
	instance->createStreams();

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
					instance->importPeerEdgeCells();
				if (MULTI_NODE)
					instance->importNetworkPeerEdgeCells();
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
			case MF_INIT_GAMMA:
				//printf(" T %d issuing MF_INIT_GAMMA\n", deviceIndex);
				instance->kernel_initGradGamma();
				break;
			case MF_UPDATE_GAMMA:
				//printf(" T %d issuing MF_UPDATE_GAMMA\n", deviceIndex);
				instance->kernel_updateGamma();
				break;
			case MF_UPDATE_POS:
				//printf(" T %d issuing MF_UPDATE_POS\n", deviceIndex);
				instance->kernel_updatePositions();
				break;
			case MF_CALC_BOUND_CONDITIONS:
				//printf(" T %d issuing MF_CALC_BOUND_CONDITIONS\n", deviceIndex);
				instance->kernel_dynamicBoundaryConditions();
				break;
			case MF_UPDATE_BOUND_VALUES:
				//printf(" T %d issuing MF_UPDATE_BOUND_VALUES\n", deviceIndex);
				instance->kernel_updateValuesAtBoundaryElements();
				break;
			case SPS:
				//printf(" T %d issuing SPS\n", deviceIndex);
				instance->kernel_sps();
				break;
			case MEAN_STRAIN:
				//printf(" T %d issuing MEAN_STRAIN\n", deviceIndex);
				instance->kernel_meanStrain();
			case REDUCE_BODIES_FORCES:
				//printf(" T %d issuing REDUCE_BODIES_FORCES\n", deviceIndex);
				instance->kernel_reduceRBForces();
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
			case UPLOAD_OBJECTS_CG:
				//printf(" T %d issuing UPLOAD_OBJECTS_CG\n", deviceIndex);
				instance->uploadBodiesCentersOfGravity();
				break;
			case UPLOAD_OBJECTS_MATRICES:
				//printf(" T %d issuing UPLOAD_OBJECTS_CG\n", deviceIndex);
				instance->uploadBodiesTransRotMatrices();
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

	// destroy streams
	instance->destroyStreams();

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

	calcHash(	m_dPos[gdata->currentPosRead],
				m_dParticleHash,
				m_dParticleIndex,
				m_dInfo[gdata->currentInfoRead],
#if HASH_KEY_SIZE >= 64
				m_dCompactDeviceMap,
#endif
				m_numParticles,
				m_simparams->periodicbound);
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
#if HASH_KEY_SIZE >= 64
							m_dSegmentStart,
#endif
							m_dPos[gdata->currentPosWrite],		 // output: sorted positions
							m_dVel[gdata->currentVelWrite],		 // output: sorted velocities
							m_dInfo[gdata->currentInfoWrite],		 // output: sorted info
							m_dBoundElement[gdata->currentBoundElementWrite],	// output: sorted boundary elements
							m_dGradGamma[gdata->currentGradGammaWrite],		// output: sorted gradient gamma
							m_dVertices[gdata->currentVerticesWrite],		// output: sorted vertices
							m_dPressure[gdata->currentPressureWrite],		// output: sorted pressure
							m_dTKE[gdata->currentTKEWrite],				// output: k for k-e model
							m_dEps[gdata->currentEpsWrite],				// output: e for k-e model
							m_dTurbVisc[gdata->currentTurbViscWrite],	// output: eddy viscosity
							m_dStrainRate[gdata->currentStrainRateWrite],	// output: strain rate
							m_dParticleHash,
							m_dParticleIndex,  // input: sorted particle indices
							m_dPos[gdata->currentPosRead],		 // input: sorted position array
							m_dVel[gdata->currentVelRead],		 // input: sorted velocity array
							m_dInfo[gdata->currentInfoRead],		 // input: sorted info array
							m_dBoundElement[gdata->currentBoundElementRead],	// input: sorted boundary elements
							m_dGradGamma[gdata->currentGradGammaRead],		// input: sorted gradient gamma
							m_dVertices[gdata->currentVerticesRead],		// input: sorted vertices
							m_dPressure[gdata->currentPressureRead],		// input: sorted pressure
							m_dTKE[gdata->currentTKERead],				// input: k for k-e model
							m_dEps[gdata->currentEpsRead],				// input: e for k-e model
							m_dTurbVisc[gdata->currentTurbViscRead],		// input: eddy viscosity
							m_dStrainRate[gdata->currentStrainRateRead],	// output: strain rate
							m_numParticles,
							m_nGridCells,
							m_dInversedParticleIndex);
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

	// if we have objects potentially shared across different devices, must reset their forces
	// and torques to avoid spurious contributions
	if (m_simparams->numODEbodies > 0 && MULTI_DEVICE) {
		uint bodiesPartsSize = m_numBodiesParticles * sizeof(float4);
		CUDA_SAFE_CALL(cudaMemset(m_dRbForces, 0.0F, bodiesPartsSize));
		CUDA_SAFE_CALL(cudaMemset(m_dRbTorques, 0.0F, bodiesPartsSize));
	}

	// first step
	if (numPartsToElaborate > 0 && firstStep)
		returned_dt = forces(  m_dPos[gdata->currentPosRead],   // pos(n)
						m_dVel[gdata->currentVelRead],   // vel(n)
						m_dForces,					// f(n
						m_dGradGamma[gdata->currentGradGammaRead],
						m_dBoundElement[gdata->currentBoundElementRead],
						m_dPressure[gdata->currentPressureRead],
						m_dRbForces,
						m_dRbTorques,
						m_dXsph,
						m_dInfo[gdata->currentInfoRead],
						m_dParticleHash,
						m_dCellStart,
						m_dNeibsList,
						m_numParticles,
						numPartsToElaborate,
						gdata->problem->m_deltap,
						m_simparams->slength,
						gdata->dt, // m_dt,
						m_simparams->dtadapt,
						m_simparams->dtadaptfactor,
						m_simparams->xsph,
						m_simparams->kerneltype,
						m_simparams->influenceRadius,
						m_simparams->visctype,
						m_physparams->visccoeff,
						m_dTurbVisc[gdata->currentTurbViscRead],	// nu_t(n)
						m_dTKE[gdata->currentTKERead],	// k(n)
						m_dEps[gdata->currentEpsRead],	// e(n)
						m_dDkDe,
						m_dCfl,
						m_dCflGamma,
						m_dCflTVisc,
						m_dTempCfl,
						m_simparams->sph_formulation,
						m_simparams->boundarytype,
						m_simparams->usedem);
	else
	// second step
	if (numPartsToElaborate > 0 && !firstStep)
		returned_dt = forces(  m_dPos[gdata->currentPosWrite],  // pos(n+1/2)
						m_dVel[gdata->currentVelWrite],  // vel(n+1/2)
						m_dForces,					// f(n+1/2)
						m_dGradGamma[gdata->currentGradGammaRead],
						m_dBoundElement[gdata->currentBoundElementRead],
						m_dPressure[gdata->currentPressureRead],
						m_dRbForces,
						m_dRbTorques,
						m_dXsph,
						m_dInfo[gdata->currentInfoRead],
						m_dParticleHash,
						m_dCellStart,
						m_dNeibsList,
						m_numParticles,
						numPartsToElaborate,
						gdata->problem->m_deltap,
						m_simparams->slength,
						gdata->dt, // m_dt,
						m_simparams->dtadapt,
						m_simparams->dtadaptfactor,
						m_simparams->xsph,
						m_simparams->kerneltype,
						m_simparams->influenceRadius,
						m_simparams->visctype,
						m_physparams->visccoeff,
						m_dTurbVisc[gdata->currentTurbViscRead],	// nu_t(n+1/2)
						m_dTKE[gdata->currentTKEWrite],	// k(n+1/2)
						m_dEps[gdata->currentEpsWrite],	// e(n+1/2)
						m_dDkDe,
						m_dCfl,
						m_dCflGamma,
						m_dCflTVisc,
						m_dTempCfl,
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
				m_dParticleHash,
				m_dVel[gdata->currentVelRead],	// vel(n)
				m_dTKE[gdata->currentTKERead],	// k(n)
				m_dEps[gdata->currentEpsRead],	// e(n)
				m_dInfo[gdata->currentInfoRead], //particleInfo
				m_dForces,						// f(n+1/2)
				m_dDkDe,					// dkde(n)
				m_dXsph,
				m_dPos[gdata->currentPosWrite],	// pos(n+1) = pos(n) + velc(n+1/2)*dt
				m_dVel[gdata->currentVelWrite],	// vel(n+1) = vel(n) + f(n+1/2)*dt
				m_dTKE[gdata->currentTKEWrite],	// k(n+1/2) = k(n) + dkde(n).x*dt/2
				m_dEps[gdata->currentEpsWrite],	// e(n+1/2) = e(n) + dkde(n).y*dt/2
				m_numParticles,
				numPartsToElaborate,
				gdata->dt, // m_dt,
				gdata->dt/2.0f, // m_dt/2.0,
				1,
				gdata->t + gdata->dt / 2.0f, // + m_dt,
				m_simparams->xsph);
	else
		euler(  m_dPos[gdata->currentPosRead],   // pos(n)
				m_dParticleHash,
				m_dVel[gdata->currentVelRead],   // vel(n)
				m_dTKE[gdata->currentTKERead],	// k(n)
				m_dEps[gdata->currentEpsRead],	// e(n)
				m_dInfo[gdata->currentInfoRead], //particleInfo
				m_dForces,					// f(n+1/2)
				m_dDkDe,					// dkde(n+1/2)
				m_dXsph,
				m_dPos[gdata->currentPosWrite],  // pos(n+1) = pos(n) + velc(n+1/2)*dt
				m_dVel[gdata->currentVelWrite],  // vel(n+1) = vel(n) + f(n+1/2)*dt
				m_dTKE[gdata->currentTKEWrite],	// k(n+1) = k(n) + dkde(n+1/2).x*dt
				m_dEps[gdata->currentEpsWrite],	// e(n+1) = e(n) + dkde(n+1/2).y*dt
				m_numParticles,
				numPartsToElaborate,
				gdata->dt, // m_dt,
				gdata->dt/2.0f, // m_dt/2.0,
				2,
				gdata->t + gdata->dt,// + m_dt,
				m_simparams->xsph);
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
			m_dParticleHash,
			m_dCellStart,
			m_dNeibsList,
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

	shepard(m_dPos[gdata->currentPosRead],
			m_dVel[gdata->currentVelRead],
			m_dVel[gdata->currentVelWrite],
			m_dInfo[gdata->currentInfoRead],
			m_dParticleHash,
			m_dCellStart,
			m_dNeibsList,
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
	vorticity(	m_dPos[gdata->currentPosRead],
				m_dVel[gdata->currentVelRead],
				m_dVort,
				m_dInfo[gdata->currentInfoRead],
				m_dParticleHash,
				m_dCellStart,
				m_dNeibsList,
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

	surfaceparticle(m_dPos[gdata->currentPosRead],
					m_dVel[gdata->currentVelRead],
					m_dNormals,
					m_dInfo[gdata->currentInfoRead],
					m_dInfo[gdata->currentInfoWrite],
					m_dParticleHash,
					m_dCellStart,
					m_dNeibsList,
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

	// pos and vel are read from curren*Read on the first step,
	// from current*Write on the second
	bool firstStep = (gdata->commandFlags & INTEGRATOR_STEP_1);
	uint posRead = firstStep ? gdata->currentPosRead : gdata->currentPosWrite;
	uint velRead = firstStep ? gdata->currentVelRead : gdata->currentVelWrite;

	sps(m_dTau,
		m_dPos[posRead],
		m_dVel[velRead],
		m_dInfo[gdata->currentInfoRead],
		m_dParticleHash,
		m_dCellStart,
		m_dNeibsList,
		m_numParticles,
		numPartsToElaborate,
		m_simparams->slength,
		m_simparams->kerneltype,
		m_simparams->influenceRadius);
}

void GPUWorker::kernel_meanStrain()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	// pos and vel are read from curren*Read on the first step,
	// from current*Write on the second
	bool firstStep = (gdata->commandFlags & INTEGRATOR_STEP_1);
	uint posRead = firstStep ? gdata->currentPosRead : gdata->currentPosWrite;
	uint velRead = firstStep ? gdata->currentVelRead : gdata->currentVelWrite;

	mean_strain_rate(
		m_dStrainRate[gdata->currentStrainRateRead],
		m_dPos[posRead],
		m_dVel[velRead],
		m_dInfo[gdata->currentInfoRead],
		m_dParticleHash,
		m_dCellStart,
		m_dNeibsList,
		m_dGradGamma[gdata->currentGradGammaRead],
		m_dBoundElement[gdata->currentBoundElementRead],
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
	bool secondStep = (gdata->commandFlags & INTEGRATOR_STEP_2);
	uint velRead = secondStep ? gdata->currentVelWrite : gdata->currentVelRead;
	uint tkeRead = secondStep ? gdata->currentTKEWrite : gdata->currentTKERead;
	uint epsRead = secondStep ? gdata->currentEpsWrite : gdata->currentEpsRead;

	updateBoundValues(	m_dVel[velRead],
				m_dPressure[gdata->currentPressureRead],
				m_dTKE[tkeRead],
				m_dEps[epsRead],
				m_dVertices[gdata->currentVerticesRead],
				m_dInfo[gdata->currentInfoRead],
				m_numParticles,
				numPartsToElaborate,
				true);
}

void GPUWorker::kernel_dynamicBoundaryConditions()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	// pos, vel, tke, eps are read from current*Read, except
	// on the second step, whe they are read from current*Write
	bool secondStep = (gdata->commandFlags & INTEGRATOR_STEP_2);
	uint posRead = secondStep ? gdata->currentPosWrite : gdata->currentPosRead;
	uint velRead = secondStep ? gdata->currentVelWrite : gdata->currentVelRead;
	uint tkeRead = secondStep ? gdata->currentTKEWrite : gdata->currentTKERead;
	uint epsRead = secondStep ? gdata->currentEpsWrite : gdata->currentEpsRead;

	dynamicBoundConditions(	m_dPos[posRead],
				m_dVel[velRead],
				m_dPressure[gdata->currentPressureRead],
				m_dTKE[tkeRead],
				m_dEps[epsRead],
				m_dInfo[gdata->currentInfoRead],
				m_dParticleHash,
				m_dCellStart,
				m_dNeibsList,
				m_numParticles,
				numPartsToElaborate,
				gdata->problem->m_deltap,
				m_simparams->slength,
				m_simparams->kerneltype,
				m_simparams->influenceRadius);
}

void GPUWorker::kernel_initGradGamma()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	initGradGamma(	m_dPos[gdata->currentPosRead],
				m_dPos[gdata->currentPosWrite],
				m_dVel[gdata->currentVelWrite],
				m_dInfo[gdata->currentInfoRead],
				m_dBoundElement[gdata->currentBoundElementRead],
				m_dGradGamma[gdata->currentGradGammaWrite],
				m_dParticleHash,
				m_dCellStart,
				m_dNeibsList,
				m_numParticles,
				numPartsToElaborate,
				gdata->problem->m_deltap,
				m_simparams->slength,
				m_simparams->influenceRadius,
				m_simparams->kerneltype);
}

void GPUWorker::kernel_updateGamma()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	bool initStep = (gdata->commandFlags & INITIALIZATION_STEP);
	// during initStep we use currentVelRead, else currentVelWrite
	uint velRead = initStep ? gdata->currentVelRead : gdata->currentVelWrite;

	updateGamma(m_dPos[gdata->currentPosRead],
				m_dPos[gdata->currentPosWrite],
				m_dVel[velRead],
				m_dInfo[gdata->currentInfoRead],
				m_dBoundElement[gdata->currentBoundElementRead],
				m_dGradGamma[gdata->currentGradGammaRead],
				m_dGradGamma[gdata->currentGradGammaWrite],
				m_dParticleHash,
				m_dCellStart,
				m_dNeibsList,
				m_numParticles,
				numPartsToElaborate,
				m_simparams->slength,
				m_simparams->influenceRadius,
				gdata->extraCommandArg,
				!initStep, // 0 during init step, else 1
				m_simparams->kerneltype);
}

void GPUWorker::kernel_updatePositions()
{
	uint numPartsToElaborate = (gdata->only_internal ? m_particleRangeEnd : m_numParticles);

	// is the device empty? (unlikely but possible before LB kicks in)
	if (numPartsToElaborate == 0) return;

	updatePositions(	m_dPos[gdata->currentPosRead],
					m_dPos[gdata->currentPosWrite],
					m_dVel[gdata->currentVelRead],
					m_dInfo[gdata->currentInfoRead],
					gdata->extraCommandArg,
					m_numParticles,
					numPartsToElaborate);
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

