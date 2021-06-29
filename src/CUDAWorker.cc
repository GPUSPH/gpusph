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
 */

// ostringstream
#include <sstream>
// FLT_MAX
#include <cfloat>

#include "CUDAWorker.h"
#include "cudautil.h"

#include "cudabuffer.h"

// round_up
#include "utils.h"

// UINT_MAX
#include "limits.h"

using namespace std;

CUDAWorker::CUDAWorker(GlobalData* _gdata, devcount_t _deviceIndex) :
	GPUWorker(_gdata, _deviceIndex),
	m_cudaDeviceNumber(gdata->device[_deviceIndex]),

	m_asyncH2DCopiesStream(0),
	m_asyncD2HCopiesStream(0),
	m_asyncPeerCopiesStream(0),
	m_halfForcesEvent(0)
{
	initializeParticleSystem<CUDABuffer>();
}

CUDAWorker::~CUDAWorker() {
	// Free everything and pthread terminate
	// should check whether the pthread is still running and force its termination?
}

void CUDAWorker::getMemoryInfo(size_t *freeMem, size_t *totMem)
{
	cudaMemGetInfo(freeMem, totMem);
}

// Start an async inter-device transfer. This will be actually P2P if device can access peer memory
// (actually, since it is currently used only to import data from other devices, the dstDevice could be omitted or implicit)
void CUDAWorker::peerAsyncTransfer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count)
{
	if (m_disableP2Ptranfers) {
		// reallocate if necessary
		if (count > m_hPeerTransferBufferSize)
			resizePeerTransferBuffer(count);
		// transfer Dsrc -> H -> Ddst
		SAFE_CALL_NOSYNC( cudaMemcpyAsync(m_hPeerTransferBuffer, src, count, cudaMemcpyDeviceToHost, m_asyncPeerCopiesStream) );
		SAFE_CALL_NOSYNC( cudaMemcpyAsync(dst, m_hPeerTransferBuffer, count, cudaMemcpyHostToDevice, m_asyncPeerCopiesStream) );
	} else
		SAFE_CALL_NOSYNC( cudaMemcpyPeerAsync(	dst, dstDevice, src, srcDevice, count, m_asyncPeerCopiesStream ) );
}

// Uploads cellStart and cellEnd from the shared arrays to the device memory.
// Parameters: fromCell is inclusive, toCell is exclusive
void CUDAWorker::asyncCellIndicesUpload(uint fromCell, uint toCell)
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
	SAFE_CALL_NOSYNC(cudaMemcpyAsync(dst, src, transferSize, cudaMemcpyHostToDevice, m_asyncH2DCopiesStream));

	dst = sorted.getData<BUFFER_CELLEND>() + fromCell;
	src = gdata->s_dCellEnds[m_deviceIndex] + fromCell;
	SAFE_CALL_NOSYNC(cudaMemcpyAsync(dst, src, transferSize, cudaMemcpyHostToDevice, m_asyncH2DCopiesStream));
}

// wrapper for NetworkManage send/receive methods
void CUDAWorker::networkTransfer(uchar peer_gdix, TransferDirection direction, void* _ptr, size_t _size, uint bid)
{
	// reallocate host buffer if necessary
	if (!gdata->clOptions->gpudirect && _size > m_hNetworkTransferBufferSize)
		resizeNetworkTransferBuffer(_size);

	if (direction == SND) {
		if (!gdata->clOptions->gpudirect) {
			// device -> host buffer, possibly async with forces kernel
			SAFE_CALL_NOSYNC( cudaMemcpyAsync(m_hNetworkTransferBuffer, _ptr, _size,
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
			SAFE_CALL_NOSYNC( cudaMemcpyAsync(_ptr, m_hNetworkTransferBuffer, _size,
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

void CUDAWorker::deviceSynchronize()
{
	cudaDeviceSynchronize();
}

void CUDAWorker::deviceReset()
{
	cudaDeviceReset();
}

void CUDAWorker::allocPinnedBuffer(void **ptr, size_t size)
{
	SAFE_CALL_NOSYNC(cudaMallocHost(ptr, size));
}

void CUDAWorker::freePinnedBuffer(void *ptr, bool sync)
{
	if (sync)
		SAFE_CALL(cudaStreamSynchronize(m_asyncPeerCopiesStream));
	cudaFreeHost(ptr);
}

void CUDAWorker::allocDeviceBuffer(void **ptr, size_t size)
{
	SAFE_CALL_NOSYNC(cudaMalloc(ptr, size));
}

void CUDAWorker::freeDeviceBuffer(void *ptr)
{
	cudaFree(ptr);
}

void CUDAWorker::clearDeviceBuffer(void *ptr, int val, size_t size)
{
	SAFE_CALL_NOSYNC(cudaMemset(ptr, val, size));
}

void CUDAWorker::memcpyHostToDevice(void *dst, const void *src, size_t bytes)
{
	SAFE_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

void CUDAWorker::memcpyDeviceToHost(void *dst, const void *src, size_t bytes)
{
	SAFE_CALL(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

void CUDAWorker::recordHalfForceEvent()
{
	SAFE_CALL_NOSYNC(cudaEventRecord(m_halfForcesEvent, 0));
}

void CUDAWorker::syncHalfForceEvent()
{
	SAFE_CALL_NOSYNC(cudaEventSynchronize(m_halfForcesEvent));
}

void CUDAWorker::createEventsAndStreams()
{
	// init streams
	cudaStreamCreateWithFlags(&m_asyncD2HCopiesStream, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&m_asyncH2DCopiesStream, cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&m_asyncPeerCopiesStream, cudaStreamNonBlocking);
	// init events
	cudaEventCreate(&m_halfForcesEvent);
}

void CUDAWorker::destroyEventsAndStreams()
{
	// destroy streams
	cudaStreamDestroy(m_asyncD2HCopiesStream);
	cudaStreamDestroy(m_asyncH2DCopiesStream);
	cudaStreamDestroy(m_asyncPeerCopiesStream);
	// destroy events
	cudaEventDestroy(m_halfForcesEvent);
}

const char * CUDAWorker::getHardwareType() const
{
	static const char * _type = "CUDA";
	return _type;
}

int CUDAWorker::getHardwareDeviceNumber() const
{
	return m_cudaDeviceNumber;
}

void CUDAWorker::setDeviceProperties()
{
	m_deviceProperties = checkCUDA(gdata, m_deviceIndex);
}

cudaDeviceProp CUDAWorker::getDeviceProperties() {
	return m_deviceProperties;
}

// enable direct p2p memory transfers by allowing the other devices to access the current device memory
void CUDAWorker::enablePeerAccess()
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

