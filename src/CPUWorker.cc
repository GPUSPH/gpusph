/*  Copyright (c) 2020 INGV, EDF, UniCT, JHU

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
 * Implementation of the CPU worker
 */

// ostringstream
#include <sstream>
// FLT_MAX
#include <cfloat>
// sysconf
#include <unistd.h>
// aligned_alloc needs C++17, so for the time being we'll use posix_memalign
#include <cstdlib>

#include "CPUWorker.h"

#include "hostbuffer.h"

// round_up
#include "utils.h"

// UINT_MAX
#include "limits.h"

using namespace std;

CPUWorker::CPUWorker(GlobalData* _gdata, devcount_t _deviceIndex) :
	GPUWorker(_gdata, _deviceIndex),
	m_cudaDeviceNumber(gdata->device[_deviceIndex])
{
	initializeParticleSystem<HostBuffer>();
}

CPUWorker::~CPUWorker() {
	// Free everything and pthread terminate
	// should check whether the pthread is still running and force its termination?
}

void CPUWorker::getMemoryInfo(size_t *freeMem, size_t *totMem)
{
	// sysconf returns a long, but we cast to size_t for our needs
	const size_t pg_size = sysconf(_SC_PAGE_SIZE);
	const size_t pg_phys = sysconf(_SC_PHYS_PAGES);
	const size_t pg_avail = sysconf(_SC_AVPHYS_PAGES);
	*totMem = pg_size*pg_phys;
	*freeMem = pg_size*pg_avail;
}

// Start an async inter-device transfer. This will be actually P2P if device can access peer memory
// (actually, since it is currently used only to import data from other devices, the dstDevice could be omitted or implicit)
void CPUWorker::peerAsyncTransfer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count)
{
	throw logic_error("peerAsyncTransfer not available for the CPUWorker");
}

// Uploads cellStart and cellEnd from the shared arrays to the device memory.
// Parameters: fromCell is inclusive, toCell is exclusive
void CPUWorker::asyncCellIndicesUpload(uint fromCell, uint toCell)
{
	const uint numCells = toCell - fromCell;
	const uint transferSize = sizeof(uint)*numCells;

	// TODO migrate s_dCellStarts to the device mechanism and provide an API
	// to copy offset data between buffers (even of different types)

	BufferList sorted = m_dBuffers.state_subset("sorted",
		BUFFER_CELLSTART | BUFFER_CELLEND);

	const uint *src;
	uint *dst;

	// TODO FIXME not really async for the time being, use std::async?
	dst = sorted.getData<BUFFER_CELLSTART>() + fromCell;
	src = gdata->s_dCellStarts[m_deviceIndex] + fromCell;
	memcpy(dst, src, transferSize);

	dst = sorted.getData<BUFFER_CELLEND>() + fromCell;
	src = gdata->s_dCellEnds[m_deviceIndex] + fromCell;
	memcpy(dst, src, transferSize);
}

// wrapper for NetworkManage send/receive methods
void CPUWorker::networkTransfer(uchar peer_gdix, TransferDirection direction, void* _ptr, size_t _size, uint bid)
{
	if (direction == SND) {
		if (gdata->clOptions->asyncNetworkTransfers)
			gdata->networkManager->sendBufferAsync(m_globalDeviceIdx, peer_gdix, _size, _ptr, bid);
		else
			gdata->networkManager->sendBuffer(m_globalDeviceIdx, peer_gdix, _size, _ptr);
	} else {
		if (gdata->clOptions->asyncNetworkTransfers)
			gdata->networkManager->receiveBufferAsync(peer_gdix, m_globalDeviceIdx, _size, _ptr, bid);
		else
			gdata->networkManager->receiveBuffer(peer_gdix, m_globalDeviceIdx, _size, _ptr);
	}
}

void CPUWorker::deviceSynchronize()
{
	// TODO FIXME thread fence or otherwise wait on all async ops
}

void CPUWorker::deviceReset()
{
	// nothing to do in this case
	return;
}

void CPUWorker::allocPinnedBuffer(void **ptr, size_t size)
{
	int err = posix_memalign(ptr, getpagesize(), size);
	if (err < 0)
		throw runtime_error("allocPinnedBuffer failed");
}

void CPUWorker::freePinnedBuffer(void *ptr, bool sync)
{
	// sync is currently unused since we have no separate command queues
	// for multi-device
	free(ptr);
}

void CPUWorker::allocDeviceBuffer(void **ptr, size_t size)
{
	void *a = malloc(size);
	if (!a)
		throw runtime_error("allocDeviceBuffer failed");
	*ptr = a;
}

void CPUWorker::freeDeviceBuffer(void *ptr)
{
	free(ptr);
}

void CPUWorker::clearDeviceBuffer(void *ptr, int val, size_t size)
{
	memset(ptr, val, size);
}

void CPUWorker::memcpyHostToDevice(void *dst, const void *src, size_t bytes)
{
	memcpy(dst, src, bytes);
}

void CPUWorker::memcpyDeviceToHost(void *dst, const void *src, size_t bytes)
{
	memcpy(dst, src, bytes);
}

void CPUWorker::recordHalfForceEvent()
{
	// TODO FIXME use async?
	throw logic_error("recordHalfForceEvent not availabel for the CPUWorker");
}

void CPUWorker::syncHalfForceEvent()
{
	// TODO FIXME use async?
	throw logic_error("syncHalfForceEvent not availabel for the CPUWorker");
}

void CPUWorker::createEventsAndStreams()
{
	// TODO FIXME async support
}

void CPUWorker::destroyEventsAndStreams()
{
	// TODO FIXME async support
}

const char * CPUWorker::getHardwareType() const
{
	static const char * _type = "CPU";
	return _type;
}

int CPUWorker::getHardwareDeviceNumber() const
{
	return m_cudaDeviceNumber; // fake
}

void CPUWorker::setDeviceProperties()
{
	// TODO FIXME
}

// enable direct p2p memory transfers by allowing the other devices to access the current device memory
void CPUWorker::enablePeerAccess()
{
	// TODO FIXME no peer acces ATM
	return;
}

