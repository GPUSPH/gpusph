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
 * Interface of the CUDA-based GPU worker
 */

#ifndef CUDAWORKER_H_
#define CUDAWORKER_H_

#include <thread>

#include "GPUWorker.h"

// In CUDAWoker we implement as "private" all functions which are meant to be called only by the simulationThread().
// Only the methods which need to be called by GPUSPH are declared public.
class CUDAWorker : public GPUWorker {
private:
	unsigned int m_cudaDeviceNumber;
	const char *getHardwareType() const override;
	int getHardwareDeviceNumber() const override;

	// it would be easier to put the device properties in a shared array in GlobalData;
	// this, however, would violate the principle that any CUDA-related code should be
	// handled by CUDAWorkers and, secondly, GPUSPH
	cudaDeviceProp m_deviceProperties;
	// the setter is private and meant to be called only by the simulation thread
	void setDeviceProperties() override;

	// stream for async memcpys
	cudaStream_t m_asyncH2DCopiesStream;
	cudaStream_t m_asyncD2HCopiesStream;
	cudaStream_t m_asyncPeerCopiesStream;

	// event to synchronize striping
	cudaEvent_t m_halfForcesEvent;

	// record/wait for the half-force enqueue event, for async forces computation
	void recordHalfForceEvent() override;
	void syncHalfForceEvent() override;

	// enable direct p2p memory transfers
	void enablePeerAccess() override;

	// aux methods for importPeerEdgeCells();
	void peerAsyncTransfer(void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count) override;
	void asyncCellIndicesUpload(uint fromCell, uint toCell) override;

	// wrapper for NetworkManage send/receive methods
	void networkTransfer(uchar peer_gdix, TransferDirection direction, void* _ptr, size_t _size, uint bid = 0) override;

	// synchronize the device
	void deviceSynchronize() override;
	// reset the device
	void deviceReset() override;

	// allocate/free a pinned, device-visible host buffer
	void allocPinnedBuffer(void **ptr, size_t size) override;
	void freePinnedBuffer(void *ptr, bool sync) override;
	// allocate a device buffer outside of the BufferList management
	void allocDeviceBuffer(void **ptr, size_t size) override;
	void freeDeviceBuffer(void *ptr) override;
	// memset a device buffer
	void clearDeviceBuffer(void *ptr, int val, size_t bytes) override;
	// copy from host to device
	void memcpyHostToDevice(void *dst, const void *src, size_t bytes) override;
	// copy from device to host
	void memcpyDeviceToHost(void *dst, const void *src, size_t bytes) override;

	void createEventsAndStreams() override;
	void destroyEventsAndStreams() override;

	void getMemoryInfo(size_t *freeMem, size_t *totMem) override;

public:
	// constructor & destructor
	CUDAWorker(GlobalData* _gdata, devcount_t _devnum);
	~CUDAWorker();

	// utility getters
	cudaDeviceProp getDeviceProperties();

};

#endif /* CUDAWORKER_H_ */
