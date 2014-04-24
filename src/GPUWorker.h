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

#ifndef GPUWORKER_H_
#define GPUWORKER_H_

#include <pthread.h>

#include "vector_types.h"
#include "common_types.h"
#include "GlobalData.h"

#include "cudautil.cuh"

// for CUDA_SAFE_CALL & co.
#include "cuda_call.h"

#include "physparams.h"
#include "simparams.h"

// buffers and buffer lists
#include "buffer.h"

// Bursts handling
#include "bursts.h"

// In GPUWoker we implement as "private" all functions which are meant to be called only by the simulationThread().
// Only the methods which need to be called by GPUSPH are declared public.
class GPUWorker {
private:
	pthread_t pthread_id;
	static void* simulationThread(void *ptr);
	GlobalData* gdata;

	unsigned int m_cudaDeviceNumber;
	unsigned int m_deviceIndex;
	unsigned int m_globalDeviceIdx;
	GlobalData* getGlobalData();
	unsigned int getCUDADeviceNumber();
	unsigned int getDeviceIndex();

	// number of particles of the assigned subset
	uint m_numParticles;
	// number of cells of the grid of the whole world
	uint m_nGridCells;
	// number of allocated particles (includes internal, external and unused slots)
	uint m_numAllocatedParticles;
	// number of internal particles, used for multi-GPU
	uint m_numInternalParticles;

	// range of particles the kernels should write to
	uint m_particleRangeBegin; // inclusive
	uint m_particleRangeEnd;   // exclusive

	// memory allocated
	size_t m_hostMemory;
	size_t m_deviceMemory;

	// it would be easier to put the device properties in a shared array in GlobalData;
	// this, however, would violate the principle that any CUDA-related code should be
	// handled by GPUWorkers and, secondly, GPUSPH
	cudaDeviceProp m_deviceProperties;
	// the setter is private and meant to be called ony by the simulation thread
	void setDeviceProperties(cudaDeviceProp _m_deviceProperties);

	// enable direct p2p memory transfers
	void enablePeerAccess();
	// explicitly stage P2P transfers on host
	bool m_disableP2Ptranfers;
	// host buffer if peer access is disabled: pointer, size, resize method
	void *m_hTransferBuffer;
	size_t m_hTransferBufferSize;
	void resizeTransferBuffer(size_t required_size);

	// utility pointers - the actual structures are in Problem
	PhysParams*	m_physparams;
	SimParams*	m_simparams;

	// CPU arrays
	//float4*			m_hPos;					// postions array
	//float4*			m_hVel;					// velocity array
	//float4*		m_hForces;				// forces array
	//particleinfo*	m_hInfo;				// info array
	//float3*		m_hVort;				// vorticity
	//float*		m_hVisc;				// viscosity
	//float4*   	m_hNormals;				// normals at free surface

	// TODO: CPU arrays used for debugging

	// GPU arrays
	BufferList	m_dBuffers;

	uint*		m_dCellStart;			// index of cell start in sorted order
	uint*		m_dCellEnd;				// index of cell end in sorted order

	// GPU arrays for rigid bodies (CPU ones are in GlobalData)
	uint		m_numBodiesParticles;	// Total number of particles belonging to rigid bodies
	float4*		m_dRbForces;			// Forces on particles belonging to rigid bodies
	float4*		m_dRbTorques;			// Torques on particles belonging to rigid bodies
	uint*		m_dRbNum;				// Key used in segmented scan

	// CPU/GPU data for moving boundaries
	uint		m_mbDataSize;			// size (in bytes) of m_dMbData array
	float4*		m_dMbData;				// device side moving boundary data

	// CPU/GPU buffers for the compact device map (2 bits per cell)
	uint*		m_hCompactDeviceMap;
	uint*		m_dCompactDeviceMap;

	// bursts of cells to be transferred
	BurstList	m_bursts;

	// where sequences of cells of the same type begin
	uint*		m_dSegmentStart;

	// number of blocks used in forces kernel runs (for delayed cfl reduction)
	uint		m_forcesKernelTotalNumBlocks;

	// stream for async memcpys
	cudaStream_t m_asyncH2DCopiesStream;
	cudaStream_t m_asyncD2HCopiesStream;
	cudaStream_t m_asyncPeerCopiesStream;

	// cuts all external particles
	void dropExternalParticles();

	// compute list of bursts
	void computeCellBursts();
	// iterate on the list and send/receive/read cell sizes
	void transferBurstsSizes();
	// iterate on the list and send/receive/read bursts of particles
	void transferBursts();

	// append or update the external cells of other devices in the device memory
	void importExternalCells();
	// aux methods for importPeerEdgeCells();
	void peerAsyncTransfer(void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count);
	void asyncCellIndicesUpload(uint fromCell, uint toCell);

	size_t allocateHostBuffers();
	size_t allocateDeviceBuffers();
	void deallocateHostBuffers();
	void deallocateDeviceBuffers();

	void createStreams();
	void destroyStreams();

	void printAllocatedMemory();

	void uploadSubdomain();
	void dumpBuffers();
	void setDeviceCellsAsEmpty();
	void downloadCellsIndices();
	void downloadSegments();
	void uploadSegments();
	void updateSegments();
	void resetSegments();

	// moving boundaries, gravity, planes
	void uploadMBData();
	void uploadGravity();
	void uploadPlanes();

	void createCompactDeviceMap();
	void uploadCompactDeviceMap();
	void uploadConstants();

	// bodies
	void uploadBodiesCentersOfGravity();
	void uploadBodiesTransRotMatrices();

	// kernels
	void kernel_calcHash();
	void kernel_sort();
	void kernel_inverseParticleIndex();
	void kernel_reorderDataAndFindCellStart();
	void kernel_buildNeibsList();
	void kernel_forces();
	void kernel_euler();
	void kernel_mls();
	void kernel_shepard();
	void kernel_vorticity();
	void kernel_surfaceParticles();
	void kernel_sps();
	void kernel_meanStrain();
	void kernel_reduceRBForces();
	void kernel_dynamicBoundaryConditions();
	void kernel_updateValuesAtBoundaryElements();
	void kernel_initGradGamma();
	void kernel_updateGamma();
	void kernel_updatePositions();
	void kernel_calcPrivate();
	void kernel_testpoints();
	/*void uploadMbData();
	void uploadGravity();*/

	// asynchronous alternative to kernel_force
	void kernel_forces_async_enqueue();
	void kernel_forces_async_complete();

	// aux methods for forces kernel striping
	uint enqueueForcesOnRange(uint fromParticle, uint toParticle, uint cflOffset);
	void bind_textures_forces();
	void unbind_textures_forces();
	float forces_dt_reduce();
public:
	// constructor & destructor
	GPUWorker(GlobalData* _gdata, unsigned int _devnum);
	~GPUWorker();

	// getters of the number of particles
	uint getNumParticles();
	uint getNumInternalParticles();
	uint getMaxParticles();

	// compute the bytes required for each particle/cell
	size_t computeMemoryPerParticle();
	size_t computeMemoryPerCell();
	// check how many particles we can allocate at most
	void computeAndSetAllocableParticles();

	// thread management
	void run_worker();
	void join_worker();

	// utility getters
	cudaDeviceProp getDeviceProperties();
	size_t getHostMemory();
	size_t getDeviceMemory();
	// for peer transfers
	const AbstractBuffer* getBuffer(flag_t) const;
};

#endif /* GPUWORKER_H_ */
