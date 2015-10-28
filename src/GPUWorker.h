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

#include "engine_neibs.h"
#include "engine_filter.h"
#include "engine_integration.h"
#include "engine_visc.h"
#include "engine_forces.h"

// buffers and buffer lists
#include "buffer.h"

// Bursts handling
#include "bursts.h"

// In GPUWoker we implement as "private" all functions which are meant to be called only by the simulationThread().
// Only the methods which need to be called by GPUSPH are declared public.
class GPUWorker {
private:
	GlobalData* gdata;

	AbstractNeibsEngine *neibsEngine;
	AbstractViscEngine *viscEngine;
	AbstractForcesEngine *forcesEngine;
	AbstractIntegrationEngine *integrationEngine;
	AbstractBoundaryConditionsEngine *bcEngine;
	FilterEngineSet const& filterEngines;
	PostProcessEngineSet const& postProcEngines;

	pthread_t pthread_id;
	static void* simulationThread(void *ptr);

	unsigned int m_cudaDeviceNumber;
	devcount_t m_deviceIndex;
	devcount_t m_globalDeviceIdx;
	GlobalData* getGlobalData();
	unsigned int getCUDADeviceNumber();
	devcount_t getDeviceIndex();

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
	// the setter is private and meant to be called only by the simulation thread
	void setDeviceProperties(cudaDeviceProp _m_deviceProperties);

	// enable direct p2p memory transfers
	void enablePeerAccess();
	// explicitly stage P2P transfers on host
	bool m_disableP2Ptranfers;
	// host buffers: pointer, size, resize method
	void *m_hPeerTransferBuffer;
	size_t m_hPeerTransferBufferSize;
	void resizePeerTransferBuffer(size_t required_size);

	// host buffers: pointer, size, resize method used if gpudirect is disabled
	void *m_hNetworkTransferBuffer;
	size_t m_hNetworkTransferBufferSize;
	void resizeNetworkTransferBuffer(size_t required_size);

	// utility pointers - the actual structures are in Problem
	PhysParams*	m_physparams;
	SimParams*	m_simparams;
	const SimFramework *m_simframework;

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
	MultiBufferList	m_dBuffers;

	uint*		m_dCellStart;			// index of cell start in sorted order
	uint*		m_dCellEnd;				// index of cell end in sorted order

	// GPU arrays for rigid bodies (CPU ones are in GlobalData)
	uint		m_numForcesBodiesParticles;		// Total number of particles belonging to rigid bodies on which we compute forces
	float4*		m_dRbForces;					// Forces on particles belonging to rigid bodies
	float4*		m_dRbTorques;					// Torques on particles belonging to rigid bodies
	uint*		m_dRbNum;						// Key used in segmented scan


	// CPU/GPU buffers for the compact device map (2 bits per cell)
	uint*		m_hCompactDeviceMap;
	uint*		m_dCompactDeviceMap;

	// bursts of cells to be transferred
	BurstList	m_bursts;

	// where sequences of cells of the same type begin
	uint*		m_dSegmentStart;

	// water depth at open boundaries
	uint*		m_dIOwaterdepth;

	// "new" number of particles for open boundaries
	uint*		m_dNewNumParticles;

	// number of blocks used in forces kernel runs (for delayed cfl reduction)
	uint		m_forcesKernelTotalNumBlocks;

	// stream for async memcpys
	cudaStream_t m_asyncH2DCopiesStream;
	cudaStream_t m_asyncD2HCopiesStream;
	cudaStream_t m_asyncPeerCopiesStream;

	// event to synchronize striping
	cudaEvent_t m_halfForcesEvent;

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

	// wrapper for NetworkManage send/receive methods
	void networkTransfer(uchar peer_gdix, TransferDirection direction, void* _ptr, size_t _size, uint bid = 0);

	size_t allocateHostBuffers();
	size_t allocateDeviceBuffers();
	void deallocateHostBuffers();
	void deallocateDeviceBuffers();

	void createEventsAndStreams();
	void destroyEventsAndStreams();

	void printAllocatedMemory();

	void initialize();
	void finalize();

	// select a BufferList based on the DBLBUFFER_* specification
	// in the command flags
	MultiBufferList::iterator getBufferListByCommandFlags(flag_t flags);

	void uploadSubdomain();
	void dumpBuffers();
	void swapBuffers();
	void setDeviceCellsAsEmpty();
	void downloadCellsIndices();
	void downloadSegments();
	void uploadSegments();
	void updateSegments();
	void resetSegments();
	void uploadNewNumParticles();
	void downloadNewNumParticles();

	// moving boundaries, gravity, planes
	void uploadGravity();
	void uploadPlanes();

	void createCompactDeviceMap();
	void uploadCompactDeviceMap();
	void uploadConstants();

	// bodies
	void uploadForcesBodiesCentersOfGravity();
	void uploadEulerBodiesCentersOfGravity();
	void uploadBodiesTransRotMatrices();
	void uploadBodiesVelocities();

	// kernels
	void kernel_calcHash();
	void kernel_sort();
	void kernel_reorderDataAndFindCellStart();
	void kernel_buildNeibsList();
	void kernel_forces();
	void kernel_euler();
	void kernel_filter();
	void kernel_postprocess();
	void kernel_compute_density();
	void kernel_sps();
	void kernel_meanStrain();
	void kernel_reduceRBForces();
	void kernel_updateVertIdIndexBuffer();
	void kernel_saSegmentBoundaryConditions();
	void kernel_saVertexBoundaryConditions();
	void kernel_saIdentifyCornerVertices();
	void kernel_saFindClosestVertex();
	void kernel_updatePositions();
	void kernel_disableOutgoingParts();
	void kernel_imposeBoundaryCondition();
	void kernel_initGamma();
	void kernel_download_iowaterdepth();
	void kernel_upload_iowaterdepth();
	/*void uploadMbData();
	void uploadGravity();*/

	void checkPartValByIndex(const char* printID, const uint pindex);
	void checkPartValById(const char* printID, const uint pid);

	// asynchronous alternative to kernel_force
	void kernel_forces_async_enqueue();
	void kernel_forces_async_complete();

	// aux methods for forces kernel striping
	uint enqueueForcesOnRange(uint fromParticle, uint toParticle, uint cflOffset);
	void bind_textures_forces();
	void unbind_textures_forces();
	float forces_dt_reduce();

	// aux method to warp signed cell coordinates when periodicity is enabled
	void periodicityWarp(int &cx, int &cy, int &cz);
	// aux method to check wether cell coords are inside the domain
	bool isCellInsideProblemDomain(int cx, int cy, int cz);
public:
	// constructor & destructor
	GPUWorker(GlobalData* _gdata, devcount_t _devnum);
	~GPUWorker();

	// getters of the number of particles
	uint getNumParticles();
	uint getNumAllocatedParticles();
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
	// for peer transfers: get the buffer `key` from the buffer list `list_idx`
	const AbstractBuffer* getBuffer(size_t list_idx, flag_t key) const;
};

#endif /* GPUWORKER_H_ */
