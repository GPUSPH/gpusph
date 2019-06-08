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
 *
 * \todo The CUDA-independent part should be split in a separate, generic
 * Worker class.
 */

#ifndef GPUWORKER_H_
#define GPUWORKER_H_

#include <thread>

#include "vector_types.h"
#include "common_types.h"
#include "GlobalData.h"

// for CUDA_SAFE_CALL & co.
#include "cuda_call.h"

#include "physparams.h"
#include "simparams.h"

#include "engine_neibs.h"
#include "engine_filter.h"
#include "engine_integration.h"
#include "engine_visc.h"
#include "engine_forces.h"
#include "engine_boundary_conditions.h"

// the particle system, cum suis
#include "ParticleSystem.h"

// Bursts handling
#include "bursts.h"

// In GPUWoker we implement as "private" all functions which are meant to be called only by the simulationThread().
// Only the methods which need to be called by GPUSPH are declared public.
class GPUWorker {
private:
	GlobalData* gdata;

	// utility pointers - the actual structures are in Problem
	const SimFramework *m_simframework;
	PhysParams*	m_physparams;
	SimParams*	m_simparams;

	AbstractNeibsEngine *neibsEngine;
	AbstractViscEngine *viscEngine;
	AbstractForcesEngine *forcesEngine;
	AbstractIntegrationEngine *integrationEngine;
	AbstractBoundaryConditionsEngine *bcEngine;
	FilterEngineSet const& filterEngines;
	PostProcessEngineSet const& postProcEngines;

	//! maximum kinematic viscosity
	/*! This is computed once for Newtonian fluids,
	 * or every time CALC_VISC is run otherwise
	 */
	float m_max_kinvisc;
	//! maximum sound speed
	/*! Computed once (in uploadConstants), and passed to the dtreduce of the forces engine
	 */
	float m_max_sound_speed;

	std::thread thread_id;
	void simulationThread();

	devcount_t m_deviceIndex;
	devcount_t m_globalDeviceIdx;
	unsigned int m_cudaDeviceNumber;
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
	// Total number of particles belonging to rigid bodies on which we compute forces
	uint		m_numForcesBodiesParticles;

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
	ParticleSystem	m_dBuffers;

	// CPU arrays (for the workers, these only hold cell-based buffers:
	// BUFFER_CELLSTART, BUFFER_CELLEND, BUFFER_COMPACT_DEV_MAP)
	BufferList m_hBuffers;

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

	/// Function template to run a specific command
	/*! There should be a specialization of the template for each
	 * (supported) command
	 */
	template<CommandName>
	void runCommand(CommandStruct const& cmd);

	/// Function template to show a specific command
	template<CommandName>
	void describeCommand(CommandStruct const& cmd);

	/// Handle the case of an unknown command being invoked
	void unknownCommand(CommandName);

	// cuts all external particles
	// runCommand<CROP> = void dropExternalParticles();

	/// compare UPDATE_EXTERNAL arguments against list of updated buffers
	void checkBufferUpdate(CommandStruct const& cmd);

	// compute list of bursts
	void computeCellBursts();
	// iterate on the list and send/receive/read cell sizes
	void transferBurstsSizes();
	// iterate on the list and send/receive/read bursts of particles
	void transferBursts(CommandStruct const& cmd);

	/// append or update the external cells of other devices in the device memory
	void importExternalCells(CommandStruct const& cmd); // runCommand<APPEND_EXTERNAL> or runCommand<UPDATE_EXTERNAL>
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

private:
	// create a textual description of the list of buffers in the command flags
	std::string describeCommandFlagsBuffers(flag_t flags);

	void uploadSubdomain();
	// runCommand<DUMP> = void dumpBuffers();
	// runCommand<SWAP_BUFFERS> = void swapBuffers();
	// runCommand<DUMP_CELLS> = void downloadCellsIndices();
	void downloadSegments();
	void uploadSegments();
	// runCommand<UPDATE_SEGMENTS> = void updateSegments();
	void resetSegments();
	void uploadNumOpenVertices();
	// runCommand<UPLOAD_NEWNUMPARTS> = void uploadNewNumParticles();
	// runCommand<DOWNLOAD_NEWNUMPARTS> = void downloadNewNumParticles();

	// moving boundaries, gravity, planes
	void uploadGravity(); // also runCommand<UPLOAD_GRAVITY>
	void uploadPlanes(); // also runCommand<UPLOAD_PLANES>

	void createCompactDeviceMap();
	void uploadCompactDeviceMap();
	void uploadConstants();

	// bodies
	void uploadForcesBodiesCentersOfGravity(); // also runCommand<FORCES_UPLOAD_OBJECTS_CG>
	void uploadEulerBodiesCentersOfGravity(); // runCommand<EULER_UPLOAD_OBJECTS_CG> 
	// runCommand<UPLOAD_OBJECTS_MATRICES> = void uploadBodiesTransRotMatrices();
	// runCommand<UPLOAD_OBJECTS_VELOCITIES> = void uploadBodiesVelocities();

	// kernels
	// runCommand<CALCHASH> = void kernel_calcHash();
	// runCommand<SORT> = void kernel_sort();
	// runCommand<REORDER> = void kernel_reorderDataAndFindCellStart();
	// runCommand<BUILDNEIBS> = void kernel_buildNeibsList();
	// runCommand<FORCES_SYNC> = void kernel_forces();
	// runCommand<EULER> = void kernel_euler();
	// runCommand<DENSITY_SUM> = void kernel_density_sum();
	// runCommand<INTEGRATE_GAMMA> = void kernel_integrate_gamma();
	// runCommand<CALC_DENSITY_DIFFUSION> = void kernel_calc_density_diffusion();
	// runCommand<APPLY_DENSITY_DIFFUSION> = void kernel_apply_density_diffusion();
	// runCommand<FILTER> = void kernel_filter();
	// runCommand<POSTPROCESS> = void kernel_postprocess();
	// runCommand<COMPUTE_DENSITY> = void kernel_compute_density();
	// runCommand<CALC_VISC> = void kernel_visc();
	void kernel_meanStrain();
	// runCommand<REDUCE_BODIES_FORCES> = void kernel_reduceRBForces();
	// runCommand<SA_CALC_SEGMENT_BOUNDARY_CONDITIONS> = void kernel_saSegmentBoundaryConditions();
	// runCommand<SA_CALC_VERTEX_BOUNDARY_CONDITIONS> = void kernel_saVertexBoundaryConditions();
	// runCommand<SA_COMPUTE_VERTEX_NORMAL> = void kernel_saComputeVertexNormal();
	// runCommand<SA_INIT_GAMMA> = void kernel_saInitGamma();
	// runCommand<IDENTIFY_CORNER_VERTICES> = void kernel_saIdentifyCornerVertices();
	void kernel_updatePositions();
	// runCommand<DISABLE_OUTGOING_PARTS> = void kernel_disableOutgoingParts();
	// runCommand<DISABLE_FREE_SURF_PARTS> = void kernel_disableFreeSurfParts();
	// runCommand<IMPOSE_OPEN_BOUNDARY_CONDITION> = void kernel_imposeBoundaryCondition();
	// runCommand<INIT_IO_MASS_VERTEX_COUNT> = void kernel_initIOmass_vertexCount();
	// runCommand<INIT_IO_MASS> = void kernel_initIOmass();
	// runCommand<DOWNLOAD_IOWATERDEPTH> = void kernel_download_iowaterdepth();
	// runCommand<UPLOAD_IOWATERDEPTH> = void kernel_upload_iowaterdepth();
	/*void uploadMbData();
	// runCommand<UPLOAD_GRAVITY> = void uploadGravity();*/

	void checkPartValByIndex(CommandStruct const& cmd,
		const char* printID, const uint pindex);

	// asynchronous alternative to kernel_force
	// runCommand<FORCES_ENQUEUE> = void kernel_forces_async_enqueue();
	// runCommand<FORCES_COMPLETE> = void kernel_forces_async_complete();

	// A pair holding the read and write buffer lists
	using BufferListPair = std::pair<const BufferList, BufferList>;

	// aux methods for forces kernel striping
	uint enqueueForcesOnRange(CommandStruct const& cmd,
		BufferListPair& buffer_lists, uint fromParticle, uint toParticle, uint cflOffset);
	// steps to do before launching a (set of) forces kernels:
	// * select the read and write buffer lists
	// * reset CFL and object forces and torque arrays
	// * bind textures
	// Returns a pair with the read and write buffer lists
	BufferListPair pre_forces(CommandStruct const& cmd, uint numPartsToElaborate);
	// steps to do after launching a (set of) forces kernels: unbinding textures, get, adaptive dt, etc
	float post_forces(CommandStruct const& cmd);

	// aux method to warp signed cell coordinates when periodicity is enabled
	void periodicityWarp(int &cx, int &cy, int &cz);
	// aux method to check wether cell coords are inside the domain
	bool isCellInsideProblemDomain(int cx, int cy, int cz);
public:
	// constructor & destructor
	GPUWorker(GlobalData* _gdata, devcount_t _devnum);
	~GPUWorker();

	// getters of the number of particles
	uint getNumParticles() const;
	uint getNumAllocatedParticles() const;
	uint getNumInternalParticles() const;
	uint getMaxParticles() const;

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
	// for peer transfers: get the buffer `key` from the given buffer state
	std::shared_ptr<const AbstractBuffer> getBuffer(std::string const& state, flag_t key) const;

#ifdef INSPECT_DEVICE_MEMORY
	const ParticleSystem& getParticleSystem() const
	{ return m_dBuffers; }
#endif
};

#endif /* GPUWORKER_H_ */
