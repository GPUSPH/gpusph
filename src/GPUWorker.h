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
 * Interface of the generic (device-independent) part of the GPU worker
 *
 * \todo Should be renamed to Worker when the split is complete
 */

#ifndef GPUWORKER_H_
#define GPUWORKER_H_

#include <thread>

#include "common_types.h"
#include "GlobalData.h"

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

// SimFramework and engine-related defines
#include "simframework.h"

// Bursts handling
#include "bursts.h"

#include "hostbuffer.h"

// In the GPUWorker we implement as "private" (protected, actually, to be accessible
// by GPUWorker subclasses) all functions which are meant to be called only by the simulationThread().
// Only the methods which need to be called by GPUSPH are declared public.
class GPUWorker {
protected:
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
	GlobalData* getGlobalData();
	devcount_t getDeviceIndex();

	virtual const char *getHardwareType() const = 0;
	virtual SupportedDeviceTypes getDeviceType() const = 0;
	virtual int getHardwareDeviceNumber() const = 0;
	virtual void setDeviceProperties() = 0;

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
	// Total number of particles belonging to objcts on which we perform Finite Elements Analysis
	uint		m_numFeaParts;
	// Total number of particles associated to Finite Elements nodes
	uint		m_numFeaNodes;

	// range of particles the kernels should write to
	uint m_particleRangeBegin; // inclusive
	uint m_particleRangeEnd;   // exclusive

	// memory allocated
	size_t m_hostMemory;
	size_t m_deviceMemory;

	// record/wait for the half-force enqueue event, for async forces computation
	virtual void recordHalfForceEvent() = 0;
	virtual void syncHalfForceEvent() = 0;

	// enable direct p2p memory transfers
	virtual void enablePeerAccess() = 0;
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

	// Device arrays
	ParticleSystem	m_dBuffers;

	// Host arrays (for the workers, these only hold cell-based buffers:
	// BUFFER_CELLSTART, BUFFER_CELLEND, BUFFER_COMPACT_DEV_MAP)
	BufferList m_hBuffers;

	//! Initialize the particle list by adding all the necessary buffers
	template<template<flag_t Key> class BufferType>
	void initializeParticleSystem()
	{
		m_dBuffers.setAllocPolicy(gdata->simframework->getAllocPolicy());

		m_dBuffers.addBuffer<BufferType, BUFFER_POS>();
		m_dBuffers.addBuffer<BufferType, BUFFER_VEL>();
		m_dBuffers.addBuffer<BufferType, BUFFER_INFO>();
		m_dBuffers.addBuffer<BufferType, BUFFER_FORCES>(0);

		if (HAS_FEA(m_simparams->simflags)) {
			m_dBuffers.addBuffer<BufferType, BUFFER_FEA_FORCES>(0);
			m_dBuffers.addBuffer<BufferType, BUFFER_FEA_VEL>(0);
		}

		if (m_simparams->numforcesbodies) {
			m_dBuffers.addBuffer<BufferType, BUFFER_RB_FORCES>(0);
			m_dBuffers.addBuffer<BufferType, BUFFER_RB_TORQUES>(0);
			m_dBuffers.addBuffer<BufferType, BUFFER_RB_KEYS>();
		}

		m_dBuffers.addBuffer<BufferType, BUFFER_CELLSTART>(-1);
		m_dBuffers.addBuffer<BufferType, BUFFER_CELLEND>(-1);
		if (MULTI_DEVICE) {
			m_dBuffers.addBuffer<BufferType, BUFFER_COMPACT_DEV_MAP>();
			m_hBuffers.addBuffer<HostBuffer, BUFFER_COMPACT_DEV_MAP>();
		}

		m_dBuffers.addBuffer<BufferType, BUFFER_HASH>();
		m_dBuffers.addBuffer<BufferType, BUFFER_PARTINDEX>();
		m_dBuffers.addBuffer<BufferType, BUFFER_NEIBSLIST>(-1); // neib list is initialized to all bits set

		if (HAS_DEM_OR_PLANES(m_simparams->simflags))
			m_dBuffers.addBuffer<BufferType, BUFFER_NEIBPLANES>(-1); // neib planes list is initialized to all bits set

		if (HAS_XSPH(m_simparams->simflags))
			m_dBuffers.addBuffer<BufferType, BUFFER_XSPH>(0);

		// TODO we may want to allocate them for delta-SPH in the debugging case
		if (HAS_CCSPH(m_simparams->simflags)) {
			m_dBuffers.addBuffer<BufferType, BUFFER_WCOEFF>(0);
			m_dBuffers.addBuffer<BufferType, BUFFER_FCOEFF>(0);
		}

		if (m_simparams->densitydiffusiontype == ANTUONO) {
			m_dBuffers.addBuffer<BufferType, BUFFER_RENORMDENS>(0);
		}

		// If the user enabled a(n actual) turbulence model, enable BUFFER_TAU, to
		// store the shear stress tensor.
		// TODO FIXME temporary: k-eps needs TAU only for temporary storage
		// across the split kernel calls in forces
		if (m_simparams->turbmodel > ARTIFICIAL)
			m_dBuffers.addBuffer<BufferType, BUFFER_TAU>(0);

		if (m_simframework->hasPostProcessOption(SURFACE_DETECTION, BUFFER_NORMALS))
			m_dBuffers.addBuffer<BufferType, BUFFER_NORMALS>();
		if (m_simframework->hasPostProcessOption(INTERFACE_DETECTION, BUFFER_NORMALS))
			m_dBuffers.addBuffer<BufferType, BUFFER_NORMALS>();

		if (m_simframework->hasPostProcessEngine(VORTICITY))
			m_dBuffers.addBuffer<BufferType, BUFFER_VORTICITY>();

		if (HAS_DTADAPT(m_simparams->simflags)) {
			m_dBuffers.addBuffer<BufferType, BUFFER_CFL>();
			m_dBuffers.addBuffer<BufferType, BUFFER_CFL_TEMP>();
			if (m_simparams->boundarytype == SA_BOUNDARY && USING_DYNAMIC_GAMMA(m_simparams->simflags))
				m_dBuffers.addBuffer<BufferType, BUFFER_CFL_GAMMA>();
			if (m_simparams->turbmodel == KEPSILON)
				m_dBuffers.addBuffer<BufferType, BUFFER_CFL_KEPS>();
		}

		if (m_simparams->boundarytype == SA_BOUNDARY) {
			m_dBuffers.addBuffer<BufferType, BUFFER_GRADGAMMA>();
			m_dBuffers.addBuffer<BufferType, BUFFER_BOUNDELEMENTS>();
			m_dBuffers.addBuffer<BufferType, BUFFER_VERTICES>();
			m_dBuffers.addBuffer<BufferType, BUFFER_VERTPOS>();
		}

		if (m_simparams->boundarytype == DUMMY_BOUNDARY) {
			m_dBuffers.addBuffer<BufferType, BUFFER_DUMMY_VEL>(0);
		}

		if (m_simparams->turbmodel == KEPSILON) {
			m_dBuffers.addBuffer<BufferType, BUFFER_TKE>();
			m_dBuffers.addBuffer<BufferType, BUFFER_EPSILON>();
			m_dBuffers.addBuffer<BufferType, BUFFER_TURBVISC>();
			m_dBuffers.addBuffer<BufferType, BUFFER_DKDE>(0);
		}

		if (m_simparams->turbmodel == SPS) {
			m_dBuffers.addBuffer<BufferType, BUFFER_SPS_TURBVISC>();
		}

		if (NEEDS_EFFECTIVE_VISC(m_simparams->rheologytype))
			m_dBuffers.addBuffer<BufferType, BUFFER_EFFVISC>();

		if (m_simparams->rheologytype == GRANULAR) {
			m_dBuffers.addBuffer<BufferType, BUFFER_EFFPRES>();
			m_dBuffers.addBuffer<BufferType, BUFFER_JACOBI>();
		}

		if (m_simparams->boundarytype == SA_BOUNDARY &&
			(HAS_INLET_OUTLET(m_simparams->simflags) || m_simparams->turbmodel == KEPSILON))
			m_dBuffers.addBuffer<BufferType, BUFFER_EULERVEL>();

		if (HAS_INLET_OUTLET(m_simparams->simflags))
			m_dBuffers.addBuffer<BufferType, BUFFER_NEXTID>();

		if (m_simparams->sph_formulation == SPH_GRENIER) {
			m_dBuffers.addBuffer<BufferType, BUFFER_VOLUME>();
			m_dBuffers.addBuffer<BufferType, BUFFER_SIGMA>();
		}

		if (m_simframework->hasPostProcessEngine(CALC_PRIVATE)) {
			m_dBuffers.addBuffer<BufferType, BUFFER_PRIVATE>();
			if (m_simframework->hasPostProcessOption(CALC_PRIVATE, BUFFER_PRIVATE2))
				m_dBuffers.addBuffer<BufferType, BUFFER_PRIVATE2>();
			if (m_simframework->hasPostProcessOption(CALC_PRIVATE, BUFFER_PRIVATE4))
				m_dBuffers.addBuffer<BufferType, BUFFER_PRIVATE4>();
		}

		if (HAS_INTERNAL_ENERGY(m_simparams->simflags)) {
			m_dBuffers.addBuffer<BufferType, BUFFER_INTERNAL_ENERGY>();
			m_dBuffers.addBuffer<BufferType, BUFFER_INTERNAL_ENERGY_UPD>(0);
		}

		// all workers begin with an "initial upload” state in their particle system,
		// to hold all the buffers that will be initialized from host
		m_dBuffers.initialize_state("initial upload");
	}

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
	virtual void peerAsyncTransfer(void* dst, int  dstDevice, const void* src, int  srcDevice, size_t count) = 0;
	virtual void asyncCellIndicesUpload(uint fromCell, uint toCell) = 0;

	// wrapper for NetworkManage send/receive methods
	virtual void networkTransfer(devcount_t peer_gdix, TransferDirection direction, void* _ptr, size_t _size, uint bid = 0) = 0;

	// synchronize the device
	virtual void deviceSynchronize() = 0;
	// reset the device
	virtual void deviceReset() = 0;

	// allocate/free a pinned, device-visible host buffer
	virtual void allocPinnedBuffer(void **ptr, size_t size) = 0;
	virtual void freePinnedBuffer(void *ptr, bool sync = false) = 0;
	// allocate/free a device buffer outside of the BufferList management
	// TODO ideally we should move everything into the BufferList
	virtual void allocDeviceBuffer(void **ptr, size_t size) = 0;
	virtual void freeDeviceBuffer(void *ptr) = 0;
	// memset a device buffer
	virtual void clearDeviceBuffer(void *ptr, int val, size_t bytes) = 0;
	// copy from host to device
	virtual void memcpyHostToDevice(void *dst, const void *src, size_t bytes) = 0;
	// copy from device to host
	virtual void memcpyDeviceToHost(void *dst, const void *src, size_t bytes) = 0;

	size_t allocateHostBuffers();
	size_t allocateDeviceBuffers();
	void deallocateHostBuffers();
	void deallocateDeviceBuffers();

	void pinGlobalHostBuffers();
	void unpinGlobalHostBuffers();

	virtual void pinHostBuffer(void *ptr, size_t bytes) = 0;
	virtual void unpinHostBuffer(void *ptr) = 0;

	virtual void createEventsAndStreams() = 0;
	virtual void destroyEventsAndStreams() = 0;

	virtual void getMemoryInfo(size_t *freeMem, size_t *totMem) = 0;

	void printAllocatedMemory();

	void initialize();
	void finalize();

protected:
	// create a textual description of the list of buffers in the command flags
	std::string describeCommandFlagsBuffers(flag_t flags);

	void uploadSubdomain();
	void downloadSegments();
	void uploadSegments();
	void resetSegments();
	void uploadNumOpenVertices();

	// moving boundaries, gravity, planes
	void uploadGravity(); // also runCommand<UPLOAD_GRAVITY>
	void uploadPlanes(); // also runCommand<UPLOAD_PLANES>

	void createCompactDeviceMap();
	void uploadCompactDeviceMap();
	void uploadConstants();

	// bodies
	void uploadForcesBodiesCentersOfGravity(); // also runCommand<FORCES_UPLOAD_OBJECTS_CG>
	void uploadEulerBodiesCentersOfGravity(); // runCommand<EULER_UPLOAD_OBJECTS_CG> 

	void checkPartValByIndex(CommandStruct const& cmd,
		const char* printID, const uint pindex);

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
	virtual ~GPUWorker();

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
