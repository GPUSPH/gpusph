/*
 * GPUWorker.h
 *
 *  Created on: Dec 19, 2012
 *      Author: rustico
 */

#ifndef GPUWORKER_H_
#define GPUWORKER_H_

class GPUWorker;

#include <pthread.h>
#include "vector_types.h"
#include "GlobalData.h"
#include "buildneibs.cuh" // for hashKey

/* We need a forward declaration of GlobalData.
 * When the compiler includes "GlobalData.h" from somewhere else, it defines _GLOBAL_DATA_
 * and in turn includes "GPUWorker.h"; but the latter does not know the GlobalData struct
 * yet and including GloblData.h again does not work since _GLOBAL_DATA_ is defined.
 * So we need to forward-declare the struct GlobalData. GPUWorker finds it and compiles.
 * The GPUWorker class needs to be forward-declared as well, since GlobalData needs it. */
struct GlobalData;

#include "cudautil.cuh"

// for CUDA_SAFE_CALL & co.
#include "cuda_call.h"

#include "physparams.h"
#include "simparams.h"

#include "forces.cuh"

// In GPUWoker we implement as "private" all functions which are meant to be called only by the simulationThread().
// Only the methods which need to be called by GPUSPH are declared public.
class GPUWorker {
private:
	pthread_t pthread_id;
	static void* simulationThread(void *ptr);
	GlobalData* gdata;

	unsigned int m_cudaDeviceNumber;
	unsigned int m_deviceIndex;
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

	// memory allocate
	unsigned long m_hostMemory;
	unsigned long m_deviceMemory;

	// it would be easier to put the device properties in a shared array in GlobalData;
	// this, however, would violate the principle that any CUDA-related code should be
	// handled by GPUWorkers and, secondly, GPUSPH
	cudaDeviceProp m_deviceProperties;
	// the setter is private and meant to be called ony by the simulation thread
	void setDeviceProperties(cudaDeviceProp _m_deviceProperties);

	void setPeerAccess();

	// utility pointers - the actual structures are in Problem
	PhysParams*	m_physparams;
	SimParams*	m_simparams;

	// CPU arrays
	float4*		m_hPos;					// postions array
	float4*		m_hVel;					// velocity array
	//float4*		m_hForces;				// forces array
	particleinfo*	m_hInfo;			// info array
	float3*		m_hVort;				// vorticity
	//float*		m_hVisc;				// viscosity
	//float4*     m_hNormals;				// normals at free surface

	// copy of CellStart and CellEnd on host, to be accessed by GPUSPH and/or by other workers
	uint*		m_hCellStart;			// index of cell start in sorted order
	uint*		m_hCellEnd;				// index of cell end in sorted order

	// TODO: CPU arrays used for debugging

	// GPU arrays
	float4*		m_dForces;				// forces array
	float4*		m_dXsph;				// mean velocity array
	float4*		m_dPos[2];				// position array
	float4*		m_dVel[2];				// velocity array
	particleinfo*	m_dInfo[2];			// particle info array
	float4*     m_dNormals;				// normal at free surface
	float3*		m_dVort;				// vorticity
	//uint		m_numPartsFmax;			// number of particles divided by BLOCK_SIZE
	float*		m_dCfl;					// cfl for each block
	float*		m_dTempCfl;				// temporary storage for cfl computation
	float*		m_dCfl2;				// test
	float2*		m_dTau[3];				// SPS stress tensor
	hashKey*	m_dParticleHash;		// hash table for sorting; 32 or 64 bit according to HASH_KEY_SIZE
	uint*		m_dParticleIndex;		// sorted particle indexes
	uint*		m_dCellStart;			// index of cell start in sorted order
	uint*		m_dCellEnd;				// index of cell end in sorted order
	//uint*		m_dSliceStart;			// index of first cell in slice
	uint*		m_dNeibsList;			// neib list with maxneibsnum neibs per particle

	// CPU/GPU arrays for rigid bodies
	uint		m_numBodiesParticles;	// Total number of particles belonging to rigid bodies
	float4*		m_dRbForces;			// Forces on particles belonging to rigid bodies
	float4*		m_dRbTorques;			// Torques on particles belonging to rigid bodies
	uint*		m_dRbNum;				// Key used in segmented scan
	uint*		m_hRbLastIndex;			// Indexes of last particles belonging to rigid bodies
	float3*		m_hRbTotalForce;		// Total force acting on each rigid body
	float3*		m_hRbTotalTorque;		// Total torque acting on each rigid body

	// CPU/GPU data for moving boundaries
	uint		m_mbDataSize;			// size (in bytes) of m_dMbData array
	float4*		m_dMbData;				// device side moving boundary data

	// CPU/GPU buffers for the compact device map (2 bits per cell)
	uint*		m_hCompactDeviceMap;
	uint*		m_dCompactDeviceMap;

	// where sequences of cells of the same type begin
	uint*		m_dSegmentStart;

	// cuts all external particles
	void dropExternalParticles();

	// append a copy of the external cells of other devices to the present array
	void importPeerEdgeCells();
	// overwrite the external edge cells with an updated copy
	void updatePeerEdgeCells();

	size_t allocateHostBuffers();
	size_t allocateDeviceBuffers();
	void deallocateHostBuffers();
	void deallocateDeviceBuffers();

	void printAllocatedMemory();

	void uploadSubdomain();
	void dumpBuffers();
	void downloadCellsIndices();

	void createCompactDeviceMap();
	void uploadCompactDeviceMap();
	void saveCompactDeviceMapTofile(std::string prefix);
	void uploadConstants();

	// kernels
	void kernel_calcHash();
	void kernel_sort();
	void kernel_reorderDataAndFindCellStart();
	void kernel_buildNeibsList();
	void kernel_forces();
	void kernel_euler();
	void kernel_mls();
	void kernel_shepard();
	/*void uploadMbData();
	void uploadGravity();*/

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
	unsigned long getHostMemory();
	unsigned long getDeviceMemory();
	// for peer transfers
	const float4** getDPosBuffers();
	const float4** getDVelBuffers();
	const particleinfo** getDInfoBuffers();
};

#endif /* GPUWORKER_H_ */
