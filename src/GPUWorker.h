/*
 * GPUWorker.h
 *
 *  Created on: Dec 19, 2012
 *      Author: rustico
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
	float4*		m_dForces;				// forces array
	float4*		m_dXsph;				// mean velocity array
	float4*		m_dPos[2];				// position array
	float4*		m_dVel[2];				// velocity array
	particleinfo*	m_dInfo[2];			// particle info array
	float4*     m_dNormals;				// normal at free surface
	float3*		m_dVort;				// vorticity
	//uint		m_numPartsFmax;			// number of particles divided by BLOCK_SIZE
	float*		m_dCfl;					// cfl for each block
	float*		m_dCflGamma;			// cfl contribution due to gamma integration
	float*		m_dCflTVisc;			// cfl contribution from eddy viscosity
	float*		m_dTempCfl;				// temporary storage for cfl computation
	float2*		m_dTau[3];				// SPS stress tensor

	float4*		m_dGradGamma[2];		// gradient of renormalization term gamma (x,y,z) and gamma itself (w)
	float4*		m_dBoundElement[2];		// normal coordinates (x,y,z) and surface (w) of boundary elements (triangles)
	vertexinfo*	m_dVertices[2];			// stores indexes of 3 vertex particles for every boundary element
	float*		m_dPressure[2];			// stores pressure for vertex and boundary particles
	float*		m_dTKE[2];				// k - turbulent kinetic energy
	float*		m_dEps[2];				// e - turbulent kinetic energy dissipation rate
	float*		m_dTurbVisc[2];			// nu_t - kinematic eddy viscosity
	float*		m_dStrainRate[2];		// S - mean scalar strain rate
	float2*		m_dDkDe;				// dk/dt and de/dt for k-e model

	hashKey*	m_dParticleHash;		// hash table for sorting; 32 or 64 bit according to HASH_KEY_SIZE
	uint*		m_dParticleIndex;		// sorted particle indexes
	uint*		m_dInversedParticleIndex;// inversed m_dParticle index array
	uint*		m_dCellStart;			// index of cell start in sorted order
	uint*		m_dCellEnd;				// index of cell end in sorted order
	//uint*		m_dSliceStart;			// index of first cell in slice
	neibdata*	m_dNeibsList;			// neib list with maxneibsnum neibs per particle

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

	// where sequences of cells of the same type begin
	uint*		m_dSegmentStart;

	// stream for async memcpys
	cudaStream_t m_asyncH2DCopiesStream;
	cudaStream_t m_asyncD2HCopiesStream;
	cudaStream_t m_asyncPeerCopiesStream;

	// cuts all external particles
	void dropExternalParticles();

	// append or update the external cells of other devices in the device memory
	void importPeerEdgeCells();
	// MPI versions of the previous method
	void importNetworkPeerEdgeCells();
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
	size_t getHostMemory();
	size_t getDeviceMemory();
	// for peer transfers
	const float4* const* getDPosBuffers() const;
	const float4* const* getDVelBuffers() const;
	const particleinfo* const* getDInfoBuffers() const;
	const float4* getDForceBuffer() const;
	const float2* const* getDTauBuffers() const;
	const hashKey* getDHashBuffer() const;
	const uint* getDPartIndexBuffer() const;
	const float4* const* getDBoundElemsBuffers() const;
	const float4* const* getDGradGammaBuffers() const;
	const vertexinfo* const* getDVerticesBuffers() const;
	const float* const* getDPressureBuffers() const;
	const float* const* getDTKEBuffers() const;
	const float* const* getDEpsBuffers() const;
	const float* const* getDTurbViscBuffers() const;
	const float* const* getDStrainRateBuffers() const;
};

#endif /* GPUWORKER_H_ */
