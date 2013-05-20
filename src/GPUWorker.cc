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

GPUWorker::GPUWorker(GlobalData* _gdata, unsigned int _devnum) {
	gdata = _gdata;
	devnum = _devnum;

	m_hPos = NULL;
	m_hVel = NULL;
	m_hInfo = NULL;
	m_hVort = NULL;

	// we know that GPUWorker is initialized when Problem was already
	m_simparams = gdata->problem->get_simparams();
	m_physparams = gdata->problem->get_physparams();

	// we also know Problem::fillparts() has already been called; however, this is
	// going to change when each worker will only manage a subset of particles
	m_numParticles = gdata->totParticles;
	m_nGridCells = gdata->nGridCells;

	m_hostMemory = m_deviceMemory = 0;
}

GPUWorker::~GPUWorker() {
	// Free everything and pthread terminate
	// should check whether the pthread is still running and force its termination?
}

// All the allocators assume that gdata is updated with the number of particles (done by problem->fillparts).
// Later this will be changed since each thread does not need to allocate the global number of particles.
size_t GPUWorker::allocateHostBuffers() {
	// common sizes
	const uint float3Size = sizeof(float3)*m_numParticles;
	const uint float4Size = sizeof(float4)*m_numParticles;
	const uint infoSize = sizeof(particleinfo)*m_numParticles;
	const uint uintCellsSize = sizeof(uint)*m_nGridCells;

	size_t allocated = 0;

	m_hPos = new float4[m_numParticles];
	memset(m_hPos, 0, float4Size);
	allocated += float4Size;

	m_hVel = new float4[m_numParticles];
	memset(m_hVel, 0, float4Size);
	allocated += float4Size;

	m_hInfo = new particleinfo[m_numParticles];
	memset(m_hInfo, 0, infoSize);
	allocated += infoSize;

	m_hCellStart = new uint[m_nGridCells];
	memset(m_hCellStart, 0, uintCellsSize);
	allocated += uintCellsSize;

	m_hCellEnd = new uint[m_nGridCells];
	memset(m_hCellEnd, 0, uintCellsSize);
	allocated += uintCellsSize;

	// TODO: only allocate when multi-GPU
	m_hCompactDeviceMap = new uint[m_nGridCells];
	memset(m_hCompactDeviceMap, 0, uintCellsSize);
	allocated += uintCellsSize;

	if (m_simparams->vorticity) {
		m_hVort = new float3[m_numParticles];
		allocated += float3Size;
		// NOTE: *not* memsetting, as in master branch
	}

	m_hostMemory += allocated;
	return allocated;
}

size_t GPUWorker::allocateDeviceBuffers() {
	// common sizes
	// compute common sizes (in bytes)
	//const uint floatSize = sizeof(float)*m_numParticles;
	const uint float2Size = sizeof(float2)*m_numParticles;
	const uint float3Size = sizeof(float3)*m_numParticles;
	const uint float4Size = sizeof(float4)*m_numParticles;
	const uint infoSize = sizeof(particleinfo)*m_numParticles;
	const uint intSize = sizeof(uint)*m_numParticles;
	const uint uintCellsSize = sizeof(uint)*m_nGridCells;
	const uint neibslistSize = sizeof(uint)*m_simparams->maxneibsnum*(m_numParticles/NEIBINDEX_INTERLEAVE + 1)*NEIBINDEX_INTERLEAVE;
	const uint hashSize = sizeof(hashKey)*m_numParticles;
	//const uint neibslistSize = sizeof(uint)*128*m_numParticles;
	//const uint sliceArraySize = sizeof(uint)*m_gridSize.PSA;

	size_t allocated = 0;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dForces, float4Size));
	CUDA_SAFE_CALL(cudaMemset(m_dForces, 0, float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dXsph, float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dPos[0], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dPos[1], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVel[0], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVel[1], float4Size));
	allocated += float4Size;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dInfo[0], infoSize));
	allocated += infoSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dInfo[1], infoSize));
	allocated += infoSize;

	// Free surface detection
	if (m_simparams->savenormals) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNormals, float4Size));
		allocated += float4Size;
	}

	if (m_simparams->vorticity) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dVort, float3Size));
		allocated += float3Size;
	}

	if (m_simparams->visctype == SPSVISC) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[0], float2Size));
		allocated += float2Size;

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[1], float2Size));
		allocated += float2Size;

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTau[2], float2Size));
		allocated += float2Size;
	}

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dParticleHash, hashSize));
	allocated += hashSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dParticleIndex, intSize));
	allocated += intSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCellStart, uintCellsSize));
	allocated += uintCellsSize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCellEnd, uintCellsSize));
	allocated += uintCellsSize;

	//CUDA_SAFE_CALL(cudaMalloc((void**)&m_dSliceStart, sliceArraySize));
	//allocated += sliceArraySize;

	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dNeibsList, neibslistSize));
	allocated += neibslistSize;

	// TODO: allocate only if multi-GPU
	// TODO: an array of uchar would suffice
	CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCompactDeviceMap, uintCellsSize));
	allocated += uintCellsSize;

	// TODO: allocation for rigid bodies

	if (m_simparams->dtadapt) {
		m_numPartsFmax = getNumPartsFmax(m_numParticles);
		const uint fmaxTableSize = m_numPartsFmax*sizeof(float);

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCfl, fmaxTableSize));
		CUDA_SAFE_CALL(cudaMemset(m_dCfl, 0, fmaxTableSize));

		const uint tempCflSize = getFmaxTempStorageSize(m_numPartsFmax);
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dTempCfl, tempCflSize));
		CUDA_SAFE_CALL(cudaMemset(m_dTempCfl, 0, tempCflSize));

		allocated += fmaxTableSize + tempCflSize;
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
	delete [] m_hPos;
	delete [] m_hVel;
	delete [] m_hInfo;
	delete [] m_hCellStart;
	delete [] m_hCellEnd;
	if (m_simparams->vorticity)
		delete [] m_hVort;
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

	CUDA_SAFE_CALL(cudaFree(m_dParticleHash));
	CUDA_SAFE_CALL(cudaFree(m_dParticleIndex));
	CUDA_SAFE_CALL(cudaFree(m_dCellStart));
	CUDA_SAFE_CALL(cudaFree(m_dCellEnd));
	CUDA_SAFE_CALL(cudaFree(m_dNeibsList));
	CUDA_SAFE_CALL(cudaFree(m_dCompactDeviceMap));

	// TODO: deallocation for rigid bodies

	if (m_simparams->dtadapt) {
		CUDA_SAFE_CALL(cudaFree(m_dCfl));
		CUDA_SAFE_CALL(cudaFree(m_dTempCfl));
	}

	// here: dem device buffers?
}

// upload subdomain, just allocated and sorted by main thread
void GPUWorker::uploadSubdomain() {
	// indices
	uint myDevNum = devnum; // global device number
	uint firstInnerParticle	= gdata->s_hStartPerDevice[myDevNum];
	uint howManyParticles	= gdata->s_hPartsPerDevice[myDevNum];

	size_t _size = 0;

	int buffer = gdata->currentPosRead;
	printf("Uploading to buffer %d\n", buffer);

	// memcpys - recalling GPU arrays are double buffered
	_size = howManyParticles * sizeof( m_hPos[ gdata->currentPosWrite ] ); // float4
	CUDA_SAFE_CALL(cudaMemcpy(m_dPos[ buffer ], m_hPos, _size, cudaMemcpyHostToDevice));
	_size = howManyParticles * sizeof( m_hVel[ gdata->currentVelWrite ] ); // float4
	CUDA_SAFE_CALL(cudaMemcpy(m_dVel[ buffer ], m_hVel, _size, cudaMemcpyHostToDevice));
	_size = howManyParticles * sizeof( m_hInfo[ gdata->currentInfoWrite ] ); // particleInfo
	CUDA_SAFE_CALL(cudaMemcpy(m_dInfo[ buffer ], m_hInfo, _size, cudaMemcpyHostToDevice));
}

// download the subdomain to the private member arrays
void GPUWorker::downloadSubdomain() {
	// indices
	uint myDevNum = devnum; // global device number
	uint firstInnerParticle	= gdata->s_hStartPerDevice[myDevNum];
	uint howManyParticles	= gdata->s_hPartsPerDevice[myDevNum];

	size_t _size = 0;

	// memcpys - recalling GPU arrays are double buffered
	_size = howManyParticles * sizeof( m_hPos[ gdata->currentPosRead ] ); // float4
	CUDA_SAFE_CALL(cudaMemcpy(	m_hPos,
								m_dPos[ gdata->currentPosRead ],
								_size, cudaMemcpyDeviceToHost));

	_size = howManyParticles * sizeof( m_hVel[ gdata->currentVelRead ] ); // float4
	CUDA_SAFE_CALL(cudaMemcpy(	m_hVel,
								m_dVel[ gdata->currentVelRead ],
								_size, cudaMemcpyDeviceToHost));

	_size = howManyParticles * sizeof( m_hInfo[ gdata->currentInfoRead ] ); // particleInfo
	CUDA_SAFE_CALL(cudaMemcpy(	m_hInfo,
								m_dInfo[ gdata->currentInfoRead ],
								_size, cudaMemcpyDeviceToHost));
}

// download the subdomain to the shared CPU arrays
void GPUWorker::downloadSubdomainToGlobalBuffer() {
	// indices
	uint myDevNum = devnum; // global device number
	uint firstInnerParticle	= gdata->s_hStartPerDevice[myDevNum];
	uint howManyParticles	= gdata->s_hPartsPerDevice[myDevNum];

	size_t _size = 0;

	// memcpys - recalling GPU arrays are double buffered
	_size = howManyParticles * sizeof( m_hPos[ gdata->currentPosRead ] ); // float4
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hPos + firstInnerParticle,
								m_dPos[ gdata->currentPosRead ],
								_size, cudaMemcpyDeviceToHost));

	_size = howManyParticles * sizeof( m_hVel[ gdata->currentVelRead ] ); // float4
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hVel + firstInnerParticle,
								m_dVel[ gdata->currentVelRead ],
								_size, cudaMemcpyDeviceToHost));

	_size = howManyParticles * sizeof( m_hInfo[ gdata->currentInfoRead ] ); // particleInfo
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hInfo + firstInnerParticle,
								m_dInfo[ gdata->currentInfoRead ],
								_size, cudaMemcpyDeviceToHost));
}

// create a compact device map, for this device, from the global one,
// with each cell being marked in the high bits
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

	// iterate on all cells of the world
	for (int ix=0; ix < gdata->gridSize.x; ix++)
		for (int iy=0; iy < gdata->gridSize.y; iy++)
			for (int iz=0; iz < gdata->gridSize.z; iz++) {
				// data of current cell
				uint cell_lin_idx = gdata->calcGridHashHost(ix, iy, iz);
				uint cell_devnum = gdata->s_hDeviceMap[cell_lin_idx];
				bool is_mine = (cell_devnum == devnum);
				// aux vars for iterating on neibs
				bool any_foreign_neib = false; // at least one neib does not belong to me?
				bool any_mine_neib = false; // at least one neib does belong to me?
				bool enough_info = false; // when true, stop iterating on neibs
				// iterate on neighbors
				for (int dx=-1; dx <= 1 && !enough_info; dx++)
					for (int dy=-1; dy <= 1 && !enough_info; dy++)
						for (int dz=-1; dz <= 1 && !enough_info; dz++)
							// check we are in the grid
							// TODO: modulus for periodic boundaries
							if (ix + dx < 0 || ix + dx >= gdata->gridSize.x ||
								iy + dy < 0 || iy + dy >= gdata->gridSize.y ||
								iz + dz < 0 || iz + dz >= gdata->gridSize.z) continue;
							else
							// do not iterate on self
							if (!(dx == 0 && dy == 0 && dz == 0)) {
								// data of neib cell
								uint neib_lin_idx = gdata->calcGridHashHost(ix + dx, iy + dy, iz + dz);
								uint neib_devnum = gdata->s_hDeviceMap[neib_lin_idx];
								any_mine_neib =		(neib_devnum == cell_devnum);
								any_foreign_neib =	(neib_devnum != cell_devnum);
								// did we read enough to decide for current cell?
								enough_info = (is_mine && any_foreign_neib) || (!is_mine && any_mine_neib);
							}
				uint cellType;
				if (is_mine && !any_foreign_neib)	cellType = INNER_CELL;
				if (is_mine && any_foreign_neib)	cellType = INNER_EDGE_CELL;
				if (!is_mine && any_mine_neib)		cellType = OUTER_EDGE_CELL;
				if (!is_mine && !any_mine_neib)		cellType = OUTER_CELL;
				m_hCompactDeviceMap[cell_lin_idx] = cellType;
			}
}

// self-explanatory
void GPUWorker::uploadCompactDeviceMap() {
	size_t _size = m_nGridCells * sizeof( m_dCompactDeviceMap[0] );
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

unsigned int GPUWorker::getDeviceNumber() {
	return devnum;
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

void GPUWorker::setDeviceProperties(cudaDeviceProp _m_deviceProperties) {
	m_deviceProperties = _m_deviceProperties;
}

// Actual thread calling GPU-methods
void* GPUWorker::simulationThread(void *ptr) {
	// INITIALIZATION PHASE

	// take the pointer of the instance starting this thread
	GPUWorker* instance = (GPUWorker*) ptr;

	// retrieve GlobalData and device number (index in process array)
	const GlobalData* gdata = instance->getGlobalData();
	const unsigned int devnum = instance->getDeviceNumber();

	instance->setDeviceProperties( checkCUDA(gdata, devnum) );

	// upload constants (PhysParames, some SimParams)
	instance->uploadConstants();

	// allocate CPU and GPU arrays
	instance->allocateHostBuffers();
	instance->allocateDeviceBuffers();

	// create and upload the compact device map (2 bits per cell)
	if (gdata->devices>1) {
		instance->createCompactDeviceMap();
		instance->uploadCompactDeviceMap();
	}

	// TODO: here set_reduction_params() will be called (to be implemented in this class). These parameters can be device-specific.

	// TODO: here setDemTexture() will be called. It is device-wide, but reading the DEM file is process wide and will be in GPUSPH class

	gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***
	//printf("Thread %d, wait for the initialization\n", devnum);

	// here GPUSPH::initialize is over and GPUSPH::runSimulation() is called

	gdata->threadSynchronizer->barrier(); // begins UPLOAD ***
	//printf("Thread %d, beings the upload\n", devnum);

	instance->uploadSubdomain();

	gdata->threadSynchronizer->barrier();  // end of UPLOAD, begins SIMULATION ***
	//printf("Thread %d, beings the simulation\n", devnum);

	// TODO
	// Here is a copy-paste from the CPU thread worker of branch cpusph, as a canvas
	while (gdata->keep_going) {
		switch (gdata->nextCommand) {
			// logging here?
			case IDLE:
				//printf("Thread %d, IDLE command\n", devnum);
				break;
			case CALCHASH:
				//printf("Thread %d, CALCHASH command\n", devnum);
				instance->kernel_calcHash();
				break;
			case SORT:
				//printf("Thread %d, SORT command\n", devnum);
				instance->kernel_sort();
				break;
			case REORDER:
				//printf("Thread %d, REORDER command\n", devnum);
				instance->kernel_reorderDataAndFindCellStart();
				break;
			case BUILDNEIBS:
				//printf("Thread %d, BUILDNEIBS command\n", devnum);
				instance->kernel_buildNeibsList();
				break;
			case FORCES:
				//printf("Thread %d, FORCE command\n", devnum);
				instance->kernel_forces();
				break;
			case EULER:
				//printf("Thread %d, EULER command\n", devnum);
				instance->kernel_euler();
				break;
			case QUIT:
				//printf("Thread %d, QUIT command\n", devnum);
				// actually, setting keep_going to false and unlocking the barrier should be enough to quit the cycle
				break;
		}
		if (gdata->keep_going) {
			// the first barrier waits for the main thread to set the next command; the second is to unlock
			//printf("Thread %d, arriving at CYCLE BARRIER 1\n", devnum);
			gdata->threadSynchronizer->barrier();  // CYCLE BARRIER 1
			//printf("Thread %d, out of CYCLE BARRIER 1\n", devnum);
			//printf("Thread %d, arriving at CYCLE BARRIER 2\n", devnum);
			gdata->threadSynchronizer->barrier();  // CYCLE BARRIER 2
			//printf("Thread %d, out of CYCLE BARRIER 2\n", devnum);
		}
	}

	gdata->threadSynchronizer->barrier();  // end of SIMULATION, begins FINALIZATION ***
	//printf("Thread %d, out of simulation\n", devnum);

	// deallocate buffers
	instance->deallocateHostBuffers();
	instance->deallocateDeviceBuffers();
	// ...what else?

	gdata->threadSynchronizer->barrier();  // end of FINALIZATION ***
	//printf("Thread %d, out of finalization\n", devnum);

	pthread_exit(NULL);
}

void GPUWorker::kernel_calcHash()
{
	calcHash(m_dPos[gdata->currentPosRead],
#if HASH_KEY_SIZE >= 64
					m_dInfo[gdata->currentInfoRead],
#endif
					m_dParticleHash,
					m_dParticleIndex,
					gdata->gridSize,
					gdata->cellSize,
					gdata->worldOrigin,
					m_numParticles);
}

void GPUWorker::kernel_sort()
{
	sort(m_dParticleHash, m_dParticleIndex, m_numParticles);
}

void GPUWorker::kernel_reorderDataAndFindCellStart()
{
	reorderDataAndFindCellStart(m_dCellStart,	  // output: cell start index
							m_dCellEnd,		// output: cell end index
							m_dPos[gdata->currentPosWrite],		 // output: sorted positions
							m_dVel[gdata->currentVelWrite],		 // output: sorted velocities
							m_dInfo[gdata->currentInfoWrite],		 // output: sorted info
							m_dParticleHash,
							m_dParticleIndex,  // input: sorted particle indices
							m_dPos[gdata->currentPosRead],		 // input: sorted position array
							m_dVel[gdata->currentVelRead],		 // input: sorted velocity array
							m_dInfo[gdata->currentInfoRead],		 // input: sorted info array
							m_numParticles,
							m_nGridCells);
}

void GPUWorker::kernel_buildNeibsList()
{
	buildNeibsList(	m_dNeibsList,
						m_dPos[gdata->currentPosRead],
						m_dInfo[gdata->currentInfoRead],
						m_dParticleHash,
						m_dCellStart,
						m_dCellEnd,
						gdata->gridSize,
						gdata->cellSize,
						gdata->worldOrigin,
						m_numParticles,
						m_nGridCells,
						m_simparams->nlInfluenceRadius,
						m_simparams->periodicbound);
}

void GPUWorker::kernel_forces()
{
	float returned_dt = 0.0F;
	bool firstStep = (gdata->step == 1);
	if (firstStep)
		returned_dt = forces(  m_dPos[gdata->currentPosRead],   // pos(n)
						m_dVel[gdata->currentVelRead],   // vel(n)
						m_dForces,					// f(n)
						0, // float* rbforces
						0, // float* rbtorques
						m_dXsph,
						m_dInfo[gdata->currentInfoRead],
						m_dNeibsList,
						m_numParticles,
						m_simparams->slength,
						gdata->dt, // m_dt,
						m_simparams->dtadapt,
						m_simparams->dtadaptfactor,
						m_simparams->xsph,
						m_simparams->kerneltype,
						m_simparams->influenceRadius,
						m_simparams->visctype,
						m_physparams->visccoeff,
						m_dCfl,
						m_dTempCfl,
						m_numPartsFmax,
						m_dTau,
						m_simparams->periodicbound,
						m_simparams->sph_formulation,
						m_simparams->boundarytype,
						m_simparams->usedem );
	else
		returned_dt = forces(  m_dPos[gdata->currentPosWrite],  // pos(n+1/2)
						m_dVel[gdata->currentVelWrite],  // vel(n+1/2)
						m_dForces,					// f(n+1/2)
						0, // float* rbforces,
						0, // float* rbtorques,
						m_dXsph,
						m_dInfo[gdata->currentInfoRead],
						m_dNeibsList,
						m_numParticles,
						m_simparams->slength,
						gdata->dt, // m_dt,
						m_simparams->dtadapt,
						m_simparams->dtadaptfactor,
						m_simparams->xsph,
						m_simparams->kerneltype,
						m_simparams->influenceRadius,
						m_simparams->visctype,
						m_physparams->visccoeff,
						m_dCfl,
						m_dTempCfl,
						m_numPartsFmax,
						m_dTau,
						m_simparams->periodicbound,
						m_simparams->sph_formulation,
						m_simparams->boundarytype,
						m_simparams->usedem );

	// gdata->dts is directly used instead of handling dt1 and dt2
	if (firstStep)
		gdata->dts[devnum] = returned_dt;
	else
		gdata->dts[devnum] = min(gdata->dts[devnum], returned_dt);
}

void GPUWorker::kernel_euler()
{
	if (gdata->step == 1)

		euler(  m_dPos[gdata->currentPosRead],   // pos(n)
				m_dVel[gdata->currentVelRead],   // vel(n)
				m_dInfo[gdata->currentInfoRead], //particleInfo
				m_dForces,					// f(n+1/2)
				m_dXsph,
				m_dPos[gdata->currentPosWrite],  // pos(n+1) = pos(n) + velc(n+1/2)*dt
				m_dVel[gdata->currentVelWrite],  // vel(n+1) = vel(n) + f(n+1/2)*dt
				m_numParticles,
				gdata->dt, // m_dt,
				gdata->dt/2.0f, // m_dt/2.0,
				1,
				gdata->t + gdata->dt / 2.0f, // + m_dt,
				m_simparams->xsph,
				m_simparams->periodicbound);
	else
		euler(  m_dPos[gdata->currentPosRead],   // pos(n)
				m_dVel[gdata->currentVelRead],   // vel(n)
				m_dInfo[gdata->currentInfoRead], //particleInfo
				m_dForces,					// f(n+1/2)
				m_dXsph,
				m_dPos[gdata->currentPosWrite],  // pos(n+1) = pos(n) + velc(n+1/2)*dt
				m_dVel[gdata->currentVelWrite],  // vel(n+1) = vel(n) + f(n+1/2)*dt
				m_numParticles,
				gdata->dt, // m_dt,
				gdata->dt/2.0f, // m_dt/2.0,
				2,
				gdata->t + gdata->dt,// + m_dt,
				m_simparams->xsph,
				m_simparams->periodicbound);
}

void GPUWorker::uploadConstants()
{
	// NOTE: visccoeff must be set before uploading the constants. This is done in GPUSPH main cycle

	// Setting kernels and kernels derivative factors
	setforcesconstants(m_simparams, m_physparams);
	seteulerconstants(m_physparams);
	setneibsconstants(m_simparams, m_physparams);
}


