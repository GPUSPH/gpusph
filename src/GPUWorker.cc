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

GPUWorker::GPUWorker(GlobalData* _gdata, unsigned int _deviceIndex) {
	gdata = _gdata;
	m_deviceIndex = _deviceIndex;
	m_cudaDeviceNumber = gdata->device[m_deviceIndex];

	m_hPos = NULL;
	m_hVel = NULL;
	m_hInfo = NULL;
	m_hVort = NULL;

	// we know that GPUWorker is initialized when Problem was already
	m_simparams = gdata->problem->get_simparams();
	m_physparams = gdata->problem->get_physparams();

	// we also know Problem::fillparts() has already been called
	m_numParticles = gdata->s_hPartsPerDevice[m_deviceIndex];
	uint _estROParts = estimateROParticles();
	m_numAlocatedParticles = m_numParticles + _estROParts;
	m_nGridCells = gdata->nGridCells;

	printf("Device idx %u (CUDA: %u) will allocate %u (assigned) + %u (estimated r.o.) = %u particles\n",
		m_deviceIndex, m_cudaDeviceNumber, m_numParticles, _estROParts, m_numAlocatedParticles);

	m_hostMemory = m_deviceMemory = 0;
}

GPUWorker::~GPUWorker() {
	// Free everything and pthread terminate
	// should check whether the pthread is still running and force its termination?
}

// Return the number of particles currently being handled (internal and r.o.)
uint GPUWorker::getNumParticles()
{
	return m_numParticles;
}

// Return the maximum number of particles the worker can handled (allocated)
uint GPUWorker::getMaxParticles()
{
	return m_numAlocatedParticles;
}

// Estimate the number of r.o. particles the worker might need
// NOTE: assuming GlobalData::totParticles, deviceMap, etc. have been already filled
// TODO: make a more realisti estimation, e.g. by counting the neighbor cells
uint GPUWorker::estimateROParticles()
{
	return gdata->s_hPartsPerDevice[m_deviceIndex] * 1.5f;
}

// Cut all particles that are not internal.
// Assuming segments have already been filled and downloaded to the shared array
void GPUWorker::dropExternalParticles()
{
	// We would like to trim out all external particles. According to the sorting criteria,
	// they should be compacted last. If there are no external particles, their segmentStart
	// is equal to m_numParticles
	uint external_start_at = min( gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_EDGE_CELL],
			gdata->s_dSegmentsStart[m_deviceIndex][CELLTYPE_OUTER_CELL] );
	if (external_start_at > m_numParticles)
		printf("WARNING: thread %u: first outer particle (%u) beyond active particles (%u)! Not cropping\n",
				m_deviceIndex, external_start_at, m_numParticles);
	else
		m_numParticles = external_start_at;
}

// All the allocators assume that gdata is updated with the number of particles (done by problem->fillparts).
// Later this will be changed since each thread does not need to allocate the global number of particles.
size_t GPUWorker::allocateHostBuffers() {
	// common sizes
	const uint float3Size = sizeof(float3) * m_numAlocatedParticles;
	const uint float4Size = sizeof(float4) * m_numAlocatedParticles;
	const uint infoSize = sizeof(particleinfo) * m_numAlocatedParticles;
	const uint uintCellsSize = sizeof(uint) * m_nGridCells;

	size_t allocated = 0;

	m_hPos = new float4[m_numAlocatedParticles];
	memset(m_hPos, 0, float4Size);
	allocated += float4Size;

	m_hVel = new float4[m_numAlocatedParticles];
	memset(m_hVel, 0, float4Size);
	allocated += float4Size;

	m_hInfo = new particleinfo[m_numAlocatedParticles];
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
		m_hVort = new float3[m_numAlocatedParticles];
		allocated += float3Size;
		// NOTE: *not* memsetting, as in master branch
	}

	m_hostMemory += allocated;
	return allocated;
}

size_t GPUWorker::allocateDeviceBuffers() {
	// common sizes
	// compute common sizes (in bytes)
	//const uint floatSize = sizeof(float) * m_numAlocatedParticles;
	const uint float2Size = sizeof(float2) * m_numAlocatedParticles;
	const uint float3Size = sizeof(float3) * m_numAlocatedParticles;
	const uint float4Size = sizeof(float4) * m_numAlocatedParticles;
	const uint infoSize = sizeof(particleinfo) * m_numAlocatedParticles;
	const uint intSize = sizeof(uint) * m_numAlocatedParticles;
	const uint uintCellsSize = sizeof(uint) * m_nGridCells;
	const uint neibslistSize = sizeof(uint) * m_simparams->maxneibsnum*(m_numAlocatedParticles/NEIBINDEX_INTERLEAVE + 1)*NEIBINDEX_INTERLEAVE;
	const uint hashSize = sizeof(hashKey) * m_numAlocatedParticles;
	const uint segmentsSize = sizeof(uint) * 4; // 4 = types of cells
	//const uint neibslistSize = sizeof(uint) * 128 * m_numAlocatedParticles;
	//const uint sliceArraySize = sizeof(uint) * m_gridSize.PSA;

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
	CUDA_SAFE_CALL(cudaMemset(m_dNeibsList, 0xffffffff, neibslistSize));
	allocated += neibslistSize;

	// TODO: an array of uchar would suffice
	if (gdata->devices > 1) {
		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dCompactDeviceMap, uintCellsSize));
		allocated += uintCellsSize;

		CUDA_SAFE_CALL(cudaMalloc((void**)&m_dSegmentStart, segmentsSize));
		allocated += segmentsSize;
	}

	// TODO: allocation for rigid bodies

	if (m_simparams->dtadapt) {
		m_numPartsFmax = getNumPartsFmax(m_numAlocatedParticles);
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

	if (gdata->devices > 1) {
		CUDA_SAFE_CALL(cudaFree(m_dCompactDeviceMap));
		CUDA_SAFE_CALL(cudaFree(m_dSegmentStart));
	}

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
	uint firstInnerParticle	= gdata->s_hStartPerDevice[m_deviceIndex];
	uint howManyParticles	= gdata->s_hPartsPerDevice[m_deviceIndex];

	size_t _size = 0;

	// memcpys - recalling GPU arrays are double buffered
	_size = howManyParticles * sizeof(float4);
	//printf("Thread %d uploading %d POS items (%u Kb) on device %d from position %d\n",
	//		m_deviceIndex, howManyParticles, (uint)_size/1000, m_cudaDeviceNumber, firstInnerParticle);
	CUDA_SAFE_CALL(cudaMemcpy(	m_dPos[ gdata->currentPosRead ],
								gdata->s_hPos + firstInnerParticle,
								_size, cudaMemcpyHostToDevice));

	_size = howManyParticles * sizeof(float4);
	//printf("Thread %d uploading %d VEL items (%u Kb) on device %d from position %d\n",
	//		m_deviceIndex, howManyParticles, (uint)_size/1000, m_cudaDeviceNumber, firstInnerParticle);
	CUDA_SAFE_CALL(cudaMemcpy(	m_dVel[ gdata->currentVelRead ],
								gdata->s_hVel + firstInnerParticle,
								_size, cudaMemcpyHostToDevice));

	_size = howManyParticles * sizeof(particleinfo);
	//printf("Thread %d uploading %d INFO items (%u Kb) on device %d from position %d\n",
	//		m_deviceIndex, howManyParticles, (uint)_size/1000, m_cudaDeviceNumber, firstInnerParticle);
	CUDA_SAFE_CALL(cudaMemcpy(	m_dInfo[ gdata->currentInfoRead ],
								gdata->s_hInfo + firstInnerParticle,
								_size, cudaMemcpyHostToDevice));
}

// DEPRECATED download the subdomain to the private member arrays
void GPUWorker::downloadSubdomain() {
	// indices
	uint firstInnerParticle	= gdata->s_hStartPerDevice[m_deviceIndex];
	uint howManyParticles	= gdata->s_hPartsPerDevice[m_deviceIndex];

	size_t _size = 0;

	// memcpys - recalling GPU arrays are double buffered
	_size = howManyParticles * sizeof(float4);
	CUDA_SAFE_CALL(cudaMemcpy(	m_hPos,
								m_dPos[ gdata->currentPosRead ],
								_size, cudaMemcpyDeviceToHost));

	_size = howManyParticles * sizeof(float4);
	CUDA_SAFE_CALL(cudaMemcpy(	m_hVel,
								m_dVel[ gdata->currentVelRead ],
								_size, cudaMemcpyDeviceToHost));

	_size = howManyParticles * sizeof(particleinfo);
	CUDA_SAFE_CALL(cudaMemcpy(	m_hInfo,
								m_dInfo[ gdata->currentInfoRead ],
								_size, cudaMemcpyDeviceToHost));
}

// download the subdomain to the shared CPU arrays
void GPUWorker::downloadSubdomainToGlobalBuffer() {
	// indices
	uint firstInnerParticle	= gdata->s_hStartPerDevice[m_deviceIndex];
	uint howManyParticles	= gdata->s_hPartsPerDevice[m_deviceIndex];

	size_t _size = 0;

	//if (m_deviceIndex==1) return;
	//printf(" - thread %d downloading stuff\n", m_deviceIndex);

	// memcpys - recalling GPU arrays are double buffered
	_size = howManyParticles * sizeof(float4);
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hPos + firstInnerParticle,
								m_dPos[ gdata->currentPosRead ],
								_size, cudaMemcpyDeviceToHost));

	_size = howManyParticles * sizeof(float4);
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hVel + firstInnerParticle,
								m_dVel[ gdata->currentVelRead ],
								_size, cudaMemcpyDeviceToHost));

	_size = howManyParticles * sizeof(particleinfo);
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_hInfo + firstInnerParticle,
								m_dInfo[ gdata->currentInfoRead ],
								_size, cudaMemcpyDeviceToHost));
}

// download cellStart and cellEnd to the shared arrays
void GPUWorker::downloadCellsIndices()
{
	size_t _size = gdata->nGridCells * sizeof(uint);
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_dCellStarts[m_deviceIndex],
								m_dCellStart,
								_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_dCellEnds[m_deviceIndex],
								m_dCellEnd,
								_size, cudaMemcpyDeviceToHost));
	_size = 4 * sizeof(uint);
	CUDA_SAFE_CALL(cudaMemcpy(	gdata->s_dSegmentsStart[m_deviceIndex],
									m_dSegmentStart,
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
				bool is_mine = (cell_devnum == m_deviceIndex);
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
								any_mine_neib	 |= (neib_devnum == cell_devnum);
								any_foreign_neib |= (neib_devnum != cell_devnum);
								// did we read enough to decide for current cell?
								enough_info = (is_mine && any_foreign_neib) || (!is_mine && any_mine_neib);
							}
				uint cellType;
				if (is_mine && !any_foreign_neib)	cellType = CELLTYPE_MASK_INNER_CELL;
				if (is_mine && any_foreign_neib)	cellType = CELLTYPE_MASK_INNER_EDGE_CELL;
				if (!is_mine && any_mine_neib)		cellType = CELLTYPE_MASK_OUTER_EDGE_CELL;
				if (!is_mine && !any_mine_neib)		cellType = CELLTYPE_MASK_OUTER_CELL;
				m_hCompactDeviceMap[cell_lin_idx] = cellType;
			}
}

// self-explanatory
void GPUWorker::uploadCompactDeviceMap() {
	size_t _size = m_nGridCells * sizeof(uint);
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

unsigned int GPUWorker::getCUDADeviceNumber()
{
	return m_cudaDeviceNumber;
}

unsigned int GPUWorker::getDeviceIndex()
{
	return m_deviceIndex;
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
	const unsigned int cudaDeviceNumber = instance->getCUDADeviceNumber();
	const unsigned int deviceIndex = instance->getDeviceIndex();

	instance->setDeviceProperties( checkCUDA(gdata, cudaDeviceNumber) );

	// upload constants (PhysParames, some SimParams)
	instance->uploadConstants();

	// allocate CPU and GPU arrays
	instance->allocateHostBuffers();
	instance->allocateDeviceBuffers();

	printf("Thread %u allocated %lu Kb on host, %lu Kb on device %u\n",
		deviceIndex, instance->getHostMemory()/1000, instance->getDeviceMemory()/1000, cudaDeviceNumber);

	// create and upload the compact device map (2 bits per cell)
	if (gdata->devices>1) {
		instance->createCompactDeviceMap();
		instance->uploadCompactDeviceMap();
	}

	// TODO: here set_reduction_params() will be called (to be implemented in this class). These parameters can be device-specific.

	// TODO: here setDemTexture() will be called. It is device-wide, but reading the DEM file is process wide and will be in GPUSPH class

	gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***

	// here GPUSPH::initialize is over and GPUSPH::runSimulation() is called

	gdata->threadSynchronizer->barrier(); // begins UPLOAD ***

	instance->uploadSubdomain();

	gdata->threadSynchronizer->barrier();  // end of UPLOAD, begins SIMULATION ***

	// TODO
	// Here is a copy-paste from the CPU thread worker of branch cpusph, as a canvas
	while (gdata->keep_going) {
		switch (gdata->nextCommand) {
			// logging here?
			case IDLE:
				break;
			case CALCHASH:
				//printf(" T %d issuing HASH\n", deviceIndex);
				instance->kernel_calcHash();
				break;
			case SORT:
				//printf(" T %d issuing SORT\n", deviceIndex);
				instance->kernel_sort();
				break;
			case CROP:
				//printf(" T %d issuing CROP\n", deviceIndex);
				instance->dropExternalParticles();
				break;
			case REORDER:
				//printf(" T %d issuing REORDER\n", deviceIndex);
				instance->kernel_reorderDataAndFindCellStart();
				break;
			case BUILDNEIBS:
				//printf(" T %d issuing BUILDNEIBS\n", deviceIndex);
				instance->kernel_buildNeibsList();
				break;
			case FORCES:
				//printf(" T %d issuing FORCES\n", deviceIndex);
				instance->kernel_forces();
				break;
			case EULER:
				//printf(" T %d issuing EULER\n", deviceIndex);
				instance->kernel_euler();
				break;
			case DUMP:
				//printf(" T %d issuing DUMP\n", deviceIndex);
				instance->downloadSubdomainToGlobalBuffer();
				break;
			case DUMP_CELLS:
				//printf(" T %d issuing DUMP_CELLS\n", deviceIndex);
				instance->downloadCellsIndices();
				break;
			case QUIT:
				//printf(" T %d issuing QUIT\n", deviceIndex);
				// actually, setting keep_going to false and unlocking the barrier should be enough to quit the cycle
				break;
		}
		if (gdata->keep_going) {
			// the first barrier waits for the main thread to set the next command; the second is to unlock
			gdata->threadSynchronizer->barrier();  // CYCLE BARRIER 1
			gdata->threadSynchronizer->barrier();  // CYCLE BARRIER 2
		}
	}

	gdata->threadSynchronizer->barrier();  // end of SIMULATION, begins FINALIZATION ***

	// deallocate buffers
	instance->deallocateHostBuffers();
	instance->deallocateDeviceBuffers();
	// ...what else?

	gdata->threadSynchronizer->barrier();  // end of FINALIZATION ***

	pthread_exit(NULL);
}

void GPUWorker::kernel_calcHash()
{
	calcHash(m_dPos[gdata->currentPosRead],
#if HASH_KEY_SIZE >= 64
					m_dInfo[gdata->currentInfoRead],
					m_dCompactDeviceMap,
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
#if HASH_KEY_SIZE >= 64
							m_dSegmentStart,
#endif
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
						m_simparams->nlSqInfluenceRadius,
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
	//printf(" Step %d, bool %d, returned %g, current %g, ",
	//	gdata->step, firstStep, returned_dt, gdata->dts[devnum]);
	if (firstStep)
		gdata->dts[m_deviceIndex] = returned_dt;
	else
		gdata->dts[m_deviceIndex] = min(gdata->dts[m_deviceIndex], returned_dt);
	//printf("set to %g\n",gdata->dts[m_deviceIndex]);
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


