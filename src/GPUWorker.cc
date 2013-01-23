/*
 * GPUWorker.cpp
 *
 *  Created on: Dec 19, 2012
 *      Author: rustico
 */

#include "GPUWorker.h"

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
}

GPUWorker::~GPUWorker() {
	// Free everything and pthread terminate
}

// All the allocators assume that gdata is updated with the number of particles (done by problem->fillparts).
// Later this will be changed since each thread does not need to allocate the global number of particles.
size_t GPUWorker::allocateHostBuffers() {
	// common sizes
	const uint float3Size = sizeof(float3)*m_numParticles;
	const uint float4Size = sizeof(float4)*m_numParticles;
	const uint infoSize = sizeof(particleinfo)*m_numParticles;

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

	if (m_simparams->vorticity) {
		m_hVort = new float3[m_numParticles];
		allocated += float3Size;
		// NOTE: *not* memsetting, as in master branch
	}

	return allocated;
}

size_t GPUWorker::allocateDeviceBuffers() {
	// stub
}

void GPUWorker::deallocateHostBuffers() {
	// stub
}

void GPUWorker::deallocateDeviceBuffers() {
	// stub
}

void GPUWorker::uploadSubdomains() {
	// upload subdomain, just allocated and sorted by main thread
}

void GPUWorker::createCompactDeviceMap() {
	// create a compact device map, for this device, from the global one,
	// with each cell being marked in the high bits

}

void GPUWorker::uploadCompactDeviceMap() {
	// create a compact device map, for this device, from the global one,
	// with each cell being marked in the high bits

}

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

	//		GPUWorkers > allocateGPU
	//		GPUWorkers > uploadSubdomains (cell by cell, light optimizations)
	//			incl. edging!
	//		GPUWorkers > createCompactDevMap (from global devmap to 2bits/dev)
	//		GPUWorkers > uploadCompactDevMap (2 bits per cell, to be elaborated on this)

	gdata->threadSynchronizer->barrier(); // end of INITIALIZATION ***
	gdata->threadSynchronizer->barrier(); // begins UPLOAD ***

	// TODO

	gdata->threadSynchronizer->barrier();  // end of UPLOAD, begins SIMULATION ***

	// TODO

	gdata->threadSynchronizer->barrier();  // end of SIMULATION, begins FINALIZATION ***

	// TODO

	gdata->threadSynchronizer->barrier();  // end of FINALIZATION ***

	pthread_exit(NULL);
}


