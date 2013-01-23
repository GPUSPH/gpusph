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

	// we know that GPUWorker is initialized when Problem was already
	m_simparams = gdata->problem->get_simparams();
	m_physparams = gdata->problem->get_physparams();
}

GPUWorker::~GPUWorker() {
	// Free everything and pthread terminate
}

// All the allocators assume that gdata is updated with the number of particles (done by problem->fillparts).
// Later this will be changed since each thread does not need to allocate the global number of particles.
size_t GPUWorker::allocateHostBuffers() {
	// stub
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


