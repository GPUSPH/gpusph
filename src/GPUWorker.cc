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
}

GPUWorker::~GPUWorker() {
	// Free everything and pthread terminate
}

size_t GPUWorker::allocateDevice() {
	// allocate GPU dev memory
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

GlobalData* GPUWorker::getGlobalData() {
	return gdata;
}

void* GPUWorker::simulationThread(void *ptr) {
	// actual thread calling GPU-methods

	// take the pointer of the instance starting this thread
	GPUWorker* instance = (GPUWorker*) ptr;

	// TODO
}


