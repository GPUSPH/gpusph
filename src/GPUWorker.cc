/*
 * GPUWorker.cpp
 *
 *  Created on: Dec 19, 2012
 *      Author: rustico
 */

#include "GPUWorker.h"

GPUWorker::GPUWorker() {
	// Init class. Fire thread?

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
	// this is a wrapper to actually call pthread_create(simulationThread, ...)
}

void GPUWorker::simulationThread() {
	// static method to be run as a separate pthread
}


