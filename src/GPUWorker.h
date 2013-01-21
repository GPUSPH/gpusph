/*
 * GPUWorker.h
 *
 *  Created on: Dec 19, 2012
 *      Author: rustico
 */

#ifndef GPUWORKER_H_
#define GPUWORKER_H_

#include <pthread.h>

class GPUWorker {
private:
	pthread_t pthread_id;
	static void* simulationThread(void *ptr);
public:
	GPUWorker();
	~GPUWorker();
	size_t allocateDevice();
	void uploadSubdomains();
	void createCompactDeviceMap();
	void uploadCompactDeviceMap();
	void run_worker();
};

#endif /* GPUWORKER_H_ */
