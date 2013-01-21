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
#include "GlobalData.h"

/* We need a forward declaration of GlobalData.
 * When the compiler includes "GlobalData.h" from somewhere else, it defines _GLOBAL_DATA_
 * and in turn includes "GPUWorker.h"; but the latter does not know the GlobalData struct
 * yet and including GloblData.h again does not work since _GLOBAL_DATA_ is defined.
 * So we need to forward-declare the struct GlobalData. GPUWorker finds it and compiles. */
struct GlobalData;

class GPUWorker {
private:
	pthread_t pthread_id;
	static void* simulationThread(void *ptr);
	GlobalData* gdata;
	unsigned int devnum;
	GlobalData* getGlobalData();
	unsigned int getDeviceNumber();
public:
	GPUWorker(GlobalData* _gdata, unsigned int _devnum);
	~GPUWorker();
	size_t allocateDevice();
	void uploadSubdomains();
	void createCompactDeviceMap();
	void uploadCompactDeviceMap();
	void run_worker();
	void join_worker();
};

#endif /* GPUWORKER_H_ */
