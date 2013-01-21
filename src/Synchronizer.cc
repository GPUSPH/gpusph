/*
 * Synchronizer.cc
 *
 *  Created on: Jan 14, 2013
 *      Author: rustico
 */

#include "Synchronizer.h"

Synchronizer::Synchronizer(unsigned int numThreads)
{
	nThreads = numThreads;
	reached = 0;
	forcesUnlockOccurred = false;
	pthread_mutex_init(&syncMutex, NULL);
	pthread_cond_init(&syncCondition, NULL);
}

Synchronizer::~Synchronizer()
{
	pthread_mutex_destroy(&syncMutex);
	pthread_cond_destroy(&syncCondition);
}

// optimized for speed: barriers are awaken only when numThreads is reached, not at any increment
void Synchronizer::barrier() {
	pthread_mutex_lock(&syncMutex);
	reached++;
	if (reached < nThreads && !forcesUnlockOccurred)
		pthread_cond_wait(&syncCondition, &syncMutex);
	else {
		reached = 0;
		pthread_cond_broadcast(&syncCondition);
	}
	pthread_mutex_unlock(&syncMutex);
}

// Emergency stop: broadcast signal to awake everyone and reset reached
// To avoid race conditions, after calling forceUnlock the synchronizer does not work
// (if a barrier is called again, it does not block anymore)
void Synchronizer::forceUnlock() {
	pthread_mutex_lock(&syncMutex);
	reached = 0;
	forcesUnlockOccurred = true;
	pthread_cond_broadcast(&syncCondition);
	pthread_mutex_unlock(&syncMutex);
}

// thread-unsafe; use only for debugging or to double check after barrier was reached
unsigned int Synchronizer::queryReachedThreads() {
	return reached;
}
