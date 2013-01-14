/*
 * Synchronizer.h
 *
 *  Created on: Jan 14, 2013
 *      Author: rustico
 */

#ifndef SYNCHRONIZER_H_
#define SYNCHRONIZER_H_

#include <pthread.h>

class Synchronizer {
private:
	unsigned int nThreads;
	unsigned int reached;
	pthread_mutex_t syncMutex;
	pthread_cond_t syncCondition;
	bool forcesUnlockOccurred;
public:
	Synchronizer(unsigned int numThreads);
	~Synchronizer();
	void barrier();
	void forceUnlock();
	unsigned int queryReachedThreads();
};

#endif /* SYNCHRONIZER_H_ */
