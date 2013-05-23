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
	unsigned int m_nThreads;
	unsigned int m_reached;
	pthread_mutex_t m_syncMutex;
	pthread_cond_t m_syncCondition;
	bool m_forcesUnlockOccurred;
public:
	Synchronizer(unsigned int numThreads);
	~Synchronizer();
	void barrier();
	void forceUnlock();
	unsigned int queryReachedThreads();
	unsigned int getNumThreads();
	bool didForceUnlockOccurr();
};

#endif /* SYNCHRONIZER_H_ */
