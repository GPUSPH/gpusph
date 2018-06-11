/*
 * Synchronizer.h
 *
 *  Created on: Jan 14, 2013
 *      Author: rustico
 */

/*! \file
 * Cross-thread syncronization
 */

#ifndef SYNCHRONIZER_H_
#define SYNCHRONIZER_H_

#include <mutex>
#include <condition_variable>

class Synchronizer {
private:
	unsigned int m_nThreads;
	unsigned int m_reached;
	std::mutex m_syncMutex;
	std::condition_variable m_syncCondition;
	bool m_forcesUnlockOccurred;
public:
	Synchronizer(unsigned int numThreads);
	void barrier();
	void forceUnlock();
	unsigned int queryReachedThreads();
	unsigned int getNumThreads();
	bool didForceUnlockOccurr();
};

#endif /* SYNCHRONIZER_H_ */
