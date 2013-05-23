/*
 * Synchronizer.cc
 *
 *  Created on: Jan 14, 2013
 *      Author: rustico
 */

#include "Synchronizer.h"

Synchronizer::Synchronizer(unsigned int numThreads)
{
	m_nThreads = numThreads;
	m_reached = 0;
	m_forcesUnlockOccurred = false;
	pthread_mutex_init(&m_syncMutex, NULL);
	pthread_cond_init(&m_syncCondition, NULL);
}

Synchronizer::~Synchronizer()
{
	pthread_mutex_destroy(&m_syncMutex);
	pthread_cond_destroy(&m_syncCondition);
}

// optimized for speed: barriers are awaken only when numThreads is reached, not at any increment
void Synchronizer::barrier() {
	pthread_mutex_lock(&m_syncMutex);
	m_reached++;
	if (m_reached < m_nThreads && !m_forcesUnlockOccurred)
		pthread_cond_wait(&m_syncCondition, &m_syncMutex);
	else {
		m_reached = 0;
		pthread_cond_broadcast(&m_syncCondition);
	}
	pthread_mutex_unlock(&m_syncMutex);
}

// Emergency stop: broadcast signal to awake everyone and reset reached
// To avoid race conditions, after calling forceUnlock the synchronizer does not work
// (if a barrier is called again, it does not block anymore)
void Synchronizer::forceUnlock() {
	pthread_mutex_lock(&m_syncMutex);
	m_reached = 0;
	m_forcesUnlockOccurred = true;
	pthread_cond_broadcast(&m_syncCondition);
	pthread_mutex_unlock(&m_syncMutex);
}

// thread-unsafe; use only for debugging or to double check after barrier was reached
unsigned int Synchronizer::queryReachedThreads() {
	return m_reached;
}

// get the number of threads needed by the barrier to unlock
unsigned int Synchronizer::getNumThreads()
{
	return m_nThreads;
}

// did we already try to force unlock?
bool Synchronizer::didForceUnlockOccurr()
{
	return m_forcesUnlockOccurred;
}
