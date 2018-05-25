/*
 * Synchronizer.cc
 *
 *  Created on: Jan 14, 2013
 *      Author: rustico
 */

#include "Synchronizer.h"
using namespace std;

Synchronizer::Synchronizer(unsigned int numThreads)
{
	m_nThreads = numThreads;
	m_reached = 0;
	m_forcesUnlockOccurred = false;
}

// optimized for speed: barriers are awaken only when numThreads is reached, not at any increment
void Synchronizer::barrier() {
	unique_lock<mutex> lock(m_syncMutex);
	m_reached++;
	if (m_reached < m_nThreads && !m_forcesUnlockOccurred)
		m_syncCondition.wait(lock);
	else {
		m_reached = 0;
		m_syncCondition.notify_all();
	}
}

// Emergency stop: broadcast signal to awake everyone and reset reached
// To avoid race conditions, after calling forceUnlock the synchronizer does not work
// (if a barrier is called again, it does not block anymore)
void Synchronizer::forceUnlock() {
	lock_guard<mutex> lock(m_syncMutex);
	m_reached = 0;
	m_forcesUnlockOccurred = true;
	m_syncCondition.notify_all();
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
