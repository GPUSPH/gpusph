/*  Copyright (c) 2013-2018 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
 */
/*
 * Synchronizer.cc
 *
 *  Created on: Jan 14, 2013
 *      Author: rustico
 */

/*! \file
 * Cross-thread syncronization implementation
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
