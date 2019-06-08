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
