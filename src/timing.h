/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

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

/* Timing info */

#ifndef _TIMING_H
#define _TIMING_H

#include <time.h>
#include <exception>

typedef unsigned int uint;
typedef unsigned long ulong;

typedef struct TimingInfo {
	// current simulation time
	//float   t;
	// current simulation timestep
	//float   dt;
	// current number of particles
	//uint	numParticles;
	// current maximum number of neibs
	uint	maxNeibs;
	// iterations done so far
	//ulong	iterations;
	// number of particle-particle interactions with current neiblist
	uint	numInteractions;
	// average number of particle-particle interactions
	//ulong	meanNumInteractions;
	// time taken to build the neiblist (latest)
	//float   timeNeibsList;
	// avg. time  to build the neiblist
	//float   meanTimeNeibsList;
	// time taken to compute interactions (latest)
	//float   timeInteract;
	// avg. time to compute interactions
	//float   meanTimeInteract;
	// time taken to integrate (latest)
	//float   timeEuler;
	// avg. time to integrate
	//double  meanTimeEuler;

	// number of iterations times number of particles

	/* Note: this is computed by adding to it the current number of particles
	 * after each iteration, to ensure the correct value even when the number of
	 * particles changes during the simulation
	 */
	//ulong	iterTimesParts;

	/*
	TimingInfo(void) : t(0), dt(0), numParticles(0), maxNeibs(0),
		iterations(0), numInteractions(0), meanNumInteractions(0),
		timeNeibsList(0), meanTimeNeibsList(0),
		timeInteract(0), meanTimeInteract(0),
		timeEuler(0), meanTimeEuler(0),
		startTime(0), iterTimesParts(0)
	{}
	*/

	TimingInfo(void) : maxNeibs(0), numInteractions(0) {}

} TimingInfo;


/*
struct SavingInfo {
	float   displayfreq;		// unit time
	uint	screenshotfreq;		// unit displayfreq
	uint	writedatafreq;		// unit displayfreq
};
*/

class IPPSCounter
{
	private:
		time_t	m_startTime;
		bool m_started;
		ulong m_iterPerParts;
	public:
		IPPSCounter():
			m_startTime(0),
			m_started(false),
			m_iterPerParts(0)
		{};

		// start the counter
		time_t start() {
			time(&m_startTime);
			m_started = true;
			return m_startTime;
		}

		// reset the counter
		time_t restart() {
			m_started = true;
			return start();
		}

		// increment the internal counter of iterationsXparticles
		void incItersTimesParts(ulong increment) {
			m_iterPerParts += increment;
		}

		// reset the internal counter of iterationsXparticles
		void resetItersTimesParts() {
			m_iterPerParts = 0;
		}

		// return the throughput computed as iterations times particles per second
		double getIPPS() const {
			if (!m_started) return 0;
			time_t now;
			time(&now);
			double timeInterval = difftime(now, m_startTime);
			if (timeInterval <= 0.0)
				return 0.0;
			else
				return (double(m_iterPerParts) / timeInterval);
		}

		// almost all devices get at least 1MIPPS, so:
		inline double getMIPPS() const {
			return getIPPS()/1000000.0;
		}
};

/* Timing error exceptions */

class TimingException: public std::exception
{

public:
	float simTime, dt;

	TimingException(float _time = nan(""), float _dt = nan("")) :
		std::exception(), simTime(_time), dt(_dt) {}

	virtual const char *what() const throw() {
		return "timing error";
	}
};

class DtZeroException: public TimingException
{
public:
	DtZeroException(float _time = nan(""), float _dt = 0) :
		TimingException(_time, _dt) {}

	virtual const char *what() const throw() {
		return "timestep zeroed!";
	}
};


#endif
