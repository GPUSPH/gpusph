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
#include <cmath> // NAN

// clock_gettime() is not implemented on OSX, so we declare it here and
// implement it in timing.cc
#ifdef __APPLE__
// missing defines, we don't really care about the values
#define CLOCK_REALTIME				0
#define CLOCK_MONOTONIC				0
#include <sys/time.h>
int clock_gettime(int /*clk_id*/, struct timespec* t);
#endif

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


class IPPSCounter
{
	private:
		timespec	m_startTime;
		bool m_started;
		ulong m_iterPerParts;
	public:
		IPPSCounter():
			m_started(false),
			m_iterPerParts(0)
		{
			m_startTime.tv_sec = m_startTime.tv_nsec = 0;
		};

		// start the counter
		timespec start() {
			clock_gettime(CLOCK_MONOTONIC, &m_startTime);
			m_started = true;
			m_iterPerParts = 0;
			return m_startTime;
		}

		// reset the counter
		timespec restart() {
			return start();
		}

		// increment the internal counter of iterationsXparticles
		void incItersTimesParts(ulong increment) {
			m_iterPerParts += increment;
		}

		// compute the difference between two timespecs and store it in ret
		inline
		void timespec_diff(timespec &end, timespec &start, timespec &ret)
		{
			ret.tv_sec = end.tv_sec - start.tv_sec;
			if (end.tv_nsec < start.tv_nsec) {
				ret.tv_sec -= 1;
				ret.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
			} else {
				ret.tv_nsec = end.tv_nsec - start.tv_nsec;
			}
		}

		// compute the difference between two timespecs (in seconds)
		inline
		double diff_seconds(timespec &end, timespec &start) {
			timespec diff;
			timespec_diff(end, start, diff);
			/* explicit casts to silence -Wconversion */
			return double(diff.tv_sec) + double(diff.tv_nsec)/1.0e9;
		}

		// returns the elapsed seconds since [re]start() was called
		double getElapsedSeconds() {
			if (!m_started) return 0;
			timespec now;
			clock_gettime(CLOCK_MONOTONIC, &now);
			double timeInterval = diff_seconds(now, m_startTime);
			if (timeInterval <= 0.0)
				return 0.0;
			else
				return timeInterval;
		}

		// return the throughput computed as iterations times particles per second
		double getIPPS() {
			double timeInterval = getElapsedSeconds();
			if (timeInterval <= 0.0)
				return 0.0;
			else
				return (double(m_iterPerParts) / timeInterval);
		}

		// almost all devices get at least 1MIPPS, so:
		inline double getMIPPS() {
			return getIPPS()/1.0e6;
		}
};

/* Timing error exceptions */

class TimingException: public std::exception
{

public:
	double simTime;
	float dt;

	TimingException(double _time = nan(""), float _dt = nan("")) :
		std::exception(), simTime(_time), dt(_dt) {}

	virtual const char *what() const throw() {
		return "timing error";
	}
};

class DtZeroException: public TimingException
{
public:
	DtZeroException(double _time = nan(""), float _dt = 0) :
		TimingException(_time, _dt) {}

	virtual const char *what() const throw() {
		return "timestep zeroed!";
	}
};


#endif
