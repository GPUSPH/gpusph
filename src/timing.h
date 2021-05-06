/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

/*! \file
 * Timing info
 */

#ifndef _TIMING_H
#define _TIMING_H

#include <chrono>
#include <exception>
#include <cmath> // NAN
#include "particleinfo.h"

typedef unsigned int uint;
typedef unsigned long ulong;

typedef struct TimingInfo {
	// current simulation time
	//float   t;
	// current simulation timestep
	//float   dt;
	// current number of particles
	//uint	numParticles;

	// maximum number of fluid+boundary neibs
	uint	maxFluidBoundaryNeibs;
	// maximum number of vertex neibs
	uint	maxVertexNeibs;
	// index of a particle with too many neibs
	int	hasTooManyNeibs;
	// number of neibs of that particle
	int	hasMaxNeibs[PT_TESTPOINT];
	// index of a cell with too many particles
	int	hasTooManyParticles;
	// how many particles are in the cell with too many particles
	int	hasHowManyParticles;

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

	TimingInfo(void) : maxFluidBoundaryNeibs(0), maxVertexNeibs(0), numInteractions(0),
		hasTooManyParticles(-1), hasHowManyParticles(0)
	{ }

} TimingInfo;


class IPPSCounter
{
	// we track runtime using the default monotonic clock
	using clock = std::chrono::steady_clock;
	using time_point = clock::time_point;
	// we track runtime in seconds (floating-point)
	using duration = std::chrono::duration<double>;
	static constexpr auto now = clock::now;

	private:
		time_point m_startTime;
		ulong m_iterPerParts;
		bool m_started;
	public:
		IPPSCounter():
			m_startTime(),
			m_iterPerParts(0),
			m_started(false)
		{};

		// start the counter
		time_point start() {
			m_startTime = now();
			m_started = true;
			m_iterPerParts = 0;
			return m_startTime;
		}

		// reset the counter
		time_point restart() {
			return start();
		}

		// increment the internal counter of iterationsXparticles
		void incItersTimesParts(ulong increment) {
			m_iterPerParts += increment;
		}

		// returns the elapsed seconds since [re]start() was called
		double getElapsedSeconds() {
			if (!m_started) return 0;
			duration elapsed_seconds = now() - m_startTime;
			double timeInterval = elapsed_seconds.count();
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
				return (m_iterPerParts / timeInterval);
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
