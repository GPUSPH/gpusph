/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

typedef unsigned int uint;
typedef unsigned long ulong;

typedef struct TimingInfo {
	// current simulation time
	float   t;
	// current simulation timestep
	float   dt;
	// current number of particles
	uint	numParticles;
	// current maximum number of neibs
	uint	maxNeibs;
	// iterations done so far
	ulong	iterations;
	// number of particle-particle interactions with current neiblist
	uint	numInteractions;
	// average number of particle-particle interactions
	ulong	meanNumInteractions;
	// time taken to build the neiblist (latest)
	float   timeNeibsList;
	// avg. time  to build the neiblist
	float   meanTimeNeibsList;
	// time taken to compute interactions (latest)
	float   timeInteract;
	// avg. time to compute interactions
	float   meanTimeInteract;
	// time taken to integrate (latest)
	float   timeEuler;
	// avg. time to integrate
	double  meanTimeEuler;

	// clock time when simulation started
	clock_t	startTime;
	// number of iterations times number of particles

	/* Note: this is computed by adding to it the current number of particles
	 * after each iteration, to ensure the correct value even when the number of
	 * particles changes during the simulation
	 */
	ulong	iterTimesParts;

	TimingInfo(void) : t(0), dt(0), numParticles(0), maxNeibs(0),
		iterations(0), numInteractions(0), meanNumInteractions(0),
		timeNeibsList(0), meanTimeNeibsList(0),
		timeInteract(0), meanTimeInteract(0),
		timeEuler(0), meanTimeEuler(0),
		startTime(0), iterTimesParts(0)
	{}


	// a method to return the throughput computed as iterations times particles per second
	double	getIPPS(void) const {
		return (double(iterTimesParts)/(clock()-startTime))*CLOCKS_PER_SEC;
	}
	// almost all devices get at least 1MIPPS, so:
	inline
	double	getMIPPS(void) const {
		return getIPPS()/1000000.0;
	}
	// set the startTime of the simulation
	clock_t	start(void) {
		startTime = clock();
		return startTime;
	}
} TimingInfo;


struct SavingInfo {
	float   displayfreq;		// unit time
	uint	screenshotfreq;		// unit displayfreq
	uint	writedatafreq;		// unit displayfreq
};

#endif
