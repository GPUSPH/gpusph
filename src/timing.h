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
	// number of particle-particle interactions with current neiblist
	uint	numInteractions;
	// iterations done so far
	ulong	iterations;
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
} TimingInfo;


struct SavingInfo {
	float   displayfreq;		// unit time
	uint	screenshotfreq;		// unit displayfreq
	uint	writedatafreq;		// unit displayfreq
};

#endif
