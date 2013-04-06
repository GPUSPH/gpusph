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
	float   t;
	float   dt;
	uint	numParticles;
	uint	maxNeibs;
	uint	numInteractions;
	ulong	iterations;
	ulong	meanNumInteractions;
	float   timeNeibsList;
	float   meanTimeNeibsList;
	float   timeInteract;
	float   meanTimeInteract;
	float   timeEuler;
	double  meanTimeEuler;
	clock_t	startTime;
} TimingInfo;


struct SavingInfo {
	float   displayfreq;		// unit time
	uint	screenshotfreq;		// unit displayfreq
	uint	writedatafreq;		// unit displayfreq
};

#endif
