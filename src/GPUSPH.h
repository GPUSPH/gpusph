/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef GPUSPH_H_
#define GPUSPH_H_

#include <cstdio>

#include "Options.h"
#include "GlobalData.h"
#include "Problem.h"

// IPPSCounter
#include "timing.h"

// The GPUSPH class is singleton. Wise tips about a correct singleton implementation are give here:
// http://stackoverflow.com/questions/1008019/c-singleton-design-pattern

// Note: this is not thread-safe, under both the singleton point of view and the destructor.
// But we aren't that paranoid, are we?

class GPUSPH {
private:
	// some pointers
	Options* clOptions;
	GlobalData* gdata;
	Problem* problem;

	// performance counters (in MIPPS)
	IPPSCounter *m_totalPerformanceCounter;
	IPPSCounter *m_intervalPerformanceCounter;
	IPPSCounter *m_multiNodePerformanceCounter; // only used if MULTI_NODE

	// Information stream where the current status
	// is output
	string m_info_stream_name; // name of the stream
	FILE *m_info_stream; // file handle

	// aux arrays for rollCallParticles()
	bool *m_rcBitmap;
	bool *m_rcNotified;
	uint *m_rcAddrs;

	// store max speed reached during the whole simulation
	// NOTE: float since network reduction currently does not support double
	float m_peakParticleSpeed;
	double m_peakParticleSpeedTime; // ...and when

	// other vars
	bool initialized;

	// constructor and copy/assignment: private for singleton scheme
	GPUSPH();
	GPUSPH(GPUSPH const&); // NOT implemented
	void operator=(GPUSPH const&); // avoid the (unlikely) case of self-assignement

	// open/close/write the info stream
	void openInfoStream();
	void closeInfoStream();

	// (de)allocation of shared host buffers
	size_t allocateGlobalHostBuffers();
	void deallocateGlobalHostBuffers();

	// sort particles by device before uploading
	void sortParticlesByHash();
	// aux function for sorting; swaps particles in s_hPos, s_hVel, s_hInfo
	void particleSwap(uint idx1, uint idx2);

	// update s_hStartPerDevice and s_hPartsPerDevice
	void updateArrayIndices();

	// perform post-filling operations
	void prepareProblem();

	// set nextCommand, unlock the threads and wait for them to complete
	void doCommand(CommandType cmd, flag_t flags=NO_FLAGS, float arg=NAN);

	// sets the correct viscosity coefficient according to the one set in SimParams
	void setViscosityCoefficient();

	// Do the multi gpu/multi node forces reduction and move bodies
	void move_bodies(const uint);

	// create the Writer
	void createWriter();

	// use the writer (optionally let writer know if
	// writing was forced)
	void doWrite(bool force);

	// callbacks for moving boundaries and variable gravity
	void doCallBacks();

	// rebuild the neighbor list
	void buildNeibList();

	// setting of boundary conditions for the semi-analytical boundaries
	void saBoundaryConditions(flag_t cFlag);

	// print information about the status of the simulation
	void printStatus(FILE *out = stdout);

	// print information about the status of the simulation
	void printParticleDistribution();

	// print peer accessibility for all devices
	void printDeviceAccessibilityTable();

	// do a roll call of particle IDs
	void rollCallParticles();

	void allocateRbArrays();
	void cleanRbArrays();

	double Wendland2D(const double, const double);

public:
	// destructor
	~GPUSPH();

	// getInstance(), constructor and operator for singleton
	static GPUSPH* getInstance();

	// (de)initialization (include allocation)
	bool initialize(GlobalData* _gdata);
	bool finalize();

	/*static*/ bool runSimulation();

};

#endif // GPUSPH_H_
