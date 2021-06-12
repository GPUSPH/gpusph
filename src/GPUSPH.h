/*  Copyright (c) 2013-2019 INGV, EDF, UniCT, JHU

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
 * Interface of the GPUSPH core
 */

#ifndef GPUSPH_H_
#define GPUSPH_H_

#include <cstdio>
#include <type_traits>

#include "Options.h"
#include "GlobalData.h"
#include "ProblemCore.h"
#include "Integrator.h"
#include "cpp11_missing.h"

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
	ProblemCore* problem;

	// performance counters (in MIPPS)
	IPPSCounter *m_totalPerformanceCounter;
	IPPSCounter *m_intervalPerformanceCounter;
	IPPSCounter *m_multiNodePerformanceCounter; // only used if MULTI_NODE

	// Information stream where the current status
	// is output
	std::string m_info_stream_name; // name of the stream
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
	bool repacked;

	std::shared_ptr<Integrator> integrator;

protected:
	friend struct TimerObject;

	using cmd_time_clock = std::chrono::steady_clock;
	using cmd_time_duration = cmd_time_clock::duration;

	// Maximum time taken to dispatch each command
	cmd_time_duration max_cmd_time[NUM_COMMANDS];
	// Total time spent executing each command
	cmd_time_duration tot_cmd_time[NUM_COMMANDS];
	unsigned long cmd_calls[NUM_COMMANDS];

private:
	// constructor and copy/assignment: private for singleton scheme
	GPUSPH();
	GPUSPH(GPUSPH const&); // NOT implemented
	void operator=(GPUSPH const&); // avoid the (unlikely) case of self-assignement

	void resetCommandTimes();
	void showCommandTimes();

	// open/close/write the info stream
	void openInfoStream();
	void closeInfoStream();

	// (de)allocation of shared host buffers
	size_t allocateGlobalHostBuffers();
	void deallocateGlobalHostBuffers();

	// check consistency of buffers across multiple GPUs
	void checkBufferConsistency(CommandStruct const&);

	// compute initial values for the IDs of the next generated particles,
	// and return the number of open boundary vertices
	uint initializeNextIDs(bool resumed);

	// sort particles by device before uploading
	void sortParticlesByHash();
	// aux function for sorting; swaps particles in s_hPos, s_hVel, s_hInfo
	void particleSwap(uint idx1, uint idx2);

	// perform post-filling operations
	void prepareProblem();

	/// Function template to run a specific command
	/*! There should be a specialization of the template for each
	 * (supported) command
	 */
	template<CommandName>
	void runCommand(CommandStruct const& cmd);

	/// Raise an errror about an unknown command
	void unknownCommand(CommandName cmd);

	// dispatch the command cmd, either by running it in GPUSPH itself,
	// or by setting nextCommand, unlocking the threads and waiting for them
	// to complete
	void dispatchCommand(CommandStruct const& cmd);
	void dispatchCommand(CommandStruct cmd, flag_t flags);

	// sets the correct viscosity coefficient according to the one set in SimParams
	void setViscosityCoefficient();

	// create the Writer
	void createWriter();

	// use the writer, with additional options about forced writes
	void doWrite(WriteFlags const& write_flags);

	// save the particle system to disk
	void saveParticles(PostProcessEngineSet const& enabledPostProcess,
		std::string const& state, WriteFlags const& write_flags);

	//! Rebuild the neighbor list
	void buildNeibList();

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

	void check_write(const bool);

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
