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

/*! \file
 * Interface of the GPUSPH core
 */

#ifndef GPUSPH_H_
#define GPUSPH_H_

#include <cstdio>
#include <type_traits>

#include "Options.h"
#include "GlobalData.h"
#include "Problem.h"
#include "cpp11_missing.h"

// IPPSCounter
#include "timing.h"

//! A sequence of commands, modelling a phase of the integrator
/*! This is essentially an std::vector<CommandStruct>, with minor changes:
 * it only exposes reserve(), constant begin() and end() methods, and
 * a push_back() method that returns a reference to back()
 */
class CommandSequence
{
	using base = std::vector<CommandStruct>;
	base m_seq;
public:
	CommandSequence() : m_seq() {}

	void reserve(size_t sz)
	{ m_seq.reserve(sz); }

	base::const_iterator begin() const
	{ return m_seq.begin(); }
	base::const_iterator end() const
	{ return m_seq.end(); }

	CommandStruct& push_back(CommandStruct const& cmd)
	{
		m_seq.push_back(cmd);
		return m_seq.back();
	}
};

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

	// Command sequences describing the major phases of the integration
	// TODO these will be moved to an actual Integrator class once we're done
	// with the refactoring
	CommandSequence neibsListCommands; // steps implementing buildNeibList()
	CommandSequence nextStepCommands[3]; // steps implementing prepareNextStep(), for each of the integrator steps
	CommandSequence predCorrCommands[2]; // steps implementing the prediction and correction phases of the integrator

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

	// check consistency of buffers across multiple GPUs
	void checkBufferConsistency();

	// compute initial values for the IDs of the next generated particles,
	// and return the number of open boundary vertices
	uint initializeNextIDs(bool resumed);

	// initialize the command sequences
	// TODO provisional during the refactoring
	void initializeCommandSequences();

	// sort particles by device before uploading
	void sortParticlesByHash();
	// aux function for sorting; swaps particles in s_hPos, s_hVel, s_hInfo
	void particleSwap(uint idx1, uint idx2);

	// update s_hStartPerDevice and s_hPartsPerDevice
	void updateArrayIndices();

	// perform post-filling operations
	void prepareProblem();

	// run a host command
	void runCommand(CommandStruct const& cmd);
	// dispatch the command cmd, either by running it in GPUSPH itself,
	// or by setting nextCommand, unlocking the threads and waiting for them
	// to complete
	void doCommand(CommandStruct const& cmd);
	void doCommand(CommandStruct cmd, flag_t flags);

	// TODO possibly temporary, while states are strings
	void doCommand(CommandStruct cmd, std::string const& src, std::string const& dst,
		flag_t flags = NO_FLAGS);
	void doCommand(CommandStruct cmd, std::string const& dst, flag_t flags = NO_FLAGS);

	// sets the correct viscosity coefficient according to the one set in SimParams
	void setViscosityCoefficient();

	// Do the multi gpu/multi node forces reduction and move bodies
	void move_bodies(flag_t integrator_step);

	// create the Writer
	void createWriter();

	// use the writer, with additional options about forced writes
	void doWrite(WriteFlags const& write_flags);

	// save the particle system to disk
	void saveParticles(PostProcessEngineSet const& enabledPostProcess,
		std::string const& state, WriteFlags const& write_flags);

	// callbacks for moving boundaries and variable gravity
	void doCallBacks(const flag_t current_integrator_step);

	//! Rebuild the neighbor list
	void buildNeibList();

	// setting of boundary conditions for the semi-analytical boundaries
	void saBoundaryConditions(flag_t cFlag);

	// prepare for the next forces computation
	void prepareNextStep(const flag_t current_integrator_step);

	// mark the beginning/end of a step, setting the state and validity
	// of READ and WRITE buffers
	void markIntegrationStep(
		std::string const& read_state, BufferValidity read_valid,
		std::string const& write_state, BufferValidity write_valid);

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

	void check_write(const bool);

	// refactored code by splitting the two integrator steps
	void runIntegratorStep(const flag_t integrator_step);
	void runEnabledFilters(const FilterFreqList& enabledFilters);

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
