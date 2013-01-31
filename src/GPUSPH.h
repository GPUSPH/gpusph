/*
 * GPUSPH.h
 *
 *  Created on: Jan 18, 2013
 *      Author: rustico
 */

#ifndef GPUSPH_H_
#define GPUSPH_H_

#include "Options.h"
#include "GlobalData.h"
#include "Problem.h"
#include "ParticleSystem.h"
// Including Synchronizer.h is acutally needed only in GPUSPH.cc.
// Should be follow our unusual convention to include in headers, or not?
#include "Synchronizer.h"

// The GPUSPH class is singleton. Wise tips about a correct singleton implementation are give here:
// http://stackoverflow.com/questions/1008019/c-singleton-design-pattern

// Note: this is not thread-safe, under both the singleoton point of view and the destructor.
// But aren't that paranoid, are we?

class GPUSPH {
private:
	// some pointers
	Options* clOptions;
	GlobalData* gdata;
	Problem* problem;
	ParticleSystem* psystem;

	// other vars
	bool initialized;

	// constructor (private for singleton scheme)
	GPUSPH();

	// (de)allocation of shared host buffers
	long unsigned int allocateGlobalHostBuffers();
	void deallocateGlobalHostBuffers();

	// sort particles by device before uploading
	void sortParticlesByHash();
	// counter: how many particles per device (also used for uploading)
	uchar m_partsPerDevice[MAX_DEVICES_PER_CLUSTER]; // TODO: can change to PER_NODE if not compiling for multinode
public:
	// destructor
	~GPUSPH();
	// getInstance(), constructor and operator for singleton
	static GPUSPH& getInstance();
	GPUSPH(GPUSPH const&) {}; // NOT implemented
    void operator=(GPUSPH const&); // avoid the (unlikely) case of self-assignement

    // (de)initialization (include allocation)
	bool initialize(GlobalData* _gdata);
	bool finalize();

	/*static*/ bool runSimulation();

};

#endif // GPUSPH_H_
