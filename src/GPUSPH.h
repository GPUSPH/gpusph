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

// The GPUSPH class is singleton. Wise tips about a correct singleton implementation are give here:
// http://stackoverflow.com/questions/1008019/c-singleton-design-pattern

// Note: this is not thread-safe, under both the singleoton point of view and the destructor.
// But aren't that paranoid, are we?

class GPUSPH {
private:
	Options *clOptions;
	GlobalData *gdata;
	Problem *problem;
	ParticleSystem *psystem;
	bool initialized;
	GPUSPH();
	long unsigned int allocateCPUBuffers();
	void deallocateCPUBuffers();
public:
	~GPUSPH();
	static GPUSPH& getInstance();
	GPUSPH(GPUSPH const&) {}; // NOT implemented
    void operator=(GPUSPH const&); // avoid the (unlikely) case of self-assignement
	bool initialize(GlobalData* _gdata);
	/*static*/ bool runSimulation();
	bool finalize();
};

#endif // GPUSPH_H_
