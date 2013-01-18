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

// The GPUSPH class is singleton. Wise tips about a correct singleton implementation are give here:
// http://stackoverflow.com/questions/1008019/c-singleton-design-pattern

// Note: this is not thread-safe, under both the singleoton point of view and the destructor.
// But aren't that paranoid, are we?

class GPUSPH {
private:
	Options *clOptions;
	GlobalData *gdata;
	Problem *problem;
	bool initialized;
	GPUSPH();
public:
	~GPUSPH();
	static GPUSPH& getInstance();
	GPUSPH(GPUSPH const&) {}; // NOT implemented
    void operator=(GPUSPH const&); // avoid the (unlikely) case of self-assignement
	bool initialize(Options *_clOptions, GlobalData *_gdata, Problem *_problem);
	/*static*/ bool runSimulation();
	bool finalize();
};

#endif // GPUSPH_H_
