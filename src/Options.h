#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include <math.h>
#include <string>

using namespace std;

struct Options {
	string problem; // problem name
	int device;  // which device to use
	string dem; // DEM file to use
	bool console; // run in console (no GUI)
	float deltap; // deltap
	float tend; // simulation end
	Options(void) :
		problem(),
		device(-1),
		dem(),
		console(false),
		deltap(NAN),
		tend(NAN)
	{};
};

#endif
