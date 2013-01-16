/*
 * main.cc
 *
 *  Created on: Jan 16, 2013
 *      Author: rustico
 */

#include "Options.h"
#include "GlobalData.h"

int newMain() {
	// stub
int newMain(int argc, char** argv) {
	// Command line options
	Options clOptions;

	// GlobalData, to be read potentially by everyone
	GlobalData gdata;

	// TODO: check uint = 2 short
	// TODO: catch signal SIGINT et al.

	// TODO: parse_options(argc, argv);
	// TODO: check options

	// TODO: equivalent of GPUThread::runMultiGPU(&clOptions, &cdata);

	return 0;
}

