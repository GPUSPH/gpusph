/*
 * main.cc
 *
 *  Created on: Jan 16, 2013
 *      Author: rustico
 */

// endl, cerr, etc.
#include <iostream>
#include "Options.h"
#include "GlobalData.h"

// TODO: cleanup, no exit
void print_usage() {
	cerr << "Syntax: " << endl;
	cerr << "\tGPUSPH [--device n[,n...]] [--dem dem_file] [--deltap VAL] [--tend VAL]\n";
	cerr << "\t       [--console] [--pthreads] [--dir directory] [--nosave] [--nobalance] [--nopause]\n";
	cerr << "\t       [--cpuonly VAL [--single]]\n";
	cerr << "\t       [--alloc-max] [--lb-threshold VAL] [--cpuonly VAL]\n";
	cerr << "\tGPUSPH --help\n\n";
	cerr << " --device n[,n...] : Use device number n; runs multi-gpu if multiple n are given\n";
	cerr << " --dem : Use given DEM (if problem supports it)\n";
	cerr << " --deltap : Use given deltap (VAL is cast to float)\n";
	cerr << " --tend: Break at given time (VAL is cast to float)\n";
	cerr << " --console : Run in console mode, no GL window\n";
	cerr << " --dir : Use given directory for dumps instead of date-based one\n";
	cerr << " --pthreads : Force use of threads even if single GPU\n";
	cerr << " --nosave : Disable all file dumps but the last\n";
	cerr << " --nobalance : Disable dynamic load balancing\n";
	cerr << " --alloc-max : Alloc total number of particles for every device\n";
	cerr << " --lb-threshold : Set custom LB activation threshold (VAL is cast to float)\n";
	cerr << " --cpuonly : Simulates on the CPU using VAL pthreads\n";
	cerr << " --single : Computes fluid-fluid interactions once per pair\n";
	cerr << " --nopause : Do *not* start paused in GL mode\n";
	cerr << " --help: Show this help and exit\n";
	exit(-1);
}

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

