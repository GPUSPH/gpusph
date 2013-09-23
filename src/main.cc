/*
 * main.cc
 *
 *  Created on: Jan 16, 2013
 *      Author: rustico
 */

// endl, cerr, etc.
#include <iostream>
// signal, sigaction, etc.
#include <signal.h>

#include "GPUSPH.h"
#include "Options.h"
#include "GlobalData.h"
#include "NetworkManager.h"

// Include only the problem selected at compile time (PROBLEM, QUOTED_PROBLEM)
#include "problem_select.opt"

// TODO: cleanup, no exit
void print_usage() {
	cerr << "Syntax: " << endl;
	cerr << "\tGPUSPH [--device n[,n...]] [--dem dem_file] [--deltap VAL] [--tend VAL]\n";
	cerr << "\t       [--pthreads] [--dir directory] [--nosave] [--nobalance] [--nopause]\n";
	cerr << "\t       [--cpuonly VAL [--single]]\n";
	cerr << "\t       [--alloc-max] [--lb-threshold VAL] [--cpuonly VAL]\n";
	cerr << "\tGPUSPH --help\n\n";
	cerr << " --device n[,n...] : Use device number n; runs multi-gpu if multiple n are given\n";
	cerr << " --dem : Use given DEM (if problem supports it)\n";
	cerr << " --deltap : Use given deltap (VAL is cast to float)\n";
	cerr << " --tend: Break at given time (VAL is cast to float)\n";
	cerr << " --dir : Use given directory for dumps instead of date-based one\n";
	cerr << " --pthreads : Force use of threads even if single GPU\n";
	cerr << " --nosave : Disable all file dumps but the last\n";
	cerr << " --nobalance : Disable dynamic load balancing\n";
	cerr << " --alloc-max : Alloc total number of particles for every device\n";
	cerr << " --lb-threshold : Set custom LB activation threshold (VAL is cast to float)\n";
	cerr << " --cpuonly : Simulates on the CPU using VAL pthreads\n";
	cerr << " --single : Computes fluid-fluid interactions once per pair\n";
	cerr << " --help: Show this help and exit\n";
	//exit(-1);
}

// if some option needs to be passed to GlobalData, remember to set it in GPUSPH::initialize()
bool parse_options(int argc, char **argv, GlobalData *gdata)
{
	const char *arg(NULL);

	if (!gdata) return NULL;
	Options* _clOptions = gdata->clOptions;

	// skip arg 0 (program name)
	argv++; argc--;

	while (argc > 0) {
		arg = *argv;
		argv++;
		argc--;
		if (!strcmp(arg, "--device")) {
			/* read the next arg as a list of integers */
			char * pch;
			pch = strtok (*argv,",");
			while (pch != NULL) {
				//printf ("%s\n",pch);
				if (gdata->devices==MAX_DEVICES_PER_NODE) {
					printf("WARNING: devices exceeding number %u will be ignored\n",
						gdata->device[MAX_DEVICES_PER_NODE-1]);
					break;
				} else {
					// inc _clOptions->devices only if scanf was successful
					if (sscanf(pch, "%u", &(gdata->device[gdata->devices]))>0) {
						gdata->devices++;
					} else {
						printf("WARNING: token %s is not a number - ignored\n", pch);
						//break;
					}
				}
				pch = strtok (NULL, " ,.-");
			}
			if (gdata->devices<1) {
				printf("ERROR: --device option given, but no device specified\n");
				exit(1);
			}
			argv++;
			argc--;
		} else if (!strcmp(arg, "--deltap")) {
			/* read the next arg as a float */
			sscanf(*argv, "%f", &(_clOptions->deltap));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--tend")) {
			/* read the next arg as a float */
			sscanf(*argv, "%f", &(_clOptions->tend));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--dem")) {
			_clOptions->dem = std::string(*argv);
			argv++;
			argc--;
		} else if (!strcmp(arg, "--dir")) {
			_clOptions->custom_dir = std::string(*argv);
			argv++;
			argc--;
		} /* else if (!strcmp(arg, "--console")) {
			_clOptions->console = true;
		} else if (!strcmp(arg, "--pthreads")) {
			_clOptions->forcePthreads = true;
		} else if (!strcmp(arg, "--cpuonly")) {
			_clOptions->cpuonly = true;
			gdata->cpuonly = true;
			sscanf(*argv, "%d", &(gdata->numCpuThreads));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--single")) {
			gdata->single_inter = true;
		} */ else if (!strcmp(arg, "--nosave")) {
			_clOptions->nosave = true;
			gdata->nosave = true;
		} /*else if (!strcmp(arg, "--nobalance")) {
			_clOptions->nobalance = true;
			gdata->nobalance = true;
		} else if (!strcmp(arg, "--alloc-max")) {
			_clOptions->alloc_max = true;
			gdata->alloc_max = true;
		} else if (!strcmp(arg, "--lb-threshold")) {
			// read the next arg as a float
			sscanf(*argv, "%f", &(_clOptions->custom_lb_threshold));
			gdata->custom_lb_threshold = _clOptions->custom_lb_threshold;
			argv++;
			argc--;
		} */ else if (!strcmp(arg, "--help")) {
			print_usage();
			//exit(0);
			return false;
		//} else if (!strcmp(arg, "--nopause")) {
		//	bPause = false;
		} else if (!strcmp(arg, "--")) {
			cout << "Skipping unsupported option " << arg << endl;
		} else {
			cout << "Fatal: Unknown option: " << arg << endl;
			// exit(0);
			return false;

			// Left for future dynamic loading:
			/*if (_clOptions->problem.empty()) {
				_clOptions->problem = std::string(arg);
			} else {
				cout << "Problem " << arg << " selected after problem " << _clOptions->problem << endl;
			}*/
		}
	}

	if (gdata->devices==0) {
		printf(" * No devices specified, falling back to default (dev 0)...\n");
		// default: use first device. May use cutGetMaxGflopsDeviceId() instead.
		gdata->device[gdata->devices++] = 0;
	}

#if HASH_KEY_SIZE < 64
	// only single-GPU possible with 32 bits keys
	if (gdata->devices > 1) {
		printf(" FATAL: multi-GPU requires the Hashkey to be at least 64 bits long\n");
		return false;
	}
#endif

	// only for single-gpu
	_clOptions->device = gdata->device[0];

	_clOptions->problem = std::string( QUOTED_PROBLEM );
	cout << "Compiled for problem \"" << QUOTED_PROBLEM << "\"" << endl;

	// Left for future dynamic loading:
	/*if (_clOptions->problem.empty()) {
		problem_list();
		exit(0);
	}*/

	return true;
}

bool check_short_length() {
	return (sizeof(uint) == 2*sizeof(short));
}

// SIGINT handler: issues a quit_request
void sigint_handler(int signum) {
	// if issued the second time, brutally terminate everything
	if (gdata_static_pointer->quit_request) {
		uint reachedt = gdata_static_pointer->threadSynchronizer->queryReachedThreads();
		uint maxt = gdata_static_pointer->threadSynchronizer->getNumThreads();
		if (reachedt > 0 && reachedt < maxt && !gdata_static_pointer->threadSynchronizer->didForceUnlockOccurr()) {
			printf("Second quit request - threads waiting: %u/%u. Forcing unlock...\n", reachedt, maxt);
			gdata_static_pointer->threadSynchronizer->forceUnlock();
		} else {
			printf("Unable to force unlock. Issuing exit(1)\n");
			exit(1);
		}
	} else {
		printf("Kindly asking to quit - please wait for the current iteration to complete\n");
		gdata_static_pointer->quit_request = true;
	}
}

void sigusr1_handler(int signum) {
	// TODO: actually issue a user save_request
	//gdata_static_pointer->save_request = true;
}

int main(int argc, char** argv) {
	if (!check_short_length()) {
		printf("Fatal: this architecture does not have uint = 2 short\n");
		exit(1);
	}

	GlobalData gdata;
	gdata_static_pointer = &gdata;

	// Command line options
	gdata.clOptions = new Options();

	// catch SIGINT and SIGUSR1
	struct sigaction int_action, usr1_action;
	int_action.sa_flags=0;
	int_action.sa_handler = sigint_handler;
	sigaction(SIGINT, &int_action, NULL);
	usr1_action.sa_flags=0;
	usr1_action.sa_handler = sigusr1_handler;
	sigaction(SIGUSR1, &usr1_action, NULL);

	// parse command-line options
	// NOTE: here gdata is almost complete, only the problem is not allocated yet
	if (!parse_options(argc, argv, &gdata))
		exit(1);

	// TODO: check options, i.e. consistency

	// NOTE: Although GPUSPH has been designed to be run with one multi-threaded process per node, it is important not to create
	// any file or lock singleton resources before initializing the network, as the process might be forked
	gdata.networkManager = new NetworkManager();
	gdata.networkManager->initNetwork();

	// the Problem could (should?) be initialized inside GPUSPH::initialize()
	gdata.problem = new PROBLEM(*(gdata.clOptions));

	// get - and actually instantiate - the existing instance of GPUSPH
	GPUSPH Simulator = GPUSPH::getInstance();

	// initialize CUDA, start workers, allocate CPU and GPU buffers
	bool result = Simulator.initialize(&gdata);
	printf("GPUSPH: %s\n", (result ? "initialized" : "NOT initialized") );

	// run the simulation until a quit request is triggered or an exception is thrown (TODO)
	Simulator.runSimulation();

	// finalize everything
	Simulator.finalize();

	// same consideration as above
	delete gdata.problem;

	// finalize MPI
	gdata.networkManager->finalizeNetwork();

	delete gdata.networkManager;

	return 0;
}

