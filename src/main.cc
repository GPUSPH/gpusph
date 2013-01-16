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

// Include only the problem selected at compile time (QUOTED_PROBLEM)
#include "problem_select.opt"

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

void parse_options(int argc, char **argv, Options *_clOptions, GlobalData *gdata)
{
	const char *arg(NULL);

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
				if (gdata->devices==MAX_DEVICES) {
					printf("WARNING: devices exceeding number %u will be ignored\n",
						gdata->device[MAX_DEVICES-1]);
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
		}
		// TODO: add --dir option from multigpu. Maybe just a cherry pick?
		/*} else if (!strcmp(arg, "--dir")) {
			_clOptions->custom_dir = std::string(*argv);
			argv++;
			argc--;
		}*/ else if (!strcmp(arg, "--console")) {
			_clOptions->console = true;
		//} else if (!strcmp(arg, "--pthreads")) {
		//	_clOptions->forcePthreads = true;
		/* } else if (!strcmp(arg, "--cpuonly")) {
			_clOptions->cpuonly = true;
			gdata->cpuonly = true;
			sscanf(*argv, "%d", &(gdata->numCpuThreads));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--single")) {
			gdata->single_inter = true;
		} else if (!strcmp(arg, "--nosave")) {
			_clOptions->nosave = true;
			gdata->nosave = true;
		} else if (!strcmp(arg, "--nobalance")) {
			_clOptions->nobalance = true;
			gdata->nobalance = true;
		} else if (!strcmp(arg, "--alloc-max")) {
			_clOptions->alloc_max = true;
			gdata->alloc_max = true;
		}*/ //else if (!strcmp(arg, "--lb-threshold")) {
			/* read the next arg as a float */
			//sscanf(*argv, "%f", &(_clOptions->custom_lb_threshold));
			//gdata->custom_lb_threshold = _clOptions->custom_lb_threshold;
			//argv++;
			//argc--;
		} else if (!strcmp(arg, "--help")) {
			print_usage();
			exit(0);
		//} else if (!strcmp(arg, "--nopause")) {
		//	bPause = false;
		} else if (!strcmp(arg, "--")) {
			cout << "Skipping unsupported option " << arg << endl;
		} else {
			cout << "Fatal: Unknown option: " << arg << endl;
			// TODO: should not brutally terminate the program here, but the method only
			exit(0);

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

	// only for single-gpu
	_clOptions->device = gdata->device[0];

	_clOptions->problem = std::string( QUOTED_PROBLEM );
	cout << "Compiled for problem \"" << QUOTED_PROBLEM << "\"" << endl;

	// Left for future dynamic loading:
	/*if (_clOptions->problem.empty()) {
		problem_list();
		exit(0);
	}*/
}

bool check_short_length() {
	return (sizeof(uint) == 2*sizeof(short));
}

int newMain(int argc, char** argv) {
	if (!check_short_length()) {
		printf("Fatal: this architecture does not have uint = 2 short\n");
		exit(1);
	}

	// Command line options
	Options clOptions;

	// GlobalData, to be read potentially by everyone
	GlobalData gdata;

	// TODO: catch signal SIGINT et al.

	parse_options(argc, argv, &clOptions, &gdata);
	// TODO: check options

	// TODO: equivalent of GPUThread::runMultiGPU(&clOptions, &cdata);

	return 0;
}

