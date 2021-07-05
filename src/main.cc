/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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
 * Entrypoint to GPUSPH: parse options, instantiate problem and engine,
 * run simulation, handle errors.
 */

// endl, cerr, etc.
#include <iostream>
// signal, sigaction, etc.
#include <signal.h>

#include "GPUSPH.h"
#include "Options.h"
#include "GlobalData.h"
#include "NetworkManager.h"

#include "debugflags.h"

#include "problem_spec.h"

/* Include all other opt file for show_version */
#include "clang_select.opt"
#include "chrono_select.opt"
#include "compute_select.opt"
#include "dbg_select.opt"
#include "fastmath_select.opt"
#include "fastdem_select.opt"
#include "gpusph_version.opt"
#include "hdf5_select.opt"
#include "mpi_select.opt"
#include "catalyst_select.opt"

/* include cuda/cache_preference.h, to show the cache preference */
#include "cuda/cache_preference.h"

using namespace std;

// this global bool is used by the simulation framework (which cannot access gdata)
// to know if the simulation is single- or multi-device
// (the information currently is only used to determine if the kernel launches should be
// synchronous or not)
bool is_multi_device;

void show_version()
{
	static const char dbg_or_rel[] =
#if defined(_DEBUG_)
		"Debug";
#else
		"Release";
#endif

	printf("GPUSPH version %s\n", GPUSPH_VERSION);
	printf("%s version %s fastmath for compute capability %u.%u (cache preference: %s cache)\n",
		dbg_or_rel,
		FASTMATH ? "with" : "without",
		COMPUTE/10, COMPUTE%10,
		PREFER_L1 ? "L1" : "texture");
	printf("\tbuilt with %s, major version %d\n",
		(CLANG_CUDA ? "clang" : "nvcc"),
		(CLANG_CUDA ? CLANG_CUDA_VERSION : CUDA_MAJOR));
	printf("DEM    : %s\n", FASTDEM ? "fast" : "symmetrized");
	printf("Chrono : %s\n", USE_CHRONO ? "enabled" : "disabled");
	printf("HDF5   : %s\n", USE_HDF5 ? "enabled" : "disabled");
	printf("MPI    : %s\n", USE_MPI ? "enabled" : "disabled");
	printf("Catalyst : %s\n", USE_CATALYST ? "enabled" : "disabled");
	printf("Compiled for problem \"%s\"\n", selected_problem.name);
}


// TODO: cleanup, no exit
void print_usage() {
	show_version();
	cout << "Syntax: " << endl;
	cout << "\tGPUSPH [--device n[,n...]] [--dem dem_file] [--deltap VAL] [--tend VAL] [--dt VAL]\n";
	cout << "\t       [--resume fname] [--checkpoint-every VAL] [--checkpoints VAL]\n";
	cout << "\t       [--dir directory] [--nosave] [--striping] [--gpudirect [--asyncmpi]]\n";
	cout << "\t       [--num-hosts VAL [--byslot-scheduling]]\n";
	cout << "\t       [--display [--display-every VAL] --display-script VAL]\n";
	cout << "\t       [--debug FLAGS]\n";
	cout << "\tGPUSPH --help\n\n";
	cout << " --resume : resume from the given file (HotStart file saved by HotWriter)\n";
	cout << " --checkpoint-every : HotStart checkpoints will be created every VAL seconds\n";
	cout << "                      of simulated time (float VAL, 0 disables)\n";
	cout << " --checkpoints : number of HotStart checkpoints to keep (integer VAL)\n";
	cout << " --device n[,n...] : Use device number n; runs multi-gpu if multiple n are given\n";
	cout << " --dem : Use given DEM (if problem supports it)\n";
	cout << " --deltap : Use given deltap (VAL is cast to float)\n";
	cout << " --tend : Break at given time (VAL is cast to float)\n";
	cout << " --dt : Use the provided fixed time-step (VAL is cast to float)\n";
	cout << " --maxiter : Break after this many iterations (integer VAL)\n";
	cout << " --dir : Use given directory for dumps instead of date-based one\n";
	cout << " --nosave : Disable all file dumps but the last\n";
	cout << " --gpudirect: Enable GPUDirect for RDMA (requires a CUDA-aware MPI library)\n";
	cout << " --striping : Enable computation/transfer overlap  in multi-GPU (usually convenient for 3+ devices)\n";
	cout << " --asyncmpi : Enable asynchronous network transfers (requires GPUDirect and 1 process per device)\n";
	cout << " --num-hosts : Specify number of hosts. To be used if #processes > #hosts (VAL is cast to uint)\n";
	cout << " --byslot-scheduling : MPI scheduler is filling hosts first, as opposite to round robin scheduling\n";
	cout << " --no-leak-warning : do not warn if #particles decreases without outlets (e.g. overtopping, leaking)\n";
	//cout << " --nobalance : Disable dynamic load balancing\n";
	//cout << " --lb-threshold : Set custom LB activation threshold (VAL is cast to float)\n";
	cout << " --display : Enable co-processing visulaization\n";
	cout << " --display-every : Simulation data will be passed to visualization every VAL seconds\n";
	cout << "                   of simulated time (VAL is cast to double, 0 or not defined - visualization for each iteration)\n";
	cout << " --display-script : Path to co-processing Python script\n";
	cout << " --debug : enable debug flags FLAGS\n";
#include "describe-debugflags.h"
	cout << " --repack : run the repacking before the simulation, beware to enable repacking in the simulation framework\n";
	cout << " --repack-only : run the repacking and stop\n";
	cout << " --repack-maxiter : repacking breaks after this many iterations (integer VAL)\n";
	cout << " --from-repack : run from a previous repack file\n";
	cout << " --help: Show this help and exit\n";
}

/// Parse command line options
/*!
 * We return the opposite of the error value if there were errors parsing the
 * command-line options,
 * 0 if the user asked for usage or version, and
 * 1 if the simulation can go on.
 */
int parse_options(int argc, char **argv, GlobalData *gdata)
{
	const char *arg(NULL);

	if (!gdata) return -ENOMEM;
	Options* _clOptions = gdata->clOptions;

	// skip arg 0 (program name)
	argv++; argc--;

	while (argc > 0) {
		arg = *argv;
		argv++;
		argc--;
		if (!strcmp(arg, "--resume")) {
			_clOptions->resume_fname = string(*argv);
			gdata->resume = true;
			argv++;
			argc--;
		} else if (!strcmp(arg, "--checkpoint-every")) {
			sscanf(*argv, "%f", &(_clOptions->checkpoint_freq));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--checkpoints")) {
			sscanf(*argv, "%d", &(_clOptions->checkpoints));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--device")) {
			/* parse the argument as a device list, and append it to any previously
			 * added devices */
			auto devs = parse_devices_string(*argv);
			if (devs.empty())
				throw std::out_of_range("--device option given, but no device specified");
			_clOptions->devices.insert(_clOptions->devices.end(),
					devs.begin(), devs.end());
			argv++;
			argc--;
		} else if (!strcmp(arg, "--deltap")) {
			/* read the next arg as a double */
			sscanf(*argv, "%lf", &(_clOptions->deltap));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--tend")) {
			/* read the next arg as a float */
			sscanf(*argv, "%f", &(_clOptions->tend));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--dt")) {
			/* read the next arg as a float */
			sscanf(*argv, "%f", &(_clOptions->dt));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--maxiter")) {
			/* read the next arg as an unsigned long */
			sscanf(*argv, "%lu", &(_clOptions->maxiter));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--dem")) {
			_clOptions->dem = string(*argv);
			argv++;
			argc--;
		} else if (!strcmp(arg, "--dir")) {
			_clOptions->dir = string(*argv);
			argv++;
			argc--;
		} else if (!strcmp(arg, "--nosave")) {
			_clOptions->nosave = true;
		} else if (!strcmp(arg, "--gpudirect")) {
			_clOptions->gpudirect = true;
		} else if (!strcmp(arg, "--striping")) {
			_clOptions->striping = true;
		} else if (!strcmp(arg, "--asyncmpi")) {
			_clOptions->asyncNetworkTransfers = true;
		} else if (!strcmp(arg, "--num-hosts") || !strcmp(arg, "--num_hosts")) {
			/* read the next arg as a uint */
			sscanf(*argv, "%u", &(_clOptions->num_hosts));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--version")) {
			show_version();
			return 0;
		} else if (!strcmp(arg, "--byslot-scheduling") || !strcmp(arg, "--byslot_scheduling")) {
			_clOptions->byslot_scheduling = true;
		} else if (!strcmp(arg, "--no-leak-warning") || !strcmp(arg, "--no_leak_warning")) {
			_clOptions->no_leak_warning = true;
#if 0 // options will be enabled later
		} else if (!strcmp(arg, "--nobalance")) {
			_clOptions->nobalance = true;
		} else if (!strcmp(arg, "--lb-threshold")) {
			// read the next arg as a float
			sscanf(*argv, "%f", &(_clOptions->custom_lb_threshold));
			argv++;
			argc--;
#endif
		} else if (!strcmp(arg, "--display")) {
		        _clOptions->visualization = true;
		} else if (!strcmp(arg, "--display-every")) {
			/* read the next arg as a double */
			sscanf(*argv, "%lf", &(_clOptions->visu_freq));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--display-script")) {
			_clOptions->pipeline_fpath = string(*argv);
			argv++;
			argc--;
		} else if (!strcmp(arg, "--debug")) {
			parse_debug_flags(*argv);
			argv++;
			argc--;
		} else if (!strcmp(arg, "--repack")) {
			_clOptions->repack = true;
		} else if (!strcmp(arg, "--repack-only")) {
			_clOptions->repack_only = true;
		} else if (!strcmp(arg, "--repack-maxiter")) {
			/* read the next arg as an unsigned long */
			sscanf(*argv, "%u", &(_clOptions->repack_maxiter));
			argv++;
			argc--;
		} else if (!strcmp(arg, "--from-repack")) {
			_clOptions->resume_fname = string(*argv);
			gdata->resume = true;
			argv++;
			argc--;
		} else if (!strcmp(arg, "--help")) {
			print_usage();
			return 0;
		} else if (!strncmp(arg, "--", 2)) {
			// TODO bool options would need to be treated specially,
			// currently they require a following 1 or 0
			_clOptions->set(arg+2, *argv);
			argv++;
			argc--;
		} else {
			cerr << "Fatal: Unknown option: " << arg << endl;
			return -EINVAL;

			// Left for future dynamic loading:
			/*if (_clOptions->problem.empty()) {
				_clOptions->problem = string(arg);
			} else {
				cout << "Problem " << arg << " selected after problem " << _clOptions->problem << endl;
			}*/
		}
	}

	// If no --device was specified by the user, get the default device(s)
	if (_clOptions->devices.empty()) {
		_clOptions->devices = get_default_devices();
	}


	for (auto dev : _clOptions->devices) {
		if (gdata->devices == MAX_DEVICES_PER_NODE) {
			printf("WARNING: devices exceeding number %u will be ignored\n",
					gdata->device[MAX_DEVICES_PER_NODE-1]);
			break;
		}
		gdata->device[gdata->devices] = dev;
		++gdata->devices;
		++gdata->totDevices;
	}

	// Check if pipeline script path is defined and the file exists
	if (_clOptions->visualization) {
		std::string script_path = _clOptions->pipeline_fpath;

		if (script_path.empty()) {
			printf("WARNING: pipeline script is not defined, visualization will be disabled.\n");
		} else {
			ifstream f(script_path.c_str());
			if (!f.good()) {
				printf("WARNING: pipeline script could not be opened, visualization will be disabled.\n");
			}
		}
	}

	_clOptions->problem = string( selected_problem.name );

	// Left for future dynamic loading:
	/*if (_clOptions->problem.empty()) {
		problem_list();
		exit(0);
	}*/

	return 1;
}

bool check_short_length() {
	return (sizeof(uint) == 2*sizeof(short));
}

// static pointer to the instance of GlobalData allocated in the main. Its aim is to make
// variables such as quit_request and save_request accessible by the signal handlers
static GlobalData *gdata_static_pointer = NULL;

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
	gdata_static_pointer->save_request = true;
}

void simulate(GlobalData *gdata, RunMode repack_or_run)
{
	gdata->run_mode = repack_or_run;

	gdata->problem = selected_problem.create(gdata);
	if (gdata->problem->simframework())
		gdata->simframework = gdata->problem->simframework();
	else
		throw invalid_argument("no simulation framework defined in the problem!");
	gdata->allocPolicy = gdata->simframework->getAllocPolicy();

	if (gdata->run_mode == REPACK && !HAS_REPACKING(gdata->problem->simparams()->simflags))
		throw invalid_argument("Repacking is not enabled in the " + gdata->clOptions->problem + " problem");


	// get - and actually instantiate - the existing instance of GPUSPH
	GPUSPH *Simulator = GPUSPH::getInstance();

	// initialize CUDA, start workers, allocate CPU and GPU buffers
	bool initialized  = Simulator->initialize(gdata);

	if (!initialized)
		throw runtime_error("GPUSPH: problem during initialization");

	printf("GPUSPH: initialized\n");

	// run the simulation until a quit request is triggered or an exception is thrown (TODO)
	bool all_ok = Simulator->runSimulation();

	// finalize everything
	Simulator->finalize();

	if (repack_or_run == REPACK)
		gdata->cleanup();

	if (!all_ok)
		gdata->ret = 1;
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

	memset(&int_action, 0, sizeof(struct sigaction));
	int_action.sa_handler = sigint_handler;
	sigaction(SIGINT, &int_action, NULL);

	memset(&usr1_action, 0, sizeof(struct sigaction));
	usr1_action.sa_handler = sigusr1_handler;
	sigaction(SIGUSR1, &usr1_action, NULL);

	// parse command-line options
	int opt_ret = parse_options(argc, argv, &gdata);
	if (opt_ret <= 0)
		exit(-opt_ret);

	show_version();

	// TODO: check options, i.e. consistency

	// NOTE: Although GPUSPH has been designed to be run with one multi-threaded process per node, it is important not to create
	// any file or lock singleton resources before initializing the network, as the process might be forked
	gdata.networkManager = new NetworkManager();
	gdata.networkManager->initNetwork();
	gdata.networkManager->printInfo();

	int nm_worldsize = gdata.networkManager->getWorldSize();
	if (nm_worldsize > MAX_NODES_PER_CLUSTER) {
		cerr << "Too many nodes in cluster: " << nm_worldsize << " > " << MAX_NODES_PER_CLUSTER << endl;
		exit(1);
	}

	gdata.mpi_nodes = devcount_t(nm_worldsize);
	gdata.mpi_rank = gdata.networkManager->getProcessRank();

	// We "shift" the cuda device indices by devIndexOffset. It is useful in case of multiple processes per node. Will write external docs about the formula
	uint devIndexOffset = 0;
	if (gdata.clOptions->num_hosts > 0) {
		if (gdata.clOptions->byslot_scheduling)
			// non round-robin scheduling: fill first node, then start assigning to the second
			devIndexOffset = (gdata.mpi_rank % ( gdata.mpi_nodes / gdata.clOptions->num_hosts ) ) * gdata.devices;
		else
			// round-robin scheduling: distribute to non-empty node only if others have at least n-1 processes already
			devIndexOffset = (gdata.mpi_rank / gdata.clOptions->num_hosts) * gdata.devices;

		for (uint d=0; d < gdata.devices; d++)
				gdata.device[d] += devIndexOffset;
	} else
		if (gdata.clOptions->byslot_scheduling)
			printf("WARNING: --byslot-scheduling was enabled, but number of hosts is zero!\n");

	gdata.totDevices = gdata.mpi_nodes * gdata.devices;
	printf(" tot devs = %u (%u * %u)\n",gdata.totDevices, gdata.mpi_nodes, gdata.devices );
	if (gdata.clOptions->num_hosts > 0)
		printf(" num-hosts was specified: %u; shifting device numbers with offset %u\n", gdata.clOptions->num_hosts, devIndexOffset);

	is_multi_device = (gdata.totDevices > 1);

	// the Problem could (should?) be initialized inside GPUSPH::initialize()
	try {
		if (gdata.clOptions->asyncNetworkTransfers) {

			// since H2D and D2H transfers have to wait for network transfers
			if (!gdata.clOptions->gpudirect)
				throw invalid_argument("asynchronous network transfers require --gpudirect");

			// since we were too lazy to implement a more complex mechanism
			if (gdata.devices > 1)
				throw invalid_argument("asynchronous network transfers only supported with 1 process per device");
		}

		if (gdata.clOptions->repack || gdata.clOptions->repack_only)
			simulate(&gdata, REPACK);
		if (!gdata.ret && !gdata.clOptions->repack_only) {
			simulate(&gdata, SIMULATE);
		}
	} catch (exception const& e) {
		cerr << "FATAL: " << e.what() << endl;
		gdata.ret = 1;
	}

	return gdata.ret;
}

