/*
 * GlobalData.h
 *
 *  Created on: Jan 16, 2013
 *      Author: rustico
 */

#ifndef _GLOBAL_DATA_
#define _GLOBAL_DATA_

// MAX_DEVICES et al.
#include "multi_gpu_defines.h"
// float4 et al
#include "vector_types.h"
// particleinfo
#include "particledefine.h"
// Problem
#include "Problem.h"
// Options
#include "Options.h"
// GPUWorker
#include "GPUWorker.h"
// Synchronizer
#include "Synchronizer.h"
// Writer
#include "Writer.h"
// ostringstream
#include <sstream>
// NetworkManager
#include "NetworkManager.h"


// Next step for workers. It could be replaced by a struct with the list of parameters to be used.
// A few explanations: DUMP requests to download pos, vel and info on shared arrays; DUMP_CELLS
// requests to download cellStart and cellEnd
enum CommandType {
	IDLE,				// do a dummy cycle
	CALCHASH,			// run calcHash kernel
	SORT,				// run thrust::sort
	CROP,				// crop out all the external particles
	REORDER,			// run reorderAndFindCellStart kernel
	BUILDNEIBS,			// run buildNeibs kernel
	FORCES,				// run forces kernel
	EULER,				// run euler kernel
	DUMP,				// dump all pos, vel and info to shared host arrays
	DUMP_CELLS,			// dump cellStart and cellEnd to shared host arrays
	UPDATE_SEGMENTS,	// dump segments to shared host array, then update the number of internal parts
	DOWNLOAD_NEWNUMPARTS,	// dump the updated number of particles (in case of inlets/outlets)
	UPLOAD_NEWNUMPARTS,		// update the "newNumParts" on device with the host value
	APPEND_EXTERNAL,	// append a copy of the external cells to the end of self device arrays
	UPDATE_EXTERNAL,	// update the r.o. copy of the external cells
	MLS,				// MLS correction
	SHEPARD,			// SHEPARD correction
	VORTICITY,			// vorticity computation
	SURFACE_PARTICLES,	// surface particle detections (including storing the normals)
	SPS,				// SPS stress matrix computation kernel
	REDUCE_BODIES_FORCES,	// reduce rigid bodies forces (sum the forces for each boy)
	UPLOAD_MBDATA,		// upload data for moving boundaries, after problem callback
	UPLOAD_GRAVITY,		// upload new value for gravity, after problem callback
	UPLOAD_PLANES,		// upload planes
	UPLOAD_OBJECTS_CG,	// upload centers of gravity of objects
	QUIT				// quits the simulation cycle
};

enum WriterType
{
	TEXTWRITER,
	VTKWRITER,
	VTKLEGACYWRITER,
	CUSTOMTEXTWRITER,
	UDPWRITER
};

// 0 reserved as "no flags"
#define NO_FLAGS	((uint)0)

// flags for forces and euler kernels
#define INTEGRATOR_STEP_1	((uint)1 << 0)
#define INTEGRATOR_STEP_2	((uint)1 << 1)

// buffer constants for swapDeviceBuffers() - serving also as flags for DUMP, UPDATE_INTERNAL, etc.
// WARNING: supported flags might vary for each command
#define BUFFER_POS	((uint)1 << 2)
#define BUFFER_VEL	((uint)1 << 3)
#define BUFFER_INFO	((uint)1 << 4)
#define BUFFER_VORTICITY	((uint)1 << 5)
#define BUFFER_NORMALS		((uint)1 << 6)
#define BUFFER_FORCES	((uint)1 << 7)
#define BUFFER_TAU	((uint)1 << 8)

// the selection of the double buffers is also encoded (in high bits)
#define DBLBUFFER_READ		((uint)1 << 30)
#define DBLBUFFER_WRITE		((uint)1 << 31)

// common shortcut
#define BUFFERS_POS_VEL_INFO	(BUFFER_POS | BUFFER_VEL | BUFFER_INFO)

// forward declaration of Writer
class Writer;

class Problem;

// The GlobalData struct can be considered as a set of pointers. Different pointers may be initialized
// by different classes in different phases of the initialization. Pointers should be used in the code
// only where we are sure they were already initialized.
struct GlobalData {
	// # of GPUs running

	// number of user-specified devices (# of GPUThreads). When multi-node, #device per node
	unsigned int devices;
	// array of cuda device numbers
	unsigned int device[MAX_DEVICES_PER_NODE];

	// MPI vars
	unsigned int mpi_nodes; // # of MPI nodes. 0 if network manager is not initialized, 1 if no other nodes (only multi-gpu)
	int mpi_rank; // MPI rank. -1 if not initialized

	// total number of devices. Same as "devices" if single-node
	unsigned int totDevices;

	// array of GPUWorkers, one per GPU
	GPUWorker** GPUWORKERS;

	Problem* problem;

	Options* clOptions;

	Synchronizer* threadSynchronizer;

	NetworkManager* networkManager;

	// NOTE: the following holds
	// s_hPartsPerDevice[x] <= processParticles[d] <= totParticles <= allocatedParticles
	// - s_hPartsPerDevice[x] is the number of particles currently being handled by the GPU
	//   (only useful in multigpu to keep track of the number of particles to dump; varies according to the fluid displacemente in the domain)
	// - processParticles[d] is the sum of all the internal particles of all the GPUs in the process of rank d
	//   (only useful in multinode to keep track of the number of particles and offset to dump them on host; varies according to the fluid displacement in the domain)
	// - totParticles is the sum of all the internal particles of all the network
	//   (equal to the sum of all the processParticles, can vary if there are inlets/outlets)
	// - allocatedParticles is the number of allocations
	//   (can be higher when there are inlets)

	// global number of particles - whole simulation
	uint totParticles;
	// number of particles of each process
	uint processParticles[MAX_NODES_PER_CLUSTER];
	// number of allocated particles *in the process*
	uint allocatedParticles;
	// global number of planes (same as local ones)
	//uint numPlanes;
	// grid size, for particle hash computation
	//uint3 gridSize;
	// maximum neighbors number
	//uint maxneibsnum;

	float3 worldSize;
	float3 worldOrigin;
	float3 cellSize;
	uint3 gridSize;
	uint nGridCells;

	// ceil(totParticles/devices)
	//uint idealSubset;

	// CPU buffers ("s" stands for "shared"). Not double buffered
	float4*			s_hPos;  // position array
	float4*			s_hVel;  // velocity array
	particleinfo*	s_hInfo; // particle info array
	float3*			s_hVorticity; // vorticity
	float4*			s_hNormals; // surface normals
	float4*			s_hForces;  // forces (alloc by 1st thread, for striping)
	uchar* 			s_hDeviceMap; // one uchar for each cell, tells  which device the cell has been assigned to

	// counter: how many particles per device
	uint s_hPartsPerDevice[MAX_DEVICES_PER_NODE]; // TODO: can change to PER_NODE if not compiling for multinode
	uint s_hStartPerDevice[MAX_DEVICES_PER_NODE]; // ditto

	// cellStart, cellEnd, segmentStart (limits of cells of the sam type) for each device.
	// Note the s(shared)_d(device) prefix, since they're device pointers
	uint** s_dCellStarts;
	uint** s_dCellEnds;
	uint** s_dSegmentsStart;

	// pinned memory var to retrieve dt asynchronously
	//float *pin_maxcfl;

	// CPU buffers for file dump
	//float4*			dump_hPos;  // position array
	//float4*			dump_hVel;  // velocity array
	//particleinfo*	dump_hInfo; // particle info array

	// number of neibs overlapping with prev and next for each GPU, and offset
	//uint s_overlapping[MAX_DEVICES][3];

	// offsets; copy for async file save (for VTKWriter if cdata != NULL)
	//uint s_last_offsets[MAX_DEVICES];

	// buffers for particles crossing GPUs
	//float4 s_pos_outgoing_buf[MAX_DEVICES][2][EXCHANGE_BUF_SIZE];
	//float4 s_vel_outgoing_buf[MAX_DEVICES][2][EXCHANGE_BUF_SIZE];
	//particleinfo s_info_outgoing_buf[MAX_DEVICES][2][EXCHANGE_BUF_SIZE];

	// last dt for each PS
	float dts[MAX_DEVICES_PER_NODE];

	// indices for double-buffered device arrays (0 or 1)
	uint currentPosRead;	// current index in m_dPos for position reading (0 or 1)
	uint currentPosWrite;	// current index in m_dPos for writing (0 or 1)
	uint currentVelRead;	// current index in m_dVel for velocity reading (0 or 1)
	uint currentVelWrite;	// current index in m_dVel for writing (0 or 1)
	uint currentInfoRead;		// current index in m_dInfo for info reading (0 or 1)
	uint currentInfoWrite;	// current index in m_dInfo for writing (0 or 1)

	// moving boundaries
	float4* s_mbData;
	uint mbDataSize;

	// planes
	uint numPlanes;
	float4* s_hPlanes;
	float *	s_hPlanesDiv;

	// variable gravity
	float3 s_varGravity;

	// simulation time control
	bool keep_going;
	bool quit_request;
	//bool save_request;
	//bool save_after_bneibs;
	//bool requestSliceStartDump;
	unsigned long iterations;
	float t;
	float dt;

	// using only cpu threads for comparison
	//bool cpuonly;
	// compute half fluid-fluid interactions per thread
	//bool single_inter;

	// how many CPU threads?
	//uint numCpuThreads;

	// next command to be executed by workers
	CommandType nextCommand;
	// step parameter, e.g. for predictor/corrector scheme
	// command flags, i.e. parameter for the command
	uint commandFlags;
	// set to true if next kernel has to be run only on internal particles
	// (need support of the worker and/or the kernel)
	bool only_internal;

	// Writer variables
	WriterType writerType;
	Writer *writer;

	// disable saving (for timing, or only for the last)
	bool nosave;

	// ids, tdatas and ranges of each cpu thread
	//pthread_t *cpuThreadIds;
	//dataForCPUThread *tdatas;
	//uint *cpuThreadFromParticle;
	//uint *cpuThreadToParticle;
	//float *cpuThreadDts;
	//uint runningCPU;
	//pthread_mutex_t mutexCPU;
	//pthread_cond_t condCPU;
	//pthread_cond_t condCPUworker;

	// objects
	//float4 *cMbData; // just pointer
	//float3 crbcg[MAXBODIES];
	//float3 crbtrans[MAXBODIES];
	//float crbsteprot[9*MAXBODIES];
	uint s_hRbLastIndex[MAXBODIES]; // last indices are the same for all workers
	float3 s_hRbTotalForce[MAX_DEVICES_PER_NODE][MAXBODIES]; // there is one partial totals force for each object in each thread
	float3 s_hRbTotalTorque[MAX_DEVICES_PER_NODE][MAXBODIES]; // ditto, for partial torques
	// gravity centers and rototranslations, which are computed by the ODE library
	float3* s_hRbGravityCenters;
	float3* s_hRbTranslations;
	float* s_hRbRotationMatrices;

	// least elegant way ever to pass phase number to threads
	//bool phase1;

	// phase control
	//bool buildNeibs;

	// load balancing control
	//bool balancing_request;

	// balancing ops counter
	//uint balancing_operations;

	// asynchronous file save control
	//bool saving;

	// disable file dump
	//bool nosave;

	// disable load balancing
	//bool nobalance;

	// custom balance threshold
	//float custom_lb_threshold;

	// in multigpu, alloc for every GPU the total number of parts
	//bool alloc_max;

	GlobalData(void):
		devices(0),
		mpi_nodes(0),
		mpi_rank(-1),
		totDevices(0),
		// GPUTHREADS(NULL),
		problem(NULL),
		clOptions(NULL),
		threadSynchronizer(NULL),
		networkManager(NULL),
		totParticles(0),
		allocatedParticles(0),
		nGridCells(0),
		//idealSubset(0),
		s_hPos(NULL),
		s_hVel(NULL),
		s_hInfo(NULL),
		s_hVorticity(NULL),
		s_hNormals(NULL),
		s_hForces(NULL),
		s_hDeviceMap(NULL),
		s_dCellStarts(NULL),
		s_dCellEnds(NULL),
		s_dSegmentsStart(NULL),
		//pin_maxcfl(NULL),
		//dump_hPos(NULL),
		//dump_hVel(NULL),
		//dump_hInfo(NULL),
		s_mbData(NULL),
		mbDataSize(0),
		numPlanes(0),
		s_hPlanes(NULL),
		s_hPlanesDiv(NULL),
		keep_going(true),
		quit_request(false),
		//save_request(false),
		//save_after_bneibs(false),
		//requestSliceStartDump(false),
		iterations(0),
		t(0.0f),
		dt(0.0f),
		//cpuonly(false),
		//single_inter(false),
		//numCpuThreads(0),
		//cpuThreadIds(NULL),
		nextCommand(IDLE),
		commandFlags(0),
		only_internal(false),
		writerType(VTKWRITER),
		writer(NULL),
		nosave(false),
		s_hRbGravityCenters(NULL),
		s_hRbTranslations(NULL),
		s_hRbRotationMatrices(NULL)
		//tdatas(NULL),
		//cpuThreadFromParticle(NULL),
		//cpuThreadToParticle(NULL),
		//cpuThreadDts(NULL),
		//runningCPU(0),
		//phase1(true),
		//buildNeibs(false),
		//balancing_request(false),
		//balancing_operations(0),
		//nobalance(false),
		//custom_lb_threshold(0.0f),
		//alloc_max(false)
	{
		// init dts
		for (uint d=0; d < MAX_DEVICES_PER_NODE; d++)
			dts[d] = 0.0F;

		// init partial forces and torques
		for (uint d=0; d < MAX_DEVICES_PER_NODE; d++)
			for (uint ob=0; ob < MAXBODIES; ob++) {
				s_hRbTotalForce[d][ob] = make_float3(0.0F);
				s_hRbTotalTorque[d][ob] = make_float3(0.0F);
			}

		// init last indices for segmented scans for objects
		for (uint ob=0; ob < MAXBODIES; ob++)
			s_hRbLastIndex[ob] = 0;

	};

	// compute the coordinates of the cell which contains the particle located at pos
	int3 calcGridPosHost(float3 pos) {
		int3 gridPos;
		gridPos.x = floor((pos.x - worldOrigin.x) / cellSize.x);
		gridPos.y = floor((pos.y - worldOrigin.y) / cellSize.y);
		gridPos.z = floor((pos.z - worldOrigin.z) / cellSize.z);
		return gridPos;
	}
	// overloaded
	int3 calcGridPosHost(float px, float py, float pz) {
		int3 gridPos;
		gridPos.x = floor((px - worldOrigin.x) / cellSize.x);
		gridPos.y = floor((py - worldOrigin.y) / cellSize.y);
		gridPos.z = floor((pz - worldOrigin.z) / cellSize.z);
		return gridPos;
	}

	// compute the linearized hash of the cell located at gridPos
	uint calcGridHashHost(int3 gridPos) {
		gridPos.x = min( max(0, gridPos.x), gridSize.x-1);
		gridPos.y = min( max(0, gridPos.y), gridSize.y-1);
		gridPos.z = min( max(0, gridPos.z), gridSize.z-1);
		return ( (gridPos.z * gridSize.y) * gridSize.x ) + (gridPos.y * gridSize.x) + gridPos.x;
	}
	// overloaded
	uint calcGridHashHost(int cellX, int cellY, int cellZ) {
		int trimmedX = min( max(0, cellX), gridSize.x-1);
		int trimmedY = min( max(0, cellY), gridSize.y-1);
		int trimmedZ = min( max(0, cellZ), gridSize.z-1);
		return ( (trimmedZ * gridSize.y) * gridSize.x ) + (trimmedY * gridSize.x) + trimmedX;
	}

	// reverse the linearized hash of the cell and return the location in gridPos
	int3 reverseGridHashHost(uint cell_lin_idx) {
		int cz = cell_lin_idx / (gridSize.y * gridSize.x);
		int cy = (cell_lin_idx - (cz * gridSize.y * gridSize.x)) / gridSize.x;
		int cx = cell_lin_idx - (cz * gridSize.y * gridSize.x) - (cy * gridSize.x);
		return make_int3(cx, cy, cz);
	}

	// compute the global device Id of the cell holding pos
	uchar calcGlobalDeviceIndex(float4 pos) {
		// do not access s_hDeviceMap if single-GPU
		if (devices == 1 && mpi_nodes == 1) return 0;
		// compute 3D cell coordinate
		int3 cellCoords = calcGridPosHost( pos.x, pos.y, pos.z );
		// compute cell linearized index
		uint linearizedCellIdx = calcGridHashHost( cellCoords );
		// read which device number was assigned
		return s_hDeviceMap[linearizedCellIdx];
	}

	// swap (indices of) double buffers for positions and velocities; optionally swaps also pInfo
	void swapDeviceBuffers(uint buffers) {
		if (buffers & BUFFER_POS)	std::swap(currentPosRead, currentPosWrite);
		if (buffers & BUFFER_VEL)	std::swap(currentVelRead, currentVelWrite);
		if (buffers & BUFFER_INFO)	std::swap(currentInfoRead, currentInfoWrite);
	}

	// convert to string and add thousand separators
	string addSeparators(long int number) {
		std::ostringstream oss;
		ulong mod, div;
		uchar separator = ',';
		// last triplet need 0 padding, if it is not the only one
		bool padding_needed = false;
		// negative?
		if (number < 0) {
			oss << "-";
			number *= -1;
		}
		uint magnitude = 1000000000;
		while (number >= 1000) {
			if (number >= magnitude) {
				div = number / magnitude;
				mod = number % magnitude;
				// padding
				if (padding_needed) {
					if (div <= 99) oss << "0";
					if (div <= 9) oss << "0";
				}
				oss << div << separator;
				number = mod;
				padding_needed = true;
			}
			magnitude /= 1000;
		}
		if (padding_needed) {
			if (number <= 99) oss << "0";
			if (number <= 9) oss << "0";
		}
		oss << number;
		return oss.str();
	}

	string to_string(uint number) {
		ostringstream ss;
		ss << number;
		return ss.str();
	}

	// returns a string in the format "r.w" with r = process rank and w = world size
	string rankString() {
		return to_string(mpi_rank) + "." + to_string(mpi_nodes);
	}

	// MPI aux methods: conversion from/to local device ids to global ones
	inline static uchar RANK(uchar globalDevId) { return (globalDevId >> DEVICE_BITS);} // discard device bits
	inline static uchar DEVICE(uchar globalDevId) { return (globalDevId & DEVICE_BITS_MASK);} // discard all but device bits
	inline static uchar GLOBAL_DEVICE_ID(uchar nodeRank, uchar localDevId) { return ((nodeRank << DEVICE_BITS) | (localDevId & DEVICE_BITS_MASK));} // compute global dev id
	// compute a simple "linearized" index of the given device, as opposite to convertDevices() does. Not static because devices is known after instantiation and initialization
	inline uchar GLOBAL_DEVICE_NUM(uchar globalDevId) { return devices * RANK( globalDevId ) + DEVICE( globalDevId ); }

	// translate the numbers in the deviceMap in the correct global device index format (5 bits node + 3 bits device)
	void convertDeviceMap() {
		for (uint n = 0; n < nGridCells; n++) {
			uchar _rank = s_hDeviceMap[n] / devices;
			uchar _dev  = s_hDeviceMap[n] % devices;
			s_hDeviceMap[n] = GLOBAL_DEVICE_ID(_rank, _dev);
		}
	}

	// Write the process device map to a CSV file. Appends process rank if multinode.
	// To open such file in Paraview: open the file; check the correct separator is set; apply "Table to points" filter;
	// set the correct fields; apply and enable visibility
	void saveDeviceMapToFile(string prefix) {
		std::ostringstream oss;
		oss << problem->get_dirname() << "/";
		if (!prefix.empty())
			oss << prefix << "_";
		oss << problem->m_name;
		oss << "_dp" << problem->m_deltap;
		if (mpi_nodes > 1) oss << "_rank" << mpi_rank << "." << mpi_nodes << "." << networkManager->getProcessorName();
		oss << ".csv";
		std::string fname = oss.str();
		FILE *fid = fopen(fname.c_str(), "w");
		fprintf(fid,"X,Y,Z,LINEARIZED,VALUE\n");
		for (int ix=0; ix < gridSize.x; ix++)
				for (int iy=0; iy < gridSize.y; iy++)
					for (int iz=0; iz < gridSize.z; iz++) {
						uint cell_lin_idx = calcGridHashHost(ix, iy, iz);
						fprintf(fid,"%u,%u,%u,%u,%u\n", ix, iy, iz, cell_lin_idx, s_hDeviceMap[cell_lin_idx]);
					}
		fclose(fid);
		printf(" > device map dumped to file %s\n",fname.c_str());
	}

	// Same as saveDeviceMapToFile() but saves the *compact* device map and, if multi-gpu, also appends the device number
	void saveCompactDeviceMapToFile(string prefix, uint srcDev, uint *compactDeviceMap) {
		std::ostringstream oss;
		oss << problem->get_dirname() << "/";
		if (!prefix.empty())
			oss << prefix << "_";
		oss << problem->m_name;
		oss << "_dp" << problem->m_deltap;
		if (devices > 1) oss << "_dev" << srcDev << "." << devices;
		oss << ".csv";
		std::string fname = oss.str();
		FILE *fid = fopen(fname.c_str(), "w");
		fprintf(fid,"X,Y,Z,LINEARIZED,VALUE\n");
		for (int ix=0; ix < gridSize.x; ix++)
				for (int iy=0; iy < gridSize.y; iy++)
					for (int iz=0; iz < gridSize.z; iz++) {
						uint cell_lin_idx = calcGridHashHost(ix, iy, iz);
						fprintf(fid,"%u,%u,%u,%u,%u\n", ix, iy, iz, cell_lin_idx, compactDeviceMap[cell_lin_idx] >> 30);
					}
		fclose(fid);
		printf(" > compact device map dumped to file %s\n",fname.c_str());
	}
};

// utility defines, improve readability
#define MULTI_NODE (gdata->mpi_nodes > 1)
#define SINGLE_NODE (!MULTI_NODE)
#define MULTI_GPU (gdata->devices > 1)
#define SINGLE_GPU (gdata->devices > 1)
#define MULTI_DEVICE (MULTI_GPU || MULTI_NODE)
#define SINGLE_DEVICE (!MULTI_DEVICE)

// static pointer to the instance of GlobalData allocated in the main. Its aim is to make
// variables such as quit_request and save_request accessible by the signal handlers
static GlobalData *gdata_static_pointer = NULL;

#endif // _GLOBAL_DATA_
