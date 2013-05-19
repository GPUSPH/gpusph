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

// Next step for workers. It could be replaced by a struct with the list of parameters to be used
enum CommandType {IDLE, CALCHASH, SORT, REORDER, BUILDNEIBS, FORCES, EULER, QUIT};

// forward declaration of Writer
class Writer;

// The GlobalData struct can be considered as a set of pointers. Different pointers may be initialized
// by different classes in different phases of the initialization. Pointers should be used in the code
// only where we are sure they were already initialized.
struct GlobalData {
	// # of GPUs running

	// number of user-specified devices (# of GPUThreads)
	unsigned int devices;
	// array of devices indices
	unsigned int device[MAX_DEVICES_PER_NODE];

	// array of GPUWorkers, one per GPU
	GPUWorker** GPUWORKERS;

	Problem* problem;

	Options* clOptions;

	Synchronizer* threadSynchronizer;

	// global number of particles - whole simulation
	uint totParticles;
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
	uint nSortingBits;

	// ceil(totParticles/devices)
	//uint idealSubset;

	// CPU buffers ("s" stands for "shared"). Not double buffered
	float4*			s_hPos;  // position array
	float4*			s_hVel;  // velocity array
	particleinfo*	s_hInfo; // particle info array
	float4*			s_hForces;  // forces (alloc by 1st thread, for striping)
	uchar* 			s_hDeviceMap; // one uchar for each cell, tells  which device the cell has been assigned to

	// counter: how many particles per device
	uint s_hPartsPerDevice[MAX_DEVICES_PER_CLUSTER]; // TODO: can change to PER_NODE if not compiling for multinode
	uint s_hStartPerDevice[MAX_DEVICES_PER_CLUSTER]; // ditto

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

	// moving boundaries data
	//float4* mbData;
	//uint mbDataSize;

	// variable gravity
	//float3 var_gravity;

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
	Problem::WriterType writerType;
	Writer *writer;

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

	// object moving stuff
	//float4 *cMbData; // just pointer
	//float3 crbcg[MAXBODIES];
	//float3 crbtrans[MAXBODIES];
	//float crbsteprot[9*MAXBODIES];

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
		// GPUTHREADS(NULL),
		problem(NULL),
		clOptions(NULL),
		threadSynchronizer(NULL),
		/*totParticles(0),
		//numPlanes(0),
		//idealSubset(0), */
		s_hPos(NULL),
		s_hVel(NULL),
		s_hInfo(NULL),
		s_hForces(NULL),
		s_hDeviceMap(NULL),
		//pin_maxcfl(NULL),
		//dump_hPos(NULL),
		//dump_hVel(NULL),
		//dump_hInfo(NULL),
		//mbData(NULL),
		//mbDataSize(0),
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
		writerType(Problem::VTKWRITER),
		writer(NULL)
		//tdatas(NULL),
		//cpuThreadFromParticle(NULL),
		//cpuThreadToParticle(NULL),
		//cpuThreadDts(NULL),
		//runningCPU(0),
		//phase1(true),
		//buildNeibs(false),
		//balancing_request(false),
		//balancing_operations(0),
		//nosave(false),
		//nobalance(false),
		//custom_lb_threshold(0.0f),
		//alloc_max(false)
	{ };

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
		gridPos.x = max(0, min(gridPos.x, gridSize.x-1));
		gridPos.y = max(0, min(gridPos.y, gridSize.y-1));
		gridPos.z = max(0, min(gridPos.z, gridSize.z-1));
		return ( (gridPos.z * gridSize.y) * gridSize.x ) + (gridPos.y * gridSize.x) + gridPos.x;
	}
	// overloaded
	uint calcGridHashHost(int cellX, int cellY, int cellZ) {
		int trimmedX = max(0, min(cellX, gridSize.x-1));
		int trimmedY = max(0, min(cellY, gridSize.y-1));
		int trimmedZ = max(0, min(cellZ, gridSize.z-1));
		return ( (trimmedZ * gridSize.y) * gridSize.x ) + (trimmedY * gridSize.x) + trimmedX;
	}

	// swap (indices of) double buffers for positions and velocities; optionally swaps also pInfo
	void swapDeviceBuffers(bool alsoInfo) {
		std::swap(currentPosRead, currentPosWrite);
		std::swap(currentVelRead, currentVelWrite);
		if (alsoInfo)
			std::swap(currentInfoRead, currentInfoWrite);
	}
};

// static pointer to the instance of GlobalData allocated in the main. Its aim is to make
// variables such as quit_request and save_request accessible by the signal handlers
static GlobalData *gdata_static_pointer;

#endif // _GLOBAL_DATA_
