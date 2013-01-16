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

struct GlobalData {
	// # of GPUs running

	// number of user-specified devices (# of GPUThreads)
	unsigned int devices;
	// array of devices indices
	unsigned int device[MAX_DEVICES];

	// array of GPUThreads, one per GPU
	//GPUThread** GPUTHREADS;

	//Problem* problem;

	// global number of particles
	//uint totParticles;
	// global number of planes (same as local ones)
	//uint numPlanes;
	// grid size, for particle hash computation
	//uint3 gridSize;
	// maximum neighbors number
	//uint maxneibsnum;

	// ceil(totParticles/devices)
	//uint idealSubset;

	// CPU buffers ("s" stands for "shared")
	float4*			s_hPos;  // position array
	float4*			s_hVel;  // velocity array
	particleinfo*	s_hInfo; // particle info array
	float4*			s_hForces;  // forces (alloc by 1st thread, for striping)

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
	float s_dt[MAX_DEVICES];

	// moving boundaries data
	//float4* mbData;
	//uint mbDataSize;

	// variable gravity
	//float3 var_gravity;

	// simulation time control
	bool keep_going;
	//bool quit_request;
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
	//CommandType nextCommand;

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
		/* GPUTHREADS(NULL),
		//problem(NULL),
		//totParticles(0),
		//numPlanes(0),
		//idealSubset(0), */
		s_hPos(NULL),
		s_hVel(NULL),
		s_hInfo(NULL),
		s_hForces(NULL),
		//pin_maxcfl(NULL),
		//dump_hPos(NULL),
		//dump_hVel(NULL),
		//dump_hInfo(NULL),
		//mbData(NULL),
		//mbDataSize(0),
		keep_going(true),
		//quit_request(false),
		//save_request(false),
		//save_after_bneibs(false),
		//requestSliceStartDump(false),
		iterations(0),
		t(0.0f),
		dt(0.0f)
		//cpuonly(false),
		//single_inter(false),
		//numCpuThreads(0),
		//cpuThreadIds(NULL),
		//nextCommand(IDLE),
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
};

#endif // _GLOBAL_DATA_
