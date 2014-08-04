/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

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

#ifndef _GLOBAL_DATA_
#define _GLOBAL_DATA_

// ostringstream
#include <sstream>

// std::map
#include <map>

// MAX_DEVICES et al.
#include "multi_gpu_defines.h"
// float4 et al
#include "vector_types.h"
// common host types
#include "common_types.h"
// Problem
#include "Problem.h"
// Options
#include "Options.h"
// TimingInfo
#include "timing.h"

// BufferList
#include "buffer.h"

// COORD1, COORD2, COORD3
#include "linearization.h"

// GPUWorker
// no need for a complete definition, a simple declaration will do
// and since GPUWorker.h needs to include GlobalData.h, it solves
// the problem of recursive inclusions
class GPUWorker;

// Synchronizer
#include "Synchronizer.h"
// Writer
#include "Writer.h"
// NetworkManager
#include "NetworkManager.h"


// Next step for workers. It could be replaced by a struct with the list of parameters to be used.
// A few explanations: DUMP requests to download pos, vel and info on shared arrays; DUMP_CELLS
// requests to download cellStart and cellEnd
enum CommandType {
	IDLE,				// do a dummy cycle
	CALCHASH,			// run calcHash kernel
	SORT,				// run thrust::sort
	INVINDEX,			// save the old index for segment connectivity
	CROP,				// crop out all the external particles
	REORDER,			// run reorderAndFindCellStart kernel
	BUILDNEIBS,			// run buildNeibs kernel
	FORCES_SYNC,		// run forces kernel in a blocking fashion (texture binds + kernel + unbinds + dt reduction)
	FORCES_ENQUEUE,		// enqueues forces kernel in an asynchronous fashion and returns (texture binds + kernel)
	FORCES_COMPLETE,	// waits for the forces kernel to complete (device sync + texture unbinds + dt reduction)
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
	SA_CALC_SEGMENT_BOUNDARY_CONDITIONS,	// compute segment boundary conditions and identify fluid particles that leave open boundaries
	SA_CALC_VERTEX_BOUNDARY_CONDITIONS,		// compute vertex boundary conditions including mass update and create new fluid particles at open boundaries; at the init step this routine also computes a preliminary grad gamma direction vector
	DELETE_OUTGOING_PARTS,	// Removes particles that went through an open boundary
	SPS,				// SPS stress matrix computation kernel
	REDUCE_BODIES_FORCES,	// reduce rigid bodies forces (sum the forces for each boy)
	UPLOAD_MBDATA,		// upload data for moving boundaries, after problem callback
	UPLOAD_GRAVITY,		// upload new value for gravity, after problem callback
	UPLOAD_PLANES,		// upload planes
	UPLOAD_OBJECTS_CG,	// upload centers of gravity of objects
	UPLOAD_OBJECTS_MATRICES, // upload translation vector and rotation matrices for objects
	CALC_PRIVATE,		// compute a private variable for debugging or additional passive values
	COMPUTE_TESTPOINTS,	// compute velocities on testpoints
	QUIT				// quits the simulation cycle
};

// 0 reserved as "no flags"
#define NO_FLAGS	((flag_t)0)

// flags for kernels that process arguments differently depending on which
// step of the simulation we are at
// (e.g. forces, euler)
// these grow from the bottom
#define INITIALIZATION_STEP	((flag_t)1)
#define INTEGRATOR_STEP_1	(INITIALIZATION_STEP << 1)
#define INTEGRATOR_STEP_2	(INTEGRATOR_STEP_1 << 1)
#define	LAST_DEFINED_STEP	INTEGRATOR_STEP_2
// if new steps are added after INTEGRATOR_STEP_2, remember to update LAST_DEFINED_STEP

// flags to select which buffer to access, in case of double-buffered arrays
// these grow from the top
#define DBLBUFFER_WRITE		((flag_t)1 << (sizeof(flag_t)*8 - 1)) // last bit of the type
#define DBLBUFFER_READ		(DBLBUFFER_WRITE >> 1)

// flags for the vertexinfo .w coordinate which specifies how many vertex particles of one segment
// is associated to an open boundary
#define VERTEX1 ((flag_t)1)
#define VERTEX2 (VERTEX1 << 1)
#define VERTEX3 (VERTEX2 << 1)
#define ALLVERTICES ((flag_t)(VERTEX1 | VERTEX2 | VERTEX3))

// now, flags used to specify the buffers to access for swaps, uploads, updates, etc.
// these start from the next available bit from the bottom and SHOULD NOT get past the highest bit available
// at the top

// start with a generic define that can be used to iterate over all buffers
#define FIRST_DEFINED_BUFFER	(LAST_DEFINED_STEP << 1)

// buffer definitions are set into their own include
#include "define_buffers.h"

// forward declaration of Writer
class Writer;

class Problem;

// maps buffer keys to indices. used for currentRead and currentWrite:
// currentRead[BUFFER_SOMETHING] is the current array to be read in the double-buffered
// set BUFFER_SOMETHING
typedef std::map<flag_t, uint> BufferIndexMap;

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

	// CPU buffers ("s" stands for "shared"). Not double buffered
	BufferList s_hBuffers;

	uchar*			s_hDeviceMap; // one uchar for each cell, tells  which device the cell has been assigned to

	// counter: how many particles per device
	uint s_hPartsPerDevice[MAX_DEVICES_PER_NODE]; // TODO: can change to PER_NODE if not compiling for multinode
	uint s_hStartPerDevice[MAX_DEVICES_PER_NODE]; // ditto

	// cellStart, cellEnd, segmentStart (limits of cells of the sam type) for each device.
	// Note the s(shared)_d(device) prefix, since they're device pointers
	uint** s_dCellStarts;
	uint** s_dCellEnds;
	uint** s_dSegmentsStart;

	// last dt for each PS
	float dts[MAX_DEVICES_PER_NODE];

	// indices for double-buffered device arrays (0 or 1)

	BufferIndexMap		currentRead;
	BufferIndexMap		currentWrite;

	// moving boundaries
	float4	*s_mbData;
	uint	mbDataSize;

	// planes
	uint numPlanes;
	float4	*s_hPlanes;
	float	*s_hPlanesDiv;

	// variable gravity
	float3 s_varGravity;

	// simulation time control
	bool keep_going;
	bool quit_request;
	bool save_request;
	unsigned long iterations;
	float t;
	float dt;

	// One TimingInfo per worker, currently used for statistics about neibs and interactions
	TimingInfo timingInfo[MAX_DEVICES_PER_NODE];
	uint lastGlobalPeakNeibsNum;
	uint lastGlobalNumInteractions;

	// next command to be executed by workers
	CommandType nextCommand;
	// step parameter, e.g. for predictor/corrector scheme
	// command flags, i.e. parameter for the command
	flag_t commandFlags;
	// additional argument to be passed to the command
	float extraCommandArg;
	// set to true if next kernel has to be run only on internal particles
	// (need support of the worker and/or the kernel)
	bool only_internal;

	// disable saving (for timing, or only for the last)
	bool nosave;

	// ODE objects
	uint s_hRbLastIndex[MAXBODIES]; // last indices are the same for all workers
	float3 s_hRbTotalForce[MAX_DEVICES_PER_NODE][MAXBODIES]; // there is one partial totals force for each object in each thread
	float3 s_hRbTotalTorque[MAX_DEVICES_PER_NODE][MAXBODIES]; // ditto, for partial torques
	// gravity centers and rototranslations, which are computed by the ODE library
	float3* s_hRbGravityCenters;
	float3* s_hRbTranslations;
	float* s_hRbRotationMatrices;

	// peer accessibility table (indexed with device indices, not CUDA dev nums)
	bool s_hDeviceCanAccessPeer[MAX_DEVICES_PER_NODE][MAX_DEVICES_PER_NODE];

	GlobalData(void):
		devices(0),
		mpi_nodes(0),
		mpi_rank(-1),
		totDevices(0),
		problem(NULL),
		clOptions(NULL),
		threadSynchronizer(NULL),
		networkManager(NULL),
		totParticles(0),
		allocatedParticles(0),
		nGridCells(0),
		s_hDeviceMap(NULL),
		s_dCellStarts(NULL),
		s_dCellEnds(NULL),
		s_dSegmentsStart(NULL),
		s_mbData(NULL),
		mbDataSize(0),
		numPlanes(0),
		s_hPlanes(NULL),
		s_hPlanesDiv(NULL),
		keep_going(true),
		quit_request(false),
		save_request(false),
		iterations(0),
		t(0.0f),
		dt(0.0f),
		lastGlobalPeakNeibsNum(0),
		lastGlobalNumInteractions(0),
		nextCommand(IDLE),
		commandFlags(NO_FLAGS),
		extraCommandArg(NAN),
		only_internal(false),
		nosave(false),
		s_hRbGravityCenters(NULL),
		s_hRbTranslations(NULL),
		s_hRbRotationMatrices(NULL)
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

		for (uint d=0; d < MAX_DEVICES_PER_NODE; d++)
			for (uint p=0; p < MAX_DEVICES_PER_NODE; p++)
				s_hDeviceCanAccessPeer[d][p] = false;

	};

	// compute the coordinates of the cell which contains the particle located at pos
	int3 calcGridPosHost(double px, double py, double pz) const {
		int3 gridPos;
		gridPos.x = floor((px - worldOrigin.x) / cellSize.x);
		gridPos.y = floor((py - worldOrigin.y) / cellSize.y);
		gridPos.z = floor((pz - worldOrigin.z) / cellSize.z);
		return gridPos;
	}
	// overloaded
	int3 calcGridPosHost(double3 pos) const {
		return calcGridPosHost(pos.x, pos.y, pos.z);
	}

	// compute the linearized hash of the cell located at gridPos
	uint calcGridHashHost(int cellX, int cellY, int cellZ) const {
		int3 trimmed;
		trimmed.x = min( max(0, cellX), gridSize.x-1);
		trimmed.y = min( max(0, cellY), gridSize.y-1);
		trimmed.z = min( max(0, cellZ), gridSize.z-1);
		return ( (trimmed.COORD3 * gridSize.COORD2) * gridSize.COORD1 ) + (trimmed.COORD2 * gridSize.COORD1) + trimmed.COORD1;
	}
	// overloaded
	uint calcGridHashHost(int3 gridPos) const {
		return calcGridHashHost(gridPos.x, gridPos.y, gridPos.z);
	}

	// TODO MERGE REVIEW. refactor with next one
	uint3 calcGridPosFromCellHash(uint cellHash) const {
		uint3 gridPos;

		gridPos.COORD3 = cellHash / (gridSize.COORD1 * gridSize.COORD2);
		gridPos.COORD2 = (cellHash - gridPos.COORD3 * gridSize.COORD1 * gridSize.COORD2) / gridSize.COORD1;
		gridPos.COORD1 = cellHash - gridPos.COORD2 * gridSize.COORD1 - gridPos.COORD3 * gridSize.COORD1 * gridSize.COORD2;

		return gridPos;
	}

	// reverse the linearized hash of the cell and return the location in gridPos
	int3 reverseGridHashHost(uint cell_lin_idx) const {
		int3 res;

		res.COORD3 = cell_lin_idx / (gridSize.COORD2 * gridSize.COORD1);
		res.COORD2 = (cell_lin_idx - (res.COORD3 * gridSize.COORD2 * gridSize.COORD1)) / gridSize.COORD1;
		res.COORD1 = cell_lin_idx - (res.COORD3 * gridSize.COORD2 * gridSize.COORD1) - (res.COORD2 * gridSize.COORD1);

		return make_int3(res.x, res.y, res.z);
	}

	// compute the global device Id of the cell holding globalPos
	// NOTE: as the name suggests, globalPos is _global_
	uchar calcGlobalDeviceIndex(double4 globalPos) const {
		// do not access s_hDeviceMap if single-GPU
		if (devices == 1 && mpi_nodes == 1) return 0;
		// compute 3D cell coordinate
		int3 cellCoords = calcGridPosHost( globalPos.x, globalPos.y, globalPos.z );
		// compute cell linearized index
		uint linearizedCellIdx = calcGridHashHost( cellCoords );
		// read which device number was assigned
		return s_hDeviceMap[linearizedCellIdx];
	}

	// swap (indices of) double buffered arrays
	void swapDeviceBuffers(flag_t buffers) {
		BufferIndexMap::iterator idxset = currentRead.begin();
		const BufferIndexMap::iterator stop = currentRead.end();
		for (; idxset != stop; ++idxset) {
			flag_t bufkey = idxset->first;
			if (!(bufkey & buffers))
				continue; // don't swap unselected buffers
			// manual swap, eh
			uint prv = idxset->second;
			currentRead[bufkey] = currentWrite[bufkey];
			currentWrite[bufkey] = prv;
		}
	}

	// pretty-print memory amounts
	string memString(size_t memory) const {
		static const char *memSuffix[] = {
			"B", "KiB", "MiB", "GiB", "TiB"
		};
		static const size_t memSuffix_els = sizeof(memSuffix)/sizeof(*memSuffix);

		double mem = memory;
		uint idx = 0;
		while (mem > 1024 && idx < memSuffix_els - 1) {
			mem /= 1024;
			++idx;
		}

		std::ostringstream oss;
		oss.precision(mem < 10 ? 3 : mem < 100 ? 4 : 5);
		oss << mem << " " << memSuffix[idx];
		return oss.str();
	}

	// convert to string and add thousand separators
	string addSeparators(long int number) const {
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

	string to_string(uint number) const {
		ostringstream ss;
		ss << number;
		return ss.str();
	}

	// returns a string in the format "r.w" with r = process rank and w = world size
	string rankString() const {
		return to_string(mpi_rank) + "." + to_string(mpi_nodes);
	}

	// *** MPI aux methods: conversion from/to local device ids to global ones
	// get rank from globalDeviceIndex
	inline static uchar RANK(uchar globalDevId) { return (globalDevId >> DEVICE_BITS);} // discard device bits
	// get deviceIndex from globalDeviceIndex
	inline static uchar DEVICE(uchar globalDevId) { return (globalDevId & DEVICE_BITS_MASK);} // discard all but device bits
	// get globalDeviceIndex from rank and deviceIndex
	inline static uchar GLOBAL_DEVICE_ID(uchar nodeRank, uchar localDevId) { return ((nodeRank << DEVICE_BITS) | (localDevId & DEVICE_BITS_MASK));} // compute global dev id
	// compute a simple "linearized" index of the given device, as opposite to convertDevices() does. Not static because devices is known after instantiation and initialization
	inline uchar GLOBAL_DEVICE_NUM(uchar globalDevId) { return devices * RANK( globalDevId ) + DEVICE( globalDevId ); }
	// opoosite of the previous: get rank
	uchar RANK_FROM_LINEARIZED_GLOBAL(uchar linearized) const { return linearized / devices; }
	// opposite of the previous: get device
	uchar DEVICE_FROM_LINEARIZED_GLOBAL(uchar linearized) const { return linearized % devices; }

	// translate the numbers in the deviceMap in the correct global device index format (5 bits node + 3 bits device)
	void convertDeviceMap() const {
		for (uint n = 0; n < nGridCells; n++) {
			uchar _rank = RANK_FROM_LINEARIZED_GLOBAL( s_hDeviceMap[n] );
			uchar _dev  = DEVICE_FROM_LINEARIZED_GLOBAL( s_hDeviceMap[n] );
			s_hDeviceMap[n] = GLOBAL_DEVICE_ID(_rank, _dev);
		}
	}

	// Write the process device map to a CSV file. Appends process rank if multinode.
	// To open such file in Paraview: open the file; check the correct separator is set; apply "Table to points" filter;
	// set the correct fields; apply and enable visibility
	void saveDeviceMapToFile(string prefix) const {
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
		for (uint ix=0; ix < gridSize.x; ix++)
				for (uint iy=0; iy < gridSize.y; iy++)
					for (uint iz=0; iz < gridSize.z; iz++) {
						uint cell_lin_idx = calcGridHashHost(ix, iy, iz);
						fprintf(fid,"%u,%u,%u,%u,%u\n", ix, iy, iz, cell_lin_idx, s_hDeviceMap[cell_lin_idx]);
					}
		fclose(fid);
		printf(" > device map dumped to file %s\n",fname.c_str());
	}

	// Same as saveDeviceMapToFile() but saves the *compact* device map and, if multi-gpu, also appends the device number
	// NOTE: values are shifted; CELLTYPE_*_CELL is written while CELLTYPE_*_CELL_SHIFTED is in memory
	void saveCompactDeviceMapToFile(string prefix, uint srcDev, uint *compactDeviceMap) const {
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
		for (uint ix=0; ix < gridSize.x; ix++)
				for (uint iy=0; iy < gridSize.y; iy++)
					for (uint iz=0; iz < gridSize.z; iz++) {
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
