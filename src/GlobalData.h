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

/*! \file GlobalData.h
 * GlobalData class and related data types and definitions
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
// planes
#include "planes.h"
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

// IGNORE_WARNINGS
#include "deprecation.h"

// commands that GPUSPH can issue to workers
#include "command_type.h"

// command flags are defined in their own include files
#include "command_flags.h"

// buffer definitions are set into their own include
#include "define_buffers.h"

// forward declaration of Writer
class Writer;

class Problem;

#include "debugflags.h"

// The GlobalData struct can be considered as a set of pointers. Different pointers may be initialized
// by different classes in different phases of the initialization. Pointers should be used in the code
// only where we are sure they were already initialized.
struct GlobalData {
	// Return value for the application
	int ret;

	DebugFlags debug;

	// # of GPUs running

	// number of user-specified devices (# of GPUThreads). When multi-node, #device per node
	devcount_t devices;
	// array of cuda device numbers
	unsigned int device[MAX_DEVICES_PER_NODE];

	// MPI vars
	devcount_t mpi_nodes; // # of MPI nodes. 0 if network manager is not initialized, 1 if no other nodes (only multi-gpu)
	int mpi_rank; // MPI rank. -1 if not initialized

	// total number of devices. Same as "devices" if single-node
	devcount_t totDevices;

	// array of GPUWorkers, one per GPU
	std::vector<std::shared_ptr<GPUWorker>> GPUWORKERS;

	Problem* problem;

	SimFramework *simframework;
	std::shared_ptr<BufferAllocPolicy> allocPolicy;

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
	/// global number of open-boundary vertices in the whole simulation
	/*! This is computed once at the beginning of the simulation, and
	 * used by the devices to compute the next ID when genering new particles
	 */
	uint numOpenVertices;
	// number of particles of each process
	uint processParticles[MAX_NODES_PER_CLUSTER];
	// number of allocated particles *in the process*
	uint allocatedParticles;

	float3 worldSize;
	float3 worldOrigin;
	float3 cellSize;
	uint3 gridSize;
	uint nGridCells;

	// CPU buffers ("s" stands for "shared"). Not double buffered
	BufferList s_hBuffers;

	devcount_t*			s_hDeviceMap; // one uchar for each cell, tells  which device the cell has been assigned to

	// counter: how many particles per device
	uint s_hPartsPerDevice[MAX_DEVICES_PER_NODE]; // TODO: can change to PER_NODE if not compiling for multinode
	uint s_hStartPerDevice[MAX_DEVICES_PER_NODE]; // ditto

	// Counters to help splitting evenly after filling.
	// NOTE: allocated only in multi-device simulations
	uint* s_hPartsPerSliceAlongX;
	uint* s_hPartsPerSliceAlongY;
	uint* s_hPartsPerSliceAlongZ;

	// cellStart, cellEnd, segmentStart (limits of cells of the sam type) for each device.
	// Note the s(shared)_d(device) prefix, since they're device pointers
	// TODO migrate them to the buffer mechanism as well
	uint** s_dCellStarts;
	uint** s_dCellEnds;
	uint** s_dSegmentsStart;

	// last dt for each PS
	float dts[MAX_DEVICES_PER_NODE];

	// indicates whether particles were created at open boundaries
	bool	particlesCreatedOnNode[MAX_DEVICES_PER_NODE];
	bool	particlesCreated;
	// keep track of #iterations in which at particlesCreated holds
	uint	createdParticlesIterations;

	/**! Planes are defined in the implicit form a x + b y + c z + d = 0,
	 * where (a, b, c) = n is the normal to the plane (oriented towards the
	 * computational domain.
	 * On GPU, the (signed) distance between a particle and plane is computed as
	 * n.(P - P0) where P0 is a point of the plane. Therefore, on GPU we need
	 * three pieces of information for each plane: the normal (float3), the
	 * grid position of P0 (int3) and the in-cell local position of P0 (float3).
	 * This is also stored on CPU for convenience, and computed after copy_planes.
	 */
	// TODO planes should be vector<double4> and fill_planes should push back to
	// it directly, or something like that. And there should be an ENABLE_PLANES
	// simflag.
	PlaneList s_hPlanes;

	// variable gravity
	float3 s_varGravity;

	// simulation time control
	bool keep_going;
	bool quit_request;
	bool save_request;
	unsigned long iterations;
	unsigned long last_buildneibs_iteration;

	// on the host, the total simulation time is a double. on the device, it
	// will be downconverted to a float. this ensures that we can run very long
	// simulations even when the timestep is too small for the device to track
	// time changes
	// TODO check how moving boundaries cope with this
	double t;
	float dt;
	//! Will we use adaptive time-stepping?
	/*! Adaptive time-stepping is enabled by default (ENABLE_DTADAPT in the framework
	 * simulation flags) and can be disabled by adding the appropriate disable_flags<>
	 * specification to the framework.
	 * However, if the user specifies a time-step on the command-line, adaptive
	 * time-stepping will be disabled unconditionally
	 */
	bool dtadapt;

	// One TimingInfo per worker, currently used for statistics about neibs and interactions
	TimingInfo timingInfo[MAX_DEVICES_PER_NODE];
	uint lastGlobalPeakFluidBoundaryNeibsNum;
	uint lastGlobalPeakVertexNeibsNum;
	uint lastGlobalNumInteractions;

	// next command to be executed by workers
	CommandStruct nextCommand;

	// ODE objects
	int* s_hRbFirstIndex; // first indices: so forces kernel knows where to write rigid body force
	uint* s_hRbLastIndex; // last indices are the same for all workers
	float3* s_hRbDeviceTotalForce; // there is one partial totals force for each object in each thread
	float3* s_hRbDeviceTotalTorque; // ditto, for partial torques

	float3* s_hRbTotalForce; // aggregate total force (sum across all devices and nodes);
	float3* s_hRbTotalTorque; // aggregate total torque (sum across all devices and nodes);

	// actual applied total forces and torques. may be different from the computed total
	// forces/torques if modified by the problem callback
	float3* s_hRbAppliedForce;
	float3* s_hRbAppliedTorque;


	// gravity centers and rototranslations, which are computed by the ODE library
	int3* s_hRbCgGridPos; // cell of the gravity center
	float3* s_hRbCgPos; // in-cell position of the gravity center

	float3* s_hRbTranslations;
	float* s_hRbRotationMatrices;
	float3* s_hRbLinearVelocities;
	float3*	s_hRbAngularVelocities;

	// waterdepth at pressure outflows: an array of numOpenBoundaries elements
	// for each device
	uint**	h_IOwaterdepth;
	// an array of numOpenBoundaries elements holding the maximum water depth
	// across all devices
	uint*   h_maxIOwaterdepth;

	// peer accessibility table (indexed with device indices, not CUDA dev nums)
	bool s_hDeviceCanAccessPeer[MAX_DEVICES_PER_NODE][MAX_DEVICES_PER_NODE];

	GlobalData(void):
		ret(0),
		debug(),
		devices(0),
		mpi_nodes(0),
		mpi_rank(-1),
		totDevices(0),
		problem(NULL),
		simframework(NULL),
		clOptions(NULL),
		threadSynchronizer(NULL),
		networkManager(NULL),
		totParticles(0),
		numOpenVertices(0),
		allocatedParticles(0),
		nGridCells(0),
		s_hDeviceMap(NULL),
		s_hPartsPerSliceAlongX(NULL),
		s_hPartsPerSliceAlongY(NULL),
		s_hPartsPerSliceAlongZ(NULL),
		s_dCellStarts(NULL),
		s_dCellEnds(NULL),
		s_dSegmentsStart(NULL),
		particlesCreated(false),
		createdParticlesIterations(0),
		s_hPlanes(),
		keep_going(true),
		quit_request(false),
		save_request(false),
		iterations(0),
		last_buildneibs_iteration(ULONG_MAX),
		t(0.0),
		dt(0.0f),
		lastGlobalPeakFluidBoundaryNeibsNum(0),
		lastGlobalPeakVertexNeibsNum(0),
		lastGlobalNumInteractions(0),
		nextCommand(IDLE),
		s_hRbFirstIndex(NULL),
		s_hRbLastIndex(NULL),
		s_hRbDeviceTotalForce(NULL),
		s_hRbDeviceTotalTorque(NULL),
		s_hRbTotalForce(NULL),
		s_hRbTotalTorque(NULL),
		s_hRbAppliedForce(NULL),
		s_hRbAppliedTorque(NULL),
		s_hRbCgGridPos(NULL),
		s_hRbCgPos(NULL),
		s_hRbTranslations(NULL),
		s_hRbRotationMatrices(NULL),
		s_hRbLinearVelocities(NULL),
		s_hRbAngularVelocities(NULL),
		h_IOwaterdepth(NULL),
		h_maxIOwaterdepth(NULL)
	{
		// init dts
		for (uint d=0; d < MAX_DEVICES_PER_NODE; d++)
			dts[d] = 0.0F;

		// init particlesCreatedOnNode
		for (uint d=0; d < MAX_DEVICES_PER_NODE; d++)
			particlesCreatedOnNode[d] = false;

		for (uint d=0; d < MAX_DEVICES_PER_NODE; d++)
			for (uint p=0; p < MAX_DEVICES_PER_NODE; p++)
				s_hDeviceCanAccessPeer[d][p] = false;

	};

	// compute the global position from grid and local pos. note that the
	// world origin needs to be added to this
	template<typename T> // T should be uint3 or int3
	double3 calcGlobalPosOffset(T const& gridPos, float3 const& pos) const {
		double3 dpos;
		dpos.x = ((double) this->cellSize.x)*(gridPos.x + 0.5) + (double) pos.x;
		dpos.y = ((double) this->cellSize.y)*(gridPos.y + 0.5) + (double) pos.y;
		dpos.z = ((double) this->cellSize.z)*(gridPos.z + 0.5) + (double) pos.z;
		return dpos;
	}

	// compute the coordinates of the cell which contains the particle located at pos
	int3 calcGridPosHost(double px, double py, double pz) const {
		int3 gridPos;
		gridPos.x = (int)floor((px - worldOrigin.x) / cellSize.x);
		gridPos.y = (int)floor((py - worldOrigin.y) / cellSize.y);
		gridPos.z = (int)floor((pz - worldOrigin.z) / cellSize.z);
		return gridPos;
	}
	// overloaded
	int3 calcGridPosHost(double3 pos) const {
		return calcGridPosHost(pos.x, pos.y, pos.z);
	}

	// compute the linearized hash of the cell located at gridPos
	uint calcGridHashHost(int cellX, int cellY, int cellZ) const {
		int3 trimmed;
		trimmed.x = std::min( std::max(0, cellX), int(gridSize.x)-1);
		trimmed.y = std::min( std::max(0, cellY), int(gridSize.y)-1);
		trimmed.z = std::min( std::max(0, cellZ), int(gridSize.z)-1);
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
	devcount_t calcGlobalDeviceIndex(double4 globalPos) const {
		// do not access s_hDeviceMap if single-GPU
		if (devices == 1 && mpi_nodes == 1) return 0;
		// compute 3D cell coordinate
		int3 cellCoords = calcGridPosHost( globalPos.x, globalPos.y, globalPos.z );
		// compute cell linearized index
		uint linearizedCellIdx = calcGridHashHost( cellCoords );
		// read which device number was assigned
		return s_hDeviceMap[linearizedCellIdx];
	}

	// pretty-print memory amounts
	std::string memString(size_t memory) const {
		static const char *memSuffix[] = {
			"B", "KiB", "MiB", "GiB", "TiB"
		};
		static const size_t memSuffix_els = sizeof(memSuffix)/sizeof(*memSuffix);

		double mem = (double)memory;
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
	std::string addSeparators(long int number) const {
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

	std::string to_string(uint number) const {
		std::ostringstream ss;
		ss << number;
		return ss.str();
	}

	// returns a string in the format "r.w" with r = process rank and w = world size
	std::string rankString() const {
		return to_string(mpi_rank) + "." + to_string(mpi_nodes);
	}


	/* disable -Wconversion warnings in this uchar manipulation sections, since GCC is a bit overeager
	 * in signaling potential issues in the upconversion from uchar to (u)int and subsequent downconversion
	 * that happen on the shifts
	 */
	IGNORE_WARNINGS(conversion)

	// *** MPI aux methods: conversion from/to local device ids to global ones
	// get rank from globalDeviceIndex
	inline static devcount_t RANK(devcount_t globalDevId) { return (globalDevId >> DEVICE_BITS);} // discard device bits
	// get deviceIndex from globalDeviceIndex
	inline static devcount_t DEVICE(devcount_t globalDevId) { return (globalDevId & DEVICE_BITS_MASK);} // discard all but device bits
	// get globalDeviceIndex from rank and deviceIndex
	inline static devcount_t GLOBAL_DEVICE_ID(devcount_t nodeRank, devcount_t localDevId) { return ((nodeRank << DEVICE_BITS) | (localDevId & DEVICE_BITS_MASK));} // compute global dev id
	// compute a simple "linearized" index of the given device, as opposite to convertDevices() does. Not static because devices is known after instantiation and initialization
	inline devcount_t GLOBAL_DEVICE_NUM(devcount_t globalDevId) { return devices * RANK( globalDevId ) + DEVICE( globalDevId ); }
	// opoosite of the previous: get rank
	devcount_t RANK_FROM_LINEARIZED_GLOBAL(devcount_t linearized) const { return linearized / devices; }
	// opposite of the previous: get device
	devcount_t DEVICE_FROM_LINEARIZED_GLOBAL(devcount_t linearized) const { return linearized % devices; }

	// translate the numbers in the deviceMap in the correct global device index format (5 bits node + 3 bits device)
	void convertDeviceMap() const {
		for (uint n = 0; n < nGridCells; n++) {
			devcount_t _rank = RANK_FROM_LINEARIZED_GLOBAL( s_hDeviceMap[n] );
			devcount_t _dev  = DEVICE_FROM_LINEARIZED_GLOBAL( s_hDeviceMap[n] );
			s_hDeviceMap[n] = GLOBAL_DEVICE_ID(_rank, _dev);
		}
	}

	RESTORE_WARNINGS

	// Write the process device map to a CSV file. Appends process rank if multinode.
	// To open such file in Paraview: open the file; check the correct separator is set; apply "Table to points" filter;
	// set the correct fields; apply and enable visibility
	void saveDeviceMapToFile(std::string prefix) const {
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
	void saveCompactDeviceMapToFile(std::string prefix, uint srcDev, uint *compactDeviceMap) const {
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
#define SINGLE_GPU (!MULTI_GPU)
#define MULTI_DEVICE (MULTI_GPU || MULTI_NODE)
#define SINGLE_DEVICE (!MULTI_DEVICE)

#endif // _GLOBAL_DATA_
