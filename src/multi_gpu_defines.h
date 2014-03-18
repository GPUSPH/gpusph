#ifndef _MULTIGPU_DEFINES_
#define _MULTIGPU_DEFINES_

// for HASH_KEY_SIZE
#include "hash_key_size_select.opt"

// we use a byte (uchar) to address a device in the cluster
#define MAX_DEVICES_PER_CLUSTER 256
// how many bits [1...8] we reserve to the node rank in the global device index
// default: 5 bits for the node, 3 for the device. Max 32 nodes with 8 devices each
#define NODE_BITS 5
#define DEVICE_BITS (8 - NODE_BITS)
#define MAX_NODES_PER_CLUSTER (1 << NODE_BITS)
#define MAX_DEVICES_PER_NODE  (1 << DEVICE_BITS)
#define DEVICE_BITS_MASK (MAX_DEVICES_PER_NODE - 1)

// if hashKey is 64 bits long, the first two bits of the cell hash are reserved; otherwise, the max number of cells is UINT_MAX
#if HASH_KEY_SIZE == 32
#define MAX_CELLS			(UINT_MAX)
#else
#define MAX_CELLS			(UINT_MAX >> 2)
#endif

// cellTypes used as array indices for the segments
#define CELLTYPE_INNER_CELL			0U
#define CELLTYPE_INNER_EDGE_CELL	1U
#define CELLTYPE_OUTER_EDGE_CELL	2U
#define CELLTYPE_OUTER_CELL			3U

// 2 high bits for cell type in the compact device map. It is important in
// the current implementation that the OUTER cells are sorted last
#define CELLTYPE_INNER_CELL_SHIFTED			(CELLTYPE_INNER_CELL<<30)
#define CELLTYPE_INNER_EDGE_CELL_SHIFTED	(CELLTYPE_INNER_EDGE_CELL<<30)
#define CELLTYPE_OUTER_EDGE_CELL_SHIFTED	(CELLTYPE_OUTER_EDGE_CELL<<30)
#define CELLTYPE_OUTER_CELL_SHIFTED			(CELLTYPE_OUTER_CELL<<30) // memset to 0xFF for making OUTER_CELL defaults

// Bitmasks used to reset the cellType (AND mask to reset the high bits, AND ~mask to extract them)
#if HASH_KEY_SIZE == 32
// mask: 111111...
#define CELLTYPE_BITMASK (UINT_MAX)
#else
// mask: 001111...
#define CELLTYPE_BITMASK		(~( 3U  << 30 ))
#define CELLTYPE_BITMASK_LONG	(~( 3LU  << 62 ))
#endif

// empty segment (uint)
#define EMPTY_SEGMENT (UINT_MAX)
// guess?
#define EMPTY_CELL (UINT_MAX)

#endif // _MULTIGPU_DEFINES_


// The spaghetti-inclusion of headers need some tricks
//#ifdef _JUST_DEVICES_

// maximum number of GPUS
//#define MAX_DEVICES 10

// it set to true, wait next buildneibs to save to file
/* #define ALIGN_DUMP_WITH_NEIBS false

// size of exchange buffer for particle transfer, in particles
#define EXCHANGE_BUF_SIZE (128*1024)

// every device allocates totParticles/devices + ALLOCATION_MARGIN
#define ALLOCATION_MARGIN_FACTOR (1.4f)

// num of elements to average (tip: even please)
#define FORCES_AVERAGE_SAMPLES (10)

// load balancing threshold trigger: multiplier for one slice required time
#define LB_THRESHOLD_MULTIPLIER (0.5f)

#else // ifdef _JUST_DEVICES_

#ifndef _MULTIGPU_DEFINES_
#define _MULTIGPU_DEFINES_

#ifndef PROBLEM
#include "problem_select.opt"
#endif

#include <cuda_runtime.h>
typedef cudaStream_t cudaStream_t;

// define Problem Simmetry Axis
#if defined(PROBLEM_SIMMETRY_PLANE_YZ)
	#define PSA x
	#define CALC_SIMMETRY_HASH return \
		__mul24(__mul24(gridPos.x, gridSize.y), gridSize.z) + \
		__mul24(gridPos.y, gridSize.z) + \
		gridPos.z;
	#define CALC_SIMMETRY_HASH_HOST return \
		(gridPos.x*cdata->gridSize.y*cdata->gridSize.z) + \
		(gridPos.y*cdata->gridSize.z) + gridPos.z;
#elif defined(PROBLEM_SIMMETRY_PLANE_XZ)
    #define PSA y
    #define CALC_SIMMETRY_HASH return \
		__mul24(__mul24(gridPos.y, gridSize.z), gridSize.x) + \
		__mul24(gridPos.z, gridSize.x) + \
		gridPos.x;
	#define CALC_SIMMETRY_HASH_HOST return \
		(gridPos.y*cdata->gridSize.z*cdata->gridSize.x) + \
		(gridPos.z*cdata->gridSize.x) + gridPos.x;
#else 
	#define PROBLEM_SIMMETRY_PLANE_XY
	#define PSA z
	#define CALC_SIMMETRY_HASH return \
		__mul24(__mul24(gridPos.z, gridSize.y), gridSize.x) + \
		__mul24(gridPos.y, gridSize.x) + \
		gridPos.x;
	#define CALC_SIMMETRY_HASH_HOST return \
		(gridPos.z*cdata->gridSize.y*cdata->gridSize.x) + \
		(gridPos.y*cdata->gridSize.x) + gridPos.x;
#endif

// quoted axis for output
// http://gcc.gnu.org/onlinedocs/cpp/Stringification.html#Stringification
#define pre_str(s) #s
#define str(s) pre_str(s)
#define QUOTED_PSA str(PSA) */
/*
// for some reasons, the above code (taken from the manual!) doesn't work,
// while the following does...
#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)
#define QUOTED_PSA EXPAND_AND_QUOTE(PSA)

// Choose which thread synchronization method to use:
//  - pthreads signals and wait OR
// NOT MANTAINED NOW
//#define SYNC_BY_SIGNAL_WAIT
//  - pthreads barriers OR
//#define SYNC_BY_BARRIERS
//  - boolean busy wait
// NOT MANTAINED NOW
//#define SYNC_BY_BUSYWAIT
// test
#define MULTIPLATFORM_BARRIERS_SYNC

#endif // _MULTIGPU_DEFINES_

#endif // ifdef _JUST_DEVICES_ else */
