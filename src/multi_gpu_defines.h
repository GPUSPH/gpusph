/*  Copyright (c) 2013-2018 INGV, EDF, UniCT, JHU

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
 * Defines for multi-device support
 */
#ifndef _MULTIGPU_DEFINES_
#define _MULTIGPU_DEFINES_

#include "backend_select.opt"
#include "mpi_select.opt"

//! We use a byte (uchar) to address a device in the cluster
//! @{
typedef unsigned char devcount_t;
#define GLOBAL_DEVICE_BITS 8
#define MAX_DEVICES_PER_CLUSTER (1U << GLOBAL_DEVICE_BITS)
//! @}

//! Distribution of device bits between per-node devices and node rank.
//! These define the maximum number of devices per node and the maximum number of nodes
//! supported by GPUSPH.
//!
//! When MPI support is disabled, we allow up to 256 devices and a single node.
//! When MPI support is enabled, the default split depends onthe backend.
//!
//! With the GPU backend, we support up to 8 devices per node (using 3 bits),
//! and the remaining bits (5) are used for the nodes, thus allowing up to 32 nodes.
//! This choice is motivated by the limit of 8 PCI devices per domain.
//! Multi-domain configurations or special situations (e.g. 8 dual GPUs)
//! may require a change of this value.
//!
//! The CPU backend by default is configured for an even split:
//! 4 bits for the device (up to 16 cores per node), and
//! 4 bits for the nodes (up to 16 nodes).
//! @{
#if !USE_MPI
#define DEVICE_BITS GLOBAL_DEVICE_BITS
#else
#if CPU_BACKEND_ENABLED
#define DEVICE_BITS 4
#else
#define DEVICE_BITS 3
#endif
#endif

#define NODE_BITS (GLOBAL_DEVICE_BITS - DEVICE_BITS)
#define MAX_NODES_PER_CLUSTER (1U << NODE_BITS)
#define MAX_DEVICES_PER_NODE  (1U << DEVICE_BITS)
#define DEVICE_BITS_MASK (MAX_DEVICES_PER_NODE - 1)
//! @}

//! The two most significant bits of the cell hash are reserved for multi-GPU
//! usage, as they are used to indicate the cell type (inner, edge, outer). Take
//! this into account in the maximum number of
//! cells we can have in a problem
#define MAX_CELLS			(UINT_MAX >> 2)

//! cellTypes used as array indices for the segments
//! @{
#define CELLTYPE_INNER_CELL			0U
#define CELLTYPE_INNER_EDGE_CELL	1U
#define CELLTYPE_OUTER_EDGE_CELL	2U
#define CELLTYPE_OUTER_CELL			3U
//! @}

//! 2 high bits for cell type in the compact device map. It is important in
//! the current implementation that the OUTER cells are sorted last
//! @{
#define CELLTYPE_INNER_CELL_SHIFTED			(CELLTYPE_INNER_CELL<<30)
#define CELLTYPE_INNER_EDGE_CELL_SHIFTED	(CELLTYPE_INNER_EDGE_CELL<<30)
#define CELLTYPE_OUTER_EDGE_CELL_SHIFTED	(CELLTYPE_OUTER_EDGE_CELL<<30)
#define CELLTYPE_OUTER_CELL_SHIFTED			(CELLTYPE_OUTER_CELL<<30) // memset to 0xFF for making OUTER_CELL defaults
//! @}

//! Bitmasks used to reset the cellType (AND mask to reset the high bits, AND
//! ~mask to extract them)
#define CELLTYPE_BITMASK		(~( 3U  << 30 ))

//! Empty segment (uint)
#define EMPTY_SEGMENT (UINT_MAX)

//! Empty cell
#define EMPTY_CELL (UINT_MAX)

#endif // _MULTIGPU_DEFINES_

// TODO: delete commented stuff ? (Alexis)
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
