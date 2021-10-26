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
 * NeibsEngine CUDA kernels
 */

#include "sycl_wrap.h"

/*
 * Device code.
 */

// TODO :
// We can also plan to have separate arrays for boundary parts
// one for the fixed boundary that is sorted only one time in the simulation
// an other one for moving boundary that will be sort with fluid particle
// and a last one for fluid particles. In this way we will compute interactions
// only on fluid particles.

#ifndef _BUILDNEIBS_KERNEL_
#define _BUILDNEIBS_KERNEL_

#include "particledefine.h"
#include "vector_math.h"

#include "atomic_type.h"

// TODO : what was CELLTYPE_MASK_* supposed to be ? Can we delete ?
// CELLTYPE_MASK_*
#include "multi_gpu_defines.h"

#include "buildneibs_params.h"

/*! \cond */
#include "cellgrid.cuh"
/*! \endcond */

/*! \cond */
#include "neibs_iteration.cuh"
/*! \endcond */

#include "geom_core.cu"

#include "posvel_struct.h"

//! We have two ways to build the neighbors list: by particle or by cell
#ifndef BUILDNEIBS_BY_PARTICLE
#define BUILDNEIBS_BY_PARTICLE CPU_BACKEND_ENABLED
#endif
#define BUILDNEIBS_BY_CELL (!BUILDNEIBS_BY_PARTICLE)


/* Important notes on block sizes:
	- a parallel reduction for max neibs number is done inside neiblist, block
	size for neiblist MUST BE A POWER OF 2
 */

/* On GPU, BLOCK_SIZE_BUILDNEIBS is built as the product of BUILDENEIBS_CELLS_PER_BLOCK and WARP_SIZE.
 * This is meaningful when building neighbors by cell, but it also affects the block size when building by particle.
 * (We could have a more complex set of defines, but I don't think it's worth it.)
 * Note that this changes the default from 256 to 128 when building by particle
 */
#define BUILDNEIBS_CELLS_PER_BLOCK	4U
#define WARP_SIZE 32U

#if CPU_BACKEND_ENABLED
	#define BLOCK_SIZE_CALCHASH		CPU_BLOCK_SIZE
	#define MIN_BLOCKS_CALCHASH		CPU_MIN_BLOCKS
	#define BLOCK_SIZE_REORDERDATA	CPU_BLOCK_SIZE
	#define MIN_BLOCKS_REORDERDATA	CPU_MIN_BLOCKS
	#define BLOCK_SIZE_BUILDNEIBS	CPU_BLOCK_SIZE
	#define MIN_BLOCKS_BUILDNEIBS	CPU_MIN_BLOCKS
#elif (__COMPUTE__ == 75 || __COMPUTE__ == 86)
	#define BLOCK_SIZE_CALCHASH		256
	#define MIN_BLOCKS_CALCHASH		4
	#define BLOCK_SIZE_REORDERDATA	256
	#define MIN_BLOCKS_REORDERDATA	4
	#define BLOCK_SIZE_BUILDNEIBS	(BUILDNEIBS_CELLS_PER_BLOCK*WARP_SIZE)
	#define MIN_BLOCKS_BUILDNEIBS	4
#elif (__COMPUTE__ >= 20)
	#define BLOCK_SIZE_CALCHASH		256
	#define MIN_BLOCKS_CALCHASH		6
	#define BLOCK_SIZE_REORDERDATA	256
	#define MIN_BLOCKS_REORDERDATA	6
	#define BLOCK_SIZE_BUILDNEIBS	(BUILDNEIBS_CELLS_PER_BLOCK*WARP_SIZE)
	#define MIN_BLOCKS_BUILDNEIBS	5
#else
	#define BLOCK_SIZE_CALCHASH		256
	#define MIN_BLOCKS_CALCHASH		1
	#define BLOCK_SIZE_REORDERDATA	256
	#define MIN_BLOCKS_REORDERDATA	1
	#define BLOCK_SIZE_BUILDNEIBS	(BUILDNEIBS_CELLS_PER_BLOCK*WARP_SIZE)
	#define MIN_BLOCKS_BUILDNEIBS	1
#endif

/** \namespace cuneibs
 *  \brief Contains all device functions/kernels/variables used for neighbor list construction
 *
 *  The namespace cuneibs contains all the device part of neighbor list construction :
 *  	- device constants/variables
 *  	- device functions
 *  	- kernels
 */
namespace cuneibs {
/** \addtogroup neibs_device_variables Device variables
 * 	\ingroup neibs
 *  Device variables used in neighbor list construction
 *  @{ */
__device__ int d_numInteractions;			///< Total number of interactions
__device__ int d_maxFluidBoundaryNeibs;		///< Computed maximum number of fluid + boundary neighbors across particles
__device__ int d_maxVertexNeibs;			///< Computed maximum number of vertex neighbors across particles
__device__ ATOMIC_TYPE(int) d_hasTooManyNeibs;			///< id of a particle with too many neighbors
__device__ int d_hasMaxNeibs[PT_TESTPOINT];	///< Number of neighbors of that particle
__device__ ATOMIC_TYPE(int) d_hasTooManyParticles; ///< Index of a cell with too many particles
__device__ int d_hasHowManyParticles; ///< How many particles are in the  cell with too many particles
/** @} */

/** \addtogroup neibs_device_functions_params Neighbor list device function variables
 * 	\ingroup neibs
 *  Templatized structures holding variables used by neibsInCell device function
 *  @{ */
/// Common variables used in neibsInCell device function
/*!	Variables used in neibsInCell device function depends on the type of
 * 	of boundary. This structure contains the variables common to all
 * 	boundary types.
 */
struct common_niC_vars
{
	const	uint	gridHash;		///< Hash value of grid position
	const	uint	bucketStart;	///< Index of first particle in cell
	const	uint	bucketEnd;		///< Index of last particle in cell
	const	float3	pos_corr;       ///< corrected position of the central particle
			uint	neib_index;		///< current neib index
	particleinfo	neib_info;		///< current neib info
	ParticleType	neib_type;		///< current neib type
			bool	encode_cell;	///< should next neib encode the cell too?

	/// Constructor
	/*!	Computes structure members value according to the grid position.
	 */
	template<BoundaryType boundarytype, flag_t simflags>
	__device__ __forceinline__
	common_niC_vars(
		buildneibs_params<boundarytype, simflags> const& bparams,
		int3 const& gridPos,	///< [in] position in the grid
		int3 const& gridOffset,
		float3 const& pos )
	:
		gridHash(calcGridHash(gridPos)),
		bucketStart(bparams.fetchCellStart(gridHash)),
		bucketEnd(bparams.fetchCellEnd(gridHash)),
		// Substract gridOffset*cellsize to pos so we don't need to do it each time
		// we compute relPos respect to potential neighbor
		pos_corr(pos - gridOffset*d_cellSize),
		neib_index(bucketStart),
		encode_cell(true)
	{}
};


/// Specific variables used in neibsInCell with SA boundary
/*!	This structure contains specific variables used in plus of the
 * 	common one in neibsInCell for SA boundary.
 */
struct sa_boundary_niC_vars
{
	vertexinfo			vertices; 		///< TODO
	const	float4		boundElement; 	///< TODO
	const	uint		j; 				///< TODO
	const	float4		coord1; 		///< TODO
	const	float4		coord2; 		///< TODO

	/// Constructor
	/*!	Computes structure members according to particle's index
	 * 	and parameters passed to buildneibs kernel.
	 *
	 * 	\param[in] index : particle index
	 * 	\param[in] bparams : TODO
	 */
	template<flag_t simflags>
	__device__ __forceinline__
	sa_boundary_niC_vars(const uint index, buildneibs_params<SA_BOUNDARY, simflags> const& bparams) :
		vertices(bparams.fetchVert(index)),
		boundElement(bparams.fetchBound(index)),
		// j is 0, 1 or 2 depending on which is smaller (in magnitude) between
		// boundElement.{x,y,z}
		j(
			(fabs(boundElement.z) < fabs(boundElement.y) &&
			fabs(boundElement.z) < fabs(boundElement.x)) ? 2 :
			(fabs(boundElement.y) < fabs(boundElement.x) ? 1 : 0)
		 ),
		// Compute the first coordinate which is a 2-D rotated version of the normal with the j-th coordinate set to 0
		coord1(
			normalize(make_float4(
			// switch over j to give: 0 -> (0, z, -y); 1 -> (-z, 0, x); 2 -> (y, -x, 0)
			-((j==1)*boundElement.z) +  (j == 2)*boundElement.y , // -z if j == 1, y if j == 2
			  (j==0)*boundElement.z  - ((j == 2)*boundElement.x), // z if j == 0, -x if j == 2
			-((j==0)*boundElement.y) +  (j == 1)*boundElement.x , // -y if j == 0, x if j == 1
			0))
			),
		// The second coordinate is the cross product between the normal and the first coordinate
		coord2( cross3(boundElement, coord1) )
		{ }
};

// This structure holds the variables that are necessary to manage the cell/warp mapping
struct warp_cell_data
{
	uint cell_index;
	uint warp_index;
	uint lane;
};

template<BuildNeibsMappingType bn_type>
struct BuildNeibsMapping :
	COND_STRUCT(bn_type == BY_CELL, warp_cell_data)
{
	static constexpr BuildNeibsMappingType type = bn_type;

	//! buildNeibsListDevice uses shared memory in two cases:
	//! 1. in count_neighbors for the reductions (total and max neighbors)
	//! 2. during the neighbors list construction as a cache when building by cell
	//!
	//! In the latter case, it needs to be typed as float4, because this type is the widest used,
	//! otherwise, uint will suffice
	using shared_t = typename std::conditional<type == BY_CELL, float4, uint>::type;

	//! get the base for a warp or block inside a chunk of shared memory
	__device__ __forceinline__
	uint warp_base() const;

	//! get the lane for indexing inside the block or warp
	__device__ __forceinline__
	uint get_lane() const;

	//! compute the cell hash from a given particle index
	template<typename params_t>
	__device__ __forceinline__
	int3 calcGridPos(params_t const& params, uint index) const;

	//! Building by particle, we need block-wide synchronization
	//! When building by cell, we only need warp-level sync
	static constexpr uint reduction_size = (type == BY_PARTICLE ? BLOCK_SIZE_BUILDNEIBS : WARP_SIZE);

	__device__ __forceinline__
	static void sync()
#if CPU_BACKEND_ENABLED
	;
#else
	{
		switch (type) {
		case BY_PARTICLE: __syncthreads(); break;
		case BY_CELL:
#if CUDA_MAJOR >= 9
			__syncwarp();
#endif
			break;
		}
	}
#endif

};

template<> __device__ __forceinline__ uint BuildNeibsMapping<BY_PARTICLE>::warp_base() const { return 0; }
template<> __device__ __forceinline__ uint BuildNeibsMapping<BY_CELL>::warp_base() const { return warp_index*WARP_SIZE; }

#if !CPU_BACKEND_ENABLED
template<> __device__ __forceinline__ uint BuildNeibsMapping<BY_PARTICLE>::get_lane() const { return threadIdx.x; }
#endif
template<> __device__ __forceinline__ uint BuildNeibsMapping<BY_CELL>::get_lane() const { return warp_cell_data::lane; }

template<>
template<typename params_t>
__device__ __forceinline__
int3
BuildNeibsMapping<BY_PARTICLE>::calcGridPos(params_t const& params, uint index) const
{
	// Get particle grid position computed from particle hash
	return calcGridPosFromParticleHash(params.particleHash[index]);
}

template<>
template<typename params_t>
__device__ __forceinline__
int3
BuildNeibsMapping<BY_CELL>::calcGridPos(params_t const& params, uint index) const
{
	// We're iterating by cell, and the hash of the particle is just the cell index
	// we already have. There isn't even a need to remove the extra bits used for multi-GPU
	// because this kernel is GPU-local
	return calcGridPosFromCellHash(cell_index);
}


extern __shared__ char buildNeibsShared[];

template<typename T, typename Mapping, typename base_t = typename Mapping::shared_t>
__device__ __forceinline__
enable_if_t< (sizeof(T) <= sizeof(base_t)), T* >
get_buildNeibsShared_base(Mapping const& mapping, int chunk_number = 0)
{
	base_t *base = (base_t*)buildNeibsShared;
	return (T*)(base + chunk_number*BLOCK_SIZE_BUILDNEIBS + mapping.warp_base());
}

/// Cache the neighbor cell particle data in shared memory
struct niC_cell_cache : BuildNeibsMapping<BY_CELL>
{
	// pointers into the buildNeibsShared shared memory region
	float4* neib_pos_cache;
	particleinfo* neib_info_cache;

	uint cache_base;
	uint cache_index;

	__device__ __forceinline__
	niC_cell_cache(BuildNeibsMapping<BY_CELL> const& mapping, uint bucketStart)
	:
		BuildNeibsMapping<BY_CELL>( mapping ),
		neib_pos_cache( get_buildNeibsShared_base<float4>(mapping) ),
		neib_info_cache( get_buildNeibsShared_base<particleinfo>(mapping, 1) ),
		cache_base(bucketStart),
		cache_index(WARP_SIZE) // force a cache priming
	{}

	__device__ __forceinline__
	void next()
	{
		++cache_index;
	}

	template<typename params_t>
	__device__ __forceinline__
	void prime(params_t const& bparams, const uint bucketEnd)
	{
		if (cache_index < WARP_SIZE)
			return;

		sync();

		const uint gidx = cache_base + lane;
		if (gidx < bucketEnd) {
			neib_info_cache[lane] = bparams.fetchInfo(gidx);
			neib_pos_cache[lane] = bparams.fetchPos(gidx);
		}

		cache_base += WARP_SIZE;
		cache_index -= WARP_SIZE;

		sync();
	}

	template<typename params_t>
	__device__ __forceinline__
	particleinfo fetch_neib_info(params_t const& bparams, uint neib_index) const
	{
		return neib_info_cache[cache_index];
	}

	template<typename params_t>
	__device__ __forceinline__
	float4 fetch_neib_pos(params_t const& bparams, uint neib_index) const
	{
		return neib_pos_cache[cache_index];
	}

};

/// Don't cache the neighbor cell particle data in shared memory
struct niC_cell_no_cache : BuildNeibsMapping<BY_PARTICLE>
{
	__device__ __forceinline__
	niC_cell_no_cache(BuildNeibsMapping<BY_PARTICLE> const& mapping, uint bucketStart) {}

	__device__ __forceinline__
	void next() {}

	template<typename params_t>
	__device__ __forceinline__
	void prime(params_t const& bparams, const uint bucketEnd)
	{}

	template<typename params_t>
	__device__ __forceinline__
	particleinfo fetch_neib_info(params_t const& bparams, uint neib_index) const
	{
		return bparams.fetchInfo(neib_index);
	}

	template<typename params_t>
	__device__ __forceinline__
	float4 fetch_neib_pos(params_t const& bparams, uint neib_index) const
	{
		return bparams.fetchPos(neib_index);
	}

};

template<BuildNeibsMappingType bn_type>
using niC_cache = typename std::conditional<bn_type == BY_CELL, niC_cell_cache, niC_cell_no_cache>::type;

/// The actual neibInCell variables structure, which concatenates the above, as appropriate
/*! This structure provides a constructor that takes as arguments the union of the
 *	variables that would ever be needed by the neibsInCell device function.
 *  It then delegates the appropriate subset of variables to the appropriate
 *  structures it derives from, in the correct order
 */
template<BoundaryType boundarytype, BuildNeibsMappingType bn_type>
struct niC_vars :
	common_niC_vars,
	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_niC_vars),
	niC_cache<bn_type>
{
	using cache = niC_cache<bn_type>;

	template<flag_t simflags>
	__device__ __forceinline__
	particleinfo fetch_neib_info(buildneibs_params<boundarytype, simflags> const& bparams)
	{
		cache::prime(bparams, bucketEnd);
		return cache::fetch_neib_info(bparams, neib_index);
	}

	template<flag_t simflags>
	__device__ __forceinline__
	float4 fetch_neib_pos(buildneibs_params<boundarytype, simflags> const& bparams)
	{
		return cache::fetch_neib_pos(bparams, neib_index);
	}

	template<flag_t simflags>
	__device__ __forceinline__
	void load_neib_info(buildneibs_params<boundarytype, simflags> const& bparams)
	{
		neib_info = fetch_neib_info(bparams);
		neib_type = PART_TYPE(neib_info);
	}

	template<flag_t simflags>
	__device__ __forceinline__
	niC_vars(
		buildneibs_params<boundarytype, simflags> const& bparams,
		BuildNeibsMapping<bn_type> const& mapping,
		const uint index,
		int3 const& gridPos,
		int3 const& gridOffset,
		float3 const& pos)
	:
		common_niC_vars(bparams, gridPos, gridOffset, pos),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_niC_vars)(index, bparams),
		cache(mapping, bucketStart)
	{}

	__device__ __forceinline__
	void next_neib()
	{
		++neib_index;
		cache::next();
	}

	//! Check if the neighbor index we're at corresponds to a neighbor of the specified type
	template<ParticleType nptype, flag_t simflags>
	__device__ __forceinline__
	bool on_neib_of_type(
		buildneibs_params<boundarytype, simflags> const& bparams)
	{
		// we're done? then the type is irrelevant 8-)
		if (neib_index >= bucketEnd) return false;

		// ok, we have a neighbor. load the data and check the type
		load_neib_info(bparams);
		return neib_type == nptype;
	}

};
/** @} */


/** \addtogroup neibs_device_functions Device functions
 * 	\ingroup neibs
 *  Device functions used in neighbor list construction
 *  @{ */
/// Clamp grid position to edges according to periodicity
/*! This function clamp grid position to edges according to the chosen
 * 	periodicity, returns the new grid position and update the grid offset.
 * 	If the offset is more than one cell the flag toofar is set to true.
 *
 *	\tparam periodicbound type of periodic boundaries (0 ... 7)
 *
 *	\return new grid position
 *	\todo: verify periodicity along multiple axis
 */
template <Periodicity periodicbound>
__device__ __forceinline__ int3
clampGridPos(	const int3& gridPos, 	///< [in] grid position to be clamped
				int3& gridOffset, 		///< [in, out] grid offset
				bool *toofar			///< [out] set to true offset > 1 cell
				)
{
	int3 newGridPos = gridPos + gridOffset;
	// For the axis involved in periodicity the new grid position reflects
	// the periodicity and should not be clamped and the grid offset remains
	// unchanged.
	// For the axis not involved in periodicity the new grid position
	// is equal to the clamped old one and the grid offset is updated.

	// Periodicity in x
	if (periodicbound & PERIODIC_X) {
		if (newGridPos.x < 0) newGridPos.x += d_gridSize.x;
		if (newGridPos.x >= d_gridSize.x) newGridPos.x -= d_gridSize.x;
	} else {
		newGridPos.x = min(max(0, newGridPos.x), d_gridSize.x-1);
		if (abs(gridOffset.x) > 1 && newGridPos.x == gridPos.x)
			*toofar = true;
		gridOffset.x = newGridPos.x - gridPos.x;
	}

	// Periodicity in y
	if (periodicbound & PERIODIC_Y) {
		if (newGridPos.y < 0) newGridPos.y += d_gridSize.y;
		if (newGridPos.y >= d_gridSize.y) newGridPos.y -= d_gridSize.y;
	} else {
		newGridPos.y = min(max(0, newGridPos.y), d_gridSize.y-1);
		if (abs(gridOffset.y) > 1 && newGridPos.y == gridPos.y)
			*toofar = true;
		gridOffset.y = newGridPos.y - gridPos.y;
	}

	// Periodicity in z
	if (periodicbound & PERIODIC_Z) {
		if (newGridPos.z < 0) newGridPos.z += d_gridSize.z;
		if (newGridPos.z >= d_gridSize.z) newGridPos.z -= d_gridSize.z;
	} else {
		newGridPos.z = min(max(0, newGridPos.z), d_gridSize.z-1);
		if (abs(gridOffset.z) > 1 && newGridPos.z == gridPos.z)
			*toofar = true;
		gridOffset.z = newGridPos.z - gridPos.z;
	}

	return newGridPos;
}


/// clampGridPos Specialization without any periodicity
/*! \see clampGridPos
 */
template <>
__device__ __forceinline__ int3
clampGridPos<PERIODIC_NONE>(const int3& gridPos, int3& gridOffset, bool *toofar)
{
	int3 newGridPos = gridPos + gridOffset;

	// Without periodicity the new grid position is clamped to edges
	newGridPos.x = min(max(0, newGridPos.x), d_gridSize.x-1);
	newGridPos.y = min(max(0, newGridPos.y), d_gridSize.y-1);
	newGridPos.z = min(max(0, newGridPos.z), d_gridSize.z-1);
	if ((abs(gridOffset.x) > 1 && newGridPos.x == gridPos.x) ||
		(abs(gridOffset.y) > 1 && newGridPos.y == gridPos.y) ||
		(abs(gridOffset.z) > 1 && newGridPos.z == gridPos.z))
		*toofar = true;

	// In case of change in grid position the grid offset is updated
	gridOffset = newGridPos - gridPos;

	return newGridPos;
}


/// Compute the grid position for a neighboring cell
/*! This function computes the grid position for a neighboring cell,
 * 	according to the given offset and periodicity specification.
 *
 *	\tparam periodicbound type of periodic boundaries (0 ... 7)
 *
 *	\return true if the new cell is in the domain, false otherwise.
 */
template <Periodicity periodicbound>
__device__ __forceinline__ bool
calcNeibCell(
		int3 &gridPos, 			///< [in, out] current grid position
		int3 const& gridOffset) ///< [in] cell offset from current grid position
{
	// Compute the grid position of the current cell
	gridPos += gridOffset;

	// With periodic boundary when the neighboring cell grid position lies
	// outside the domain size: we wrap it to the d_gridSize or 0 according
	// with the chosen periodicity
	// TODO: verify periodicity along multiple axis
	if (periodicbound) {
		// Periodicity along x axis
		if (gridPos.x < 0) {
			if (periodicbound & PERIODIC_X)
				gridPos.x = d_gridSize.x - 1;
			else
				return false;
		}
		else if (gridPos.x >= d_gridSize.x) {
			if (periodicbound & PERIODIC_X)
				gridPos.x = 0;
			else
				return false;
		}

		// Periodicity along y axis
		if (gridPos.y < 0) {
			if (periodicbound & PERIODIC_Y)
				gridPos.y = d_gridSize.y - 1;
			else
				return false;
		}
		else if (gridPos.y >= d_gridSize.y) {
			if (periodicbound & PERIODIC_Y)
				gridPos.y = 0;
			else
				return false;
		}

		// Periodicity along z axis
		if (gridPos.z < 0) {
			if (periodicbound & PERIODIC_Z)
				gridPos.z = d_gridSize.z - 1;
			else
				return false;
		}
		else if (gridPos.z >= d_gridSize.z) {
			if (periodicbound & PERIODIC_Z)
				gridPos.z = 0;
			else
				return false;
		}
	}
	// Without periodic boundary when the neighboring cell grid position lies
	// outside the domain size there is nothing to do
	else {
		if ((gridPos.x < 0) || (gridPos.x >= d_gridSize.x) ||
			(gridPos.y < 0) || (gridPos.y >= d_gridSize.y) ||
			(gridPos.z < 0) || (gridPos.z >= d_gridSize.z))
				return false;
	}

	// If we get here, the new gridPos was computed correctly, we are
	// still in the domain
	return true;

}

/// Check if a particle is close enough to be considered for neibslist inclusion
/*! Compares the squared distance between two particles to the squared influence
 * 	radius.
 *
 * 	\param[in] relPos : relative position vector
 * 	\return : true if the distance is < to the squared influence radius, false otherwise
 * 	\tparam boundarytype : the boundary model used
 */
template<ParticleType nptype, BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<boundarytype != SA_BOUNDARY || nptype != PT_BOUNDARY, bool>
isCloseEnough(float3 const& relPos, particleinfo const& neib_info,
	buildneibs_params<boundarytype, simflags> const& params)
{
	// Default : check against the influence radius
	return sqlength(relPos) < params.sqinfluenceradius;
}

/// Specialization of isCloseEnough for SA boundaries and PT_BOUNDARY neighbors
/// In this case, the comparison is madeagainst boundNlSqInflRad,
/// which is typically larger (because the boundary particle 'representing'
/// an intersected boundary element may be further than the influence radius
template<ParticleType nptype, BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<boundarytype == SA_BOUNDARY && nptype == PT_BOUNDARY, bool>
isCloseEnough(float3 const& relPos, particleinfo const& neib_info,
	buildneibs_params<boundarytype, simflags> const& params)
{
	// Include boundary neighbors which are a little further than sqinfluenceradius
	return sqlength(relPos) < params.boundNlSqInflRad;
}


/// Process SA segments in neibsInCell
/*! Do special treatment for segments when using SA boundaries. Obviously
 * 	don't do anything at all in the standard case..
 *
 * 	\param[in] index : particle index
 * 	\param[in] neib_index : neighbor index
 * 	\param[in] relPos : relative position vector
 * 	\param[in] params : build neibs parameters
 * 	\param[in] vars : neib in cell variables
 * 	\return : true if the distance is < to the squared influence radius, false otherwise
 * 	\tparam boundarytype : the boundary model used
 */
template<BoundaryType boundarytype, flag_t simflags, BuildNeibsMappingType bn_type>
__device__ __forceinline__
enable_if_t<boundarytype != SA_BOUNDARY>
process_niC_segment(const uint index, const uint neib_id, float3 const& relPos,
	buildneibs_params<boundarytype, simflags> const& params,
	niC_vars<boundarytype, bn_type> const& var)
{ /* Do nothing by default */ }


/// Specialization of process_niC_segment for SA boundaries
/// \see process_niC_segment
template<BoundaryType boundarytype, flag_t simflags, BuildNeibsMappingType bn_type>
__device__ __forceinline__
enable_if_t<boundarytype == SA_BOUNDARY>
process_niC_segment(const uint index, const uint neib_id, float3 const& relPos,
	buildneibs_params<boundarytype, simflags> const& params,
	niC_vars<boundarytype, bn_type> const& var)
{
	int i = -1;
	if (neib_id == var.vertices.x)
		i = 0;
	else if (neib_id == var.vertices.y)
		i = 1;
	else if (neib_id == var.vertices.z)
		i = 2;
	if (i>-1) {
		// relPosProj is the projected relative position of the vertex to the segment.
		// the first coordinate system is given by the following two vectors:
		// 1. set one coordinate to 0 and rotate the remaining 2-D vector
		// 2. cross product between coord1 and the normal of the boundary element
		float2 relPosProj = make_float2(0.0);
		relPosProj.x = dot3(relPos, var.coord1);
		relPosProj.y = dot3(relPos, var.coord2);
		// save relPosProj in vertPos buffer
		if (i==0)
			params.vertPos0[index] = relPosProj;
		else if (i==1)
			params.vertPos1[index] = relPosProj;
		else
			params.vertPos2[index] = relPosProj;
	}
}

/// Offset location of the nth neighbor of type neib_type
template<ParticleType nptype>
__device__ __forceinline__
constexpr uint neibListOffset(uint neib_num)
{
	return	(nptype == PT_FLUID) ? neib_num :
			(nptype == PT_BOUNDARY) ? d_neibboundpos - neib_num :
			/* nptype == PT_VERTEX */ neib_num + d_neibboundpos + 1;
}

/// Same as above, picking the neighbor index from the type-indexed neibs_num array
template<ParticleType nptype>
__device__ __forceinline__
uint neibListOffset(uint* neibs_num)
{
	return	neibListOffset<nptype>(neibs_num[nptype]);
}

/// Check if we have too many neighbors of the given type
/*! Since PT_FLUID neighbors go from 0 upwards, while PT_BOUNDARY go from d_neibboundpos
 * downwards, the sum number of neighbors in the two must be strictly less than d_neibboundpos
 * (leaving room for 1 NEIBS_END marker separating the lists).
 * PT_VERTEX neighbors go from d_neibboundpos + 1 upwards, so there can be at most
 * d_neiblistsize - d_neibboundpos - 1 neighbors.
 * If we also want to enforce there being a NEIBS_END marker at the end (for
 * consistency with the other particle types, and thus simplifying loops over
 * neighbors), the effective number must be strictly less than that limit.
 */
template<ParticleType nptype>
__device__ __forceinline__
bool too_many_neibs(const uint* neibs_num)
{
	return
		/* We give priority to the PT_FLUID particles, so we consider an overflow
		 * for them withount considering the number of boundary particles.
		 * Full checks will find the overflow anyway from the PT_BOUNDARY check.
		 */
		nptype == PT_FLUID    ? !(neibs_num[PT_FLUID] < d_neibboundpos) :
		nptype == PT_BOUNDARY ? !(neibs_num[PT_FLUID] + neibs_num[PT_BOUNDARY] < d_neibboundpos) :
		nptype == PT_VERTEX   ? !(neibs_num[PT_VERTEX] < d_neiblistsize - d_neibboundpos - 1) :
		true;
}

/// Find neighbors of a given type in a given cell
/*! This function handles the neighbor-type specific aspect of \see neibsInCell,
 * for everything except PT_VERTEX particles on non-SA boundaries
 */
template<ParticleType nptype,
	SPHFormulation sph_formulation, typename ViscSpec, BoundaryType boundarytype, flag_t simflags,
	BuildNeibsMappingType bn_type>
__device__ __forceinline__
enable_if_t< (boundarytype == SA_BOUNDARY) || (nptype != PT_VERTEX) >
neibsInCellOfType(
			buildneibs_params<boundarytype, simflags> const& params,			// build neibs parameters
			niC_vars<boundarytype, bn_type>& var,
			const uchar		cell,		// cell number (0 ... 26)
			const uint		index,		// current particle index
			uint*			neibs_num,	// number of neighbors for the current particle
			const bool		build_nl,
			const bool		central_is_boundary,
			const bool		central_is_fea)
{
	var.encode_cell = true;

	for ( ; var.template on_neib_of_type<nptype>(params); var.next_neib()) {

		// nothing to do if we don't need to build the neighbors list
		// (i.e. we're thread running just to load neighbor data into the cache)
		if (!build_nl) continue;

		// Prevent self-interaction
		if (var.neib_index == index) continue;

		// LJ boundary particles should not have any boundary neighbor, except when
		// rheologytype is GRANULAR.
		// If we are here is because a FLOATING LJ boundary needs neibs.
		if (central_is_boundary)
			if (nptype == PT_BOUNDARY && boundarytype == LJ_BOUNDARY && ViscSpec::rheologytype != GRANULAR)
				continue;

		// With dynamic boundaries, boundary parts don't interact with other boundary parts
		// except for Grenier's formulation, where the sigma computation needs all neighbors
		// to be enumerated
		// TODO FIXME FEA why the discrepancy between LJ and DYN?
		if (central_is_boundary && !central_is_fea && !DEFORMABLE(var.neib_info))
			if (nptype == PT_BOUNDARY && boundarytype == DYN_BOUNDARY && sph_formulation != SPH_GRENIER)
				continue;

		const pos_mass neib = var.fetch_neib_pos(params);

		// Skip inactive particles
		if (is_inactive(neib))
			continue;

		// Compute relative position between particle and potential neighbor
		const float3 relPos = var.pos_corr - neib.pos;

		// Check if the squared distance is smaller than the squared influence radius
		// used for neighbor list construction
		const bool close_enough = isCloseEnough<nptype>(relPos, var.neib_info, params);

		if (!close_enough)
			continue;

		// OK, it's close enough

		/* The previous number of neighbors is the index of the current neighbor,
		 * use it to get the offset where it will be placed in the list */
		const uint offset = neibListOffset<nptype>(neibs_num);
		neibs_num[nptype]++;

		/* We only store the neighbor if we don't have too many.
		 * End-of-list markers and overflow counts will be managed
		 * after the list has been built */
		if (!too_many_neibs<nptype>(neibs_num))
		{
			const int neib_bucket_offset = var.neib_index - var.bucketStart;
			const int encode_offset = var.encode_cell ? ENCODE_CELL(cell) : 0;
			params.neibsList[ITH_NEIGHBOR_DEVICE(index, offset)] = neib_bucket_offset + encode_offset;
			var.encode_cell = false;
		}

		// SA-specific: in this case central_is_boundary tells us that this is a boundary segment
		// (or rather the particle representing it), that needs to updates its relative position projection
		// to its adjacent vertices
		if (boundarytype == SA_BOUNDARY && nptype == PT_VERTEX && central_is_boundary)
			process_niC_segment(index, id(var.neib_info), relPos, params, var);
	}

}

template<ParticleType nptype,
	SPHFormulation sph_formulation, typename ViscSpec, BoundaryType boundarytype, flag_t simflags,
	BuildNeibsMappingType bn_type>
__device__ __forceinline__
enable_if_t< (boundarytype != SA_BOUNDARY) && (nptype == PT_VERTEX) >
neibsInCellOfType(
			buildneibs_params<boundarytype, simflags> const& params,			// build neibs parameters
			niC_vars<boundarytype, bn_type>& var,
			const uchar		cell,		// cell number (0 ... 26)
			const uint		index,		// current particle index
			uint*			neibs_num,	// number of neighbors for the current particle
			const bool		build_nl,
			const bool		central_is_boundary,
			const bool		central_isfea)
{
	/* This can't happen */
}



/// Find neighbors in a given cell
/*! This function look for neighbors of the current particle in
 * 	a given cell.
 * 	The parameter params is built on specialized version of
 * 	build_neibs_params according to template values.
 * 	If the current particle belongs to a segment, the function
 * 	will also look for the tree associated vertices.
 *
 *	\param[in, out] buildneibs_params : build neibs parameters
 *	\param[in] gridPos : current particle grid position
 *	\param[in] gridOffset : cell offset from current particle cell
 *	\param[in] cell : cell number
 *	\param[in] index : index of the current particle
 *	\param[in] pos : position of the current particle
 *	\param[in, out] neibs_num : current number of neighbors found for current particle
 *	\param[in] segment : true if the current particle belongs to a segment
 *
 *	\tparam boundarytype : the boundary model used
 *	\tparam periodicbound : type of periodic boundaries (0 ... 7)
 *
 * First and last particle index for grid cells and particle's information
 * are read through texture fetches.
 */
template <SPHFormulation sph_formulation, typename ViscSpec, BoundaryType boundarytype, Periodicity periodicbound,
		 flag_t simflags, BuildNeibsMappingType bn_type>
__device__ __forceinline__ void
neibsInCell(
			buildneibs_params<boundarytype, simflags> const& params,			// build neibs parameters
			int3			gridPos,	// current particle grid position
			int3 const&		gridOffset,	// cell offset from current particle grid position
			const uchar		cell,		// cell number (0 ... 26)
			const uint		index,		// current particle index
			float3 const&	pos,		// current particle position
			uint*			neibs_num,	// number of neighbors for the current particle
			const bool		build_nl,   // build the NL for this particle (false = use it only to load neib data)
			BuildNeibsMapping<bn_type> const& mapping,
			const bool		central_is_boundary,
			const bool		central_is_fea)
{
	// Compute the grid position of the current cell, and return if it's
	// outside the domain
	if (!calcNeibCell<periodicbound>(gridPos, gridOffset))
		return;

	// Internal variables used by neibsInCell. Structure built on
	// specialized template of niC_vars.
	niC_vars<boundarytype, bn_type> var(params, mapping, index, gridPos, gridOffset, pos);

	// Return if the cell is empty
	if (var.bucketStart == CELL_EMPTY)
		return;

	var.load_neib_info(params);

	// Process neighbors by type, leveraging the fact that within cell they are sorted by type
	if (var.neib_type == PT_FLUID)
		neibsInCellOfType<PT_FLUID, sph_formulation, ViscSpec, boundarytype, simflags, bn_type>
			(params, var, cell, index, neibs_num, build_nl, central_is_boundary, central_is_fea);

	if (var.neib_type == PT_BOUNDARY)
		neibsInCellOfType<PT_BOUNDARY, sph_formulation, ViscSpec, boundarytype, simflags, bn_type>
			(params, var, cell, index, neibs_num, build_nl, central_is_boundary, central_is_fea);

	if (boundarytype == SA_BOUNDARY && var.neib_type == PT_VERTEX)
		neibsInCellOfType<PT_VERTEX, sph_formulation, ViscSpec, boundarytype, simflags, bn_type>
			(params, var, cell, index, neibs_num, build_nl, central_is_boundary, central_is_fea);

	// Testpoints have a neighbor list, but are not considered in the neighbor list of other points.
	// Moreover, since they are sorted last, we can stop iterating over the cell when we come across one
}
/** @} */


/** \addtogroup neibs_kernel Kernels
 * 	\ingroup neibs
 *  Kernels used in neighbor list construction
 *  @{ */
/// Updates particles hash value of particles and prepare the index table
/*! This kernel should be called before the sort. It
 * 	- updates hash values and relative positions for fluid and
 * 	object particles
 * 	- fill the particle's indexes array with current indexes
 *	\tparam periodicbound : type of periodic boundaries (0 ... 7)
 */
template <Periodicity periodicbound>
struct calcHashDevice
{
	static constexpr unsigned BLOCK_SIZE = BLOCK_SIZE_CALCHASH;
	static constexpr unsigned MIN_BLOCKS = MIN_BLOCKS_CALCHASH;

	float4*				posArray;			///< [in,out] particle's positions
	hashKey*			particleHash;		///< [in,out] particle's hashes
	uint*				particleIndex;		///< [out] particle's indexes
	const particleinfo*	particleInfo;		///< [in] particle's informations
	const uint*			compactDeviceMap;	///< [in] type of the cells belonging to the device
	const uint			numParticles;		///< [in] total number of particles

	calcHashDevice(
		BufferList const& bufread,
		BufferList& bufwrite,
		const uint numParticles_)
	:
		posArray(bufwrite.getData<BUFFER_POS>()),
		particleHash(bufwrite.getData<BUFFER_HASH>()),
		particleIndex(bufwrite.getData<BUFFER_PARTINDEX>()),
		particleInfo(bufread.getData<BUFFER_INFO>()),
		compactDeviceMap(bufread.getData<BUFFER_COMPACT_DEV_MAP>()),
		numParticles(numParticles_)
	{}

	__device__ void operator()(simple_work_item item) const
{
	const uint index = item.get_id();

	if (index >= numParticles)
		return;

	const particleinfo info = particleInfo[index];

	// Get the old grid hash
	uint gridHash = cellHashFromParticleHash( particleHash[index] );

	// We compute new hash only for fluid and moving not fluid particles (object, moving boundaries),
	// and surface boundaries in case of repacking
	if (FLUID(info) || MOVING(info) || (SURFACE(info) && !FLUID(info)) || DEFORMABLE(info) || FEA_NODE(info)) {
		// Getting new pos relative to old cell
		pos_mass pdata = posArray[index];

		// Getting grid address of old cell (computed from old hash)
		const int3 gridPos = calcGridPosFromCellHash(gridHash);

		// pos is the local position with respect to the cell center, and should always be
		// in the range [-cellSize/2, cellSize/2[ —otherwise, it means that the particle moved
		// to a different cell, and we should compute which one. The gridOffset computed below
		// is essentially the delta to the original gridPos: gridPos+gridOffset (modulo periodicity)
		// should be the 3D index of the new cell.
		// The trivial approach to compute the offset would be
		//     floor((pos + 0.5f*cellSize)/cellSize)
		// but in critical circumstances this leads to a particle “vibrating” between two cells; rewriting
		// the formula as
		//     floor(pos/cellSize + 0.5f)
		// reveals that the issue arises when pos/cellSize is _exactly_ 0.49999997f; why this magic value?
		// because that's the representable value immediately preceding 0.5f, so that adding it to 0.5f
		// (by rounding) leads to 1.0f.
		//
		// (A good writeup on this issue can be found in the three-part blog series starting from
		// http://blog.frama-c.com/index.php?post/2013/05/02/nearbyintf1 etc.)
		//
		// In our case, the consequence of this is that the particle gets shifted to the next cell, except
		// that on the next cell its gridOffset will be computed as -1.0f so it goes back to the original one,
		// on the next hash refresh, even if it didn't move, etc. Under these circumstances,
		// multiple consecutive hash computations would be unstable (something that reflects e.g.
		// on the inability to resume exactly from a hotfile).
		//
		// There are a number of possible solutions to this issue. One, suggested by the above post,
		// is to compute the offset as
		//     floor(pos/cellSize + 0.49999997f)
		// so that pos/cellSize needs to be at least 0.5f to become a positive offset, as required.
		// However, this constant makes a particle wiggle when pos/cellSize == -0.5f (since the
		// argument to the floor() function will then be -2.98023224e-08).
		// Our solution is to use different constants for positive and negative pos
		const float3 half_check = make_float3( // urgh I want select() with vector ops like in OpenCL 8-P
			pdata.pos.x < 0 ? 0.5f : 0.49999997f,
			pdata.pos.y < 0 ? 0.5f : 0.49999997f,
			pdata.pos.z < 0 ? 0.5f : 0.49999997f);
		int3 gridOffset = make_int3(floor(pdata.pos/d_cellSize + half_check));

		// #if 1 and change the check if there's a need to further debug gridOffset computation
#if 0
		if (id(info) == 3060) {
			printf(	"(%.9g %.9g %.9g) / "
				"(%.9g %.9g %.9g) => "
				"(%.9g %.9g %.9g) => "
				"(%.9g %.9g %.9g) => "
				"(%d %d %d)\n",
				pos.x, pos.y, pos.z,
				d_cellSize.x, d_cellSize.y, d_cellSize.z,
				pos.x/d_cellSize.x, pos.y/d_cellSize.y, pos.z/d_cellSize.z,
				pos.x/d_cellSize.x + half_check.x, pos.y/d_cellSize.y + half_check.y, pos.z/d_cellSize.z + half_check.y,
				gridOffset.x, gridOffset.y, gridOffset.z);
			printf("%.9g %.9g\n", floor(pos.y/d_cellSize.y), floor(pos.y/d_cellSize.y + half_check.y));
		}
#endif

		// Has the particle flown out of the domain by more than a cell? Clamping
		// its position will set this to true if necessary
		bool toofar = false;
		// Compute new grid pos relative to cell, adjust grid offset and compute new cell hash
		gridHash = calcGridHash(clampGridPos<periodicbound>(gridPos, gridOffset, &toofar));

		// Adjust position
		pdata.pos -= gridOffset*d_cellSize;

		// If the particle would have flown out of the domain by more than a cell, disable it
		if (toofar)
			disable_particle(pdata);

		// Mark with special hash if inactive.
		// NOTE: it could have been marked as inactive outside this kernel.
		if (is_inactive(pdata))
			gridHash = CELL_HASH_MAX;

		posArray[index] = pdata;
	}

	// Mark the cell as inner/outer and/or edge by setting the high bits
	// the value in the compact device map is a CELLTYPE_*_SHIFTED, so 32 bit with high bits set.
	// See multi_gpu_defines.h for the definition of these macros.
	if (compactDeviceMap && gridHash != CELL_HASH_MAX)
		gridHash |= compactDeviceMap[gridHash];

	// Store grid hash
	particleHash[index] = gridHash;

	// Preparing particle index array for the sort phase
	particleIndex[index] = index;
}
};


/// Updates high bits of cell hash with compact device map
/*! This kernel is specific for MULTI_DEVICE simulations
 * 	and should be called at the 1st iteration.
 * 	He computes the high bits of particle hash according to the
 * 	compact device map. Also initialize particleIndex.
 * 	\tparam periodicbound : type of periodic boundaries (0 ... 7)
 */
struct fixHashDevice
{
	static constexpr unsigned BLOCK_SIZE = BLOCK_SIZE_CALCHASH;
	static constexpr unsigned MIN_BLOCKS = MIN_BLOCKS_CALCHASH;

	hashKey*		particleHash;			///< [in;out] particle's hashes
	uint*				particleIndex;		///< [out] particle's indexes
	const particleinfo* particleInfo;		///< [in] particle's informations
	const uint*			compactDeviceMap;	///< [in] type of the cells belonging to the device
	const uint			numParticles;		///< [in] total number of particles

	fixHashDevice(
		BufferList const& bufread,
		BufferList& bufwrite,
		const uint numParticles_)
	:
		particleHash(bufwrite.getData<BUFFER_HASH>()),
		particleIndex(bufwrite.getData<BUFFER_PARTINDEX>()),
		particleInfo(bufread.getData<BUFFER_INFO>()),
		compactDeviceMap(bufread.getData<BUFFER_COMPACT_DEV_MAP>()),
		numParticles(numParticles_)
	{}

	__device__ void operator()(simple_work_item item) const
{
	const uint index = item.get_id();

	if (index >= numParticles)
		return;

	if (particleHash) {

		uint gridHash = cellHashFromParticleHash( particleHash[index] );

		// Mark the cell as inner/outer and/or edge by setting the high bits
		// the value in the compact device map is a CELLTYPE_*_SHIFTED, so 32 bit with high bits set
		if (compactDeviceMap)
			particleHash[index] = particleHash[index] | compactDeviceMap[gridHash];
	}

	// Preparing particle index array for the sort phase
	particleIndex[index] = index;
}
};

//! Find the index of the first and last particle in each cell
/*! The algorithm is pretty simple: a particle marks the start of a cell when
 *  its hash is different from the previous, and the end of a cell when its hash
 *  if different from the next.
 */

/*! In this version of the algorithm, we use shared memory as a cache */
#if CUDA_BACKEND_ENABLED
struct HashCacheShared
{
	uint sharedHash[BLOCK_SIZE_REORDERDATA+1];

	__device__ __forceinline__
	uint init(uint index, uint numParticles, const hashKey *particleHash)
	{
		uint cellHash;

		if (index < numParticles) {
			// To find where cells start/end we only need the cell part of the hash.
			// Note: we do not reset the high bits since we need them to find the segments
			// (aka where the outer particles begin)
			cellHash = cellHashFromParticleHash(particleHash[index], true);

			// Load hash data into shared memory so that we can look
			// at neighboring particle's hash value without loading
			// two hash values per thread
			sharedHash[threadIdx.x + 1] = cellHash;

			if (index > 0 && threadIdx.x == 0) {
				// first thread in block must load neighbor particle hash
				sharedHash[0] = cellHashFromParticleHash(particleHash[index - 1], true);
			}
		}

		__syncthreads();

		return cellHash;
	}

	__device__ __forceinline__
	uint prevHash() const
	{ return sharedHash[threadIdx.x]; }

};
#endif

/*! In this version, we rely on the hardware cache */
struct HashCacheL1
{
	uint _prevHash;

	__device__ __forceinline__
	uint init(uint index, uint numParticles, const hashKey *particleHash)
	{
		uint cellHash;

		if (index < numParticles) {
			// To find where cells start/end we only need the cell part of the hash.
			// Note: we do not reset the high bits since we need them to find the segments
			// (aka where the outer particles begin)
			cellHash = cellHashFromParticleHash(particleHash[index], true);

			if (index > 0)
				_prevHash = cellHashFromParticleHash(particleHash[index - 1], true);

		}

		return cellHash;
	}

	__device__ __forceinline__
	uint prevHash() const
	{ return _prevHash; }
};


/// Reorders particles data after the sort and updates cells informations
/*! This kernel should be called after the sort. It
 * 		- computes the index of the first and last particle of
 * 		each grid cell
 * 		- reorders the particle's data (position, velocity, ...)
 * 		according to particles index that have been previously
 * 		sorted during the sort phase
 * 		- compute the new number of particles accounting for those
 * 		marked for deletion
 *
 *  In order to avoid Write-After-Read issues we use double buffering.
 *  We used to this through textures, but newer architectures now share
 *  the L1 cache with the texture cache when using const restrict pointers,
 *  so we use those everywhere instead, wrapped in the reorder_data
 *  structure template that gets passed as first argument to this kernel.
 *
// \todo k goes with e, make it a float2. (Alexis).
// \todo document segmentStart (Alexis).
 */
template<typename RP> /* reorder_params specialization */
struct reorderDataAndFindCellStartDevice :
/* the reorder_params structure used to be an rparams parameter to the kernel,
 * but with our transition to the SYCL-style calling, we can make the kernel
 * structure derive from the reorder_params structure instead:
 */
	RP														///< [in/out] data to be reordered
{
	static constexpr uint BLOCK_SIZE = BLOCK_SIZE_REORDERDATA;
	static constexpr uint MIN_BLOCKS = MIN_BLOCKS_REORDERDATA;

			uint* __restrict__				cellStart;		///< [out] index of cells first particle
			uint* __restrict__				cellEnd;		///< [out] index of cells last particle
			uint* __restrict__				segmentStart;	///< [out] multi-GPU segments
	const	particleinfo * __restrict__		particleInfo;	///< [in] previously sorted particle's informations
	const	hashKey* __restrict__			particleHash;	///< [in] previously sorted particle's hashes
	const	uint* __restrict__				particleIndex;	///< [in] previously sorted particle's indexes
	const	uint							numParticles;	///< [in] total number of particles
			uint* __restrict__				newNumParticles;	///< [out] device pointer to new number of active particles

	reorderDataAndFindCellStartDevice(
		uint *segmentStart_,
		BufferList& sorted_buffers,
		BufferList const& unsorted_buffers,
		const uint numParticles_,
		uint *newNumParticles_)
	:
		/* all arrays to be sorted */
		RP(sorted_buffers, unsorted_buffers),
		// index of cells first and last particles (computed by the kernel)
		cellStart(sorted_buffers.getData<BUFFER_CELLSTART>()),
		cellEnd(sorted_buffers.getData<BUFFER_CELLEND>()),
		// multi-GPU segments
		segmentStart(segmentStart_),
		// already-sorted data, used to compute the rest
		particleInfo(sorted_buffers.getConstData<BUFFER_INFO>()),
		particleHash(sorted_buffers.getConstData<BUFFER_HASH>()),
		particleIndex(sorted_buffers.getConstData<BUFFER_PARTINDEX>()),
		numParticles(numParticles_),
		newNumParticles(newNumParticles_)
	{}

	__device__ void operator()(simple_work_item item) const
{
	// Wrapper for the cell start/end finder
#if CUDA_BACKEND_ENABLED && USE_SHARED_CELL_START_FINDER
	__shared__ HashCacheShared hashCache;
#else
	HashCacheL1 hashCache;
#endif

	const uint index = item.get_id();

	// Initialize segmentStarts
	if (segmentStart && index < 4) segmentStart[index] = EMPTY_SEGMENT;

	const uint cellHash = hashCache.init(index, numParticles, particleHash);

	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell
		// or the first inactive particle.
		// Store the index of this particle as the new cell start and as
		// the previous cell end

		const uint prevHash = hashCache.prevHash();

		// Note: we need to reset the high bits of the cell hash if the particle hash is 64 bits wide
		// every time we use a cell hash to access an element of CellStart or CellEnd

		if (index == 0 || cellHash != prevHash) {

			// New cell, otherwise, it's the number of active particles (short hash: compare with 32 bits max)
			if (cellHash != CELL_HASH_MAX)
				// If it isn't an inactive particle, it is also the start of the cell
				cellStart[cellHash & CELLTYPE_BITMASK] = index;
			else
				*newNumParticles = index;

			// If it isn't the first particle, it must also be the end of the previous cell
			if (index > 0)
				cellEnd[prevHash & CELLTYPE_BITMASK] = index;
		}

		// If we are an inactive particle, we're done (short hash: compare with 32 bits max)
		if (cellHash == CELL_HASH_MAX)
			return;

		if (index == numParticles - 1) {
			// Ditto
			cellEnd[cellHash & CELLTYPE_BITMASK] = index + 1;
			*newNumParticles = numParticles;
		}

		if (segmentStart) {
			// If segment start is given, hash key size is 64 and we detect the segments
			uchar curr_type = cellHash >> 30;
			uchar prev_type = prevHash >> 30;
			if (index == 0 || curr_type != prev_type)
				segmentStart[curr_type] = index;
		}

		// Now use the sorted index to reorder particle's data
		const uint sortedIndex = particleIndex[index];

		// The particleInfo fetch is only actually needed by the BUFFER_VERTICES sorter,
		// TODO measure the impact of this usage
		RP::reorder(index, sortedIndex, particleInfo[index]);
	}
}
};

/// Warn if a particle is behind the DEM
/*! The check is enabled only with the --debug planes option,
 *  otherwise no actual check is done
 */
template<bool debug_planes, typename Params_t>
__device__ __forceinline__
enable_if_t<debug_planes>
warnIfBehindDEM(Params_t const& params, float r, int index)
{
	if (r < 0) {
		int part_id = id(params.fetchInfo(index));
		printf("Particle %d id %d behind DEM!\n", index, part_id);
	}
}

/// Don't warn if particle is behind the DEM
template<bool debug_planes, typename Params_t>
__device__ __forceinline__
enable_if_t<!debug_planes>
warnIfBehindDEM(Params_t const& params, float r, int index)
{ return; }

/// Warn if a particle is behind a plane
/*! The check is enabled only with the --debug planes option,
 *  otherwise no actual check is done
 */
template<bool debug_planes, typename Params_t>
__device__ __forceinline__
enable_if_t<debug_planes>
warnIfBehindPlane(Params_t const& params, float r, int index, int p)
{
	if (r < 0) {
		int part_id = id(params.fetchInfo(index));
		printf("Particle %d id %d behind plane %d!\n", index, part_id, p);
	}
}

/// Don't warn if a particle is behind a plane
template<bool debug_planes, typename Params_t>
__device__ __forceinline__
enable_if_t<!debug_planes>
warnIfBehindPlane(Params_t const& params, float r, int index, int p)
{ return; }

/// Find the planes within the influence radius of the particle
/*! If neither ENABLE_PLANES nor ENABLE_DEM are active, do nothing
 */
template<bool debug_planes, BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<!HAS_DEM_OR_PLANES(simflags)>
findNeighboringPlanes(
	buildneibs_params<boundarytype, simflags> params,
	const int3& gridPos,
	const float3& pos,
	int index)
{ /* do nothing */ }

/// Check if the DEM is in range
/*! Do nothing if not ENABLE_DEM */
template<bool debug_planes, BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<!HAS_DEM(simflags), bool>
isDemInRange(
	buildneibs_params<boundarytype, simflags> params,
	const int3& gridPos,
	const float3& pos,
	int index)
{ return false; /* result won't be used anyway */ }

/// Check if the DEM is in range
/*! Actual check in case DEM is enabled */
template<bool debug_planes, BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<HAS_DEM(simflags), bool>
isDemInRange(
	buildneibs_params<boundarytype, simflags> params,
	const int3& gridPos,
	const float3& pos,
	int index)
{
	const float2 demPos = cugeom::DemPos(gridPos, pos);
	// TODO find a way to be more accurate about this
	const float globalZ = d_worldOrigin.z + (gridPos.z + 0.5f)*d_cellSize.z + pos.z;
	const float globalZ0 = cugeom::DemInterpol(params, demPos);
	const float r = globalZ - globalZ0;
	warnIfBehindDEM<debug_planes>(params, r, index);
	return (r*r < params.sqinfluenceradius);
}

/// Find the planes within the influence radius of the particle
/*! If ENABLE_PLANES or ENABLE_DEM are active, go over each defined plane
 * and see if the distance is within the required radius
 */
template<bool debug_planes, BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<HAS_DEM_OR_PLANES(simflags)>
findNeighboringPlanes(
	buildneibs_params<boundarytype, simflags> params,
	const int3& gridPos,
	const float3& pos,
	int index)
{
	using namespace cugeom;
	int neib_planes = 0;
	// we cheat a bit: we know that an int4 stores x, y, z, w consecutively,
	// so we can take the address of the x component, and the other
	// will follow
	int *store = &(params.neibPlanes[index].x);

	if ( HAS_PLANES(simflags) ) {
		for (int p = 0; p < d_numplanes; ++p) {
			float r = signedPlaneDistance(gridPos, pos, d_plane[p]);
			warnIfBehindPlane<debug_planes>(params, r, index, p);
			if (r*r < params.sqinfluenceradius) {
				store[neib_planes++] = p;
				if (neib_planes > 3) break;
			}
		}
	}

	if ( HAS_DEM(simflags) && neib_planes < 4 ) {
		if (isDemInRange<debug_planes>(params, gridPos, pos, index)) {
			store[neib_planes++] = MAX_PLANES;
		}
	}

	while (neib_planes < 4) {
		store[neib_planes++] = -1;
	}
}

//! Count the total number of neighbors (and hence the overall number of _interactions_)
//! and the max number of neighbors of each type
template< bool neibcount // should we actually count?
	, int num_sm_neibs_max // number of neighbor type groups (2 if SA, 1 otherwise)
	, typename Mapping
	>
__device__ __forceinline__
enable_if_t<neibcount == true>
count_neighbors(const uint *neibs_num, Mapping const& mapping) // computed number of neighbors per type
{
#if CUDA_BACKEND_ENABLED
	uint *sm_total_neibs_num = get_buildNeibsShared_base<uint>(mapping);
	uint *sm_neibs_max = get_buildNeibsShared_base<uint>(mapping, 1);
#endif

	uint neibs_max[num_sm_neibs_max];
	uint total_neibs_num = neibs_max[0] = neibs_num[PT_FLUID] + neibs_num[PT_BOUNDARY];
	// SA_BOUNDARY
	if (num_sm_neibs_max > 1) {
		neibs_max[1] = neibs_num[PT_VERTEX];
		total_neibs_num += neibs_num[PT_VERTEX];
	}

#if CPU_BACKEND_ENABLED
#if OPENMP_SCOPED_REDUCTIONS
#pragma omp scope reduction(max: d_maxFluidBoundaryNeibs) reduction(max: d_maxVertexNeibs) reduction(+: d_numInteractions)
#else
#pragma omp critical
	{
		d_maxFluidBoundaryNeibs = max(d_maxFluidBoundaryNeibs, neibs_max[0]);
		if (num_sm_neibs_max > 1) {
			d_maxVertexNeibs = max(d_maxVertexNeibs, neibs_max[1]);
		}
		d_numInteractions += total_neibs_num;
	}
#endif
#else // CUDA_BACKEND_ENABLED
	Mapping::sync(); // make sure we're not conflicting with the cache usage

#define LANE mapping.get_lane()

	sm_total_neibs_num[LANE] = total_neibs_num;
	sm_neibs_max[LANE] = neibs_max[0];
	if (num_sm_neibs_max > 1)
		sm_neibs_max[LANE + BLOCK_SIZE_BUILDNEIBS] = neibs_max[1];

	for (unsigned int i = Mapping::reduction_size/2; i > 0; i /= 2) {
		Mapping::sync();
		if (LANE < i) {
			total_neibs_num += sm_total_neibs_num[LANE + i];
			sm_total_neibs_num[LANE] = total_neibs_num;

#pragma unroll
			for (int o = 0; o < num_sm_neibs_max; ++o) {
				const uint n2 = sm_neibs_max[LANE + i + o*BLOCK_SIZE_BUILDNEIBS];
				if (n2 > neibs_max[o]) {
					sm_neibs_max[LANE + o*BLOCK_SIZE_BUILDNEIBS] = neibs_max[o] = n2;
				}
			}
		}
	}

	if (!LANE) {
		atomicMax(&d_maxFluidBoundaryNeibs, neibs_max[0]);
		if (num_sm_neibs_max > 1) {
			atomicMax(&d_maxVertexNeibs, neibs_max[1]);
		}
		atomicAdd(&d_numInteractions, total_neibs_num);
	}
#endif
};

template<bool neibcount, int num_sm_neibs_max, typename Mapping>
__device__ __forceinline__
enable_if_t<neibcount == false>
count_neighbors(const uint *neibs_num, Mapping const&)
{ /* no counting case */ };


/// Builds particles neighbors list
/*! This function builds the list of neighbors' indices of a given particles.
 *  The parameter params is built on specialized version of build_neibs_params
 * 	according to template values.
 *	The neighbor list is now organized by neighboring particle type :
 *	index	0					neibboundpos		neiblistsize-1
 *			|						  |				 |
 *			v						  v				 v
 *		   |PT_FLUID->...<-PT_BOUNDARY PT_VERTEX->...|
 *  First boundary particle is at index neibboundpos and first vertex particle is at
 *  neibboundpos + 1.
 *	This is made possible by the sort where particles are sorted by cell AND
 *	by particle type according to the ordering PT_FLUID < PT_BOUNDARY < PT_VERTEX.
 *
 *	\param[in, out] params: build neibs parameters
 *	\tparam boundarytype : boundary type (determines which particles have a neib list)
 *	\tparam periodicbound : type periodic boundaries (0 ... 7)
 *	\tparam neibcount : if true we compute maximum neighbor number
 */
template<Dimensionality dimensions, SPHFormulation sph_formulation, typename ViscSpec, BoundaryType boundarytype,
	Periodicity periodicbound, flag_t simflags, BuildNeibsMappingType bn_type,
	bool neibcount,
	bool debug_planes,
	/* nmber of dimensions */
	int dims = space_dimensions_for(dimensions),
	/* range of z to search for neibs */
	int zrange = (dims < 3 ? 0 : 1),
	/* range of y to search for neibs */
	int yrange = (dims < 2 ? 0 : 1),
	/* Number of shared arrays for the maximum number of neighbors:
	 * this is 1 (counting fluid + boundary) for all boundary types, except
	 * SA which also has another one for vertices */
	int num_sm_neibs_max = (1 + (boundarytype == SA_BOUNDARY)),
	// parameter structure we use
	typename params_t = buildneibs_params<boundarytype, simflags>
>
__device__ __forceinline__
void
buildNeibsListOfParticle(params_t const& params, const uint index, const uint numParticlesInBlock,
	BuildNeibsMapping<bn_type> const& mapping)
{
	// Number of neighbors for the current particle for each neighbor type.
	// One slot per supported particle type
	uint neibs_num[(boundarytype == SA_BOUNDARY ? PT_TESTPOINT : PT_VERTEX)] = {0};

	// Rather than nesting if's, use a do { } while (0) loop with breaks
	// for early bail outs
	// NOTE: the early bail-outs are only used when building neighbors by particle
	// neibsInCell, since in the by-cell processing all particles are needed,
	// to load neighbors data through shared memory
	do {
		const bool IS_VALID = (index < numParticlesInBlock);

		if (bn_type == BY_PARTICLE && !IS_VALID)
			break;

		// Read particle info from texture
		const particleinfo info = IS_VALID ? params.fetchInfo(index) : particleinfo{};

		// The way the neighbors list is constructed depends on
		// the boundary type used in the simulation.
		// 	* For Lennard-Jones boundaries :
		//		we construct a neighbors list for fluid, test points
		//		and particles belonging to a floating body or a moving
		//		boundary on which we want to compute forces.
		//	* For SA boundaries :
		//		same as Lennard-Jones plus vertice and boundary particles
		//	* For dynamic boundaries :
		//		we construct a neighbors list for all particles.
		//TODO: optimze test. (Alexis).
		bool build_nl = FLUID(info) || TESTPOINT(info) || FLOATING(info) || COMPUTE_FORCE(info);
		if (HAS_FEA(simflags))
			build_nl = build_nl || DEFORMABLE(info);
		if (boundarytype == SA_BOUNDARY)
			build_nl = build_nl || VERTEX(info) || BOUNDARY(info);
		if (boundarytype == DYN_BOUNDARY || boundarytype == DUMMY_BOUNDARY)
			build_nl = true;
		if ((boundarytype == LJ_BOUNDARY || boundarytype == MK_BOUNDARY) &&
		    ViscSpec::rheologytype == GRANULAR)
			build_nl = build_nl || BOUNDARY(info);

		// Exit if we have nothing to do
		if (bn_type == BY_PARTICLE && !build_nl)
			break;

		// Don't build the neighbors list unless valid
		build_nl = build_nl && IS_VALID;

		// Get particle position
		const pos_mass pdata = IS_VALID ? params.fetchPos(index) : float4{};

		// If the particle is inactive we have nothing to do
		if (is_inactive(pdata)) {
			if (bn_type == BY_PARTICLE)
				break;
			build_nl = false;
		}

		const int3 gridPos = mapping.calcGridPos(params, index);

		// neighbors list construction needs to know if the central particle is a boundary particle
		// (in SA, a boundary segment), or a FEA node, because these influence the construction of the list
		// check here onace and for all
		const bool central_is_boundary = IS_VALID ? BOUNDARY(info) : false;
		const bool central_is_fea = IS_VALID ? (HAS_FEA(simflags) && FEA_NODE(info)) : false;

		for(int z=-zrange; z<=zrange; z++) {
			for(int y=-yrange; y<=yrange; y++) {
				for(int x=-1; x<=1; x++) {
					neibsInCell<sph_formulation, ViscSpec, boundarytype, periodicbound>(params,
						gridPos,
						make_int3(x, y, z),
						(x + 1) + (y + 1)*3 + (z + 1)*9,
						index,
						pdata.pos,
						neibs_num,
						build_nl,
						mapping,
						central_is_boundary,
						central_is_fea);
				}
			}
		}

		if (build_nl)
			findNeighboringPlanes<debug_planes>(params, gridPos, pdata.pos, index);
	} while (0);

	// Each of the sections of the neighbor list is terminated by a NEIBS_END. This allow
	// iterating over all neighbors of a given type by a simple while (true) conditionally
	// breaking on neib_data being NEIBS_END. This is particularly important when no neighbors
	// of a given type were encountered, or when too many neighbors were found, to avoid overflowing.
	// In the latter case, we truncate the list at the last useful place, and record one of the particles
	// so that diagnostic information can be printed about it.
	if (index < numParticlesInBlock) {
		bool overflow = too_many_neibs<PT_FLUID>(neibs_num);

		/* If PT_FLUID neighbors overflowed, we put the marker at the last position, which
		 * means no PT_BOUNDARY neighbors will be registered
		 */
		int marker_pos = overflow ? d_neibboundpos : neibs_num[PT_FLUID];
		params.neibsList[ITH_NEIGHBOR_DEVICE(index, marker_pos)] = NEIBS_END;

		overflow |= too_many_neibs<PT_BOUNDARY>(neibs_num);
		/* A marker here is needed only if we didn't overflow, since otherwise the PT_FLUID marker will work
		 * for PT_BOUNDARY too */
		if (!overflow) {
			marker_pos = neibListOffset<PT_BOUNDARY>(neibs_num);
			params.neibsList[ITH_NEIGHBOR_DEVICE(index, marker_pos)] = NEIBS_END;
		}

		if (boundarytype == SA_BOUNDARY) {
			overflow |= too_many_neibs<PT_VERTEX>(neibs_num);
			marker_pos = overflow ? d_neiblistsize - 1 : d_neibboundpos + 1 + neibs_num[PT_VERTEX];
			params.neibsList[ITH_NEIGHBOR_DEVICE(index, marker_pos)] = NEIBS_END;
		}

		if (overflow) {
			const particleinfo info = params.fetchInfo(index);
			const int old = atomicCAS(&d_hasTooManyNeibs, -1, (int)id(info));
			if (old == -1) {
				d_hasMaxNeibs[PT_FLUID] = neibs_num[PT_FLUID];
				d_hasMaxNeibs[PT_BOUNDARY] = neibs_num[PT_BOUNDARY];
				if (boundarytype == SA_BOUNDARY)
					d_hasMaxNeibs[PT_VERTEX] = neibs_num[PT_VERTEX];
			}
		}
	}

	count_neighbors<neibcount, num_sm_neibs_max>(neibs_num, mapping);
}

/// Build neighbors list for the particles
/// \see buildNeibsListOfParticle for details.
template<Dimensionality dimensions, SPHFormulation sph_formulation, typename ViscSpec, BoundaryType boundarytype, Periodicity periodicbound,
	flag_t simflags,
	BuildNeibsMappingType bn_type,
	bool neibcount,
	bool debug_planes,
	// parameter structure we use
	typename params_t = buildneibs_params<boundarytype, simflags>
	>
struct buildNeibsListDevice : params_t
{
	static constexpr unsigned BLOCK_SIZE = BLOCK_SIZE_BUILDNEIBS;
	static constexpr unsigned MIN_BLOCKS = MIN_BLOCKS_BUILDNEIBS;

	buildNeibsListDevice(
		const	BufferList&	bufread,
				BufferList& bufwrite,
		const	uint		numParticles,
		const	uint		numCells,
		const	float		sqinfluenceradius,

		// SA_BOUNDARY
		const	float	boundNlSqInflRad)
	:
		params_t(bufread, bufwrite, numParticles, numCells, sqinfluenceradius, boundNlSqInflRad)
	{}

	template<typename Mapping>
	__device__
	enable_if_t<Mapping::type == BY_PARTICLE>
	do_mapping(params_t const& params, Mapping& mapping, simple_work_item const& item) const
	{
		buildNeibsListOfParticle<dimensions, sph_formulation, ViscSpec, boundarytype, periodicbound, simflags,
			bn_type, neibcount, debug_planes>(params, item.get_id(), params.numParticles, mapping);
	}

	template<typename Mapping>
	__device__
	enable_if_t<Mapping::type == BY_CELL>
	do_mapping(params_t const& params, Mapping& mapping, simple_work_item const& item) const
	{
#if CPU_BACKEND_ENABLED
		throw std::runtime_error("cell-based neighbor building not supported on CPU device");
#else
		// When building neighbors by cell, each work-group takes care of BUILDNEIBS_CELLS_PER_BLOCK cells,
		// with each WARP taking care of a cell
		// TODO we might want to find a way to detect empty cells and distribute the workload
		// in a more homogeneous manner
		// TODO cells typically have less than WARP_SIZE particles in fact, so we're going to waste some resources

		mapping.warp_index = threadIdx.x / WARP_SIZE; // index of the warp in the block
		mapping.lane = threadIdx.x & (WARP_SIZE - 1); // lane inside the warp
		mapping.cell_index = blockIdx.x * BUILDNEIBS_CELLS_PER_BLOCK + mapping.warp_index;

		if (mapping.cell_index >= params.numCells) return;

		// First thing first, find the range of particles that this block needs to take care of
		uint base_idx = params.fetchCellStart(mapping.cell_index);
		const uint end_idx = params.fetchCellEnd(mapping.cell_index);

		// if the cell is empty, we're done
		if (base_idx == CELL_EMPTY) return;

		// Each work-item in the work-group takes care of one particle:
		// note that we pass end_idx so that the particle neighbors list construction
		// will abort if we're past the end of the current cell
		while (base_idx < end_idx) {
			buildNeibsListOfParticle<dimensions, sph_formulation, ViscSpec, boundarytype, periodicbound, simflags,
				bn_type, neibcount, debug_planes>(params, base_idx + mapping.lane, end_idx, mapping);
			base_idx += WARP_SIZE;
		}
#endif
	}

	__device__ void operator()(simple_work_item item) const
{
	BuildNeibsMapping<bn_type> mapping;
	do_mapping(*this, mapping, item);
}
};

/// Check if any cells have more particles that can be enumerated in CELLNUM_SHIFT
struct checkCellSizeDevice : cell_params
{
	static constexpr unsigned BLOCK_SIZE = BLOCK_SIZE_BUILDNEIBS;
	static constexpr unsigned MIN_BLOCKS = MIN_BLOCKS_BUILDNEIBS;

	uint nCells;

	checkCellSizeDevice(BufferList const& bufread, uint nCells_) :
		cell_params(bufread),
		nCells(nCells_)
	{}

	__device__ void operator()(simple_work_item item) const
{
	const uint index = item.get_id();

	if (index >= nCells)
		return;

	const uint start = fetchCellStart(index);
	const uint end = fetchCellEnd(index);

	const uint delta = end - start;

	if (delta > NEIBINDEX_MASK) {
		int old = atomicCAS(&d_hasTooManyParticles, -1, (int)index);
		if (old == -1)
			d_hasHowManyParticles = delta;
	}
}
};

/** @} */
}
#endif
