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


/*
 * Device code.
 */

/* Important notes on block sizes:
	- a parallel reduction for max neibs number is done inside neiblist, block
	size for neiblist MUST BE A POWER OF 2
 */
#if (__COMPUTE__ == 75 || __COMPUTE__ == 86)
	#define BLOCK_SIZE_CALCHASH		256
	#define MIN_BLOCKS_CALCHASH		4
	#define BLOCK_SIZE_REORDERDATA	256
	#define MIN_BLOCKS_REORDERDATA	4
	#define BLOCK_SIZE_BUILDNEIBS	256
	#define MIN_BLOCKS_BUILDNEIBS	4
#elif (__COMPUTE__ >= 20)
	#define BLOCK_SIZE_CALCHASH		256
	#define MIN_BLOCKS_CALCHASH		6
	#define BLOCK_SIZE_REORDERDATA	256
	#define MIN_BLOCKS_REORDERDATA	6
	#define BLOCK_SIZE_BUILDNEIBS	256
	#define MIN_BLOCKS_BUILDNEIBS	5
#else
	#define BLOCK_SIZE_CALCHASH		256
	#define MIN_BLOCKS_CALCHASH		1
	#define BLOCK_SIZE_REORDERDATA	256
	#define MIN_BLOCKS_REORDERDATA	1
	#define BLOCK_SIZE_BUILDNEIBS	256
	#define MIN_BLOCKS_BUILDNEIBS	1
#endif



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
__device__ int d_hasTooManyNeibs;			///< id of a particle with too many neighbors
__device__ int d_hasMaxNeibs[PT_TESTPOINT];	///< Number of neighbors of that particle
__device__ int d_hasTooManyParticles; ///< Index of a cell with too many particles
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

	/// Constructor
	/*!	Computes structure members value according to the grid position.
	 */
	template<BoundaryType boundarytype, flag_t simflags>
	__device__ __forceinline__
	common_niC_vars(
		buildneibs_params<boundarytype, simflags> const& bparams,
		int3 const& gridPos		///< [in] position in the grid
					) :
		gridHash(calcGridHash(gridPos)),
		bucketStart(bparams.fetchCellStart(gridHash)),
		bucketEnd(bparams.fetchCellEnd(gridHash))
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


/// The actual neibInCell variables structure, which concatenates the above, as appropriate
/*! This structure provides a constructor that takes as arguments the union of the
 *	variables that would ever be needed by the neibsInCell device function.
 *  It then delegates the appropriate subset of variables to the appropriate
 *  structures it derives from, in the correct order
 */
template<BoundaryType boundarytype>
struct niC_vars :
	common_niC_vars,
	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_niC_vars)
{
	template<flag_t simflags>
	__device__ __forceinline__
	niC_vars(
		int3 const& gridPos,
		const uint index,
		buildneibs_params<boundarytype, simflags> const& bparams) :
		common_niC_vars(bparams, gridPos),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_niC_vars)(index, bparams)
	{}
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
template<BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<boundarytype != SA_BOUNDARY, bool>
isCloseEnough(float3 const& relPos, particleinfo const& neib_info,
	buildneibs_params<boundarytype, simflags> const& params)
{
	// Default : check against the influence radius
	return sqlength(relPos) < params.sqinfluenceradius;
}

/// Specialization of isCloseEnough for SA boundaries
/// \see isCloseEnough
template<BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<boundarytype == SA_BOUNDARY, bool>
isCloseEnough(float3 const& relPos, particleinfo const& neib_info,
	buildneibs_params<boundarytype, simflags> const& params)
{
	const float rp2(sqlength(relPos));
	// Include boundary neighbors which are a little further than sqinfluenceradius
	return (rp2 < params.sqinfluenceradius ||
		(rp2 < params.boundNlSqInflRad && BOUNDARY(neib_info)));
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
template<BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<boundarytype != SA_BOUNDARY>
process_niC_segment(const uint index, const uint neib_id, float3 const& relPos,
	buildneibs_params<boundarytype, simflags> const& params,
	niC_vars<boundarytype> const& var)
{ /* Do nothing by default */ }


/// Specialization of process_niC_segment for SA boundaries
/// \see process_niC_segment
template<BoundaryType boundarytype, flag_t simflags>
__device__ __forceinline__
enable_if_t<boundarytype == SA_BOUNDARY>
process_niC_segment(const uint index, const uint neib_id, float3 const& relPos,
	buildneibs_params<boundarytype, simflags> const& params,
	niC_vars<boundarytype> const& var)
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
__device__ __forceinline__
constexpr uint neibListOffset(uint neib_num, ParticleType neib_type)
{
	return	(neib_type == PT_FLUID) ? neib_num :
			(neib_type == PT_BOUNDARY) ? d_neibboundpos - neib_num :
			/* neib_type == PT_VERTEX */ neib_num + d_neibboundpos + 1;
}

/// Same as above, picking the neighbor index from the type-indexed neibs_num array
__device__ __forceinline__
uint neibListOffset(uint* neibs_num, ParticleType neib_type)
{
	return	neibListOffset(neibs_num[neib_type], neib_type);
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
__device__ __forceinline__
bool too_many_neibs(const uint* neibs_num, ParticleType neib_type)
{
	switch (neib_type) {
	case PT_FLUID:
		/* We give priority to the PT_FLUID particles, so we consider an overflow
		 * for them withount considering the number of boundary particles.
		 * Full checks will find the overflow anyway from the PT_BOUNDARY check.
		 */
		return !(neibs_num[PT_FLUID] < d_neibboundpos);
	case PT_BOUNDARY:
		return !(neibs_num[PT_FLUID] + neibs_num[PT_BOUNDARY] < d_neibboundpos);
	case PT_VERTEX:
		// TODO : possible optimization: there's no need to check for PT_VERTEX if not SA
		return !(neibs_num[PT_VERTEX] < d_neiblistsize - d_neibboundpos - 1);
	default:
		// should never happen
		return true;
	}
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
		 flag_t simflags>
__device__ __forceinline__ void
neibsInCell(
			buildneibs_params<boundarytype, simflags> const& params,			// build neibs parameters
			int3			gridPos,	// current particle grid position
			const int3		gridOffset,	// cell offset from current particle grid position
			const uchar		cell,		// cell number (0 ... 26)
			const uint		index,		// current particle index
			float3			pos,		// current particle position
			uint*			neibs_num,	// number of neighbors for the current particle
			const bool		segment,	// true if the current particle belongs to a segment
			const bool		boundary)	// true if the current particle is a boundary particle
{
	// Compute the grid position of the current cell, and return if it's
	// outside the domain
	if (!calcNeibCell<periodicbound>(gridPos, gridOffset))
		return;

	// Internal variables used by neibsInCell. Structure built on
	// specialized template of niC_vars.
	niC_vars<boundarytype> var(gridPos, index, params);

	// Return if the cell is empty
	if (var.bucketStart == CELL_EMPTY)
		return;

	// Substract gridOffset*cellsize to pos so we don't need to do it each time
	// we compute relPos respect to potential neighbor
	pos -= gridOffset*d_cellSize;

	// Iterate over all particles in the cell
	bool encode_cell = true;

	ParticleType neib_type = PT_FLUID;
	for (uint neib_index = var.bucketStart; neib_index < var.bucketEnd; neib_index++) {

		// Prevent self-interaction
		if (neib_index == index)
			continue;

		const particleinfo neib_info = params.fetchInfo(neib_index);

		// Testpoints have a neighbor list, but are not considered in the neighbor list
		// of other points
		if (TESTPOINT(neib_info))
			continue;

		// Force cell encode at each neib type change
		if (!encode_cell && neib_type != PART_TYPE(neib_info))
			encode_cell = true;
		neib_type = PART_TYPE(neib_info);

		// LJ boundary particles should not have any boundary neighbor, except when
		// rheologytype is GRANULAR.
		// If we are here is because a FLOATING LJ boundary needs neibs.
		if (boundarytype == LJ_BOUNDARY && boundary && BOUNDARY(neib_info) &&
		    ViscSpec::rheologytype != GRANULAR)
			continue;

		// With dynamic boundaries, boundary parts don't interact with other boundary parts
		// except for Grenier's formulation, where the sigma computation needs all neighbors
		// to be enumerated
		if (boundarytype == DYN_BOUNDARY && sph_formulation != SPH_GRENIER) {
			if (boundary && BOUNDARY(neib_info))
				continue;
		}

		// Compute relative position between particle and potential neighbor
		const pos_mass neib = params.fetchPos(neib_index);

		// Skip inactive particles
		if (is_inactive(neib))
			continue;

		const float3 relPos = pos - neib.pos;

		// Check if the squared distance is smaller than the squared influence radius
		// used for neighbor list construction
		bool close_enough = isCloseEnough(relPos, neib_info, params);

		if (close_enough) {
			/* The previous number of neighbors is the index of the current neighbor,
			 * use it to get the offset where it will be placed in the list */
			const uint offset = neibListOffset(neibs_num, neib_type);
			neibs_num[neib_type]++;

			/* Store the neighbor, if there's room. End-of-list markers and overflow
			 * counts will be managed after the list has been built */
			if (!too_many_neibs(neibs_num, neib_type)) {
				int neib_bucket_offset = neib_index - var.bucketStart;
				int encode_offset = encode_cell ? ENCODE_CELL(cell) : 0;
				params.neibsList[offset*d_neiblist_stride + index] =
					neib_bucket_offset + encode_offset;
				encode_cell = false;
			}
		}
		if (segment) {
			process_niC_segment(index, id(neib_info), relPos, params, var);
		}

	}

	return;
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
		float4*				posArray_,
		hashKey*			particleHash_,
		uint*				particleIndex_,
		const particleinfo*	particelInfo_,
		const uint*			compactDeviceMap_,
		const uint			numParticles_)
	:
		posArray(posArray_),
		particleHash(particleHash_),
		particleIndex(particleIndex_),
		particleInfo(particelInfo_),
		compactDeviceMap(compactDeviceMap_),
		numParticles(numParticles_)
	{}

	__device__ void operator()(simple_work_item item) const;
};

template <Periodicity periodicbound>
__device__
void calcHashDevice<periodicbound>::operator()(simple_work_item item) const
{
	const uint index = item.get_id();

	if (index >= numParticles)
		return;

	const particleinfo info = particleInfo[index];

	// Get the old grid hash
	uint gridHash = cellHashFromParticleHash( particleHash[index] );

	// We compute new hash only for fluid and moving not fluid particles (object, moving boundaries),
	// and surface boundaries in case of repacking
	if (FLUID(info) || MOVING(info) || (SURFACE(info) && !FLUID(info))) {
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
	const particleinfo* particelInfo;		///< [in] particle's informations
	const uint*			compactDeviceMap;	///< [in] type of the cells belonging to the device
	const uint			numParticles;		///< [in] total number of particles

	fixHashDevice(
		hashKey*		particleHash_,
		uint*				particleIndex_,
		const particleinfo* particelInfo_,
		const uint*			compactDeviceMap_,
		const uint			numParticles_)
	:
		particleHash(particleHash_),
		particleIndex(particleIndex_),
		particelInfo(particelInfo_),
		compactDeviceMap(compactDeviceMap_),
		numParticles(numParticles_)
	{}

	__device__ void operator()(simple_work_item item) const;
};

__device__
void fixHashDevice::operator()(simple_work_item item) const
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
			RP								rparams,
			uint* __restrict__				cellStart_,
			uint* __restrict__				cellEnd_,
			uint* __restrict__				segmentStart_,
	const	particleinfo * __restrict__		particleInfo_,
	const	hashKey* __restrict__			particleHash_,
	const	uint* __restrict__				particleIndex_,
	const	uint							numParticles_,
			uint* __restrict__				newNumParticles_)
	:
		RP(rparams),
		cellStart(cellStart_),
		cellEnd(cellEnd_),
		segmentStart(segmentStart_),
		particleInfo(particleInfo_),
		particleHash(particleHash_),
		particleIndex(particleIndex_),
		numParticles(numParticles_),
		newNumParticles(newNumParticles_)
	{}

	__device__ void operator()(simple_work_item item) const;
};

template<typename RP>
__device__ void reorderDataAndFindCellStartDevice<RP>::operator()(simple_work_item item) const
{
	// Shared hash array of dimension blockSize + 1
	extern __shared__ uint sharedHash[];

	const uint index = item.get_id();

	// Initialize segmentStarts
	if (segmentStart && index < 4) segmentStart[index] = EMPTY_SEGMENT;

	uint cellHash;
	// Handle the case when number of particles is not multiple of block size
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

	if (index < numParticles) {
		// If this particle has a different cell index to the previous
		// particle then it must be the first particle in the cell
		// or the first inactive particle.
		// Store the index of this particle as the new cell start and as
		// the previous cell end

		// Note: we need to reset the high bits of the cell hash if the particle hash is 64 bits wide
		// every time we use a cell hash to access an element of CellStart or CellEnd

		if (index == 0 || cellHash != sharedHash[threadIdx.x]) {

			// New cell, otherwise, it's the number of active particles (short hash: compare with 32 bits max)
			if (cellHash != CELL_HASH_MAX)
				// If it isn't an inactive particle, it is also the start of the cell
				cellStart[cellHash & CELLTYPE_BITMASK] = index;
			else
				*newNumParticles = index;

			// If it isn't the first particle, it must also be the end of the previous cell
			if (index > 0)
				cellEnd[sharedHash[threadIdx.x] & CELLTYPE_BITMASK] = index;
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
			uchar prev_type = sharedHash[threadIdx.x] >> 30;
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

/// Find the planes within the influence radius of the particle
/*! If neither ENABLE_PLANES nor ENABLE_DEM are active, do nothing
 */
template<BoundaryType boundarytype, flag_t simflags>
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
template<BoundaryType boundarytype, flag_t simflags>
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
template<BoundaryType boundarytype, flag_t simflags>
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
	if (r < 0)
		printf("Particle %d id %d behind DEM!\n", index, (int)id(params.fetchInfo(index)));
	return (r*r < params.sqinfluenceradius);
}

/// Find the planes within the influence radius of the particle
/*! If ENABLE_PLANES or ENABLE_DEM are active, go over each defined plane
 * and see if the distance is within the required radius
 */
template<BoundaryType boundarytype, flag_t simflags>
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
			if (r < 0)
				printf("Particle %d behind plane %d!\n", index, p);
			if (r*r < params.sqinfluenceradius) {
				store[neib_planes++] = p;
				if (neib_planes > 3) break;
			}
		}
	}

	if ( HAS_DEM(simflags) && neib_planes < 4 ) {
		if (isDemInRange(params, gridPos, pos, index)) {
			store[neib_planes++] = MAX_PLANES;
		}
	}

	while (neib_planes < 4) {
		store[neib_planes++] = -1;
	}
}



/// Builds particles neighbors list
/*! This kernel builds the neighbor's indexes of all particles. The
 * 	parameter params is built on specialized version of build_neibs_params
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
 *
 *	First and last particle index for grid cells and particle's informations
 *	are read through texture fetches.
 *
 *	TODO: finish implementation for SA_BOUNDARY (include PT_VERTEX)
 */
template<SPHFormulation sph_formulation, typename ViscSpec, BoundaryType boundarytype, Periodicity periodicbound,
	flag_t simflags,
	bool neibcount,
	/* Number of shared arrays for the maximum number of neighbors:
	 * this is 1 (counting fluid + boundary) for all boundary types, except
	 * SA which also has another one for vertices */
	int num_sm_neibs_max = (1 + (boundarytype == SA_BOUNDARY))>
__global__ void
/*! \cond */
__launch_bounds__( BLOCK_SIZE_BUILDNEIBS, MIN_BLOCKS_BUILDNEIBS)
/*! \endcond */
buildNeibsListDevice(buildneibs_params<boundarytype, simflags> params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	// Number of neighbors for the current particle for each neighbor type
	uint neibs_num[PT_TESTPOINT] = {0};

	// Rather than nesting if's, use a do { } while (0) loop with breaks
	// for early bail outs
	do {
		if (index >= params.numParticles)
			break;

		// Read particle info from texture
		const particleinfo info = params.fetchInfo(index);

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
		if (boundarytype == SA_BOUNDARY)
			build_nl = build_nl || VERTEX(info) || BOUNDARY(info);
		if (boundarytype == DYN_BOUNDARY || boundarytype == DUMMY_BOUNDARY)
			build_nl = true;
		if ((boundarytype == LJ_BOUNDARY || boundarytype == MK_BOUNDARY) &&
		    ViscSpec::rheologytype == GRANULAR)
			build_nl = build_nl || BOUNDARY(info);

		// Exit if we have nothing to do
		if (!build_nl)
			break;

		// Get particle position
		const pos_mass pdata = params.fetchPos(index);

		// If the particle is inactive we have nothing to do
		if (is_inactive(pdata))
			break;

		// Get particle grid position computed from particle hash
		const int3 gridPos = calcGridPosFromParticleHash(params.particleHash[index]);

		for(int z=-1; z<=1; z++) {
			for(int y=-1; y<=1; y++) {
				for(int x=-1; x<=1; x++) {
					neibsInCell<sph_formulation, ViscSpec, boundarytype, periodicbound>(params,
						gridPos,
						make_int3(x, y, z),
						(x + 1) + (y + 1)*3 + (z + 1)*9,
						index,
						pdata.pos,
						neibs_num,
						BOUNDARY(info),
						BOUNDARY(info));
				}
			}
		}

		findNeighboringPlanes(params, gridPos, pdata.pos, index);
	} while (0);

	// Each of the sections of the neighbor list is terminated by a NEIBS_END. This allow
	// iterating over all neighbors of a given type by a simple while (true) conditionally
	// breaking on neib_data being NEIBS_END. This is particularly important when no neighbors
	// of a given type were encountered, or when too many neighbors were found, to avoid overflowing.
	// In the latter case, we truncate the list at the last useful place, and record one of the particles
	// so that diagnostic information can be printed about it.
	if (index < params.numParticles) {
		bool overflow = too_many_neibs(neibs_num, PT_FLUID);

		/* If PT_FLUID neighbors overflowed, we put the marker at the last position, which
		 * means no PT_BOUNDARY neighbors will be registered
		 */
		int marker_pos = overflow ? d_neibboundpos : neibs_num[PT_FLUID];
		params.neibsList[marker_pos*d_neiblist_stride + index] = NEIBS_END;

		overflow |= too_many_neibs(neibs_num, PT_BOUNDARY);
		/* A marker here is needed only if we didn't overflow, since otherwise the PT_FLUID marker will work
		 * for PT_BOUNDARY too */
		if (!overflow)
			params.neibsList[neibListOffset(neibs_num, PT_BOUNDARY)*d_neiblist_stride + index] = NEIBS_END;

		if (boundarytype == SA_BOUNDARY) {
			overflow |= too_many_neibs(neibs_num, PT_VERTEX);
			marker_pos = overflow ? d_neiblistsize - 1 : d_neibboundpos + 1 + neibs_num[PT_VERTEX];
			params.neibsList[marker_pos*d_neiblist_stride + index] = NEIBS_END;
		}

		if (overflow) {
			const particleinfo info = params.fetchInfo(index);
			const int old = atomicCAS(&d_hasTooManyNeibs, -1, (int)id(info));
			if (old == -1) {
				d_hasMaxNeibs[PT_FLUID] = neibs_num[PT_FLUID];
				d_hasMaxNeibs[PT_BOUNDARY] = neibs_num[PT_BOUNDARY];
				d_hasMaxNeibs[PT_VERTEX] = neibs_num[PT_VERTEX];
			}
		}
	}

	if (neibcount) {
		// Shared memory reduction of per block maximum number of neighbors
		// We count both the total number of neighbors (and hence the overall number of _interactions_)
		// and the max number of neighbors of each type
		__shared__ volatile uint sm_total_neibs_num[BLOCK_SIZE_BUILDNEIBS];
		__shared__ volatile uint sm_neibs_max[BLOCK_SIZE_BUILDNEIBS*num_sm_neibs_max];

		uint neibs_max[num_sm_neibs_max];
		neibs_max[0] = neibs_num[PT_FLUID] + neibs_num[PT_BOUNDARY];
		if (boundarytype == SA_BOUNDARY)
			neibs_max[1] = neibs_num[PT_VERTEX];
		uint total_neibs_num = neibs_max[0] + neibs_num[PT_VERTEX];

		sm_total_neibs_num[threadIdx.x] = total_neibs_num;

		sm_neibs_max[threadIdx.x] = neibs_max[0];
		if (boundarytype == SA_BOUNDARY)
			sm_neibs_max[threadIdx.x + blockDim.x] = neibs_max[1];

		uint i = blockDim.x/2;
		while (i != 0) {
			__syncthreads();
			if (threadIdx.x < i) {
				total_neibs_num += sm_total_neibs_num[threadIdx.x + i];
				sm_total_neibs_num[threadIdx.x] = total_neibs_num;

#pragma unroll
				for (int o = 0; o < num_sm_neibs_max; ++o) {
					const uint n2 = sm_neibs_max[threadIdx.x + i + o*blockDim.x];
					if (n2 > neibs_max[o]) {
						sm_neibs_max[threadIdx.x + o*blockDim.x] = neibs_max[o] = n2;
					}
				}
			}
			i /= 2;
		}

		if (!threadIdx.x) {
			atomicMax(&d_maxFluidBoundaryNeibs, neibs_max[0]);
			if (boundarytype == SA_BOUNDARY)
				atomicMax(&d_maxVertexNeibs, neibs_max[1]);
			atomicAdd(&d_numInteractions, total_neibs_num);
		}
	}
	return;
}

/// Check if any cells have more particles that can be enumerated in CELLNUM_SHIFT
__global__ void
checkCellSizeDevice(cell_params params, uint nCells)
{
	const uint index = INTMUL(blockIdx.x, blockDim.x) + threadIdx.x;

	if (index >= nCells)
		return;

	const uint start = params.fetchCellStart(index);
	const uint end = params.fetchCellEnd(index);

	const uint delta = end - start;

	if (delta > NEIBINDEX_MASK) {
		int old = atomicCAS(&d_hasTooManyParticles, -1, index);
		if (old == -1)
			d_hasHowManyParticles = delta;
	}


}
/** @} */
}
#endif
