/*  Copyright 2015 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/* Device functions and constants pertaining open boundaries */

#ifndef _BOUNDS_KERNEL_
#define _BOUNDS_KERNEL_

#include "particledefine.h"

/*!
 * \namespace cubounds
 * \brief Contains all device functions/kernels/constants related to open boundaries and domain geometry.
 *
 * The namespace contains the device side of boundary handling
 *	- domain size, origin and cell grid properties and related functions
 *	- open boundaries properties and related functions
 */
namespace cubounds {

// Grid data
#include "cellgrid.cuh"
#include "planes.h"

/// \name Device constants
/// @{

texture<float, 2, cudaReadModeElementType> demTex;	// DEM

/* DEM constants */
// TODO switch to float2s
__constant__ float	d_ewres;		///< east-west resolution (x)
__constant__ float	d_nsres;		///< north-south resolution (y)
__constant__ float	d_demdx;		///< ∆x increment of particle position for normal computation
__constant__ float	d_demdy;		///< ∆y increment of particle position for normal computation
__constant__ float	d_demdxdy;		///< ∆x*∆y
__constant__ float	d_demzmin;		///< minimum distance from DEM for normal computation

/* Constants for geometrical planar boundaries */
__constant__ uint	d_numplanes;
__constant__ plane_t d_plane[MAX_PLANES];

/// Number of open boundaries (both inlets and outlets)
__constant__ uint d_numOpenBoundaries;

// host-computed id offset used for id generation
__constant__ uint	d_newIDsOffset;

/// @}

/** \name Device functions
 *  @{ */

//! Given a point in grid + pos coordinates, and a plane defined by
//! a normal and a point (in grid + pos coordinates) on the plane,
//! returns the (signed) distance of the point to the plane.
//! NOTE: 2*signedDistance*plane.normal gives the distance vector
//! to the reflection of the point across the plane
__device__ __forceinline__ float
signedPlaneDistance(
	const int3&		gridPos,
	const float3&	pos,
	const plane_t&	plane)
{
	// Relative position of the point to the reference point of the plane
	const float3 relPos = globalDistance(gridPos, pos,
		plane.gridPos, plane.pos);

	return dot(relPos, plane.normal);
}

//! \see signedPlaneDistance, but returns the (unsigned) distance
__device__ __forceinline__ float
PlaneDistance(	const int3&		gridPos,
				const float3&	pos,
				const plane_t&	plane)
{
	return abs(signedPlaneDistance(gridPos, pos, plane));
}

//! reflect a vector through a plane.
__device__ __forceinline__ float3
reflectVector(const float3& vec, const plane_t& plane)
{
	return vec - 2*dot(vec, plane.normal)*plane.normal;
}

//! reflect the spatial components of a vector through a plane.
__device__ __forceinline__ float4
reflectVector(const float4& vec, const plane_t& plane)
{
	return make_float4(
		reflectVector(as_float3(vec), plane),
		vec.w);
}

/**! Convert an xy grid + local position into a DEM cell position
 * This is done assuming that the worldOrigin is at DEM coordinates (0, 0).
 * NOTE: the function accepts anything as grid and local pos,
 * but GridPosType should be an int2 or int3 and LocalPosType should be
 * a float2 or float3.
 * TODO use type traits to enforce this.
 */
template<typename GridPosType, typename LocalPosType>
__device__ __forceinline__ float2
DemPos(GridPosType const& gridPos, LocalPosType const& pos)
{
	// note that we separate the grid conversion part from the pos conversion part,
	// for improved accuracy. The final 0.5f is because texture values are assumed to be
	// at the center of the DEM cell.
	return make_float2(
		(gridPos.x + 0.5f)*(d_cellSize.x/d_ewres) + pos.x/d_ewres + 0.5f,
		(gridPos.y + 0.5f)*(d_cellSize.y/d_nsres) + pos.y/d_nsres + 0.5f);
}

/**! Interpolate DEM texref for a point at DEM cell pos demPos,
  plus an optional multiple of (∆x, ∆y).
  NOTE: the returned z coordinate is GLOBAL, not LOCAL!
  TODO for improved homogeneous accuracy, maybe have a texture for grid cells and a
  texture for local z coordinates?
 */
__device__ __forceinline__ float
DemInterpol(const texture<float, 2, cudaReadModeElementType> texref,
	const float2& demPos, int dx=0, int dy=0)
{
	return tex2D(texref, demPos.x + dx*d_demdx/d_ewres, demPos.y + dy*d_demdy/d_nsres);
}

//! Find the plane tangent to a DEM near a given position, assuming demPos and Z0
//! were already computed
__device__ __forceinline__ plane_t
DemTangentPlane(const texture<float, 2, cudaReadModeElementType> texref,
	const int3&	gridPos,
	const float3&	pos,
	const float2& demPos, const float globalZ0)
{
	// TODO this method to generate the interpolating plane is suboptimal, as it
	// breaks any possible symmetry in the original DEM. A better (but more expensive)
	// approach would be to sample four points, one on each side of our point (in both
	// directions)
	const float globalZ1 = DemInterpol(texref, demPos, 1, 0);
	const float globalZ2 = DemInterpol(texref, demPos, 0, 1);

	// TODO find a more accurate way to compute the normal
	const float a = d_demdy*(globalZ0 - globalZ1);
	const float b = d_demdx*(globalZ0 - globalZ2);
	const float c = d_demdxdy;
	const float l = sqrt(a*a+b*b+c*c);

	// our plane point is the one at globalZ0: this has the same (x, y) grid and local
	// position as our particle, and the z grid and local position to be computed
	// from globalZ0
	const int3 demPointGridPos = make_int3(gridPos.x, gridPos.y,
		(int)floor((globalZ0 - d_worldOrigin.z)/d_cellSize.z));
	const float3 demPointLocalPos = make_float3(pos.x, pos.y,
		globalZ0 - d_worldOrigin.z - (demPointGridPos.z + 0.5f)*d_cellSize.z);
	return make_plane(
			make_float3(a, b, c)/l, demPointGridPos, demPointLocalPos);
}

//! Find the plane tangent to a DEM near a given particle at position grid+pos
__device__ __forceinline__ plane_t
DemTangentPlane(const texture<float, 2, cudaReadModeElementType> texref,
	const int3&	gridPos,
	const float3&	pos)
{
	const float2 demPos = DemPos(gridPos, pos);
	const float globalZ0 = DemInterpol(texref, demPos);
	return DemTangentPlane(texref, gridPos, pos, demPos, globalZ0);
}

/*!
 * Create a new particle, cloning an existing particle
 * This returns the index of the generated particle, initializing new_info
 * for a FLUID particle of the same fluid as the generator, no associated
 * object or inlet, and a new id generated in a way which is multi-GPU
 * compatible.
 *
 * All other particle properties (position, velocity, etc) should be
 * set by the caller.
 */
__device__ __forceinline__
uint
createNewFluidParticle(
	/// [out] particle info of the generated particle
			particleinfo	&new_info,
	/// [in] particle info of the generator particle
	const	particleinfo	&info,
	/// [in] number of particles at the start of the current timestep
	const	uint			numParticles,
	/// [in] number of devices
	const	uint			numDevices,
	/// [in,out] number of particles including all the ones already created in this timestep
			uint			*newNumParticles)
{
	const uint new_index = atomicAdd(newNumParticles, 1);
	// number of new particles that were created on this device in this
	// time step
	const uint newNumPartsOnDevice = new_index + 1 - numParticles;
	// the i-th device can only allocate an id that satisfies id%n == i, where
	// n = number of total devices
	const uint new_id = newNumPartsOnDevice*numDevices + d_newIDsOffset;

	new_info = make_particleinfo_by_ids(
		PT_FLUID,
		fluid_num(info), 0, // copy the fluid number, not the object number
		new_id);
	return new_index;
}

/** @} */

} // namespace cubounds

#endif
