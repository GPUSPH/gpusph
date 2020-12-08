/*  Copyright (c) 2015-2019 INGV, EDF, UniCT, JHU

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
 * Device constants and functions for planes and DEM management
 */

#ifndef GEOM_CORE_CU
#define GEOM_CORE_CU

#include "fastdem_select.opt"

#include "planes.h"

#include "neibs_iteration.cuh" // for d_cellSize

thread_local cudaArray*  dDem = NULL;
thread_local cudaTextureObject_t demTex = 0;

namespace cugeom {

using namespace cuneibs;

/// \name Device constants
/// @{

/// DEM constants
/// @{
/// TODO switch to float2s

//! Correction to be applied to the particle position to obtain the DEM-relative position
/*! Our DEM stores vertex-based information,
 * with the first point, at DEM-relative coordinate 0, corresponding to the westernmost/southernmost edge,
 * and the last point, at DEM-relative coordinate (nrows-1)*ewres or (ncols-1)*nsres,
 * corresponding to the easternmost/northernmost edge.
 *
 * The DEM has a given origin that maps its coordinate 0 to the global reference system.
 *
 * To map a particle global position gPos = world_origin + (gridPos+0.5)*gridSize + lPos
 * to DEM coordinates, we need to subtract the DEM_origin, so that
 * dPos = (world_origin - DEM_origin) + (gridPos + 0.5)*gridSize + lPos.
 *
 * Additionally, for the actual texture fetch, the two componets need to be scaled by
 * ewres/nsres, so d_dem_pos_fixup will store (world_origin - DEM_origin)/res.
 */
__constant__ float2 d_dem_pos_fixup;

__constant__ float2 d_dem_scaled_cellSize; ///< d_cellSize scaled by the ewres/nsres
__constant__ float2 d_dem_scaled_dx; ///< d_demdx/d_demdy scaled by the ewres/nsres
__constant__ float	d_ewres;		///< east-west resolution (x)
__constant__ float	d_nsres;		///< north-south resolution (y)
__constant__ float	d_demdx;		///< ∆x increment of particle position for normal computation
__constant__ float	d_demdy;		///< ∆y increment of particle position for normal computation
__constant__ float	d_demdxdy;		///< ∆x*∆y
__constant__ float	d_demzmin;		///< minimum distance from DEM for normal computation
/// @}

/* Constants for geometrical planar boundaries */
__constant__ uint	d_numplanes;
__constant__ plane_t d_plane[MAX_PLANES];

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
	return d_dem_pos_fixup + make_float2(
		(gridPos.x + 0.5f)*d_dem_scaled_cellSize.x + pos.x/d_ewres + 0.5f,
		(gridPos.y + 0.5f)*d_dem_scaled_cellSize.y + pos.y/d_nsres + 0.5f);
}

/**! Interpolate DEM demTex for a point at DEM cell pos demPos,
  plus an optional multiple of (∆x, ∆y).
  NOTE: the returned z coordinate is GLOBAL, not LOCAL!
  TODO for improved homogeneous accuracy, maybe have a texture for grid cells and a
  texture for local z coordinates?
 */
__device__ __forceinline__ float
DemInterpol(cudaTextureObject_t demTex,
	const float2& demPos, int dx=0, int dy=0)
{
	return tex2D<float>(demTex, demPos.x + dx*d_dem_scaled_dx.x, demPos.y + dy*d_dem_scaled_dx.y);
}

//! Find the plane tangent to a DEM near a given position, assuming demPos and Z0
//! were already computed
__device__ __forceinline__ plane_t
DemTangentPlane(cudaTextureObject_t demTex,
	const int3&	gridPos,
	const float3&	pos,
	const float2& demPos, const float globalZ0)
{
	// TODO find way to compute the normal without passing through the global pos
	const float globalZpx = DemInterpol(demTex, demPos,  1,  0);
	const float globalZpy = DemInterpol(demTex, demPos,  0,  1);
#if FASTDEM
	// 'Classic', 'fast' computation: find the plane through three points,
	// where thre three points are z0 (projection) and two other points
	// obtained by sampling the DEM at distance dx along both the x and y axes.
	// Note that this disregards any DEM symmetry information.

	const float a = d_demdy*(globalZ0 - globalZpx);
	const float b = d_demdx*(globalZ0 - globalZpy);
	const float c = d_demdxdy;
#else
	const float globalZmx = DemInterpol(demTex, demPos, -1,  0);
	const float globalZmy = DemInterpol(demTex, demPos,  0, -1);

	// Compared to the host version, we only simplify dividing by 2, since
	// here we use the actual DEM dx and y, which may be (slightly) different
	const float a = d_demdy*(globalZmx - globalZpx);
	const float b = d_demdx*(globalZmy - globalZpy);
	const float c = 2*d_demdxdy;
#endif
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
DemTangentPlane(cudaTextureObject_t demTex,
	const int3&	gridPos,
	const float3&	pos)
{
	const float2 demPos = DemPos(gridPos, pos);
	const float globalZ0 = DemInterpol(demTex, demPos);
	return DemTangentPlane(demTex, gridPos, pos, demPos, globalZ0);
}

/** @} */
}

#endif
