/*  Copyright (c) 2013-2019 INGV, EDF, UniCT, JHU

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

/* Auxiliary functions to compute a unsigned integer hash
 * from a cell grid position, and conversely */

/* IMPORTANT NOTE: this header should be included _inside_ the namespace,
 * of each _kernel.cu file, because of the __constant__s defined below
 */

#include "hashkey.h"
#include "linearization.h"

#ifndef CELLGRID_CUH
#define CELLGRID_CUH

namespace cuneibs
{

/* The neighbor cell num ranges from 1 to 27 (included), so it fits in
 * 5 bits, which we put in the upper 5 bits of the neibdata, which is
 * 16-bit wide.
 * TODO actually compute this from sizeof(neibdata)
 */
#define CELLNUM_SHIFT	11
#define CELLNUM_ENCODED	(1U<<CELLNUM_SHIFT)
#define NEIBINDEX_MASK	(CELLNUM_ENCODED-1)
#define ENCODE_CELL(cell) ((cell + 1) << CELLNUM_SHIFT)
#define DECODE_CELL(data) ((data >> CELLNUM_SHIFT) - 1)


/** \addtogroup cellgrid Common cell/grid related device functions and variables/constants
 * 	\ingroup neibs
 * 	Device constant/variables and functions involved in computing particles hash, relative positions
 * 	and used in several namespaces (actually \ref cuneibs and \ref cuforces)
 *  @{ */

/** \addtogroup cellgrid_device_constants Common position/neighbor related device constants
 * 	\ingroup cellgrid
 * 	Device constant involved in computing particles hash, relative positions
 * 	and used in several namespaces (actually \ref cuneibs and \ref cuforces)
 *  @{ */
__constant__ float3	d_worldOrigin;			///< Origin of the simulation domain
__constant__ float3	d_cellSize;				///< Size of cells used for the neighbor search
__constant__ uint3	d_gridSize;				///< Size of the simulation domain expressed in terms of cell number
__constant__ char3	d_cell_to_offset[27];	///< Neighbor cell index to 3D offset (in cells) map
/** @} */

/** \addtogroup cellgrid_devices_functions Common position/neighbor related device functions
 * 	\ingroup cellgrid
 *  Contains all the device function needed for computing particles hash, relative positions, ...
 * 	and used in several namespaces (actually cuneibs and cuforces)
 *  @{ */
/// Compute hash value from grid position
/*! Compute the hash value from grid position according to the chosen
 * 	linearization (starting from x, y or z direction). The link
 * 	between COORD1,2,3 and .x, .y and .z is defined in linearization.h
 *
 * \return hash value
 */
__device__ __forceinline__ uint
calcGridHash(	int3 const& gridPos	///< [in] grid position
				)
{
	return INTMUL(INTMUL(gridPos.COORD3, d_gridSize.COORD2), d_gridSize.COORD1)
			+ INTMUL(gridPos.COORD2, d_gridSize.COORD1) + gridPos.COORD1;
}


/// Compute grid position from cell hash value
/*! Compute the grid position corresponding to the given cell hash. The position
 *  should be in the range [0, d_gridSize.x - 1]x[0, d_gridSize.y - 1]x[0, d_gridSize.z - 1].
 *
 * \return grid position
 *
 * \note no test is done by this function to ensure that hash value is valid.
 */
__device__ __forceinline__ int3
calcGridPosFromCellHash(	const uint cellHash	///< [in] cell hash value
							)
{
	int3 gridPos;
	int temp = INTMUL(d_gridSize.COORD2, d_gridSize.COORD1);
	gridPos.COORD3 = cellHash / temp;
	temp = cellHash - gridPos.COORD3 * temp;
	gridPos.COORD2 = temp / d_gridSize.COORD1;
	gridPos.COORD1 = temp - gridPos.COORD2 * d_gridSize.COORD1;

	return gridPos;
}

/// Compute grid position from particle hash value
/*! Compute the grid position corresponding to the given particle hash. The position
 *  should be in the range [0, d_gridSize.x - 1]x[0, d_gridSize.y - 1]x[0, d_gridSize.z - 1].
 *
 * \return grid position
 *
 * \note
 * 	- no test is done by this function to ensure that hash value is valid.
 * 	- when hashKey is 32bit long, this is equivalent to calcGridPosFromCellHash()
 */
__device__ __forceinline__ int3
calcGridPosFromParticleHash(	const hashKey particleHash	///< [in] particle hash value
								)
{
	// Read the cellHash out of the particleHash
	const uint cellHash = cellHashFromParticleHash(particleHash);
	return calcGridPosFromCellHash(cellHash);
}

/// Compute relative distance vector between points
/*! Compute the relative distance between two points
 *
 * \return relative distance
 */
template<typename PosT1, typename PosT2> // both should be either float3 or float4
__device__ __forceinline__ float3
globalDistance(	int3 const& gridPos1,	///< [in] grid cell of point 1
				PosT1 const& pos1,		///< [in] cell relative position of point 1
				int3 const& gridPos2,	///< [in] grid cell of point 2
				PosT2 const& pos2		///< [in] cell relative position of point 2
				)
{
	return (gridPos1 - gridPos2)*d_cellSize + make_float3(
		pos1.x - pos2.x, pos1.y - pos2.y, pos1.z - pos2.z);
}

/// Compute hash value from grid position
/*! Compute the hash value corresponding to the given position. If the position
 *  is not in the range [0, gridSize.x - 1]x[0, gridSize.y - 1]x[0, gridSize.z - 1]
 *  we have periodic boundary and the grid position is updated according to the
 *  chosen periodicity.
 *
 *  \return hash value
 *
 *	\note no test is done by this function to ensure that grid position is within the
 *	range and no clamping is done
 */
__device__ __forceinline__ uint
calcGridHashPeriodic(
						int3 gridPos	///< grid position
						)
{
	if (gridPos.x < 0) gridPos.x = d_gridSize.x - 1;
	if (gridPos.x >= d_gridSize.x) gridPos.x = 0;
	if (gridPos.y < 0) gridPos.y = d_gridSize.y - 1;
	if (gridPos.y >= d_gridSize.y) gridPos.y = 0;
	if (gridPos.z < 0) gridPos.z = d_gridSize.z - 1;
	if (gridPos.z >= d_gridSize.z) gridPos.z = 0;
	return calcGridHash(gridPos);
}


/// Return neighbor index and add cell offset vector to current position
/*! For given neighbor data this function compute the neighbor index
 *  and subtract, if necessary, the neighbor cell offset vector to the
 *  current particle position. This last operation is done only
 *  when the neighbor cell change and result is stored in pos_corr.
 *
 * \return neighbor index
 *
 * \note neib_cell_num and neib_cell_base_index must be persistent along
 * getNeibIndex calls.
 */
__device__ __forceinline__ uint
getNeibIndex(	float3 const&	pos,					///< [in] current particle cell relative position
				float3&			pos_corr,				///< [out] offset between particle cell and current neighbor cell
				const uint*		cellStart,				///< [in] cells first particle index
				neibdata		neib_data,				///< [in] neighbor data
				int3 const&		gridPos,				///< [in] current particle cell position
				uchar&			neib_cellnum,			///< [in,out] current neighbor cell index (0...26)
				uint&			neib_cell_base_index	///< [in,out] neib_cell_base_index : index of first particle of the current cell
				)
{
	if (neib_data >= CELLNUM_ENCODED) {
		// Update current neib cell number
		neib_cellnum = DECODE_CELL(neib_data);

		// Compute neighbor index relative to belonging cell
		neib_data &= NEIBINDEX_MASK;

		// Substract current cell offset vector to pos
		pos_corr = pos - d_cell_to_offset[neib_cellnum]*d_cellSize;

		// Compute index of the first particle in the current cell
		// use calcGridHashPeriodic because we can only have an out-of-grid cell with neighbors
		// only in the periodic case.
		neib_cell_base_index = cellStart[calcGridHashPeriodic(gridPos + d_cell_to_offset[neib_cellnum])];
	}

	// Compute and return neighbor index
	return neib_cell_base_index + neib_data;
}
/** @} */
/** @} */

}

#endif
