/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A.
 	Dalrymple, Eugenio Rustico, Ciro Del Negro

	Conservatoire National des Arts et Metiers, Paris, France

	Istituto Nazionale di Geofisica e Vulcanologia,
    Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

/* Auxiliary functions to compute a unsigned integer hash
 * from a cell grid position, and conversely */

/* IMPORTANT NOTE: this header should be included _inside_ the namespace,
 * of each _kernel.cu file, because of the __constant__s defined below
 */

#include "hashkey.h"
#include "linearization.h"

/** \name Device constants
 *  @{ */
__constant__ float3 d_worldOrigin;			///< Origin of the simulation domain
__constant__ float3 d_cellSize;				///< Size of cells used for the neighbor search
__constant__ uint3 d_gridSize;				///< Size of the simulation domain expressed in terms of cell number
__constant__ char3	d_cell_to_offset[27];	///< Neibdata cell number to offset
/** @} */

/** \name Device functions
 *  @{ */
/// Compute hash value from grid position
/*! Compute the hash value from grid position according to the chosen
 * 	linearization (starting from x, y or z direction). The link
 * 	between COORD1,2,3 and .x, .y and .z is defined in linearization.h
 *
 *	\param[in] gridPos : grid position
 *
 *	\return hash value
 */
__device__ __forceinline__ uint
calcGridHash(int3 const& gridPos)
{
	return INTMUL(INTMUL(gridPos.COORD3, d_gridSize.COORD2), d_gridSize.COORD1)
			+ INTMUL(gridPos.COORD2, d_gridSize.COORD1) + gridPos.COORD1;
}


/// Compute grid position from cell hash value
/*! Compute the grid position corresponding to the given cell hash. The position
 * 	should be in the range [0, d_gridSize.x - 1]x[0, d_gridSize.y - 1]x[0, d_gridSize.z - 1].
 *
 *	\param[in] cellHash : cell hash value
 *
 *	\return grid position
 *
 *	/note no test is done by this function to ensure that hash value is valid.
 */
__device__ __forceinline__ int3
calcGridPosFromCellHash(const uint cellHash)
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
 * 	should be in the range [0, d_gridSize.x - 1]x[0, d_gridSize.y - 1]x[0, d_gridSize.z - 1].
 *
 *	\param[in] particleHash : particle hash value
 *
 *	\return grid position
 *
 *	\note
 *		- no test is done by this function to ensure that hash value is valid.
 *		- when hashKey is 32bit long, this is equivalent to calcGridPosFromCellHash()
 */
__device__ __forceinline__ int3
calcGridPosFromParticleHash(const hashKey particleHash)
{
	// Read the cellHash out of the particleHash
	const uint cellHash = cellHashFromParticleHash(particleHash);
	return calcGridPosFromCellHash(cellHash);
}
/** @} */


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

/********************************* Neighbor data access management ******************************************/

/// Compute hash value from grid position
/*! Compute the hash value corresponding to the given position. If the position
 * 	is not in the range [0, gridSize.x - 1]x[0, gridSize.y - 1]x[0, gridSize.z - 1]
 * 	we have periodic boundary and the grid position is updated according to the
 * 	chosen periodicity.
 *
 *	\param[in] gridPos : grid position
 *
 *	\return hash value
 *
 *	Note : no test is done by this function to ensure that grid position is within the
 *	range and no clamping is done
 */
// TODO: verify periodicity along multiple axis and templatize
__device__ __forceinline__ uint
calcGridHashPeriodic(int3 gridPos)
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
 *	\param[in] pos : current particle's positions
 *	\param[out] pos_corr : pos - current neighbor cell offset
 *	\param[in] cellStart : cells first particle index
 *	\param[in] neibdata : neighbor data
 *	\param[in,out] neib_cellnum : current neighbor cell number (0...27)
 *	\param[in,out] neib_cell_base_index : index of first particle of the current cell
 *
 * 	\return neighbor index
 *
 * Note: neib_cell_num and neib_cell_base_index must be persistent along
 * getNeibIndex calls.
 */
__device__ __forceinline__ uint
getNeibIndex(float4 const&	pos,
			float3&			pos_corr,
			const uint*		cellStart,
			neibdata		neib_data,
			int3 const&		gridPos,
			char&			neib_cellnum,
			uint&			neib_cell_base_index)
{
	if (neib_data >= CELLNUM_ENCODED) {
		// Update current neib cell number
		neib_cellnum = DECODE_CELL(neib_data);

		// Compute neighbor index relative to belonging cell
		neib_data &= NEIBINDEX_MASK;

		// Substract current cell offset vector to pos
		pos_corr = as_float3(pos) - d_cell_to_offset[neib_cellnum]*d_cellSize;

		// Compute index of the first particle in the current cell
		// use calcGridHashPeriodic because we can only have an out-of-grid cell with neighbors
		// only in the periodic case.
		neib_cell_base_index = cellStart[calcGridHashPeriodic(gridPos + d_cell_to_offset[neib_cellnum])];
	}

	// Compute and return neighbor index
	return neib_cell_base_index + neib_data;
}

/************************************************************************************************************/
