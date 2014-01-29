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

/* Auxiliary functions to compute a unsigned integer hash
 * from a cell grid position, and conversely */

/* IMPORTANT NOTE: this header should be included _inside_ the namespace,
 * of each _kernel.cu file, because of the __constant__s defined below
 */

#ifndef _CELLGRID_H
#define _CELLGRID_H

#include "hashkey.h"

__constant__ float3 d_worldOrigin;
__constant__ float3 d_cellSize;
__constant__ uint3 d_gridSize;

// Compute cell hash (linearized index) from cell coordinates. No clamping/periodicity checks are done.
__device__ __forceinline__ uint
calcGridHash(int3 const& gridPos)
{
	return INTMUL(INTMUL(gridPos.z, d_gridSize.y), d_gridSize.x) + INTMUL(gridPos.y, d_gridSize.x) + gridPos.x;
}

/// Compute grid position from cell hash value
/*! Compute the grid position corresponding to the given cell hash. The position
 * 	should be in the range [0, d_gridSize.x - 1]x[0, d_gridSize.y - 1]x[0, d_gridSize.z - 1].
 *
 *	\param[in] cellHash : cell hash value
 *
 *	\return grid position
 *
 *	Note : no test is done by this function to ensure that hash value is valid.
 */
__device__ __forceinline__ int3
calcGridPosFromCellHash(const uint cellHash)
{
	int3 gridPos;
	int temp = INTMUL(d_gridSize.y, d_gridSize.x);
	gridPos.z = cellHash/temp;
	temp = cellHash - gridPos.z*temp;
	gridPos.y = temp/d_gridSize.x;
	gridPos.x = temp - gridPos.y*d_gridSize.x;

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
 *	Note : no test is done by this function to ensure that hash value is valid.
 *	Note : when hashKey is 32bit long, this is equivalent to calcGridPosFromCellHash()
 */
__device__ __forceinline__ int3
calcGridPosFromParticleHash(const hashKey particleHash)
{
	// read the cellHash out of the particleHash
	const uint cellHash = cellHashFromParticleHash(particleHash);
	return calcGridPosFromCellHash(cellHash);
}

/* the neighbor cell num ranges from 1 to 27 (included), so it fits in
 * 5 bits, which we put in the upper 5 bits of the neibdata, which is
 * 16-bit wide.
 * TODO actually compute this from sizeof(neibdata)
 */
#define CELLNUM_SHIFT	11
#define CELLNUM_ENCODED	(1U<<CELLNUM_SHIFT)
#define NEIBINDEX_MASK	(CELLNUM_ENCODED-1)
#define ENCODE_CELL(cell) ((cell + 1) << CELLNUM_SHIFT)
#define DECODE_CELL(data) ((data >> CELLNUM_SHIFT) - 1)


#endif

