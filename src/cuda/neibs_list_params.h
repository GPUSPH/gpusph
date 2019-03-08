/*  Copyright 2018 Giuseppe Bilotta, Alexis Hérault, Robert A. Dalrymple, Ciro Del Negro

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

/*! \file
 * Common parameter structure when neighbors traversal is needed
 */

#ifndef NEIBS_LIST_PARAMS_H
#define NEIBS_LIST_PARAMS_H

#include "particledefine.h"

/// Parameters needed to interact with neighbors
struct neibs_list_params
{
	const float4* __restrict__		posArray;
	const hashKey* __restrict__		particleHash;
	const uint* __restrict__		cellStart;
	const neibdata* __restrict__	neibsList;
	const uint		numParticles;
	const float		slength;
	const float		influenceradius;

	/// Constructor / initializer
	neibs_list_params(
		const	float4	* __restrict__ _posArray,
		const	hashKey	* __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,
		const	uint	_numParticles,
		const	float	_slength,
		const	float	_influenceradius) :
		posArray(_posArray),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		numParticles(_numParticles),
		slength(_slength),
		influenceradius(_influenceradius)
	{}
};

#endif
