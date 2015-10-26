/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _BUILDNEIBS_PARAMS_H
#define _BUILDNEIBS_PARAMS_H

#include "cond_params.h"
#include "particledefine.h"

/// Parameters common to all buildneibs kernel specializations
struct common_buildneibs_params
{
			neibdata	*neibsList;				///< neighbor's list (out)
#if (__COMPUTE__ >= 20)
	const	float4		*posArray;				///< particle's positions (in)
#endif
	const	hashKey		*particleHash;			///< particle's hashes (in)
	const	uint		numParticles;			///< total number of particles
	const	float		sqinfluenceradius;		///< squared influence radius

	common_buildneibs_params(
				neibdata	*_neibsList,
		const	float4		*_pos,
		const	hashKey		*_particleHash,
		const	uint		_numParticles,
		const	float		_sqinfluenceradius) :
		neibsList(_neibsList),
#if (__COMPUTE__ >= 20)
		posArray(_pos),
#endif
		particleHash(_particleHash),
		numParticles(_numParticles),
		sqinfluenceradius(_sqinfluenceradius)
	{}
};

/// Parameters used only with SA_BOUNDARY buildneibs specialization
struct sa_boundary_buildneibs_params
{
			float2	*vertPos0;				///< relative position of vertex to segment, first vertex
			float2	*vertPos1;				///< relative position of vertex to segment, second vertex
			float2	*vertPos2;				///< relative position of vertex to segment, third vertex
	const	float	boundNlSqInflRad;		///< neighbour search radius for boundary segments
	const	uint	*vertIDToIndex;			///< vertex ID to particleIndex lookup table

	sa_boundary_buildneibs_params(
				float2	*_vertPos[],
		const	uint	*_vertIDToIndex,
		const	float	_boundNlSqInflRad) :
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2]),
		vertIDToIndex(_vertIDToIndex),
		boundNlSqInflRad(_boundNlSqInflRad)
	{}
};

/// The actual forces_params struct, which concatenates the above, as appropriate
template<BoundaryType boundarytype>
struct buildneibs_params :
	common_buildneibs_params,
	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_buildneibs_params)
{
	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	buildneibs_params(
		// common
				neibdata	*_neibsList,
		const	float4		*_pos,
		const	hashKey		*_particleHash,
		const	uint		_numParticles,
		const	float		_sqinfluenceradius,

		// SA_BOUNDARY
				float2	*_vertPos[],
		const	uint	*_vertIDToIndex,
		const	float	_boundNlSqInflRad) :
		common_buildneibs_params(_neibsList, _pos, _particleHash,
			_numParticles, _sqinfluenceradius),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_buildneibs_params)(
			_vertPos, _vertIDToIndex, _boundNlSqInflRad)
	{}
};



#endif // _BUILDNEIBS_PARAMS_H

