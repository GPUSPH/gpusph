/*  Copyright (c) 2018-2019 INGV, EDF, UniCT, JHU

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
 * Common parameter structure when neighbors traversal is needed
 */

#ifndef NEIBS_LIST_PARAMS_H
#define NEIBS_LIST_PARAMS_H

#include "particledefine.h"

#include "common_params.h"
#include "dem_params.h"

#include "cond_params.h"

/// Parameters needed to iterate over the neighbors list
struct neibs_list_params : public pos_info_wrapper
{
	const hashKey* __restrict__		particleHash;
	const uint* __restrict__		cellStart;
	const neibdata* __restrict__	neibsList;
	const uint		numParticles;
	const float		slength;
	const float		influenceradius;

	/// Constructor / initializer
	neibs_list_params(
		BufferList const& bufread,
		const	uint	_numParticles,
		const	float	_slength,
		const	float	_influenceradius) :
		pos_info_wrapper(bufread),
		particleHash(bufread.getData<BUFFER_HASH>()),
		cellStart(bufread.getData<BUFFER_CELLSTART>()),
		neibsList(bufread.getData<BUFFER_NEIBSLIST>()),
		numParticles(_numParticles),
		slength(_slength),
		influenceradius(_influenceradius)
	{}
};

/// Parameters needed to iterate over the neighbors list
template<BoundaryType boundarytype
	, typename sa_params =
		typename COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_params)
>
struct neibs_interaction_params :
	neibs_list_params,
	vel_wrapper,
	sa_params
{
	neibs_interaction_params(
		BufferList const& bufread,
		const	uint	numParticles,
		const	float	slength,
		const	float	influenceradius)
	:
		neibs_list_params(bufread, numParticles, slength, influenceradius),
		vel_wrapper(bufread),
		sa_params(bufread)
	{}
};

/// Parameters needed to iterate over the neighboring particles and planes
template<BoundaryType boundarytype, flag_t simflags
	, typename planes_params =
		typename COND_STRUCT(HAS_DEM_OR_PLANES(simflags), neib_planes_params)
	, typename dem_cond =
		typename COND_STRUCT(HAS_DEM(simflags), dem_params)
>
struct neibs_planes_interaction_params :
	neibs_interaction_params<boundarytype>,
	planes_params,
	dem_cond
{
	neibs_planes_interaction_params(
		BufferList const& bufread,
		const	uint	numParticles,
		const	float	slength,
		const	float	influenceradius)
	:
		neibs_interaction_params<boundarytype>(bufread, numParticles, slength, influenceradius),
		planes_params(bufread),
		dem_cond() // automatically initialized from the global DEM object
	{}
};



#endif
