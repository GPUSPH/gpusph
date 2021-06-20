/*  Copyright (c) 2014-2018 INGV, EDF, UniCT, JHU

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

#ifndef _BUILDNEIBS_PARAMS_H
#define _BUILDNEIBS_PARAMS_H

#include "common_params.h"
#include "particledefine.h"

#include "buffer.h"
#include "define_buffers.h"
#include "cudabuffer.h"

/** \addtogroup neibs_buildnibskernel_params Neighbor list kernel parameters
 * 	\ingroup neibs
 *  Templatized structures holding parameters passed to buildneibs kernel
 *  @{ */

DEFINE_BUFFER_WRAPPER(cellStart_wrapper, BUFFER_CELLSTART, cellStart, CellStart);
DEFINE_BUFFER_WRAPPER(cellEnd_wrapper, BUFFER_CELLEND, cellEnd, CellEnd);

struct cell_params : cellStart_wrapper, cellEnd_wrapper
{
	cell_params(const BufferList& bufread) :
		cellStart_wrapper(bufread),
		cellEnd_wrapper(bufread)
	{}
};

/// Common parameters used in buildneibs kernel
/*!	Parameters passed to buildneibs device function depends on the type of
 * 	of boundary used. This structure contains the parameters common to all
 * 	boundary types.
 */
struct common_buildneibs_params :
	pos_info_wrapper, ///< particle's positions and info (in)
	cell_params
{
	const	hashKey		* __restrict__ particleHash;			///< particle's hashes (in)
			neibdata	* __restrict__ neibsList;				///< neighbor's list (out)
	const	uint		numParticles;			///< total number of particles
	const	float		sqinfluenceradius;		///< squared influence radius

	common_buildneibs_params(
		const	BufferList&	bufread,
				BufferList& bufwrite,
		const	uint		_numParticles,
		const	float		_sqinfluenceradius)
	:
		pos_info_wrapper(bufread),
		cell_params(bufread),
		particleHash(bufread.getData<BUFFER_HASH>()),
		neibsList(bufwrite.getData<BUFFER_NEIBSLIST>()),
		numParticles(_numParticles),
		sqinfluenceradius(_sqinfluenceradius)
	{}
};

/// Parameters used only when ENABLE_PLANES or ENBLE_DEM
struct planes_buildneibs_params
{
			int4	* __restrict__ neibPlanes; ///< list of neighboring planes

	planes_buildneibs_params(BufferList& bufwrite) :
		neibPlanes(bufwrite.getData<BUFFER_NEIBPLANES>())
	{}
};

/*! Wrapper for const acces to BUFFER_VERTICES
 *  Access is provided by the fetchVert(index) method, without revealing details about
 *  the storage form (texture or linear array)
 */
DEFINE_BUFFER_WRAPPER(vertices_wrapper, BUFFER_VERTICES, vert, Vert);


/// Parameters used only with SA_BOUNDARY buildneibs specialization
struct sa_boundary_buildneibs_params :
	vertices_wrapper, ///< ID of the vertices of each boundary element (in)
	boundelements_wrapper, ///< boundary elements (in)
	vertPos_params<true>  ///< relative position of vertex to segment, one float2 array per vertex (out)
{
	const	float	boundNlSqInflRad;		///< neighbor search radius for PT_FLUID <-> PT_BOUNDARY interaction

	sa_boundary_buildneibs_params(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	float	_boundNlSqInflRad) :
		vertices_wrapper(bufread),
		boundelements_wrapper(bufread),
		vertPos_params<true>(bufwrite),
		boundNlSqInflRad(_boundNlSqInflRad)
	{}
};

/// The actual buildneibs parameters structure, which concatenates the above, as appropriate
/*! This structure provides a constructor that takes as arguments the union of the
 *	parameters that would ever be passed to the forces kernel.
 *  It then delegates the appropriate subset of arguments to the appropriate
 *  structures it derives from, in the correct order
 */
template<BoundaryType boundarytype, flag_t simflags,
	typename cond_sa_params = typename COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_buildneibs_params),
	typename cond_planes_params = typename COND_STRUCT(QUERY_ANY_FLAGS(simflags, ENABLE_PLANES | ENABLE_DEM),
		planes_buildneibs_params),
	typename cond_dem_params = typename COND_STRUCT(QUERY_ANY_FLAGS(simflags, ENABLE_DEM), dem_params)>
struct buildneibs_params :
	common_buildneibs_params,
	cond_planes_params,
	cond_dem_params,
	cond_sa_params
{
	buildneibs_params(
		const	BufferList&	bufread,
				BufferList& bufwrite,
		const	uint		_numParticles,
		const	float		_sqinfluenceradius,

		// SA_BOUNDARY
		const	float	_boundNlSqInflRad) :
		common_buildneibs_params(bufread, bufwrite,
			_numParticles, _sqinfluenceradius),
		cond_planes_params(bufwrite),
		cond_dem_params(), // dem_params automatically initialize from the global DEM object
		cond_sa_params(bufread, bufwrite, _boundNlSqInflRad)
	{}
};
/** @} */

#endif // _BUILDNEIBS_PARAMS_H

