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

#include "cond_params.h"
#include "particledefine.h"

#include "buffer.h"
#include "define_buffers.h"
#include "cudabuffer.h"

/** \addtogroup neibs_buildnibskernel_params Neighbor list kernel parameters
 * 	\ingroup neibs
 *  Templatized structures holding parameters passed to buildneibs kernel
 *  @{ */

/// Wrapper for posArray access (linear or texture, based on PREFER_L1
struct pos_wrapper
{
#if PREFER_L1
	const	float4		* __restrict__ posArray;				///< particle's positions (in)
#else
	cudaTextureObject_t posTexObj;
#endif

	pos_wrapper(const BufferList& bufread) :
#if PREFER_L1
		posArray(bufread.getData<BUFFER_POS>())
#else
		posTexObj(getTextureObject<BUFFER_POS>(bufread))
#endif
	{}

	__device__ __forceinline__ float4
	fetchPos(const uint index) const
	{
#if PREFER_L1
		return posArray[index];
#else
		return tex1Dfetch<float4>(posTexObj, index);
#endif
	}
};

/// Common parameters used in buildneibs kernel
/*!	Parameters passed to buildneibs device function depends on the type of
 * 	of boundary used. This structure contains the parameters common to all
 * 	boundary types.
 */
struct common_buildneibs_params :
	pos_wrapper ///< particle's positions (in)
{
	cudaTextureObject_t infoTexObj;
	cudaTextureObject_t cellStartTexObj;
	cudaTextureObject_t cellEndTexObj;

	const	hashKey		*particleHash;			///< particle's hashes (in)
			neibdata	*neibsList;				///< neighbor's list (out)
	const	uint		numParticles;			///< total number of particles
	const	float		sqinfluenceradius;		///< squared influence radius

	common_buildneibs_params(
		const	BufferList&	bufread,
				BufferList& bufwrite,
		const	uint		_numParticles,
		const	float		_sqinfluenceradius)
	:
		pos_wrapper(bufread),
		infoTexObj(getTextureObject<BUFFER_INFO>(bufread)),
		cellStartTexObj(getTextureObject<BUFFER_CELLSTART>(bufread)),
		cellEndTexObj(getTextureObject<BUFFER_CELLEND>(bufread)),
		particleHash(bufread.getData<BUFFER_HASH>()),
		neibsList(bufwrite.getData<BUFFER_NEIBSLIST>()),
		numParticles(_numParticles),
		sqinfluenceradius(_sqinfluenceradius)
	{}

	__device__ __forceinline__ particleinfo
	fetchInfo(const uint index) const
	{ return tex1Dfetch<particleinfo>(infoTexObj, index); }

	__device__ __forceinline__ uint
	fetchCellStart(const uint index) const
	{ return tex1Dfetch<uint>(cellStartTexObj, index); }

	__device__ __forceinline__ uint
	fetchCellEnd(const uint index) const
	{ return tex1Dfetch<uint>(cellEndTexObj, index); }
};

/// Parameters used only with SA_BOUNDARY buildneibs specialization
struct sa_boundary_buildneibs_params
{
			float2	*vertPos0;				///< relative position of vertex to segment, first vertex
			float2	*vertPos1;				///< relative position of vertex to segment, second vertex
			float2	*vertPos2;				///< relative position of vertex to segment, third vertex
	const	float	boundNlSqInflRad;		///< neighbor search radius for PT_FLUID <-> PT_BOUNDARY interaction

	sa_boundary_buildneibs_params(
				float2	*_vertPos[],
		const	float	_boundNlSqInflRad) :
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2]),
		boundNlSqInflRad(_boundNlSqInflRad)
	{}

	sa_boundary_buildneibs_params(
				BufferList& bufwrite,
		const	float	_boundNlSqInflRad) :
		sa_boundary_buildneibs_params(bufwrite.getRawPtr<BUFFER_VERTPOS>(),
			_boundNlSqInflRad)
	{}
};

/// The actual buildneibs parameters structure, which concatenates the above, as appropriate
/*! This structure provides a constructor that takes as arguments the union of the
 *	parameters that would ever be passed to the forces kernel.
 *  It then delegates the appropriate subset of arguments to the appropriate
 *  structures it derives from, in the correct order
 */
template<BoundaryType boundarytype>
struct buildneibs_params :
	common_buildneibs_params,
	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_buildneibs_params)
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
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_buildneibs_params)(
			bufwrite, _boundNlSqInflRad)
	{}
};
/** @} */

#endif // _BUILDNEIBS_PARAMS_H

