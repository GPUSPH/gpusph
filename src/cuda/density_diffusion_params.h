/*  Copyright (c) 2018 INGV, EDF, UniCT, JHU

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

#ifndef _DENSITY_DIFFUSION_PARAMS_H
#define _DENSITY_DIFFUSION_PARAMS_H

#include "particledefine.h"
#include "simflags.h"
#include "common_params.h"

// TODO FIXME redesign in terms of neibs_interaction_params

struct common_density_diffusion_params
{
			float4	* __restrict__ forces;
	const	float4	* __restrict__ posArray;
	const	float4	* __restrict__ velArray;
	const	particleinfo	* __restrict__ infoArray;
	const	hashKey * __restrict__ particleHash;
	const	uint	* __restrict__ cellStart;
	const	neibdata	* __restrict__ neibsList;

	const	uint	particleRangeEnd;

	const	float	deltap;
	const	float	slength;
	const	float	influenceradius;
	const	float	dt;

	common_density_diffusion_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
	const	uint	_particleRangeEnd,
	const	float	_deltap,
	const	float	_slength,
	const	float	_influenceradius,
	const	float	_dt) :
		forces(bufwrite.getData<BUFFER_FORCES>()),
		posArray(bufread.getData<BUFFER_POS>()),
		velArray(bufread.getData<BUFFER_VEL>()),
		infoArray(bufread.getData<BUFFER_INFO>()),
		particleHash(bufread.getData<BUFFER_HASH>()),
		cellStart(bufread.getData<BUFFER_CELLSTART>()),
		neibsList(bufread.getData<BUFFER_NEIBSLIST>()),
		particleRangeEnd(_particleRangeEnd),
		deltap(_deltap),
		slength(_slength),
		influenceradius(_influenceradius),
		dt(_dt)
	{}

	__device__ __forceinline__
	float4 fetchVel(const uint index) const
	{ return velArray[index]; }
};

template<
	KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	DensityDiffusionType _densitydiffusiontype,
	BoundaryType _boundarytype,
	ParticleType _cptype>
struct density_diffusion_params :
	common_density_diffusion_params,
	COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_boundary_params)
{
	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr SPHFormulation sph_formulation = _sph_formulation;
	static constexpr DensityDiffusionType densitydiffusiontype = _densitydiffusiontype;
	static constexpr BoundaryType boundarytype = _boundarytype;
	static constexpr ParticleType cptype = _cptype;

	density_diffusion_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
	const	uint	_particleRangeEnd,
	const	float	_deltap,
	const	float	_slength,
	const	float	_influenceradius,
	const	float	_dt)
		: common_density_diffusion_params
			(bufread, bufwrite, _particleRangeEnd, _deltap, _slength, _influenceradius, _dt)
		, COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_params)(bufread)
	{}
};

template<BoundaryType boundarytype>
struct delta_sph_density_grad_params : neibs_interaction_params<boundarytype>, cspm_params<false>
{
	float4*	__restrict__		renormDensGradArray;

	delta_sph_density_grad_params(
		BufferList const& bufread,
		BufferList bufwrite,
		const	uint	numParticles,
		const	float	slength,
		const	float	influenceradius)
	:
		neibs_interaction_params<boundarytype>(bufread, numParticles, slength, influenceradius),
		cspm_params<false>(bufread),
		renormDensGradArray(bufwrite.getData<BUFFER_RENORMDENS>())
	{}
};
#endif
