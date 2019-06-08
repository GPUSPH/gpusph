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
#include "cond_params.h"

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
			float4	* __restrict__ _forces,
	const	float4	* __restrict__ _posArray,
	const	float4	* __restrict__ _velArray,
	const	particleinfo	* __restrict__ _infoArray,
	const	hashKey * __restrict__ _particleHash,
	const	uint	* __restrict__ _cellStart,
	const	neibdata	* __restrict__ _neibsList,
	const	uint	_particleRangeEnd,
	const	float	_deltap,
	const	float	_slength,
	const	float	_influenceradius,
	const	float	_dt) :
		forces(_forces),
		posArray(_posArray),
		velArray(_velArray),
		infoArray(_infoArray),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		particleRangeEnd(_particleRangeEnd),
		deltap(_deltap),
		slength(_slength),
		influenceradius(_influenceradius),
		dt(_dt)
	{}
};

struct sa_boundary_density_diffusion_params
{
	const	float4	* __restrict__	ggam;
	const	float2	* __restrict__	vertPos0;
	const	float2	* __restrict__	vertPos1;
	const	float2	* __restrict__	vertPos2;

	sa_boundary_density_diffusion_params(const float4 * __restrict__ _ggam,
		const float2 * __restrict__ const _vertPos[]) :
		ggam(_ggam),
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2])
	{}
};

template<
	KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	DensityDiffusionType _densitydiffusiontype,
	BoundaryType _boundarytype,
	ParticleType _cptype>
struct density_diffusion_params :
	common_density_diffusion_params,
	COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_boundary_density_diffusion_params)
{
	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr SPHFormulation sph_formulation = _sph_formulation;
	static constexpr DensityDiffusionType densitydiffusiontype = _densitydiffusiontype;
	static constexpr BoundaryType boundarytype = _boundarytype;
	static constexpr ParticleType cptype = _cptype;

	density_diffusion_params(
			float4	* __restrict__ _forces,
	const	float4	* __restrict__ _posArray,
	const	float4	* __restrict__ _velArray,
	const	particleinfo	* __restrict__ _infoArray,
	const	hashKey * __restrict__ _particleHash,
	const	uint	* __restrict__ _cellStart,
	const	neibdata	* __restrict__ _neibsList,
	const	float4	* __restrict__ _ggam,
	const	float2	* __restrict__ const _vertPos[],
	const	uint	_particleRangeEnd,
	const	float	_deltap,
	const	float	_slength,
	const	float	_influenceradius,
	const	float	_dt) :
		common_density_diffusion_params(_forces, _posArray, _velArray, _infoArray, _particleHash, _cellStart, _neibsList, _particleRangeEnd,
			_deltap, _slength, _influenceradius, _dt),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_density_diffusion_params)(_ggam, _vertPos)
	{}
};

#endif
