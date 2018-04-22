/*  Copyright 2018 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _SA_SEGMENT_BC_PARAMS_H
#define _SA_SEGMENT_BC_PARAMS_H

#include "particledefine.h"
#include "simflags.h"

/* Templatized parameters structure for the saSegmentBoundaryConditions kernel. */

#include "cond_params.h"

/// Parameters common to all saSegmentBoundaryConditions specializations
struct common_sa_segment_bc_params
{
	const	float4	* __restrict__ pos;
			float4	* __restrict__ vel;
	const	hashKey * __restrict__ particleHash;
	const	uint	* __restrict__ cellStart;
	const	neibdata* __restrict__ neibsList;

			float4	* __restrict__ gGam;
	const	vertexinfo*	__restrict__ vertices;
	const	float2	* __restrict__ vertPos0;
	const	float2	* __restrict__ vertPos1;
	const	float2	* __restrict__ vertPos2;

	const	uint	numParticles;

	// TODO these should probably go into constant memory
	const	float	deltap;
	const	float	slength;
	const	float	influenceradius;

	// Constructor / initializer
	common_sa_segment_bc_params(
		const	float4	* __restrict__ _pos,
				float4	* __restrict__ _vel,
		const	hashKey * __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,

				float4	* __restrict__ _gGam,
		const	vertexinfo*	__restrict__ _vertices,
		const	float2	* const *__restrict__ _vertPos,

		const	uint	_numParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius)
	:
		pos(_pos),
		vel(_vel),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		gGam(_gGam),
		vertices(_vertices),
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2]),
		numParticles(_numParticles),
		deltap(_deltap),
		slength(_slength),
		influenceradius(_influenceradius)
	{}
};

// Carries the eulerVel buffer, only used for I/O and k-eps
struct eulervel_sa_segment_bc_params
{
	float4 * __restrict__ eulerVel;

	eulervel_sa_segment_bc_params(float4 * _eulerVel) :
		eulerVel(_eulerVel)
	{}
};

struct keps_sa_segment_bc_params
{
	float * __restrict__ tke;
	float * __restrict__ eps;

	keps_sa_segment_bc_params(
		float *__restrict__ _tke,
		float *__restrict__ _eps)
	:
		tke(_tke), eps(_eps)
	{}
};

template<ViscosityType _visctype, flag_t _simflags,
	bool _has_io = (_simflags & ENABLE_INLET_OUTLET),
	bool _has_keps = (_visctype == KEPSVISC),
	bool _has_moving = (_simflags & ENABLE_MOVING_BODIES),
	bool has_eulerVel = (_has_io || _has_keps),
	typename eulervel_struct =
		typename COND_STRUCT(has_eulerVel, eulervel_sa_segment_bc_params),
	typename keps_struct =
		typename COND_STRUCT(_has_keps, keps_sa_segment_bc_params)
	>
struct sa_segment_bc_params :
	common_sa_segment_bc_params,
	eulervel_struct,
	keps_struct
{
	static constexpr ViscosityType visctype = _visctype;
	static constexpr flag_t simflags = _simflags;
	static constexpr bool has_io = _has_io;
	static constexpr bool has_keps = _has_keps;
	static constexpr bool has_moving = _has_moving;

	sa_segment_bc_params(
		const	float4	* __restrict__ _pos,
				float4	* __restrict__ _vel,
		const	hashKey * __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,

				float4	* __restrict__ _gGam,
		const	vertexinfo*	__restrict__ _vertices,
		const	float2	* const *__restrict__ _vertPos,

				float4	* __restrict__ _eulerVel,
				float	* __restrict__ _tke,
				float	* __restrict__ _eps,

		const	uint	_numParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius)
	:
		common_sa_segment_bc_params(
			_pos, _vel, _particleHash, _cellStart, _neibsList,
			_gGam, _vertices, _vertPos,
			_numParticles, -deltap, _slength, _influenceradius),
		eulervel_struct(_eulerVel),
		keps_struct(_tke, _eps)
	{}
};

#endif // _SA_SEGMENT_BC_PARAMS_H

