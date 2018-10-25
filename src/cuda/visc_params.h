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
 * Parameter structures for the visc kernels
 */

#ifndef _VISC_PARAMS_H
#define _VISC_PARAMS_H

#include "cond_params.h"

#include "particledefine.h"
#include "simflags.h"

/// Parameters common to all SPS kernel specializations
struct common_sps_params
{
	const float4* __restrict__		pos;
	const hashKey* __restrict__		particleHash;
	const uint* __restrict__		cellStart;
	const neibdata* __restrict__	neibsList;
	const uint		numParticles;
	const float		slength;
	const float		influenceradius;

	// Constructor / initializer
	common_sps_params(
		const	float4	* __restrict__ _pos,
		const	hashKey	* __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,
		const	uint	_numParticles,
		const	float	_slength,
		const	float	_influenceradius) :
		pos(_pos),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		numParticles(_numParticles),
		slength(_slength),
		influenceradius(_influenceradius)
	{}
};

/// Additional parameters passed only if simflag SPS_STORE_TAU is set
struct tau_sps_params
{
	float2* __restrict__		tau0;
	float2* __restrict__		tau1;
	float2* __restrict__		tau2;

	tau_sps_params(float2 * __restrict__ _tau0, float2 * __restrict__ _tau1, float2 * __restrict__ _tau2) :
		tau0(_tau0), tau1(_tau1), tau2(_tau2)
	{}
};

/// Additional parameters passed only if simflag SPS_STORE_TURBVISC is set
struct turbvisc_sps_params
{
	float	* __restrict__ turbvisc;
	turbvisc_sps_params(float * __restrict__ _turbvisc) :
		turbvisc(_turbvisc)
	{}
};


/// The actual forces_params struct, which concatenates all of the above, as appropriate.
template<KernelType kerneltype,
	BoundaryType boundarytype,
	uint simflags>
struct sps_params :
	common_sps_params,
	COND_STRUCT(simflags & SPSK_STORE_TAU, tau_sps_params),
	COND_STRUCT(simflags & SPSK_STORE_TURBVISC, turbvisc_sps_params)
{
	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	sps_params(
		// common
			const	float4* __restrict__	_pos,
			const	hashKey* __restrict__	_particleHash,
			const	uint* __restrict__		_cellStart,
			const	neibdata* __restrict__	_neibsList,
			const	uint		_numParticles,
			const	float		_slength,
			const	float		_influenceradius,
		// tau
					float2* __restrict__		_tau0,
					float2* __restrict__		_tau1,
					float2* __restrict__		_tau2,
		// turbvisc
					float* __restrict__		_turbvisc
		) :
		common_sps_params(_pos, _particleHash, _cellStart,
			_neibsList, _numParticles, _slength, _influenceradius),
		COND_STRUCT(simflags & SPSK_STORE_TAU, tau_sps_params)(_tau0, _tau1, _tau2),
		COND_STRUCT(simflags & SPSK_STORE_TURBVISC, turbvisc_sps_params)(_turbvisc)
	{}
};

#endif // _VISC_PARAMS_H
