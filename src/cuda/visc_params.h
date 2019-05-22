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
#include "neibs_list_params.h"

#include "simflags.h"

/// Parameters passed to the SPS kernel only if simflag SPS_STORE_TAU is set
struct tau_sps_params
{
	float2* __restrict__		tau0;
	float2* __restrict__		tau1;
	float2* __restrict__		tau2;

	tau_sps_params(float2 * __restrict__ _tau0, float2 * __restrict__ _tau1, float2 * __restrict__ _tau2) :
		tau0(_tau0), tau1(_tau1), tau2(_tau2)
	{}
};

/// Parameters passed to the SPS kernel only if simflag SPS_STORE_TURBVISC is set
struct turbvisc_sps_params
{
	float	* __restrict__ turbvisc;
	turbvisc_sps_params(float * __restrict__ _turbvisc) :
		turbvisc(_turbvisc)
	{}
};


/// The actual sps_params struct, which concatenates all of the above, as appropriate.
template<KernelType kerneltype,
	BoundaryType boundarytype,
	uint simflags>
struct sps_params :
	neibs_list_params,
	COND_STRUCT(simflags & SPSK_STORE_TAU, tau_sps_params),
	COND_STRUCT(simflags & SPSK_STORE_TURBVISC, turbvisc_sps_params)
{
	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	sps_params(
		// common
			const	float4* __restrict__	_posArray,
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
		neibs_list_params(_posArray, _particleHash, _cellStart,
			_neibsList, _numParticles, _slength, _influenceradius),
		COND_STRUCT(simflags & SPSK_STORE_TAU, tau_sps_params)(_tau0, _tau1, _tau2),
		COND_STRUCT(simflags & SPSK_STORE_TURBVISC, turbvisc_sps_params)(_turbvisc)
	{}
};

//! Parameters needed when reducing the kinematic visc to find its maximum value
struct visc_reduce_params
{
	float * __restrict__	cfl;
	visc_reduce_params(float* __restrict__ _cfl) :
		cfl(_cfl)
	{}
};

//! Additional parameters passed only with SA_BOUNDARY
struct sa_boundary_rheology_params
{
	const	float4	* __restrict__ gGam;
	const	float2	* __restrict__ vertPos0;
	const	float2	* __restrict__ vertPos1;
	const	float2	* __restrict__ vertPos2;
	sa_boundary_rheology_params(const float4 * __restrict__ const _gGam, const   float2  * __restrict__  const _vertPos[])
	{
		if (!_gGam) throw std::invalid_argument("no gGam for sa_boundary_visc_params");
		if (!_vertPos) throw std::invalid_argument("no vertPos for sa_boundary_visc_params");
		gGam = _gGam;
		vertPos0 = _vertPos[0];
		vertPos1 = _vertPos[1];
		vertPos2 = _vertPos[2];
	}
};

//! Effective viscosity kernel parameters
/** in addition to the standard neibs_list_params, it only includes
 * the array where the effective viscosity is written
 */
template<KernelType _kerneltype,
	BoundaryType _boundarytype,
	typename _ViscSpec,
	flag_t _simflags,
	typename reduce_params =
		typename COND_STRUCT(_simflags & ENABLE_DTADAPT, visc_reduce_params),
	typename sa_params =
		typename COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_boundary_rheology_params)
	>
struct effvisc_params :
	neibs_list_params,
	reduce_params,
	sa_params
{
	float * __restrict__	effvisc;

	using ViscSpec = _ViscSpec;

	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr BoundaryType boundarytype = _boundarytype;
	static constexpr RheologyType rheologytype = ViscSpec::rheologytype;
	static constexpr flag_t simflags = _simflags;

	effvisc_params(
		// common
			const	float4* __restrict__	_posArray,
			const	hashKey* __restrict__	_particleHash,
			const	uint* __restrict__		_cellStart,
			const	neibdata* __restrict__	_neibsList,
			const	uint		_numParticles,
			const	float		_slength,
			const	float		_influenceradius,
		// SA_BOUNDARY params
			const	float4* __restrict__	_gGam,
			const	float2* const *_vertPos,
		// effective viscosity
					float*	__restrict__	_effvisc,
					float*	__restrict__	_cfl) :
	neibs_list_params(_posArray, _particleHash, _cellStart, _neibsList, _numParticles,
		_slength, _influenceradius),
	reduce_params(_cfl),
	sa_params(_gGam, _vertPos),
	effvisc(_effvisc)
	{}
};

//////////////////////////////////:
/// Parameters common to all SPS kernel specializations
struct common_rheology_params
{
	const float4* __restrict__		pos;
	const hashKey* __restrict__		particleHash;
	const uint* __restrict__		cellStart;
	const neibdata* __restrict__	neibsList;
	const uint		numParticles;
	const float		deltap;
	const float		slength;
	const float		influenceradius;

	// Constructor / initializer
	common_rheology_params(
		const	float4	* __restrict__ _pos,
		const	hashKey	* __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,
		const	uint	_numParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius) :
		pos(_pos),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		numParticles(_numParticles),
		deltap(_deltap),
		slength(_slength),
		influenceradius(_influenceradius)
	{}
};

/// Additional parameters passed only if simflag RHEOLOGY is set
struct effvisc_rheology_params
{
	float	* __restrict__ effvisc;
	effvisc_rheology_params(float * __restrict__ _effvisc) :
		effvisc(_effvisc)
	{}
};

/// Additional parameters passed only if simflag RHEOLOGY is set
struct effpres_rheology_params
{
	float	* __restrict__ effpres;
	effpres_rheology_params(float * __restrict__ _effpres) :
		effpres(_effpres)
	{}
};

/// The actual forces_params struct, which concatenates all of the above, as appropriate.
//template<KernelType kerneltype,
//	BoundaryType boundarytype>
//struct rheology_params :
//	common_rheology_params,
//	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_finalize_forces_params),
//	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_rheology_params),
//	COND_STRUCT(true, effvisc_rheology_params)
//{
//	// This structure provides a constructor that takes as arguments the union of the
//	// parameters that would ever be passed to the forces kernel.
//	// It then delegates the appropriate subset of arguments to the appropriate
//	// structs it derives from, in the correct order
//	rheology_params(
//		// common
//			const	float4* __restrict__ 	_pos,
//			const	hashKey* __restrict__ 	_particleHash,
//			const	uint* __restrict__ 		_cellStart,
//			const	neibdata* __restrict__ 	_neibsList,
//			const	uint		_numParticles,
//			const	float		_deltap,
//			const	float		_slength,
//			const	float		_influenceradius,
//			// SA_BOUNDARY finalize
//			const	float4	*_gGam,
//			// SA_BOUNDARY
//			const	float2	* __restrict__ const _vertPos[],
//			// effvisc
//			float* __restrict__ 		_effvisc
//		) :
//		common_rheology_params(_pos, _particleHash, _cellStart,
//			_neibsList, _numParticles, _deltap, _slength, _influenceradius),
//		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_finalize_forces_params) (_gGam),
//		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_rheology_params) (_vertPos),
//		COND_STRUCT(true, effvisc_rheology_params)(_effvisc)
//	{}
//};

/// The actual forces_params struct, which concatenates all of the above, as appropriate.
template<KernelType kerneltype,
	BoundaryType boundarytype>
struct viscengine_rheology_params :
	common_rheology_params,
	COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_rheology_params),
	effvisc_rheology_params,
	effpres_rheology_params
{
	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	viscengine_rheology_params(
		// common
			const	float4* __restrict__ 	_pos,
			const	hashKey* __restrict__ 	_particleHash,
			const	uint* __restrict__ 		_cellStart,
			const	neibdata* __restrict__ 	_neibsList,
			const	uint		_numParticles,
			const	float		_deltap,
			const	float		_slength,
			const	float		_influenceradius,
			// effvisc
			float* __restrict__ 		_effvisc,
			// effpres
			float* __restrict__ 		_effpres,
			// SA_BOUNDARY
			const	float4	*_gGam,
			const	float2	* __restrict__ const _vertPos[]
		) :
		common_rheology_params(_pos, _particleHash, _cellStart,
			_neibsList, _numParticles, _deltap, _slength, _influenceradius),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_rheology_params) (_gGam, _vertPos),
		effvisc_rheology_params(_effvisc),
		effpres_rheology_params(_effpres)
	{}	
};


#endif // _VISC_PARAMS_H
