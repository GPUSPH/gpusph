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

#ifndef _FORCES_PARAMS_H
#define _FORCES_PARAMS_H

#include "particledefine.h"
#include "simflags.h"

/* The forces computation kernel is probably the most complex beast in GPUSPH.
   To achieve good performance, each combination of kernel, boundary, formulation
   etc is a different specialization, something which in itself makes for a huge
   number of kernels.
   To make things more complicated, some incarnations require a different set
   of parameters compared to other, and instantiate different sets of variables.
   All of this should be managed as automagically as possible while trying to
   rely only on the most readable features of C++ and, if possible, using the
   preprocessor as little as possible.

   To this end, we set up a mechanism that allows us to build structure
   templates in which the number and quality of members depends on the
   specialization.
*/

#include "cond_params.h"

// We now have the tools to assemble the structure that will be used to pass parameters to the forces kernel

/* Now we define structures that hold the parameters to be passed
   to the forces kernel. These are defined in chunks, and then ‘merged’
   into the template structure that is actually used as argument to the kernel.
   Each struct must define an appropriate constructor / initializer for its const
   members
*/

/// Parameters common to all forces kernel specializations
struct common_forces_params
{
			float4	* __restrict__ forces;
			float4	* __restrict__ rbforces;
			float4	* __restrict__ rbtorques;
	const	float4	* __restrict__ posArray;
	const	hashKey * __restrict__ particleHash;
	const	uint	* __restrict__ cellStart;
	const	neibdata	* __restrict__ neibsList;

	// Particle range to work on. toParticle is _exclusive_
	const	uint	fromParticle;
	const	uint	toParticle;

	// TODO these should probably go into constant memory
	const	float	deltap;
	const	float	slength;
	const	float	influenceradius;
	const	uint	step;
	const	float	dt;

	// Constructor / initializer
	common_forces_params(
				float4	* __restrict__ _forces,
				float4	* __restrict__ _rbforces,
				float4	* __restrict__ _rbtorques,
		const	float4	* __restrict__ _posArray,
		const	hashKey * __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,
		const	uint	_fromParticle,
		const	uint	_toParticle,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius,
		const	uint	_step,
		const	float	_dt) :
		forces(_forces),
		rbforces(_rbforces),
		rbtorques(_rbtorques),
		posArray(_posArray),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		fromParticle(_fromParticle),
		toParticle(_toParticle),
		deltap(_deltap),
		slength(_slength),
		influenceradius(_influenceradius),
		step(_step),
		dt(_dt)
	{}
};

/// Parameters common to all finalize forces kernel specializations
struct common_finalize_forces_params
{
			float4	*forces;
			float4	*rbforces;
			float4	*rbtorques;
	const	float4	*posArray;
	const	float4	*velArray;
	const	hashKey *particleHash;
	const	uint	*cellStart;

	// Particle range to work on. toParticle is _exclusive_
	const	uint	fromParticle;
	const	uint	toParticle;

	const	float	slength;

	// Constructor / initializer
	common_finalize_forces_params(
				float4	*_forces,
				float4	*_rbforces,
				float4	*_rbtorques,
		const	float4	*_posArray,
		const 	float4	*_velArray,
		const	hashKey *_particleHash,
		const	uint	*_cellStart,
		const	uint	_fromParticle,
		const	uint	_toParticle,
		const 	float	_slength) :
		forces(_forces),
		rbforces(_rbforces),
		rbtorques(_rbtorques),
		posArray(_posArray),
		velArray(_velArray),
		particleHash(_particleHash),
		cellStart(_cellStart),
		fromParticle(_fromParticle),
		toParticle(_toParticle),
		slength(_slength)
	{}
};

/// Additional parameters passed only to kernels with dynamic timestepping
struct dyndt_finalize_forces_params
{
	float	* __restrict__ cfl_forces;
	float	* __restrict__ cfl_gamma;
	float	* __restrict__ cfl_keps;
	uint	cflOffset;

	dyndt_finalize_forces_params(float * __restrict__ _cfl_forces, float * __restrict__ _cfl_gamma, float * __restrict__ _cfl_keps, uint _cflOffset) :
		cfl_forces(_cfl_forces), cfl_gamma(_cfl_gamma), cfl_keps(_cfl_keps), cflOffset(_cflOffset)
	{}
};

/// Additional parameters passed only to kernels with XSPH enabled
struct xsph_forces_params
{
	float4	* __restrict__ xsph;
	xsph_forces_params(float4 * __restrict__ _xsph) :
		xsph(_xsph)
	{}
};

/// Additional parameters passed only to kernels with SPH_GRENIER formulation
struct grenier_forces_params
{
	const float	* __restrict__ sigmaArray;
	grenier_forces_params(const float * __restrict__ _sigmaArray) : sigmaArray(_sigmaArray)
	{}
};

/// Additional parameters passed only finalize forces with SPH_GRENIER formulation
struct grenier_finalize_forces_params
{
	const float	*sigmaArray;

	grenier_finalize_forces_params(const float *_sigmaArray) :
		sigmaArray(_sigmaArray)
	{}
};

/// Used by formulations that have volume
struct volume_forces_params
{
	const float4	* __restrict__ volArray;
	volume_forces_params(const float4 * __restrict__ _volArray) : volArray(_volArray)
	{}
};

/// Additional parameters passed only to kernels with SA_BOUNDARY
/// in case of of a fluid/boudary interaction
struct sa_boundary_forces_params
{
			float4	* __restrict__ newGGam;
			float	* __restrict__ dgamdt;
			float	* __restrict__ cfl_gamma;
	const	float2	* __restrict__ vertPos0;
	const	float2	* __restrict__ vertPos1;
	const	float2	* __restrict__ vertPos2;
	const	float	epsilon;

	// Constructor / initializer
	sa_boundary_forces_params(
				float4	* __restrict__ _newGGam,
				float	* __restrict__ _dgamdt,
				float	* __restrict__ _cfl_gamma,
		const	float2	* __restrict__  const _vertPos[],
		const	float	_epsilon) :
		newGGam(_newGGam),
		dgamdt(_dgamdt),
		cfl_gamma(_cfl_gamma),
		epsilon(_epsilon)
	{
		if (_vertPos) {
			vertPos0 = _vertPos[0];
			vertPos1 = _vertPos[1];
			vertPos2 = _vertPos[2];
		} else {
			vertPos0 = vertPos1 = vertPos2 = NULL;
		}
	}
};

/// Additional parameters passed only finalize forces with SA_BOUNDARY formulation
struct sa_finalize_forces_params
{
	const	float4	*gGam;
	const	float4	*oldGGam;
			float	*dgamdt;

	// Constructor / initializer
	sa_finalize_forces_params(
		const	float4	*_gGam,
		const	float4	*_oldGGam,
				float	*_dgamdt) :
		gGam(_gGam),
		oldGGam(_oldGGam),
		dgamdt(_dgamdt)
	{}
};

/// Additional parameters passed only to kernels with SA_BOUNDARY
/// in case of of a fluid/fluid interaction
struct sa_forces_params
{
	float	*dgamdt;

	// Constructor / initializer
	sa_forces_params(float	*_dgamdt):dgamdt(_dgamdt)
	{}
};

/// Additional parameters passed only to kernels with ENABLE_WATER_DEPTH
struct water_depth_forces_params
{
	uint	* __restrict__ IOwaterdepth;

	water_depth_forces_params(uint * __restrict__ _IOwaterdepth) : IOwaterdepth(_IOwaterdepth)
	{}
};

/// Additional parameters passed only to kernels with KEPSVISC
struct kepsvisc_forces_params
{
	float3	* __restrict__ keps_dkde;
	const float	* __restrict__ turbvisc;
	float2	* __restrict__ tau0;
	float2	* __restrict__ tau1;
	float2	* __restrict__ tau2;
	kepsvisc_forces_params(float3 * __restrict__ _keps_dkde, const float * __restrict__ _turbvisc,
		float2 **tau) :
		keps_dkde(_keps_dkde),
		turbvisc(_turbvisc),
		tau0(tau[0]),
		tau1(tau[1]),
		tau2(tau[2])
	{}
};

/// Additional parameters only used to kernels with ENABLE_INTERNAL_ENERGY
struct internal_energy_forces_params
{
	float	* __restrict__ DEDt; // derivative of the internal energy with respect to time
	internal_energy_forces_params(float * __restrict__ _DEDt) :
		DEDt(_DEDt)
	{}
};

/// The actual forces_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	DensityDiffusionType _densitydiffusiontype,
	BoundaryType _boundarytype,
	ViscosityType _visctype,
	flag_t _simflags,
	ParticleType _cptype,
	ParticleType _nptype>
struct forces_params :
	common_forces_params,
	COND_STRUCT((_simflags & ENABLE_XSPH) && _cptype == _nptype, xsph_forces_params),
	COND_STRUCT(_sph_formulation == SPH_GRENIER &&
		_densitydiffusiontype == COLAGROSSI, volume_forces_params),
	COND_STRUCT(_sph_formulation == SPH_GRENIER, grenier_forces_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY && _cptype == PT_FLUID && _nptype == PT_FLUID, sa_forces_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY && _cptype != _nptype, sa_boundary_forces_params),
	COND_STRUCT(_simflags & ENABLE_WATER_DEPTH, water_depth_forces_params),
	COND_STRUCT(_visctype == KEPSVISC, kepsvisc_forces_params),
	COND_STRUCT(_simflags & ENABLE_INTERNAL_ENERGY, internal_energy_forces_params)
{
	static const KernelType kerneltype = _kerneltype;
	static const SPHFormulation sph_formulation = _sph_formulation;
	static const DensityDiffusionType densitydiffusiontype = _densitydiffusiontype;
	static const BoundaryType boundarytype = _boundarytype;
	static const ViscosityType visctype = _visctype;
	static const flag_t simflags = _simflags;
	static const ParticleType cptype = _cptype;
	static const ParticleType nptype = _nptype;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	forces_params(
		// common
				float4	* __restrict__ _forces,
				float4	* __restrict__ _rbforces,
				float4	* __restrict__ _rbtorques,
		const	float4	* __restrict__ _pos,
		const	hashKey	* __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata* __restrict__ _neibsList,
				uint	_fromParticle,
				uint	_toParticle,

		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius,
		const	uint	_step,
		const	float	_dt,

		// XSPH
				float4	* __restrict__ _xsph,

		// SPH_GRENIER
		const	float4	* __restrict__ _volArray,
		const	float	* __restrict__ _sigmaArray,

		// SA_BOUNDARY
				float4	* __restrict__ _newGGam,
				float	* __restrict__ _dgamdt,
				float	* __restrict__ _cfl_gamma,
		const	float2	* __restrict__ const _vertPos[],
		const	float	_epsilon,

		// ENABLE_WATER_DEPTH
				uint	* __restrict__ _IOwaterdepth,

		// KEPSVISC
				float3	* __restrict__ _keps_dkde,
		const	float	* __restrict__ _turbvisc,
				float2	** tau,
		// ENABLE_INTERNAL_ENERGY
				float	* __restrict__ _DEDt
		) :
		common_forces_params(_forces, _rbforces, _rbtorques,
			_pos, _particleHash, _cellStart,
			_neibsList, _fromParticle, _toParticle,
			_deltap, _slength, _influenceradius, _step, _dt),
		COND_STRUCT(simflags & ENABLE_XSPH, xsph_forces_params)(_xsph),
		COND_STRUCT(_sph_formulation == SPH_GRENIER &&
			densitydiffusiontype == COLAGROSSI, volume_forces_params)(_volArray),
		COND_STRUCT(sph_formulation == SPH_GRENIER, grenier_forces_params)(_sigmaArray),
		COND_STRUCT(boundarytype == SA_BOUNDARY && cptype == PT_FLUID && nptype == PT_FLUID, sa_forces_params)
			(_dgamdt),
		COND_STRUCT(boundarytype == SA_BOUNDARY && cptype != nptype, sa_boundary_forces_params)
			(_newGGam, _dgamdt, _cfl_gamma, _vertPos, _epsilon),
		COND_STRUCT(simflags & ENABLE_WATER_DEPTH, water_depth_forces_params)(_IOwaterdepth),
		COND_STRUCT(visctype == KEPSVISC, kepsvisc_forces_params)(_keps_dkde, _turbvisc, tau),
		COND_STRUCT(simflags & ENABLE_INTERNAL_ENERGY, internal_energy_forces_params)(_DEDt)
	{}
};


/// The actual finalize_forces_params struct, which concatenates all of the above, as appropriate.
template<SPHFormulation _sph_formulation,
	BoundaryType _boundarytype,
	ViscosityType _visctype,
	flag_t _simflags>
struct finalize_forces_params :
	common_finalize_forces_params,
	COND_STRUCT(_simflags & ENABLE_DTADAPT, dyndt_finalize_forces_params),
	COND_STRUCT(_sph_formulation == SPH_GRENIER, grenier_finalize_forces_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_finalize_forces_params),
	COND_STRUCT(_simflags & ENABLE_WATER_DEPTH, water_depth_forces_params),
	COND_STRUCT(_visctype == KEPSVISC, kepsvisc_forces_params),
	COND_STRUCT(_simflags & ENABLE_INTERNAL_ENERGY, internal_energy_forces_params)
{
	static const SPHFormulation sph_formulation = _sph_formulation;
	static const BoundaryType boundarytype = _boundarytype;
	static const ViscosityType visctype = _visctype;
	static const flag_t simflags = _simflags;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the finalize forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	finalize_forces_params(
		// common
				float4	*_forces,
				float4	*_rbforces,
				float4	*_rbtorques,
		const	float4	*_posArray,
		const	float4	*_velArray,
		const	hashKey	*_particleHash,
		const	uint	*_cellStart,
				uint	_fromParticle,
				uint	_toParticle,

		const 	float 	_slength,
		// dyndt
				float	*_cfl_forces,
				float	*_cfl_gamma,
				float	*_cfl_keps,
				uint	_cflOffset,

		// SPH_GRENIER
		const	float	*_sigmaArray,

		// SA_BOUNDARY
		const	float4	*_gGam,
		const	float4	*_oldGGam,
				float	*_dgamdt,

		// ENABLE_WATER_DEPTH
				uint	*_IOwaterdepth,

		// KEPSVISC
				float3	*_keps_dkde,
		const	float	*_turbvisc,
				float2	**tau,
		// ENABLE_INTERNAL_ENERGY
				float	* __restrict__ _DEDt
		) :
		common_finalize_forces_params(_forces, _rbforces, _rbtorques,
			_posArray, _velArray, _particleHash, _cellStart,
			 _fromParticle, _toParticle, _slength),
		COND_STRUCT(simflags & ENABLE_DTADAPT, dyndt_finalize_forces_params)
			(_cfl_forces, _cfl_gamma, _cfl_keps, _cflOffset),
		COND_STRUCT(sph_formulation == SPH_GRENIER, grenier_finalize_forces_params)(_sigmaArray),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_finalize_forces_params) (_gGam, _oldGGam, _dgamdt),
		COND_STRUCT(simflags & ENABLE_WATER_DEPTH, water_depth_forces_params)(_IOwaterdepth),
		COND_STRUCT(visctype == KEPSVISC, kepsvisc_forces_params)(_keps_dkde, _turbvisc, tau),
		COND_STRUCT(_simflags & ENABLE_INTERNAL_ENERGY, internal_energy_forces_params)(_DEDt)
	{}
};

/// Parameters common to all SPS kernel specializations
struct common_sps_params
{
	const float4* __restrict__ 	pos;
	const hashKey* __restrict__ 	particleHash;
	const uint* __restrict__ 		cellStart;
	const neibdata* __restrict__ 	neibsList;
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
	float2* __restrict__ 		tau0;
	float2* __restrict__ 		tau1;
	float2* __restrict__ 		tau2;

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
			const	float4* __restrict__ 	_pos,
			const	hashKey* __restrict__ 	_particleHash,
			const	uint* __restrict__ 		_cellStart,
			const	neibdata* __restrict__ 	_neibsList,
			const	uint		_numParticles,
			const	float		_slength,
			const	float		_influenceradius,
		// tau
					float2* __restrict__ 		_tau0,
					float2* __restrict__ 		_tau1,
					float2* __restrict__ 		_tau2,
		// turbvisc
					float* __restrict__ 		_turbvisc
		) :
		common_sps_params(_pos, _particleHash, _cellStart,
			_neibsList, _numParticles, _slength, _influenceradius),
		COND_STRUCT(simflags & SPSK_STORE_TAU, tau_sps_params)(_tau0, _tau1, _tau2),
		COND_STRUCT(simflags & SPSK_STORE_TURBVISC, turbvisc_sps_params)(_turbvisc)
	{}
};
#endif // _FORCES_PARAMS_H

