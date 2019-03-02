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

/*! \file
 * Parameter structures for the forces kernel
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

#include "common_params.h"

// We now have the tools to assemble the structure that will be used to pass parameters to the forces kernel

/* Now we define structures that hold the parameters to be passed
   to the forces kernel. These are defined in chunks, and then ‘merged’
   into the template structure that is actually used as argument to the kernel.
   Each struct must define an appropriate constructor / initializer for its const
   members
*/

/// Parameters common to both forcesDevice and finalize kernels
struct stage_common_forces_params
{
			float4	* __restrict__ forces;
	const	float4	* __restrict__ posArray;
	const	hashKey * __restrict__ particleHash;
	const	uint	* __restrict__ cellStart;

	// Particle range to work on. toParticle is _exclusive_
	const	uint	fromParticle;
	const	uint	toParticle;

	// TODO these should probably go into constant memory
	const	float	slength;

	stage_common_forces_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint	_fromParticle,
		const	uint	_toParticle,
		const	float	_slength)
	:
		forces(bufwrite.getData<BUFFER_FORCES>()),
		posArray(bufread.getData<BUFFER_POS>()),
		particleHash(bufread.getData<BUFFER_HASH>()),
		cellStart(bufread.getData<BUFFER_CELLSTART>()),
		fromParticle(_fromParticle),
		toParticle(_toParticle),
		slength(_slength)
	{}
};

/// Parameters common to all forces kernel specializations
struct common_forces_params :
	stage_common_forces_params
{
	const	neibdata	* __restrict__ neibsList;

	// TODO these should probably go into constant memory
	const	float	deltap;
	const	float	influenceradius;
	const	uint	step;
	const	float	dt;

	// Constructor / initializer
	common_forces_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint	_fromParticle,
		const	uint	_toParticle,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius,
		const	uint	_step,
		const	float	_dt)
	:
		stage_common_forces_params(bufread, bufwrite,
			_fromParticle, _toParticle, _slength),
		neibsList(bufread.getData<BUFFER_NEIBSLIST>()),
		deltap(_deltap),
		influenceradius(_influenceradius),
		step(_step),
		dt(_dt)
	{}
};

/// Parameters common to all finalize forces kernel specializations
struct common_finalize_forces_params :
	stage_common_forces_params
{
	// TODO these are only needed when force feedback is enabled,
	// but currently we do not have a way to enable force feedback
	// without also enabling moving bodies (which is much heavier),
	// so let us just enable it unconditionally for the time being
			float4	* __restrict__ rbforces;
			float4	* __restrict__ rbtorques;

	const	float4	* __restrict__ velArray;

	// Constructor / initializer
	common_finalize_forces_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint	_fromParticle,
		const	uint	_toParticle,
		const	float	_slength)
	:
		stage_common_forces_params(bufread, bufwrite,
			_fromParticle, _toParticle, _slength),
		rbforces(bufwrite.getData<BUFFER_RB_FORCES>()),
		rbtorques(bufwrite.getData<BUFFER_RB_TORQUES>()),
		velArray(bufread.getData<BUFFER_VEL>())
	{}
};

/// Additional parameters passed only to kernels with dynamic timestepping
struct dyndt_finalize_forces_params
{
	float	* __restrict__ cfl_forces;
	float	* __restrict__ cfl_gamma;
	float	* __restrict__ cfl_keps;
	uint	cflOffset;
	uint	cflGammaOffset;

	dyndt_finalize_forces_params(
		BufferList &bufwrite,
		uint _numParticles,
		uint _cflOffset)
	:
		cfl_forces(bufwrite.getData<BUFFER_CFL>()),
		cfl_gamma(bufwrite.getData<BUFFER_CFL_GAMMA>()),
		cfl_keps(bufwrite.getData<BUFFER_CFL_KEPS>()),
		cflOffset(_cflOffset),
		cflGammaOffset(round_up(_numParticles, 4U) + cflOffset)
	{}
};

/// Additional parameters passed only to kernels with XSPH enabled
struct xsph_forces_params
{
	float4	* __restrict__ xsph;
	xsph_forces_params(BufferList &bufwrite) :
		xsph(bufwrite.getData<BUFFER_XSPH>())
	{}
};

/// Additional parameters passed only to kernels with SPH_GRENIER formulation
struct grenier_forces_params
{
	const float	* __restrict__ sigmaArray;
	grenier_forces_params(BufferList const& bufread) :
		sigmaArray(bufread.getData<BUFFER_SIGMA>())
	{}
};

/// Used by formulations that have volume
struct volume_forces_params
{
	const float4	* __restrict__ volArray;
	volume_forces_params(BufferList const& bufread) :
		volArray(bufread.getData<BUFFER_VOLUME>())
	{}
};

/// Additional parameters passed only to kernels with SA_BOUNDARY
/// in case of fluid/boundary interaction
struct sa_boundary_forces_params :
	vertPos_params<false> // const vertPos[012]
{
			float	* __restrict__ cfl_gamma;
	const	float	epsilon;

	// Constructor / initializer
	sa_boundary_forces_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	float	_epsilon)
	:
		vertPos_params<false>(bufread),
		cfl_gamma(bufwrite.getData<BUFFER_CFL_GAMMA>()),
		epsilon(_epsilon)
	{}
};

/// Additional parameters passed only finalize forces with SA_BOUNDARY formulation
struct sa_finalize_forces_params
{
	const	float4	* __restrict__ gGam;

	// Constructor / initializer
	sa_finalize_forces_params(BufferList const& bufread) :
		gGam(bufread.getData<BUFFER_GRADGAMMA>())
	{}
};

/// Additional parameters passed only to kernels with ENABLE_WATER_DEPTH
struct water_depth_forces_params
{
	uint	* __restrict__ IOwaterdepth;

	water_depth_forces_params(uint * __restrict__ _IOwaterdepth) :
		IOwaterdepth(_IOwaterdepth)
	{}
};

/// Additional parameters passed only to kernels with KEPSILON
struct keps_forces_params :
	tau_params<true> // writable tau[012]
{
	float3	* __restrict__ keps_dkde;
	const float	* __restrict__ turbvisc;
	keps_forces_params(BufferList const& bufread, BufferList & bufwrite) :
		tau_params<true>(bufwrite),
		keps_dkde(bufwrite.getData<BUFFER_DKDE>()),
		turbvisc(bufread.getData<BUFFER_TURBVISC>())
	{}
};

/// Additional parameters only used to kernels with ENABLE_INTERNAL_ENERGY
struct internal_energy_forces_params
{
	float	* __restrict__ DEDt; // derivative of the internal energy with respect to time
	internal_energy_forces_params(BufferList & bufwrite) :
		DEDt(bufwrite.getData<BUFFER_INTERNAL_ENERGY_UPD>())
	{}
};

/// The actual forces_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	DensityDiffusionType _densitydiffusiontype,
	BoundaryType _boundarytype,
	typename _ViscSpec,
	flag_t _simflags,
	ParticleType _cptype,
	ParticleType _nptype,
	bool _has_keps = _ViscSpec::turbmodel == KEPSILON>
struct forces_params : _ViscSpec,
	common_forces_params,
	COND_STRUCT((_simflags & ENABLE_XSPH) && _cptype == _nptype, xsph_forces_params),
	COND_STRUCT(_sph_formulation == SPH_GRENIER &&
		_densitydiffusiontype == COLAGROSSI, volume_forces_params),
	COND_STRUCT(_sph_formulation == SPH_GRENIER, grenier_forces_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY && _cptype != _nptype, sa_boundary_forces_params),
	COND_STRUCT(_simflags & ENABLE_WATER_DEPTH, water_depth_forces_params),
	COND_STRUCT(_has_keps, keps_forces_params),
	COND_STRUCT(_simflags & ENABLE_INTERNAL_ENERGY, internal_energy_forces_params)
{
	static const KernelType kerneltype = _kerneltype;
	static const SPHFormulation sph_formulation = _sph_formulation;
	static const DensityDiffusionType densitydiffusiontype = _densitydiffusiontype;
	static const BoundaryType boundarytype = _boundarytype;

	using ViscSpec = _ViscSpec;
	static const RheologyType rheologytype = ViscSpec::rheologytype;
	static const TurbulenceModel turbmodel = ViscSpec::turbmodel;
	static const ViscousModel viscmodel = ViscSpec::viscmodel;

	static const flag_t simflags = _simflags;
	static const ParticleType cptype = _cptype;
	static const ParticleType nptype = _nptype;

	static const bool has_keps = _has_keps;
	static const bool inviscid = rheologytype == INVISCID;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	forces_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
				uint	_fromParticle,
				uint	_toParticle,

		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius,
		const	uint	_step,
		const	float	_dt,

		// SA_BOUNDARY
		const	float	_epsilon,

		// ENABLE_WATER_DEPTH
				uint	* __restrict__ _IOwaterdepth
		) :
		common_forces_params(bufread, bufwrite,
			_fromParticle, _toParticle,
			_deltap, _slength, _influenceradius, _step, _dt),
		COND_STRUCT(simflags & ENABLE_XSPH, xsph_forces_params)(bufwrite),
		COND_STRUCT(_sph_formulation == SPH_GRENIER &&
			densitydiffusiontype == COLAGROSSI, volume_forces_params)(bufread),
		COND_STRUCT(sph_formulation == SPH_GRENIER, grenier_forces_params)(bufread),
		COND_STRUCT(boundarytype == SA_BOUNDARY && cptype != nptype, sa_boundary_forces_params)
			(bufread, bufwrite, _epsilon),
		COND_STRUCT(simflags & ENABLE_WATER_DEPTH, water_depth_forces_params)(_IOwaterdepth),
		COND_STRUCT(has_keps, keps_forces_params)(bufread, bufwrite),
		COND_STRUCT(simflags & ENABLE_INTERNAL_ENERGY, internal_energy_forces_params)(bufwrite)
	{}
};


/// The actual finalize_forces_params struct, which concatenates all of the above, as appropriate.
template<SPHFormulation _sph_formulation,
	BoundaryType _boundarytype,
	typename _ViscSpec,
	flag_t _simflags,
	bool _has_keps = _ViscSpec::turbmodel == KEPSILON,
	bool _inviscid = _ViscSpec::rheologytype == INVISCID>
struct finalize_forces_params :
	common_finalize_forces_params,
	COND_STRUCT(_simflags & ENABLE_DTADAPT, dyndt_finalize_forces_params),
	COND_STRUCT(_sph_formulation == SPH_GRENIER, grenier_forces_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_finalize_forces_params),
	COND_STRUCT(_simflags & ENABLE_WATER_DEPTH, water_depth_forces_params),
	COND_STRUCT(_has_keps, keps_forces_params),
	COND_STRUCT(_simflags & ENABLE_INTERNAL_ENERGY, internal_energy_forces_params)
{
	static const SPHFormulation sph_formulation = _sph_formulation;
	static const BoundaryType boundarytype = _boundarytype;

	using ViscSpec = _ViscSpec;
	static const RheologyType rheologytype = ViscSpec::rheologytype;
	static const TurbulenceModel turbmodel = ViscSpec::turbmodel;
	static const ViscousModel viscmodel = ViscSpec::viscmodel;

	static const flag_t simflags = _simflags;

	static const bool has_keps = _has_keps;
	static const bool inviscid = _inviscid;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the finalize forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	finalize_forces_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
				uint	_numParticles,
				uint	_fromParticle,
				uint	_toParticle,

		const	float	_slength,
		const	uint	_cflOffset,
				uint	* __restrict__ _IOwaterdepth)
	:
		common_finalize_forces_params(bufread, bufwrite,
			 _fromParticle, _toParticle, _slength),
		COND_STRUCT(simflags & ENABLE_DTADAPT, dyndt_finalize_forces_params)
			(bufwrite, _numParticles, _cflOffset),
		COND_STRUCT(sph_formulation == SPH_GRENIER, grenier_forces_params)(bufread),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_finalize_forces_params)(bufread),
		COND_STRUCT(simflags & ENABLE_WATER_DEPTH, water_depth_forces_params)(_IOwaterdepth),
		COND_STRUCT(has_keps, keps_forces_params)(bufread, bufwrite),
		COND_STRUCT(_simflags & ENABLE_INTERNAL_ENERGY, internal_energy_forces_params)(bufwrite)
	{}
};

#endif // _FORCES_PARAMS_H

