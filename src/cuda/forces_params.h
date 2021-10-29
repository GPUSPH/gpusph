/*  Copyright (c) 2014-2019 INGV, EDF, UniCT, JHU

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
 * Parameter structures for the forces kernel
 */

#ifndef _FORCES_PARAMS_H
#define _FORCES_PARAMS_H

#include "particledefine.h"
#include "simflags.h"

#include "utils.h" // round_up

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
#include "dem_params.h"
#include "atomic_type.h"

// We now have the tools to assemble the structure that will be used to pass parameters to the forces kernel

/* Now we define structures that hold the parameters to be passed
   to the forces kernel. These are defined in chunks, and then ‘merged’
   into the template structure that is actually used as argument to the kernel.
   Each struct must define an appropriate constructor / initializer for its const
   members
*/

/// Parameters common to both forcesDevice and finalize kernels
struct stage_common_forces_params :
	pos_info_wrapper,
	vel_wrapper
{
			float4	* __restrict__ forces;
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
		pos_info_wrapper(bufread),
		vel_wrapper(bufread),
		forces(bufwrite.getData<BUFFER_FORCES>()),
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

template<RunMode run_mode>
struct common_finalize_forces_params;

/// Parameters common to all finalize forces kernel specializations during simulation
template<>
struct common_finalize_forces_params<SIMULATE> :
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
		const	float	_slength,
		const	float	_deltap /* unused in SIMULATE mode */)
	:
		stage_common_forces_params(bufread, bufwrite,
			_fromParticle, _toParticle, _slength),
		rbforces(bufwrite.getData<BUFFER_RB_FORCES>()),
		rbtorques(bufwrite.getData<BUFFER_RB_TORQUES>()),
		velArray(bufread.getData<BUFFER_VEL>())
	{}
};

/// Parameters common to all finalize forces kernel specializations during repacking
template<>
struct common_finalize_forces_params<REPACK> :
	stage_common_forces_params
{
	const	float4	* __restrict__ velArray;
	const	float	deltap;
			float4	* __restrict__ rbforces;
			float4	* __restrict__ rbtorques;

	// Constructor / initializer
	common_finalize_forces_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint	_fromParticle,
		const	uint	_toParticle,
		const	float	_slength,
		const	float	_deltap)
	:
		stage_common_forces_params(bufread, bufwrite,
			_fromParticle, _toParticle, _slength),
		velArray(bufread.getData<BUFFER_VEL>()),
		deltap(_deltap),
		rbforces(bufwrite.getData<BUFFER_RB_FORCES>()),
		rbtorques(bufwrite.getData<BUFFER_RB_TORQUES>())
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
	sa_boundary_params
{
			float	* __restrict__ cfl_gamma;
	const	float	epsilon;

	// Constructor / initializer
	sa_boundary_forces_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	float	_epsilon)
	:
		sa_boundary_params(bufread),
		cfl_gamma(bufwrite.getData<BUFFER_CFL_GAMMA>()),
		epsilon(_epsilon)
	{}
};

/// Additional parameters passed only to kernels that require BUFFER_EULERVEL.
/// This is currently only done if SA_BOUNDARY and either KEPSILON turbulence model
/// or ENABLE_INLET_OUTLET, and only in SIMULATE mode
DEFINE_BUFFER_WRAPPER_ARRAY(eulerVel_forces_params, BUFFER_EULERVEL, eulerVel, EulerVel);

/// Additional parameters passed only to kernels with DUMMY_BOUNDARY
/// in case of fluid/boundary interaction
struct dummy_boundary_forces_params
{
	const	float4	*  __restrict__ dummyVel;

	// Constructor / initializer
	dummy_boundary_forces_params(
		BufferList const&	bufread)
	:
		dummyVel(bufread.getData<BUFFER_DUMMY_VEL>())
	{}
};

/// Additional parameters passed only finalize forces with SA_BOUNDARY formulation
struct sa_finalize_forces_params :
	boundelements_wrapper
{
	const	float4	* __restrict__ gGam;

	// Constructor / initializer
	sa_finalize_forces_params(BufferList const& bufread) :
		boundelements_wrapper(bufread),
		gGam(bufread.getData<BUFFER_GRADGAMMA>())
	{}
};

/// Additional parameters passed only to kernels with ENABLE_WATER_DEPTH
struct water_depth_forces_params
{
	ATOMIC_TYPE(uint)	* __restrict__ IOwaterdepth;

	water_depth_forces_params(ATOMIC_TYPE(uint) * __restrict__ _IOwaterdepth) :
		IOwaterdepth(_IOwaterdepth)
	{}
};

/// Additional parameters passed only to kernels with KEPSILON
struct keps_forces_params :
	keps_tex_params, // read-only keps_{tke,eps}
	tau_params<true> // writable tau[012]
{
	float3	* __restrict__ keps_dkde;
	const float	* __restrict__ turbvisc;
	keps_forces_params(BufferList const& bufread, BufferList & bufwrite) :
		keps_tex_params(bufread),
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

struct effective_visc_forces_params
{
	const float * __restrict__ effective_visc;
	effective_visc_forces_params(BufferList const& bufread) :
		effective_visc(bufread.getData<BUFFER_EFFVISC>())
	{}
};

struct fea_forces_params
{
	const float4 * __restrict__ normsArray;
	float4	* __restrict__ feaforces;

	fea_forces_params(BufferList const& bufread, BufferList &bufwrite) :
		normsArray(bufread.getData<BUFFER_NORMALS>()),
		feaforces(bufwrite.getData<BUFFER_FEA_FORCES>())
	{}
};

/// Buffers with renormalized density gradient for Delta-SPH 
struct delta_sph_forces_params
{
	const float4 * __restrict__ deltaDensGrad;

	delta_sph_forces_params(BufferList const& bufread) :
		deltaDensGrad(bufread.getData<BUFFER_RENORMDENS>())
	{}
};

/// The actual forces_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	DensityDiffusionType _densitydiffusiontype,
	BoundaryType _boundarytype,
	typename _ViscSpec,
	flag_t _simflags,
	Dimensionality _dimensions,
	ParticleType _cptype,
	ParticleType _nptype,
	RunMode _run_mode = SIMULATE,
	bool _repacking = (_run_mode == REPACK),
	bool _has_keps = _ViscSpec::turbmodel == KEPSILON,
	bool _has_sps = _ViscSpec::turbmodel == SPS,
	bool _has_effective_visc = NEEDS_EFFECTIVE_VISC(_ViscSpec::rheologytype),
	typename xsph_cond =
		typename COND_STRUCT(!_repacking && HAS_XSPH(_simflags) && _cptype == _nptype, xsph_forces_params),
	typename fea_cond =
		typename COND_STRUCT(!_repacking && HAS_FEA(_simflags), fea_forces_params),
	typename vol_cond =
		typename COND_STRUCT(!_repacking && _sph_formulation == SPH_GRENIER &&
			_densitydiffusiontype == COLAGROSSI, volume_forces_params),
	typename grenier_cond =
		typename COND_STRUCT(!_repacking && _sph_formulation == SPH_GRENIER, grenier_forces_params),
	typename sa_cond =
		typename COND_STRUCT(_boundarytype == SA_BOUNDARY && _cptype != _nptype, sa_boundary_forces_params),
	typename dummy_cond =
		typename COND_STRUCT(_boundarytype == DUMMY_BOUNDARY && _cptype == PT_FLUID && _nptype == PT_BOUNDARY, dummy_boundary_forces_params),
	typename water_depth_cond =
		typename COND_STRUCT(!_repacking && HAS_WATER_DEPTH(_simflags), water_depth_forces_params),
	typename keps_cond =
		typename COND_STRUCT(!_repacking && _has_keps, keps_forces_params),
	typename sps_cond =
		typename COND_STRUCT(!_repacking && _has_sps, tau_tex_params),
	// eulerian velocity only used in case of keps or with open boundaries
	typename eulerVel_cond = typename
		COND_STRUCT(!_repacking && _boundarytype == SA_BOUNDARY && _cptype != _nptype
				&& (_has_keps || HAS_INLET_OUTLET(_simflags)) , // TODO this only works for SA_BOUNDARY atm
			eulerVel_forces_params),
	typename energy_cond =
		typename COND_STRUCT(!_repacking && HAS_INTERNAL_ENERGY(_simflags),
			internal_energy_forces_params),
	typename visc_cond =
		typename COND_STRUCT(!_repacking && _has_effective_visc, effective_visc_forces_params),
	typename cspm_cond =
		typename COND_STRUCT(HAS_CCSPH(_simflags), cspm_params<false>),
	typename delta_sph_cond =
		typename COND_STRUCT(_densitydiffusiontype == ANTUONO, delta_sph_forces_params)
	>
struct forces_params : _ViscSpec,
	common_forces_params,
	xsph_cond,
	fea_cond,
	vol_cond,
	grenier_cond,
	sa_cond,
	dummy_cond,
	water_depth_cond,
	keps_cond,
	sps_cond,
	eulerVel_cond,
	energy_cond,
	visc_cond,
	cspm_cond,
	delta_sph_cond
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
	static const Dimensionality dimensions = _dimensions;
	static const ParticleType cptype = _cptype;
	static const ParticleType nptype = _nptype;

	static const RunMode run_mode = _run_mode;
	static const bool repacking = _repacking;
	static const bool has_keps = _has_keps;
	static const bool has_effective_visc = _has_effective_visc;
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
		ATOMIC_TYPE(uint)	* __restrict__ _IOwaterdepth
		)
	:
		common_forces_params(bufread, bufwrite,
			_fromParticle, _toParticle,
			_deltap, _slength, _influenceradius, _step, _dt),
		xsph_cond(bufwrite),
		fea_cond(bufread, bufwrite),
		vol_cond(bufread),
		grenier_cond(bufread),
		sa_cond(bufread, bufwrite, _epsilon),
		dummy_cond(bufread),
		water_depth_cond(_IOwaterdepth),
		keps_cond(bufread, bufwrite),
		sps_cond(bufread),
		eulerVel_cond(bufread),
		energy_cond(bufwrite),
		visc_cond(bufread),
		cspm_cond(bufread),
		delta_sph_cond(bufread)
	{}
};

template<flag_t simflags>
using repackViscSpec = FullViscSpec<NEWTONIAN, LAMINAR_FLOW, KINEMATIC, MORRIS, ARITHMETIC, simflags, true>;

template<KernelType _kerneltype,
	BoundaryType _boundarytype,
	flag_t _simflags,
	Dimensionality _dimensions,
	ParticleType _cptype,
	ParticleType _nptype>
using repack_params = forces_params<_kerneltype, SPH_F1, DENSITY_DIFFUSION_NONE,
	  _boundarytype, repackViscSpec<_simflags>, _simflags, _dimensions, _cptype, _nptype, REPACK>;

/// The actual finalize_forces_params struct, which concatenates all of the above, as appropriate.
template<SPHFormulation _sph_formulation,
	BoundaryType _boundarytype,
	typename _ViscSpec,
	flag_t _simflags,
	RunMode _run_mode = SIMULATE,
	bool _repacking = (_run_mode == REPACK),
	bool _has_keps = _ViscSpec::turbmodel == KEPSILON,
	bool _inviscid = _ViscSpec::rheologytype == INVISCID,
	bool _has_effective_visc = NEEDS_EFFECTIVE_VISC(_ViscSpec::rheologytype),
	bool _has_planes = HAS_PLANES(_simflags),
	bool _has_dem = HAS_DEM(_simflags),
	typename planes_cond =
		typename COND_STRUCT(_has_planes || _has_dem, neib_planes_params),
	// DEM specifically also needs the demTex texture object
	typename dem_cond =
		typename COND_STRUCT(_has_dem, dem_params),
	typename dyndt_cond =
		typename COND_STRUCT(HAS_DTADAPT(_simflags), dyndt_finalize_forces_params),
	typename fea_cond =
		typename COND_STRUCT(!_repacking && HAS_FEA(_simflags), fea_forces_params),
	typename grenier_cond =
		typename COND_STRUCT(!_repacking && (_sph_formulation == SPH_GRENIER), grenier_forces_params),
	typename sa_cond =
		typename COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_finalize_forces_params),
	typename water_depth_cond =
		typename COND_STRUCT(!_repacking && HAS_WATER_DEPTH(_simflags), water_depth_forces_params),
	typename keps_cond = typename COND_STRUCT(!_repacking && _has_keps, keps_forces_params),
	typename energy_cond =
		typename COND_STRUCT(!_repacking && HAS_INTERNAL_ENERGY(_simflags), internal_energy_forces_params),
	typename visc_cond = typename COND_STRUCT(!_repacking && _has_effective_visc, effective_visc_forces_params)
	>
struct finalize_forces_params :
	common_finalize_forces_params<_run_mode>,
	planes_cond,
	dem_cond,
	dyndt_cond,
	fea_cond,
	grenier_cond,
	sa_cond,
	water_depth_cond,
	keps_cond,
	energy_cond,
	visc_cond
{
	static const SPHFormulation sph_formulation = _sph_formulation;
	static const BoundaryType boundarytype = _boundarytype;

	using ViscSpec = _ViscSpec;
	static const RheologyType rheologytype = ViscSpec::rheologytype;
	static const TurbulenceModel turbmodel = ViscSpec::turbmodel;
	static const ViscousModel viscmodel = ViscSpec::viscmodel;

	static const flag_t simflags = _simflags;

	static const RunMode run_mode = _run_mode;
	static const bool repacking = _repacking;
	static const bool has_planes = _has_planes;
	static const bool has_dem = _has_dem;
	static const bool has_keps = _has_keps;
	static const bool has_effective_visc = _has_effective_visc;
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
		const	float	_deltap,
		const	uint	_cflOffset,
		ATOMIC_TYPE(uint)	*_IOwaterdepth)
	:
		common_finalize_forces_params<run_mode>(bufread, bufwrite,
			 _fromParticle, _toParticle, _slength, _deltap),
		planes_cond(bufread),
		dem_cond(), // dem_params automatically initialize from the global DEM object
		dyndt_cond(bufwrite, _numParticles, _cflOffset),
		fea_cond(bufread, bufwrite),
		grenier_cond(bufread),
		sa_cond(bufread),
		water_depth_cond(_IOwaterdepth),
		keps_cond(bufread, bufwrite),
		energy_cond(bufwrite),
		visc_cond(bufread)
	{}
};

template<BoundaryType _boundarytype, flag_t _simflags>
using finalize_repack_params = finalize_forces_params<SPH_F1,
	  _boundarytype, repackViscSpec<_simflags>, _simflags, REPACK>;


#endif // _FORCES_PARAMS_H

