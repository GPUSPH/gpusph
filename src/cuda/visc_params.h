/*  Copyright (c) 2018-2019 INGV, EDF, UniCT, JHU

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
template<KernelType _kerneltype,
	BoundaryType _boundarytype,
	uint _sps_simflags>
struct sps_params :
	neibs_list_params,
	COND_STRUCT(_sps_simflags & SPSK_STORE_TAU, tau_sps_params),
	COND_STRUCT(_sps_simflags & SPSK_STORE_TURBVISC, turbvisc_sps_params)
{
	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr BoundaryType boundarytype = _boundarytype;
	static const uint sps_simflags = _sps_simflags;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the forces kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	sps_params(
		// common
		BufferList const& bufread,
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
		neibs_list_params(bufread, _numParticles, _slength, _influenceradius),
		COND_STRUCT(sps_simflags & SPSK_STORE_TAU, tau_sps_params)(_tau0, _tau1, _tau2),
		COND_STRUCT(sps_simflags & SPSK_STORE_TURBVISC, turbvisc_sps_params)(_turbvisc)
	{}
};

//! Parameters needed when reducing the kinematic visc to find its maximum value
struct visc_reduce_params
{
	float * __restrict__	cfl;
	visc_reduce_params(BufferList& bufwrite)
	{
		// We clobber both CFL buffers, even though
		// we only use the main one here: the other
		// will be used by a call to cflmax following the kernel call
		auto cfl_buf = bufwrite.get<BUFFER_CFL>();
		auto tempCfl_buf = bufwrite.get<BUFFER_CFL_TEMP>();
		cfl_buf->clobber();
		tempCfl_buf->clobber();

		cfl = cfl_buf->get();
	}
};

//! Additional parameters passed only with SA_BOUNDARY
struct sa_boundary_rheology_params
{
	cudaTextureObject_t		boundTexObj;
	const	float4	* __restrict__	gGam;
	const	float2	* __restrict__	vertPos0;
	const	float2	* __restrict__	vertPos1;
	const	float2	* __restrict__	vertPos2;


	sa_boundary_rheology_params(
		cudaTextureObject_t boundTexObj_,
		const float4 * __restrict__ const _gGam,
		const   float2  * __restrict__  const _vertPos[])
	:
		boundTexObj(boundTexObj_),
		gGam(_gGam),
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2])
	{}

	sa_boundary_rheology_params(BufferList const& bufread) :
		sa_boundary_rheology_params(
			getTextureObject<BUFFER_BOUNDELEMENTS>(bufread),
			bufread.getData<BUFFER_GRADGAMMA>(),
			bufread.getRawPtr<BUFFER_VERTPOS>())
	{}

	__device__ __forceinline__
	float4 fetchBound(const uint index) const
	{ return tex1Dfetch<float4>(boundTexObj, index); }
};

//! Additional parameters passed to include the effective pressure texture object
struct effpres_texture_params
{
	cudaTextureObject_t effPresTexObj;

	effpres_texture_params(BufferList const& bufread) :
		effPresTexObj(getTextureObject<BUFFER_EFFPRES>(bufread))
	{}

	__device__ __forceinline__
	float fetchEffPres(const uint index) const
	{ return tex1Dfetch<float>(effPresTexObj, index); }
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
		typename COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_boundary_rheology_params),
	typename granular_params =
		typename COND_STRUCT(_ViscSpec::rheologytype == GRANULAR, effpres_texture_params)
	>
struct effvisc_params :
	neibs_list_params,
	reduce_params,
	sa_params,
	granular_params
{
	float * __restrict__	effvisc;
	const float				deltap;

	using ViscSpec = _ViscSpec;

	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr BoundaryType boundarytype = _boundarytype;
	static constexpr RheologyType rheologytype = ViscSpec::rheologytype;
	static constexpr flag_t simflags = _simflags;

	// TODO switch everything to BufferList
	effvisc_params(
		// common
		BufferList const&	bufread,
		BufferList		bufwrite,
			const	uint		_numParticles,
			const	float		_slength,
			const	float		_influenceradius,
			const	float		_deltap) :
	neibs_list_params(bufread, _numParticles, _slength, _influenceradius),
	deltap(_deltap),
	reduce_params(bufwrite),
	sa_params(bufread),
	granular_params(bufread),
	effvisc(bufwrite.getData<BUFFER_EFFVISC>())
	{}
};

//! Common parameters for the kernels that solve for the effective pressure
/** This is essentially the standard neibs_list_params, plus optionally
 * the old effective pressure as a texture object
 * the array where the effective pressure is written
 */
template<KernelType _kerneltype,
	BoundaryType _boundarytype,
	// a boolean that determines if the old effective pressure should be made available
	// separately from the writeable effpres array
	bool has_old_effpres = true,
	typename old_effpres = typename COND_STRUCT(has_old_effpres, effpres_texture_params),
	typename sa_params =
		typename COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_boundary_rheology_params)
	>
struct common_effpres_params :
	neibs_list_params,
	old_effpres,
	sa_params
{
	const float				deltap;

	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr BoundaryType boundarytype = _boundarytype;

	common_effpres_params(
		// common
		BufferList const&	bufread,
			const	uint		_numParticles,
			const	float		_slength,
			const	float		_influenceradius,
			const	float		_deltap) :
	neibs_list_params(bufread, _numParticles, _slength, _influenceradius),
	old_effpres(bufread),
	sa_params(bufread),
	deltap(_deltap)
	{}
};

//! Effective pressure kernel parameters
/** in addition to the standard neibs_list_params, it only includes
 * the array where the effective pressure is written
 */
template<KernelType _kerneltype, BoundaryType _boundarytype>
struct jacobi_wall_boundary_params :
	common_effpres_params<_kerneltype, _boundarytype, false>,
	visc_reduce_params
{
	float * __restrict__	effpres;

	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr BoundaryType boundarytype = _boundarytype;

	jacobi_wall_boundary_params(
		// common
		BufferList const&	bufread,
		BufferList		bufwrite,
			const	uint		_numParticles,
			const	float		_slength,
			const	float		_influenceradius,
			const	float		_deltap) :
	common_effpres_params<_kerneltype, _boundarytype, false>(bufread, _numParticles,
		_slength, _influenceradius, _deltap),
	visc_reduce_params(bufwrite),
	effpres(bufwrite.getData<BUFFER_EFFPRES>())
	{}
};

template<KernelType _kerneltype, BoundaryType _boundarytype>
using jacobi_build_vectors_params = common_effpres_params<_kerneltype, _boundarytype, true>;

struct jacobi_update_params : info_wrapper
{
	const	float4 * __restrict__	jacobiBuffer;
		float  * __restrict__	effpres;
		float  * __restrict__	cfl;
		uint			numParticles;

	jacobi_update_params(
		BufferList const& bufread,
		BufferList	bufwrite,
		uint		numParticles_)
	:
		info_wrapper(bufread),
		jacobiBuffer(bufread.getData<BUFFER_JACOBI>()),
		effpres(bufwrite.getData<BUFFER_EFFPRES>()),
		numParticles(numParticles_)
	{
		// Clobber the residual CFL buffers (recycled to compute the residual)
		// before use
		auto cfl_buf = bufwrite.get<BUFFER_CFL>();
		auto tempCfl_buf = bufwrite.get<BUFFER_CFL_TEMP>();

		cfl_buf->clobber();
		tempCfl_buf->clobber();

		// get the (typed) pointers
		cfl = cfl_buf->get();
	}

};

#endif // _VISC_PARAMS_H
