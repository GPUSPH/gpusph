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
 * Template implementation of the ViscEngine in CUDA
 */

#include "textures.cuh"

#include "utils.h"
#include "engine_visc.h"
#include "cuda_call.h"
#include "simflags.h"

#include "define_buffers.h"

#include "visc_params.h"

#include "visc_kernel.cu"

/// CUDAViscEngine class.
///
/// Generally, the kernel and boundary type will be passed through to the
/// calc_visc() to call the appropriate kernels, and the main selector would be
/// just the ViscSpec. We cannot have partial function/method template
/// specialization, so our CUDAViscEngine::calc_visc delegates to a helper function,
/// calc_visc_implementation(), which can use SFINAE to do the necessary specialization.

template<typename _ViscSpec,
	KernelType _kerneltype,
	BoundaryType _boundarytype,
	flag_t _simflags>
class CUDAViscEngine : public AbstractViscEngine, public _ViscSpec
{
	using ViscSpec = _ViscSpec;

	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr BoundaryType boundarytype = _boundarytype;
	static constexpr flag_t simflags = _simflags;

	/// Viscous engine implementation, general case.
	/// Note that the SFINAE is done on a generic typename,
	/// which will be the type of the class itself.
	/// This is to avoid the issues associated with SFINAE not being possible
	/// when the specializations can only be differentiate by return type.
	template<typename This>
	enable_if_t<This::turbmodel != SPS && !NEEDS_EFFECTIVE_VISC(This::rheologytype), float>
	calc_visc_implementation(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
	{ return NAN; }

	/// Viscous engine implementation, specialized for the generalized Newtonian rheologies
	template<typename This>
	enable_if_t<NEEDS_EFFECTIVE_VISC(This::rheologytype), float>
	calc_visc_implementation(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
	{
		float *effvisc = bufwrite.getData<BUFFER_EFFVISC>();
		/* We recycle the CFL arrays to determine the maximum kinematic viscosity
		 * in the adaptive timestepping case
		 */
		float *cfl = NULL;
		float *tempCfl = NULL;

		auto cfl_buf = bufwrite.get<BUFFER_CFL>();
		if (cfl_buf) {
			auto tempCfl_buf = bufwrite.get<BUFFER_CFL_TEMP>();

			cfl_buf->clobber();
			tempCfl_buf->clobber();

			// get the (typed) pointers
			cfl = cfl_buf->get();
			tempCfl = tempCfl_buf->get();
		}

		const float4 *pos = bufread.getData<BUFFER_POS>();
		const float4 *vel = bufread.getData<BUFFER_VEL>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();
		const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
		const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
		const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

#if !PREFER_L1
		CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
#endif
		CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

		uint numThreads = BLOCK_SIZE_SPS;
		// number of blocks, rounded up to next multiple of 4 to improve reductions
		uint numBlocks = round_up(div_up(particleRangeEnd, numThreads), 4U);

		effvisc_params<kerneltype, boundarytype, ViscSpec, simflags> params(
			pos, particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			effvisc, cfl);

		cuvisc::effectiveViscDevice<<<numBlocks, numThreads>>>(params);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
#if !PREFER_L1
		CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
#endif

		if (This::simflags & ENABLE_DTADAPT) {
			return cflmax(numBlocks, cfl, tempCfl);
		} else {
			return NAN;
		}
	}

	/// Viscous engine implementation, specialized for the SPS turbulence model.
	template<typename This>
	enable_if_t<This::turbmodel == SPS, float>
	calc_visc_implementation(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
	{
		float2 **tau = bufwrite.getRawPtr<BUFFER_TAU>();
		float *turbvisc = bufwrite.getData<BUFFER_SPS_TURBVISC>();

		const float4 *pos = bufread.getData<BUFFER_POS>();
		const float4 *vel = bufread.getData<BUFFER_VEL>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();
		const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
		const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
		const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

		int dummy_shared = 0;
		// bind textures to read all particles, not only internal ones
#if !PREFER_L1
		CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
#endif
		CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);

#if (__COMPUTE__ == 20)
		dummy_shared = 2560;
#endif

		sps_params<kerneltype, boundarytype, (SPSK_STORE_TAU | SPSK_STORE_TURBVISC)> params(
			pos, particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			tau[0], tau[1], tau[2], turbvisc);

		cuvisc::SPSstressMatrixDevice<kerneltype, boundarytype, (SPSK_STORE_TAU | SPSK_STORE_TURBVISC)>
			<<<numBlocks, numThreads, dummy_shared>>>(params);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
#if !PREFER_L1
		CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
#endif

		CUDA_SAFE_CALL(cudaBindTexture(0, tau0Tex, tau[0], numParticles*sizeof(float2)));
		CUDA_SAFE_CALL(cudaBindTexture(0, tau1Tex, tau[1], numParticles*sizeof(float2)));
		CUDA_SAFE_CALL(cudaBindTexture(0, tau2Tex, tau[2], numParticles*sizeof(float2)));

		// TODO return SPS turbvisc?
		return NAN;
	}

	// TODO when we will be in a separate namespace from forces
	void setconstants() {}
	void getconstants() {}

	float
	calc_visc(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	slength,
		const	float	influenceradius)
	{
		return calc_visc_implementation(bufread, bufwrite,
			numParticles, particleRangeEnd, slength, influenceradius, this);
	}

};

