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
		const	float	deltap,
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
		const	float	deltap,
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

		// for granular flows
		const float *effpres = bufread.getData<BUFFER_EFFPRES>();
		if (ViscSpec::rheologytype == GRANULAR)
			CUDA_SAFE_CALL(cudaBindTexture(0, effpresTex, effpres, numParticles*sizeof(float)));

		// for SA
		const float4 *gGam = bufread.getData<BUFFER_GRADGAMMA>();
		const float4  *boundelement(bufread.getData<BUFFER_BOUNDELEMENTS>());
		const float2 * const *vertPos = bufread.getRawPtr<BUFFER_VERTPOS>();
		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));

		uint numThreads = BLOCK_SIZE_SPS;
		// number of blocks, rounded up to next multiple of 4 to improve reductions
		uint numBlocks = round_up(div_up(particleRangeEnd, numThreads), 4U);

		effvisc_params<kerneltype, boundarytype, ViscSpec, simflags> params(
			pos, particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			deltap,
			gGam, vertPos,
			effvisc, cfl);

		cuvisc::effectiveViscDevice<<<numBlocks, numThreads>>>(params);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

		if (ViscSpec::rheologytype == GRANULAR)
			CUDA_SAFE_CALL(cudaUnbindTexture(effpresTex));

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
		const	float	deltap,
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
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius)
	{
		return calc_visc_implementation(bufread, bufwrite,
			numParticles, particleRangeEnd, deltap, slength, influenceradius, this);
	}

	/* First step of the Jacobi solver for the effective pressure:
	 * the Neuman homogeneous boundary condition in enforced on vertex particles
	 * interpolating the effective pressure from the free particles of sediment.
	*/
	template<typename This>
	enable_if_t<This::rheologytype != GRANULAR, void>
	enforce_jacobi_fs_boundary_conditions_implementation(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
		{ /* do nothing */ }

	template<typename This>
	enable_if_t<This::rheologytype == GRANULAR, void>
	enforce_jacobi_fs_boundary_conditions_implementation(
		const	BufferList& bufread,
			BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
	{
		const float4 *pos = bufread.getData<BUFFER_POS>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();

		float *effpres(bufwrite.getData<BUFFER_EFFPRES>());

		int dummy_shared = 0;
		// bind textures to read all particles, not only internal ones
#if !PREFER_L1
		CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
#endif
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);
#if (__COMPUTE__ == 20)
		dummy_shared = 2560;
#endif

		// Enforce FSboundary conditions
		cuvisc::jacobiFSBoundaryConditionsDevice
			<<<numBlocks, numThreads, dummy_shared>>>(pos, effpres, numParticles, deltap);

		KERNEL_CHECK_ERROR;

		// Unbind textures
		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
#if !PREFER_L1
		CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
#endif
	}

	void
	enforce_jacobi_fs_boundary_conditions(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius)
	{
		return enforce_jacobi_fs_boundary_conditions_implementation(
			bufread,
			bufwrite,
			numParticles,
			particleRangeEnd,
			deltap,
			slength,
			influenceradius,
			this);
	}



	template<typename This>
	enable_if_t<This::rheologytype != GRANULAR,float>
	enforce_jacobi_wall_boundary_conditions_implementation(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
		{ return NAN; /* do nothing */ }

	template<typename This>
	enable_if_t<This::rheologytype == GRANULAR, float >
	enforce_jacobi_wall_boundary_conditions_implementation(
		const	BufferList& bufread,
			BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
	{
		const float4 *pos = bufread.getData<BUFFER_POS>();
		const float4 *vel = bufread.getData<BUFFER_VEL>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();
		const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
		const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
		const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

		// for SA
		const	float4 *gGam = bufread.getData<BUFFER_GRADGAMMA>();
		const	float2 * const *vertPos = bufread.getRawPtr<BUFFER_VERTPOS>();
		const   float4  *boundelement(bufread.getData<BUFFER_BOUNDELEMENTS>());
		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));
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

		float *effpres(bufwrite.getData<BUFFER_EFFPRES>());

		int dummy_shared = 0;
		// bind textures to read all particles, not only internal ones
#if !PREFER_L1
		CUDA_SAFE_CALL(cudaBindTexture(0, posTex, pos, numParticles*sizeof(float4)));
#endif
		CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);
		numBlocks = round_up(numBlocks, 4U);
#if (__COMPUTE__ == 20)
		dummy_shared = 2560;
#endif

		effpres_params<kerneltype, boundarytype> params(
			pos, particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			deltap,
			gGam, vertPos,
			effpres, cfl);

		/* The backward error on vertex effective pressure is used as an additional
		 * stopping criterion (the residual being the main criterion). This helps in particular
		 * at the initialization step where A.x = B can be approximately verified when effective
		 * pressure is initialized to zero eveywhere.
		 */

		// Enforce boundary conditions from the previous time step
		cuvisc::jacobiWallBoundaryConditionsDevice<kerneltype, boundarytype>
			<<<numBlocks, numThreads, dummy_shared>>>(params);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		// Unbind textures
		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(velTex));

		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
#if !PREFER_L1
		CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
#endif
		return cflmax(numBlocks, cfl, tempCfl);
	}
	
	float
	enforce_jacobi_wall_boundary_conditions(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius)
	{
			
		return enforce_jacobi_wall_boundary_conditions_implementation(
			bufread,
			bufwrite,
			numParticles,
			particleRangeEnd,
			deltap,
			slength,
			influenceradius,
			this);
	}


	template<typename This>
	enable_if_t<This::rheologytype != GRANULAR,void>
	build_jacobi_vectors_implementation(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
	{ }

	template<typename This>
	enable_if_t<This::rheologytype == GRANULAR, void>
	build_jacobi_vectors_implementation(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
	{
		const float4 *pos = bufread.getData<BUFFER_POS>();
		const float4 *vel = bufread.getData<BUFFER_VEL>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();
		const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
		const uint *cellStart = bufread.getData<BUFFER_CELLSTART>(); 
		const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

		// for SA
		const	float4 *gGam = bufread.getData<BUFFER_GRADGAMMA>();
		const 	float2 * const *vertPos = bufread.getRawPtr<BUFFER_VERTPOS>();
		const   float4  *boundelement(bufread.getData<BUFFER_BOUNDELEMENTS>());
		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));

		const	float *effpres(bufread.getData<BUFFER_EFFPRES>());
		float4	*jacobiBuffer = bufwrite.getData<BUFFER_JACOBI>();

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


		CUDA_SAFE_CALL(cudaBindTexture(0, effpresTex, effpres, numParticles*sizeof(float)));
		effpres_params<kerneltype, boundarytype> params(
			pos, particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			deltap,
			gGam, vertPos,
			NULL, NULL);

		/* Jacobi solver
		 *---------------
		 * The problem A.x = B is solved with A
		 * a matrix decomposed in a diagonal matrix D
		 * and a remainder matrix R:
		 * 	A = D + R
		 * The variable Rx contains the vector resulting from the matrix
		 * vector product between R and x:
		 *	Rx = R.x
		 */
		float	*D; // vector containing the matrix M diagonal elements
		float	*Rx; // vector containing the result of R.x matrix-vector product
		float	*dB; // right hand-side device vector

		// Allocate GPU memory
		CUDA_SAFE_CALL(cudaMalloc((void**)&D, numParticles*sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&Rx, numParticles*sizeof(float)));
		CUDA_SAFE_CALL(cudaMalloc((void**)&dB, numParticles*sizeof(float)));

		// Copy the updated effpres (with Dirichlet and Neumann conditions enforced)
		// into effpresTex
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		CUDA_SAFE_CALL(cudaBindTexture(0, effpresTex, effpres, numParticles*sizeof(float)));

		// Build Jacobi vectors D, Rx and B.
		cuvisc::jacobiBuildVectorsDevice<kerneltype, boundarytype>
			<<<numBlocks, numThreads, dummy_shared>>>(params, D, Rx, dB);

		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// Copy Jacobi vectors D, Rx and B to jacobiBuffer 
		cuvisc::copyJacobiVectorsToJacobiBufferDevice
			<<<numBlocks, numThreads, dummy_shared>>>(params.numParticles, D, Rx, dB, jacobiBuffer);

		CUDA_SAFE_CALL(cudaDeviceSynchronize());

		// Free GPU memory
		cudaFree(D);
		cudaFree(Rx);
		cudaFree(dB);

		// Unbind textures
		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(effpresTex));
		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
#if !PREFER_L1
		CUDA_SAFE_CALL(cudaUnbindTexture(posTex));
#endif
	}

	void
	build_jacobi_vectors(
		const	BufferList& bufread,
				BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius)
	{
	build_jacobi_vectors_implementation(
		bufread,
		bufwrite,
		numParticles,
		particleRangeEnd,
		deltap,
		slength,
		influenceradius,
		this);

	 }


	template<typename This>
	enable_if_t<This::rheologytype != GRANULAR,float>
	update_jacobi_effpres_implementation(
		const	BufferList& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
	{ return 0.; /* do nothing */}

	template<typename This>
	enable_if_t<This::rheologytype == GRANULAR, float>
	update_jacobi_effpres_implementation(
		const	BufferList& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius,
		const	This *)
	{
		/* We recycle the CFL arrays to determine the maximum residual
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

		const particleinfo *info = bufread.getData<BUFFER_INFO>();

		float *effpres(bufwrite.getData<BUFFER_EFFPRES>());

		const float4 * jacobiBuffer = bufread.getData<BUFFER_JACOBI>();

		int dummy_shared = 0;
		// bind textures to read all particles, not only internal ones
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = round_up(div_up(particleRangeEnd, numThreads), 4U);
#if (__COMPUTE__ == 20)
		dummy_shared = 2560;
#endif

		/* Jacobi solver
		 *---------------
		 * The problem A.x = B is solved with A
		 * a matrix decomposed in a diagonal matrix D
		 * and a remainder matrix R:
		 * 	A = D + R
		 * The variable Rx contains the vector resulting from the matrix
		 * vector product between R and x:
		 *	Rx = R.x
		 */

		// Update effpres and compute the residual per particle
		cuvisc::jacobiUpdateEffPresDevice
			<<<numBlocks, numThreads, dummy_shared>>>(jacobiBuffer, effpres, cfl, numParticles);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		// Unbind textures
		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
		return cflmax(numBlocks, cfl, tempCfl);
	}

	float
	update_jacobi_effpres(
		const	BufferList& bufread,
		BufferList& bufwrite,
		const	uint	numParticles,
		const	uint	particleRangeEnd,
		const	float	deltap,
		const	float	slength,
		const	float	influenceradius)
	{
		return update_jacobi_effpres_implementation(
				bufread,
				bufwrite,
				numParticles,
				particleRangeEnd,
				deltap,
				slength,
				influenceradius,
				this);
	}


};

