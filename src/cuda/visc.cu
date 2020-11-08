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

		const float4 *vel = bufread.getData<BUFFER_VEL>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();
		const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
		const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
		const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

		CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

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
			bufread, bufwrite,
			particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			deltap,
			gGam, vertPos,
			effvisc);

		cuvisc::effectiveViscDevice<<<numBlocks, numThreads>>>(params);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(velTex));

		/* We recycle the CFL arrays to determine the maximum kinematic viscosity
		 * in the adaptive timestepping case
		 */
		if (This::simflags & ENABLE_DTADAPT) {
			return cflmax(numBlocks,
				bufwrite.getData<BUFFER_CFL>(),
				bufwrite.getData<BUFFER_CFL_TEMP>());
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

		const float4 *vel = bufread.getData<BUFFER_VEL>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();
		const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
		const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
		const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

		int dummy_shared = 0;
		// bind textures to read all particles, not only internal ones
		CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);

#if (__COMPUTE__ == 20)
		dummy_shared = 2560;
#endif

		sps_params<kerneltype, boundarytype, (SPSK_STORE_TAU | SPSK_STORE_TURBVISC)> params(
			bufread, particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			tau[0], tau[1], tau[2], turbvisc);

		cuvisc::SPSstressMatrixDevice<kerneltype, boundarytype, (SPSK_STORE_TAU | SPSK_STORE_TURBVISC)>
			<<<numBlocks, numThreads, dummy_shared>>>(params);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(velTex));

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
	 * the Dirichlet condition is enforced of fluid particle at the free-surface or
	 * at the interface. This is run only once before the iterative loop.
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
		const particleinfo *info = bufread.getData<BUFFER_INFO>();

		float *effpres(bufwrite.getData<BUFFER_EFFPRES>());

		int dummy_shared = 0;
		// bind textures to read all particles, not only internal ones
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);

		// Enforce FSboundary conditions
		cuvisc::jacobiFSBoundaryConditionsDevice
			<<<numBlocks, numThreads, dummy_shared>>>(pos_wrapper(bufread), effpres, numParticles, deltap);

		KERNEL_CHECK_ERROR;

		// Unbind textures
		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
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

	/* Second step of the Jacobi solver.
	 * The Neuman homogeneous boundary condition in enforced on boundary particles 
	 * (vertex for SA) interpolating the effective pressure from the free particles of sediment.
	 * This is run once before the itrative loop, and at the end of every iteration.
	 * This returns a float being the backward error, used to evaluate the convergence at boundaries.
	*/
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
		const float4 *vel = bufread.getData<BUFFER_VEL>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();
		const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
		const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
		const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

		// for SA
		const	float4 *gGam = bufread.getData<BUFFER_GRADGAMMA>();
		const	float2 * const *vertPos = bufread.getRawPtr<BUFFER_VERTPOS>();
		const   float4  *boundelement(bufread.getData<BUFFER_BOUNDELEMENTS>());

		float *effpres(bufwrite.getData<BUFFER_EFFPRES>());

		int dummy_shared = 0;
		// bind textures to read all particles, not only internal ones
		CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));

		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);
		numBlocks = round_up(numBlocks, 4U);

		jacobi_wall_boundary_params<kerneltype, boundarytype> params(
			bufread, bufwrite,
			particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			deltap,
			gGam, vertPos,
			effpres);

		/* The backward error on vertex effective pressure is used as an additional
		 * stopping criterion (the residual being the main criterion). This helps in particular
		 * at the initialization step where A.x = B can be approximately verified when effective
		 * pressure is initialized to zero eveywhere.
		 */

		// Enforce boundary conditions from the previous time step
		cuvisc::jacobiWallBoundaryConditionsDevice
			<<<numBlocks, numThreads, dummy_shared>>>(params);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		// Unbind textures
		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(velTex));

		return cflmax(numBlocks,
			bufwrite.getData<BUFFER_CFL>(),
			bufwrite.getData<BUFFER_CFL_TEMP>());
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
		const float4 *vel = bufread.getData<BUFFER_VEL>();
		const particleinfo *info = bufread.getData<BUFFER_INFO>();
		const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
		const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
		const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

		// for SA
		const	float4 *gGam = bufread.getData<BUFFER_GRADGAMMA>();
		const	float2 * const *vertPos = bufread.getRawPtr<BUFFER_VERTPOS>();
		const   float4  *boundelement(bufread.getData<BUFFER_BOUNDELEMENTS>());

		const	float *effpres(bufread.getData<BUFFER_EFFPRES>());
		float4	*jacobiBuffer = bufwrite.getData<BUFFER_JACOBI>();

		int dummy_shared = 0;
		// bind textures to read all particles, not only internal ones
		CUDA_SAFE_CALL(cudaBindTexture(0, velTex, vel, numParticles*sizeof(float4)));
		CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));
		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));


		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);

		jacobi_build_vectors_params<kerneltype, boundarytype> params(
			bufread,
			particleHash, cellStart, neibsList, numParticles, slength, influenceradius,
			deltap,
			gGam, vertPos);

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

		// Build Jacobi vectors D, Rx and B.
		cuvisc::jacobiBuildVectorsDevice
			<<<numBlocks, numThreads, dummy_shared>>>(params, jacobiBuffer);

		KERNEL_CHECK_ERROR;

		// Unbind textures
		if (boundarytype == SA_BOUNDARY)
			CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
		CUDA_SAFE_CALL(cudaUnbindTexture(velTex));
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

