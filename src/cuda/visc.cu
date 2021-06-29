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

#include "utils.h"
#include "engine_visc.h"
#include "safe_call.h"
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
		uint numThreads = BLOCK_SIZE_SPS;
		// number of blocks, rounded up to next multiple of 4 to improve reductions
		uint numBlocks = round_up(div_up(particleRangeEnd, numThreads), 4U);

		using effvisc_params = effvisc_params<kerneltype, boundarytype, ViscSpec, simflags>;

		execute_kernel(
			cuvisc::effectiveViscDevice<effvisc_params>(
				bufread, bufwrite,
				numParticles, slength, influenceradius,
				deltap),
			numBlocks, numThreads);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		/* We recycle the CFL arrays to determine the maximum kinematic viscosity
		 * in the adaptive timestepping case
		 */
		if (HAS_DTADAPT(This::simflags)) {
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
		int dummy_shared = 0;

		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);

#if (__COMPUTE__ == 20)
		dummy_shared = 2560;
#endif

		using sps_params = sps_params<kerneltype, boundarytype, (SPSK_STORE_TAU | SPSK_STORE_TURBVISC)>;
		execute_kernel(
			cuvisc::SPSstressMatrixDevice<kerneltype, boundarytype, (SPSK_STORE_TAU | SPSK_STORE_TURBVISC)>(
				bufread, bufwrite, numParticles, slength, influenceradius),
			numBlocks, numThreads, dummy_shared);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

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
		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);

		// Enforce FSboundary conditions
		execute_kernel(
			cuvisc::jacobiFSBoundaryConditionsDevice(bufread, bufwrite, numParticles, deltap),
			numBlocks, numThreads);

		KERNEL_CHECK_ERROR;
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
		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);
		numBlocks = round_up(numBlocks, 4U);

		/* The backward error on vertex effective pressure is used as an additional
		 * stopping criterion (the residual being the main criterion). This helps in particular
		 * at the initialization step where A.x = B can be approximately verified when effective
		 * pressure is initialized to zero eveywhere.
		 */

		// Enforce boundary conditions from the previous time step
		execute_kernel(
			cuvisc::jacobiWallBoundaryConditionsDevice<kerneltype, boundarytype>(
				bufread, bufwrite,
				numParticles, slength, influenceradius,
				deltap),
			numBlocks, numThreads);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

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
		uint numThreads = BLOCK_SIZE_SPS;
		uint numBlocks = div_up(particleRangeEnd, numThreads);

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

		using KP = jacobi_build_vectors_params<kerneltype, boundarytype>;

		// Build Jacobi vectors D, Rx and B.
		execute_kernel(
			cuvisc::jacobiBuildVectorsDevice<KP>(bufread, bufwrite,
				numParticles, slength, influenceradius, deltap),
			numBlocks, numThreads);

		KERNEL_CHECK_ERROR;
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
		execute_kernel(cuvisc::jacobiUpdateEffPresDevice(bufread, bufwrite, numParticles),
			numBlocks, numThreads);

		// check if kernel invocation generated an error
		KERNEL_CHECK_ERROR;

		return cflmax(numBlocks,
			bufwrite.getData<BUFFER_CFL>(),
			bufwrite.getData<BUFFER_CFL_TEMP>());
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

