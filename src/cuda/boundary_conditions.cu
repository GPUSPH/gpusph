/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

/* Boundary conditions engine implementation */

#include <stdio.h>
#include <stdexcept>

#include "engine_boundary_conditions.h"
#include "simflags.h"

#include "utils.h"
#include "safe_call.h"

#include "buffer.h"
#include "define_buffers.h"

#include "sa_bc_params.h"

// TODO Rename and optimize
#define BLOCK_SIZE_SA_BOUND		128
#define MIN_BLOCKS_SA_BOUND		6
#define BLOCK_SIZE_DUMMY_BOUND		128
#define MIN_BLOCKS_DUMMY_BOUND		6

#include "boundary_conditions_kernel.cu"


/// Boundary conditions computation for boundary particles is a no-op for all cases
/// except DUMMY_BOUNDARY. Again, auxiliary functor does the job, to allow
/// partial specialization

/// General case: do nothing
template<KernelType kerneltype, BoundaryType boundarytype>
struct CUDABoundaryHelper {
	static void
	process(
		BufferList const& bufread,
		BufferList &bufwrite,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius)
	{ /* do nothing by default */ }
};

/// DUMMY_BOUNDARY specialization: compute pressure on boundary particles
/// from a Shepard-filtered average of the neighboring fluid particles.
/// The density of the neighbors (to compute the pressure) is taken from
/// the WRITE buffer, which is updated in-place, storing the density which
/// would give the smoothed pressure in vel.w
/// Boundary particles velocity are also computed as a Shepard-filtered average
/// of the velocity of the neighboring fluid particles, to give no-slip boundary
/// conditions.
template<KernelType kerneltype>
struct CUDABoundaryHelper<kerneltype, DUMMY_BOUNDARY> {
	static void process(
		BufferList const& bufread,
		BufferList &bufwrite,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius)
{
	uint numThreads = BLOCK_SIZE_DUMMY_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	execute_kernel(
		cubounds::ComputeDummyParticlesDevice<kerneltype>
			(bufread, bufwrite, particleRangeEnd, slength, influenceradius),
		numBlocks, numThreads);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}
};

/// Boundary conditions engines

// TODO FIXME at this time this is just a horrible hack to group the boundary-conditions
// methods needed for SA, it needs a heavy-duty refactoring of course

template<KernelType kerneltype, typename ViscSpec,
	BoundaryType boundarytype, flag_t simflags>
class CUDABoundaryConditionsEngine : public AbstractBoundaryConditionsEngine
{
public:

/// Set the number of open-boundary vertices present in the whole simulation
/*! This value is computed on host once (at the beginning of the simulation)
 * and it is then uploaded to all devices, which will use it to compute the
 * new IDs.
 */
virtual void
uploadNumOpenVertices(const uint &numOpenVertices) override
{
	COPY_TO_SYMBOL(cubounds::d_numOpenVertices, numOpenVertices, 1);
}

/// Disables particles that went through boundaries when open boundaries are used
void
disableOutgoingParts(const	BufferList& bufread,
							BufferList& bufwrite,
					const	uint			numParticles,
					const	uint			particleRangeEnd) override
{
	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	//execute kernel
	execute_kernel(
		cubounds::disableOutgoingPartsDevice(bufread, bufwrite, particleRangeEnd),
		numBlocks, numThreads);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

/// Compute boundary conditions
void
compute_boundary_conditions(
		BufferList const& bufread,
		BufferList &bufwrite,
				uint	numParticles,
				uint	particleRangeEnd,
				float	slength,
				float	influenceradius) override
{
	CUDABoundaryHelper<kerneltype, boundarytype>::process
		(bufread, bufwrite, numParticles, particleRangeEnd,
		 slength, influenceradius);
	return;
}
//! SFINAE implementation of saSegmentBoundaryConditions
/** Due to the limited or non-existant support for kernels different from Wendland
 * for semi-analytical boundary conditions, we want to avoid compiling the SA boundary
 * conditions methods altogether. The implementation is thus refactored into methods
 * that can be SFINAEd, called by the public interface.
 */
template<BoundaryType _boundarytype>
enable_if_t<_boundarytype == SA_BOUNDARY>
saSegmentBoundaryConditionsImpl(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	// step will be 0 for the initialization step,
	// and 1 or 2 for the first and second step during integration
	const	int			step,
	const	RunMode		run_mode)
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
#define SA_SEGMENT_BC_STEP(step) case step: \
	if (run_mode == REPACK) { \
		using RepackParams = sa_segment_bc_repack_params<kerneltype, ViscSpec, simflags, step>; \
		execute_kernel( \
			cubounds::saSegmentBoundaryConditionsRepackDevice<RepackParams>( \
				bufread, bufwrite, particleRangeEnd, deltap, slength, influenceradius), \
		numBlocks, numThreads, dummy_shared); \
	} else { \
		using RunParams = sa_segment_bc_params<kerneltype, ViscSpec, simflags, step>; \
		execute_kernel( \
			cubounds::saSegmentBoundaryConditionsRepackDevice<RunParams>( \
				bufread, bufwrite, particleRangeEnd, deltap, slength, influenceradius), \
		numBlocks, numThreads, dummy_shared); \
	} \
	break;

	switch (step) {
	case -1: // step -1 is the same as step 0 (initialization, but at the end of the repacking
		SA_SEGMENT_BC_STEP(0);
		SA_SEGMENT_BC_STEP(1);
		SA_SEGMENT_BC_STEP(2);
	default:
		throw std::runtime_error("unsupported step");
	}
	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

//! Non-SA case for the implementation of saSegmentBoundaryConditions
/** In this case, we should never be called, so throw
 */
template<BoundaryType _boundarytype>
enable_if_t<_boundarytype != SA_BOUNDARY>
saSegmentBoundaryConditionsImpl(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	// step will be 0 for the initialization step,
	// and 1 or 2 for the first and second step during integration
	const	int			step,
	const	RunMode		run_mode)
{
	throw std::runtime_error("saSegmentBoundaryConditions called without SA_BOUNDARY");
}

/// Computes the boundary conditions on segments using the information from the fluid
/** For solid walls this is used to impose Neuman boundary conditions.
 *  For open boundaries it imposes the appropriate inflow velocity solving the associated
 *  Riemann problem.
 */
void
saSegmentBoundaryConditions(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	// step will be 0 for the initialization step,
	// and 1 or 2 for the first and second step during integration
	const	int			step,
	const	RunMode		run_mode) override
{
	saSegmentBoundaryConditionsImpl<boundarytype>(bufwrite, bufread, numParticles,
		particleRangeEnd, deltap, slength, influenceradius, step, run_mode);
}

/// Detect particles that cross an open boundary and find the boundary element they have crossed
void
findOutgoingSegment(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius) override
{
	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	execute_kernel(
		cubounds::findOutgoingSegmentDevice<kerneltype>(
			bufread, bufwrite, particleRangeEnd, slength, influenceradius),
		numBlocks, numThreads);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

//! SFINAE implementation of saVertexBoundaryConditions
/** Due to the limited or non-existant support for kernels different from Wendland
 * for semi-analytical boundary conditions, we want to avoid compiling the SA boundary
 * conditions methods altogether. The implementation is thus refactored into methods
 * that can be SFINAEd, called by the public interface.
 */
template<BoundaryType _boundarytype>
enable_if_t<_boundarytype == SA_BOUNDARY>
saVertexBoundaryConditionsImpl(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	// step will be 0 for the initialization step,
	// and 1 or 2 for the first and second step during integration
	const	int				step,
	const	bool			resume, // TODO FIXME check if still needed
	const	float			dt, // for open boundaries
	// These are the cloning-related members
			uint*			newNumParticles,
	const	uint			deviceId,
	const	uint			numDevices,
	const	uint			totParticles,
	const	RunMode			run_mode)
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
#define SA_VERTEX_BC_STEP(step) case step: \
	if (run_mode == REPACK) { \
		using RepackParams = sa_vertex_bc_repack_params<kerneltype, ViscSpec, simflags, step>; \
		execute_kernel( \
			cubounds::saVertexBoundaryConditionsRepackDevice<RepackParams>( \
				bufread, bufwrite, newNumParticles, numParticles, totParticles, \
				deltap, slength, influenceradius, deviceId, numDevices, dt), \
		numBlocks, numThreads, dummy_shared); \
	} else { \
		using RunParams = sa_vertex_bc_params<kerneltype, ViscSpec, simflags, step>; \
		execute_kernel( \
			cubounds::saVertexBoundaryConditionsDevice<RunParams>( \
				bufread, bufwrite, newNumParticles, numParticles, totParticles, \
				deltap, slength, influenceradius, deviceId, numDevices, dt), \
		numBlocks, numThreads, dummy_shared); \
	} \
	break;

	switch (step) {
		case -1: // step -1 is the same as step 0 (initialization, but at the end of the repacking
			SA_VERTEX_BC_STEP(0);
			SA_VERTEX_BC_STEP(1);
			SA_VERTEX_BC_STEP(2);
		default:
			throw std::runtime_error("unsupported step");
	}
	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

template<BoundaryType _boundarytype>
enable_if_t<_boundarytype != SA_BOUNDARY>
saVertexBoundaryConditionsImpl(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	// step will be 0 for the initialization step,
	// and 1 or 2 for the first and second step during integration
	const	int				step,
	const	bool			resume, // TODO FIXME check if still needed
	const	float			dt, // for open boundaries
	// These are the cloning-related members
			uint*			newNumParticles,
	const	uint			deviceId,
	const	uint			numDevices,
	const	uint			totParticles,
	const	RunMode			run_mode)
{
	throw std::runtime_error("saVertexBoundaryConditions called without SA_BOUNDARY");
}

/// Apply boundary conditions to vertex particles.
// There is no need to use two velocity arrays (read and write) and swap them after.
// Computes the boundary conditions on vertex particles using the values from the segments associated to it. Also creates particles for inflow boundary conditions.
// Data is only read from fluid and segments and written only on vertices.
void
saVertexBoundaryConditions(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	// step will be 0 for the initialization step,
	// and 1 or 2 for the first and second step during integration
	const	int				step,
	const	bool			resume, // TODO FIXME check if still needed
	const	float			dt, // for open boundaries
	// These are the cloning-related members
			uint*			newNumParticles,
	const	uint			deviceId,
	const	uint			numDevices,
	const	uint			totParticles,
	const RunMode			run_mode) override
{
	saVertexBoundaryConditionsImpl<boundarytype>(bufwrite, bufread,
		numParticles, particleRangeEnd, deltap, slength, influenceradius,
		step, resume, dt,
		newNumParticles, deviceId, numDevices, totParticles, run_mode);
}

/// Compute normal for vertices in initialization step
/*! This kernel updates BUFFER_BOUNDELEMENTS,
 *  computing the normals for each vertex as the average of the normals
 *  of the adjacent boundary elements, weighted by the respective surface.
 *  Since we only write the vertex normals and only read the boundary normals,
 *  the update can be done in-place
 */
void
computeVertexNormal(
	BufferList const&	bufread,
	BufferList&		bufwrite,
	const	uint			numParticles,
	const	uint			particleRangeEnd) override
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
	execute_kernel(
		cubounds::computeVertexNormalDevice<kerneltype>(bufread, bufwrite, particleRangeEnd),
		numBlocks, numThreads, dummy_shared);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

//! SFINAE implementation of saInitGamma
/** Due to the limited or non-existant support for kernels different from Wendland
 * for semi-analytical boundary conditions, we want to avoid compiling the SA boundary
 * conditions methods altogether. The implementation is thus refactored into methods
 * that can be SFINAEd, called by the public interface.
 */
template<BoundaryType _boundarytype>
enable_if_t<_boundarytype == SA_BOUNDARY>
saInitGammaImpl(
	BufferList const&	bufread,
	BufferList&		bufwrite,
	const	float			slength,
	const	float			influenceradius,
	const	float			deltap,
	const	float			epsilon,
	const	uint			numParticles,
	const	uint			particleRangeEnd)
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel for fluid particles
	execute_kernel(
		cubounds::initGammaDevice<kerneltype, PT_FLUID>(
			bufread, bufwrite, particleRangeEnd, slength, influenceradius,
			deltap, epsilon),
		numBlocks, numThreads, dummy_shared);

	// TODO verify if this split kernele execution works in the multi-device case,
	// or if we need to update_external the fluid data first

	// execute the kernel for vertex particles
	execute_kernel(
		cubounds::initGammaDevice<kerneltype, PT_VERTEX>(
			bufread, bufwrite, particleRangeEnd, slength, influenceradius,
			deltap, epsilon),
		numBlocks, numThreads, dummy_shared);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

template<BoundaryType _boundarytype>
enable_if_t<_boundarytype != SA_BOUNDARY>
saInitGammaImpl(
	BufferList const&	bufread,
	BufferList&		bufwrite,
	const	float			slength,
	const	float			influenceradius,
	const	float			deltap,
	const	float			epsilon,
	const	uint			numParticles,
	const	uint			particleRangeEnd)
{
	throw std::runtime_error("saInitGamma called without SA_BOUNDARY");
}


/// Initialize gamma
void
saInitGamma(
	BufferList const&	bufread,
	BufferList&		bufwrite,
	const	float			slength,
	const	float			influenceradius,
	const	float			deltap,
	const	float			epsilon,
	const	uint			numParticles,
	const	uint			particleRangeEnd) override
{
	saInitGammaImpl<boundarytype>(bufread, bufwrite,
		slength, influenceradius, deltap, epsilon,
		numParticles, particleRangeEnd);
}


// counts vertices that belong to IO and same segment as other IO vertex
void
initIOmass_vertexCount(
	BufferList& bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd) override
{
	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
	execute_kernel(
		cubounds::initIOmass_vertexCountDevice<kerneltype>(bufread, bufwrite, particleRangeEnd),
		numBlocks, numThreads, dummy_shared);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

/// Adjusts the initial mass of vertex particles on open boundaries
void
initIOmass(
	BufferList& bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap) override
{
	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
	execute_kernel(cubounds::initIOmassDevice<kerneltype>(bufread, bufwrite, particleRangeEnd, deltap),
		numBlocks, numThreads, dummy_shared);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

// Downloads the per device waterdepth from the GPU
void
downloadIOwaterdepth(
			uint*	h_IOwaterdepth,
	const	uint*	d_IOwaterdepth,
	const	uint	numOpenBoundaries) override
{
	SAFE_CALL(cudaMemcpy(h_IOwaterdepth, d_IOwaterdepth, numOpenBoundaries*sizeof(int), cudaMemcpyDeviceToHost));
}


// Upload the global waterdepth to the GPU
void
uploadIOwaterdepth(
	const	uint*	h_IOwaterdepth,
			uint*	d_IOwaterdepth,
	const	uint	numOpenBoundaries) override
{
	SAFE_CALL(cudaMemcpy(d_IOwaterdepth, h_IOwaterdepth, numOpenBoundaries*sizeof(int), cudaMemcpyHostToDevice));
}

// Identifies vertices at the corners of open boundaries
void
saIdentifyCornerVertices(
	const	BufferList&	bufread,
			BufferList&	bufwrite,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			eps) override
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
	execute_kernel(
		cubounds::saIdentifyCornerVerticesDevice(bufread, bufwrite, particleRangeEnd, deltap, eps),
		numBlocks, numThreads, dummy_shared);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}
};
