/*  Copyright 2015 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/* Boundary conditions engine implementation */

#include <stdio.h>
#include <stdexcept>

#include "textures.cuh"

#include "engine_boundary_conditions.h"
#include "simflags.h"

#include "utils.h"
#include "cuda_call.h"

#include "buffer.h"
#include "define_buffers.h"

#include "sa_bc_params.h"

// TODO Rename and optimize
#define BLOCK_SIZE_SA_BOUND		128
#define MIN_BLOCKS_SA_BOUND		6

#include "boundary_conditions_kernel.cu"

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
uploadNumOpenVertices(const uint &numOpenVertices)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_numOpenVertices, &numOpenVertices, sizeof(uint)));
}

/// Disables particles that went through boundaries when open boundaries are used
void
disableOutgoingParts(const	BufferList& bufread,
							BufferList& bufwrite,
					const	uint			numParticles,
					const	uint			particleRangeEnd)
{
	const particleinfo *info = bufread.getData<BUFFER_INFO>();
	float4 *pos = bufwrite.getData<BUFFER_POS>();
	// We abuse the VERTICES array, which is otherwise unused by fluid particles,
	// to store the vertices of the boundary element crossed by outgoing particles.
	// Since the VERTICES array is shared across states unless we also have moving
	// objects, accessing it for writing here would not be allowed; but we know
	// we can do it anyway, so we use the “unsafe” version of getData
	vertexinfo *vertices = bufwrite.getData<BUFFER_VERTICES,
		BufferList::AccessSafety::MULTISTATE_SAFE>();

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	//execute kernel
	cubounds::disableOutgoingPartsDevice<<<numBlocks, numThreads>>>
		(	pos,
			vertices,
			numParticles);

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
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

	const	float4			*pos(bufread.getData<BUFFER_POS>());
	const	particleinfo	*info(bufread.getData<BUFFER_INFO>());

	const	hashKey			*particleHash(bufread.getData<BUFFER_HASH>());
	const	uint			*cellStart(bufread.getData<BUFFER_CELLSTART>());
	const	neibdata		*neibsList(bufread.getData<BUFFER_NEIBSLIST>());
	const	float2	* const *vertPos(bufread.getRawPtr<BUFFER_VERTPOS>());
	const	float4	*boundelement(bufread.getData<BUFFER_BOUNDELEMENTS>());
	const	vertexinfo	*vertices(bufread.getData<BUFFER_VERTICES>());

	float4	*vel(bufwrite.getData<BUFFER_VEL>());
	float	*tke(bufwrite.getData<BUFFER_TKE>());
	float	*eps(bufwrite.getData<BUFFER_EPSILON>());
	float4	*eulerVel(bufwrite.getData<BUFFER_EULERVEL>());
	float4  *gGam(bufwrite.getData<BUFFER_GRADGAMMA>());

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
#define SA_SEGMENT_BC_STEP(step) case step: \
	if (run_mode == REPACK) { \
		sa_segment_bc_repack_params<kerneltype, ViscSpec, simflags, step> params( \
			pos, vel, particleHash, cellStart, neibsList, \
			gGam, vertices, vertPos, \
			eulerVel, tke, eps, \
			particleRangeEnd, deltap, slength, influenceradius); \
		cubounds::saSegmentBoundaryConditionsRepackDevice<<< numBlocks, numThreads, dummy_shared >>>(params); \
	} else { \
		sa_segment_bc_params<kerneltype, ViscSpec, simflags, step> params( \
			pos, vel, particleHash, cellStart, neibsList, \
			gGam, vertices, vertPos, \
			eulerVel, tke, eps, \
			particleRangeEnd, deltap, slength, influenceradius); \
		cubounds::saSegmentBoundaryConditionsDevice<<< numBlocks, numThreads, dummy_shared >>>(params); \
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

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

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
	const	RunMode		run_mode)
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

	const	float4			*pos(bufread.getData<BUFFER_POS>());
	const	float4			*vel(bufread.getData<BUFFER_VEL>());
	const	particleinfo	*info(bufread.getData<BUFFER_INFO>());
	const	hashKey			*particleHash(bufread.getData<BUFFER_HASH>());
	const	uint			*cellStart(bufread.getData<BUFFER_CELLSTART>());
	const	neibdata		*neibsList(bufread.getData<BUFFER_NEIBSLIST>());
	const	float2	* const *vertPos(bufread.getRawPtr<BUFFER_VERTPOS>());
	const	float4	*boundelement(bufread.getData<BUFFER_BOUNDELEMENTS>());

	float4  *gGam(bufwrite.getData<BUFFER_GRADGAMMA>());
	// See note about vertices in disableOutgoingParts
	vertexinfo	*vertices(bufwrite.getData<BUFFER_VERTICES,
		BufferList::AccessSafety::MULTISTATE_SAFE>());

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	cubounds::findOutgoingSegmentDevice<kerneltype><<<numBlocks, numThreads>>>(
		pos, vel, vertices, gGam,
		vertPos[0], vertPos[1], vertPos[2],
		particleHash, cellStart, neibsList,
		particleRangeEnd, influenceradius);

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

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

	const	float4	*boundelement(bufread.getData<BUFFER_BOUNDELEMENTS>());
	const	particleinfo	*info(bufread.getData<BUFFER_INFO>());

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
#define SA_VERTEX_BC_STEP(step) case step: \
	if (run_mode == REPACK) { \
		sa_vertex_bc_repack_params<kerneltype, ViscSpec, simflags, step> params( \
				bufread, bufwrite, newNumParticles, numParticles, totParticles, \
				deltap, slength, influenceradius, deviceId, numDevices, dt); \
		cubounds::saVertexBoundaryConditionsRepackDevice<<< numBlocks, numThreads, dummy_shared >>>(params); \
	} else { \
		sa_vertex_bc_params<kerneltype, ViscSpec, simflags, step> params( \
				bufread, bufwrite, newNumParticles, numParticles, totParticles, \
				deltap, slength, influenceradius, deviceId, numDevices, dt); \
		cubounds::saVertexBoundaryConditionsDevice<<< numBlocks, numThreads, dummy_shared >>>(params); \
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

	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

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
	const RunMode   		run_mode)
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
	const	uint			particleRangeEnd)
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	float4 *boundelement = bufwrite.getData<BUFFER_BOUNDELEMENTS>();

	const vertexinfo *vertices = bufread.getData<BUFFER_VERTICES>();
	const particleinfo *pinfo = bufread.getData<BUFFER_INFO>();
	const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
	const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
	const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
	cubounds::computeVertexNormalDevice<kerneltype><<< numBlocks, numThreads, dummy_shared >>> (
		boundelement,
		vertices,
		pinfo,
		particleHash,
		cellStart,
		neibsList,
		particleRangeEnd);

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

	float4 *newGGam = bufwrite.getData<BUFFER_GRADGAMMA>();
	const float4 *oldGGam = bufread.getData<BUFFER_GRADGAMMA>();

	const float4 *oldPos = bufread.getData<BUFFER_POS>();
	const float4 *boundelement = bufread.getData<BUFFER_BOUNDELEMENTS>();
	const particleinfo *pinfo = bufread.getData<BUFFER_INFO>();
	const hashKey *particleHash = bufread.getData<BUFFER_HASH>();
	const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
	const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();
	const float2 * const *vertPos = bufread.getRawPtr<BUFFER_VERTPOS>();

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel for fluid particles
	cubounds::initGammaDevice<kerneltype, PT_FLUID><<< numBlocks, numThreads, dummy_shared >>> (
		newGGam,
		oldGGam,
		oldPos,
		boundelement,
		vertPos[0],
		vertPos[1],
		vertPos[2],
		pinfo,
		particleHash,
		cellStart,
		neibsList,
		slength,
		influenceradius,
		deltap,
		epsilon,
		particleRangeEnd);

	// TODO verify if this split kernele execution works in the multi-device case,
	// or if we need to update_external the fluid data first

	// execute the kernel for vertex particles
	cubounds::initGammaDevice<kerneltype, PT_VERTEX><<< numBlocks, numThreads, dummy_shared >>> (
		newGGam,
		oldGGam,
		oldPos,
		boundelement,
		vertPos[0],
		vertPos[1],
		vertPos[2],
		pinfo,
		particleHash,
		cellStart,
		neibsList,
		slength,
		influenceradius,
		deltap,
		epsilon,
		particleRangeEnd);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
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
	const	uint			particleRangeEnd)
{
	saInitGammaImpl<boundarytype>(bufread, bufwrite,
		slength, influenceradius, deltap, epsilon,
		numParticles, particleRangeEnd);
}


// counts vertices that belong to IO and same segment as other IO vertex
virtual
void
initIOmass_vertexCount(
	BufferList& bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd)
{
	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	const particleinfo *info = bufread.getData<BUFFER_INFO>();
	const hashKey *pHash = bufread.getData<BUFFER_HASH>();
	const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
	const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();
	const vertexinfo *vertices = bufread.getData<BUFFER_VERTICES>();
	float4 *forces = bufwrite.getData<BUFFER_FORCES>();

	// execute the kernel
	cubounds::initIOmass_vertexCountDevice<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
		(vertices, pHash, info, cellStart, neibsList, forces, particleRangeEnd);

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
	const	float			deltap)
{
	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	const float4 *oldPos = bufread.getData<BUFFER_POS>();
	const float4 *forces = bufread.getData<BUFFER_FORCES>();
	const particleinfo *info = bufread.getData<BUFFER_INFO>();
	const hashKey *pHash = bufread.getData<BUFFER_HASH>();
	const uint *cellStart = bufread.getData<BUFFER_CELLSTART>();
	const neibdata *neibsList = bufread.getData<BUFFER_NEIBSLIST>();
	const vertexinfo *vertices = bufread.getData<BUFFER_VERTICES>();

	float4 *newPos = bufwrite.getData<BUFFER_POS>();

	// execute the kernel
	cubounds::initIOmassDevice<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
		(oldPos, forces, vertices, pHash, info, cellStart, neibsList, newPos, particleRangeEnd, deltap);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

// Downloads the per device waterdepth from the GPU
void
downloadIOwaterdepth(
			uint*	h_IOwaterdepth,
	const	uint*	d_IOwaterdepth,
	const	uint	numOpenBoundaries)
{
	CUDA_SAFE_CALL(cudaMemcpy(h_IOwaterdepth, d_IOwaterdepth, numOpenBoundaries*sizeof(int), cudaMemcpyDeviceToHost));
}


// Upload the global waterdepth to the GPU
void
uploadIOwaterdepth(
	const	uint*	h_IOwaterdepth,
			uint*	d_IOwaterdepth,
	const	uint	numOpenBoundaries)
{
	CUDA_SAFE_CALL(cudaMemcpy(d_IOwaterdepth, h_IOwaterdepth, numOpenBoundaries*sizeof(int), cudaMemcpyHostToDevice));
}

// Identifies vertices at the corners of open boundaries
void
saIdentifyCornerVertices(
	const	BufferList&	bufread,
			BufferList&	bufwrite,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			eps)
{
	const float4* oldPos = bufread.getData<BUFFER_POS>();
	const float4* boundelement = bufread.getData<BUFFER_BOUNDELEMENTS>();
	const hashKey* particleHash = bufread.getData<BUFFER_HASH>();
	const vertexinfo* vertices = bufread.getData<BUFFER_VERTICES>();
	const uint* cellStart = bufread.getData<BUFFER_CELLSTART>();
	const neibdata* neibsList = bufread.getData<BUFFER_NEIBSLIST>();

	particleinfo*	info = bufwrite.getData<BUFFER_INFO>();

	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif
	// execute the kernel
	cubounds::saIdentifyCornerVerticesDevice<<< numBlocks, numThreads, dummy_shared >>> (
		oldPos,
		info,
		particleHash,
		vertices,
		cellStart,
		neibsList,
		numParticles,
		deltap,
		eps);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

}
};
