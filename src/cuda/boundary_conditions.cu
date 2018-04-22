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

#include "define_buffers.h"

#include "sa_segment_bc_params.h"

// TODO Rename and optimize
#define BLOCK_SIZE_SA_BOUND		128
#define MIN_BLOCKS_SA_BOUND		6

#include "boundary_conditions_kernel.cu"

/// Boundary conditions engines

// TODO FIXME at this time this is just a horrible hack to group the boundary-conditions
// methods needed for SA, it needs a heavy-duty refactoring of course

template<KernelType kerneltype, ViscosityType visctype,
	BoundaryType boundarytype, flag_t simflags>
class CUDABoundaryConditionsEngine : public AbstractBoundaryConditionsEngine
{
public:

void
updateNewIDsOffset(const uint &newIDsOffset)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cubounds::d_newIDsOffset, &newIDsOffset, sizeof(uint)));
}

/// Disables particles that went through boundaries when open boundaries are used
void
disableOutgoingParts(		float4*			pos,
							vertexinfo*		vertices,
					const	particleinfo*	info,
					const	uint			numParticles,
					const	uint			particleRangeEnd)
{
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

/// Computes the boundary conditions on segments using the information from the fluid (on solid walls used for Neumann boundary conditions).
void
saSegmentBoundaryConditions(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint*			cellStart,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	// step will be 0 for the initialization step,
	// and 1 or 2 for the first and second step during integration
	const	uint			step)
{
	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	// TODO we take pos from bufwrite, but it's actually read-only for us
	const	float4			*pos(bufwrite.getData<BUFFER_POS>());
	const	particleinfo	*info(bufread.getData<BUFFER_INFO>());
	const	hashKey			*particleHash(bufread.getData<BUFFER_HASH>());
	const	neibdata		*neibsList(bufread.getData<BUFFER_NEIBSLIST>());
	const	float2	* const *vertPos(bufread.getRawPtr<BUFFER_VERTPOS>());
	const	float4	*boundelement(bufread.getData<BUFFER_BOUNDELEMENTS>());

	float4	*vel(bufwrite.getData<BUFFER_VEL>());
	float	*tke(bufwrite.getData<BUFFER_TKE>());
	float	*eps(bufwrite.getData<BUFFER_EPSILON>());
	float4	*eulerVel(bufwrite.getData<BUFFER_EULERVEL>());
	float4  *gGam(bufwrite.getData<BUFFER_GRADGAMMA>());
	vertexinfo	*vertices(bufwrite.getData<BUFFER_VERTICES>());

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	sa_segment_bc_params<visctype, simflags> params(
		pos, vel, particleHash, cellStart, neibsList,
		gGam, vertices, vertPos,
		eulerVel, tke, eps,
		particleRangeEnd, deltap, slength, influenceradius);

	// execute the kernel
#define SA_SEGMENT_BC_STEP(step) case step: \
	cubounds::saSegmentBoundaryConditionsDevice<kerneltype, step><<< numBlocks, numThreads, dummy_shared >>>(params); break

	switch (step) {
		SA_SEGMENT_BC_STEP(0);
		SA_SEGMENT_BC_STEP(1);
		SA_SEGMENT_BC_STEP(2);
	default:
		throw std::runtime_error("unsupported step");
	}

	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));
	CUDA_SAFE_CALL(cudaUnbindTexture(infoTex));

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

/// Apply boundary conditions to vertex particles.
// There is no need to use two velocity arrays (read and write) and swap them after.
// Computes the boundary conditions on vertex particles using the values from the segments associated to it. Also creates particles for inflow boundary conditions.
// Data is only read from fluid and segments and written only on vertices.
void
saVertexBoundaryConditions(
			float4*			oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			float4*			oldGGam,
			float4*			oldEulerVel,
			float4*			forces,
	const	float4*			boundelement,
			vertexinfo*		vertices,
	const	float2			* const vertPos[],
			particleinfo*	info,
			hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
			uint*			newNumParticles,
	const	uint			particleRangeEnd,
	const	float			dt,
	const	int				step,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	const	bool			initStep,
	const	bool			resume,
	const	uint			deviceId,
	const	uint			numDevices,
	const	uint			totParticles)
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
	cubounds::saVertexBoundaryConditions<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
		(oldPos, oldVel, oldTKE, oldEps, oldGGam, oldEulerVel, forces, vertices, vertPos[0], vertPos[1], vertPos[2], info, particleHash, cellStart, neibsList,
		 particleRangeEnd, newNumParticles, dt, step, deltap, slength, influenceradius, initStep, resume, deviceId, numDevices);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

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
	MultiBufferList::const_iterator	bufread,
	MultiBufferList::iterator		bufwrite,
	const	uint*			cellStart,
	const	uint			numParticles,
	const	uint			particleRangeEnd)
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	float4 *boundelement = bufwrite->getData<BUFFER_BOUNDELEMENTS>();

	const vertexinfo *vertices = bufread->getData<BUFFER_VERTICES>();
	const particleinfo *pinfo = bufread->getData<BUFFER_INFO>();
	const hashKey *particleHash = bufread->getData<BUFFER_HASH>();
	const neibdata *neibsList = bufread->getData<BUFFER_NEIBSLIST>();

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel
	cubounds::computeVertexNormal<kerneltype><<< numBlocks, numThreads, dummy_shared >>> (
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


/// Initialize gamma
void
saInitGamma(
	MultiBufferList::const_iterator	bufread,
	MultiBufferList::iterator		bufwrite,
	const	uint*			cellStart,
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

	float4 *newGGam = bufwrite->getData<BUFFER_GRADGAMMA>();

	const float4 *oldPos = bufread->getData<BUFFER_POS>();
	const float4 *boundelement = bufread->getData<BUFFER_BOUNDELEMENTS>();
	const particleinfo *pinfo = bufread->getData<BUFFER_INFO>();
	const hashKey *particleHash = bufread->getData<BUFFER_HASH>();
	const neibdata *neibsList = bufread->getData<BUFFER_NEIBSLIST>();
	const float2 * const *vertPos = bufread->getRawPtr<BUFFER_VERTPOS>();

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	// execute the kernel for fluid particles
	cubounds::initGamma<kerneltype, PT_FLUID><<< numBlocks, numThreads, dummy_shared >>> (
		newGGam,
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

	// execute the kernel for vertex particles
	cubounds::initGamma<kerneltype, PT_VERTEX><<< numBlocks, numThreads, dummy_shared >>> (
		newGGam,
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

// counts vertices that belong to IO and same segment as other IO vertex
virtual
void
initIOmass_vertexCount(
	MultiBufferList::iterator bufwrite,
	MultiBufferList::const_iterator bufread,
	const	uint			numParticles,
	const	uint*			cellStart,
	const	uint			particleRangeEnd)
{
	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	const particleinfo *info = bufread->getData<BUFFER_INFO>();
	const hashKey *pHash = bufread->getData<BUFFER_HASH>();
	const neibdata *neibsList = bufread->getData<BUFFER_NEIBSLIST>();
	const vertexinfo *vertices = bufread->getData<BUFFER_VERTICES>();
	float4 *forces = bufwrite->getData<BUFFER_FORCES>();

	// execute the kernel
	cubounds::initIOmass_vertexCount<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
		(vertices, pHash, info, cellStart, neibsList, forces, particleRangeEnd);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;
}

/// Adjusts the initial mass of vertex particles on open boundaries
void
initIOmass(
	MultiBufferList::iterator bufwrite,
	MultiBufferList::const_iterator bufread,
	const	uint			numParticles,
	const	uint*			cellStart,
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

	const float4 *oldPos = bufread->getData<BUFFER_POS>();
	const float4 *forces = bufread->getData<BUFFER_FORCES>();
	const particleinfo *info = bufread->getData<BUFFER_INFO>();
	const hashKey *pHash = bufread->getData<BUFFER_HASH>();
	const neibdata *neibsList = bufread->getData<BUFFER_NEIBSLIST>();
	const vertexinfo *vertices = bufread->getData<BUFFER_VERTICES>();

	float4 *newPos = bufwrite->getData<BUFFER_POS>();

	// execute the kernel
	cubounds::initIOmass<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
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
	const	float4*			oldPos,
	const	float4*			boundelement,
			particleinfo*	info,
	const	hashKey*		particleHash,
	const	vertexinfo*		vertices,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			eps)
{
	int dummy_shared = 0;

	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));

	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif
	// execute the kernel
	cubounds::saIdentifyCornerVertices<<< numBlocks, numThreads, dummy_shared >>> (
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
