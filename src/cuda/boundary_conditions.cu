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

#include "boundary_conditions_kernel.cu"

// TODO Rename and optimize
#define BLOCK_SIZE_SA_BOUND		128
#define MIN_BLOCKS_SA_BOUND		6

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
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(cuboundaryconditions::d_newIDsOffset, &newIDsOffset, sizeof(uint)));
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
	cuboundaryconditions::disableOutgoingPartsDevice<<<numBlocks, numThreads>>>
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
			float4*			oldPos,
			float4*			oldVel,
			float*			oldTKE,
			float*			oldEps,
			float4*			oldEulerVel,
			float4*			oldGGam,
			vertexinfo*		vertices,
	const	uint*			vertIDToIndex,
	const	float2	* const vertPos[],
	const	float4*			boundelement,
	const	particleinfo*	info,
	const	hashKey*		particleHash,
	const	uint*			cellStart,
	const	neibdata*		neibsList,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius,
	const	bool			initStep,
	const	uint			step)
{
	uint numThreads = BLOCK_SIZE_SA_BOUND;
	uint numBlocks = div_up(particleRangeEnd, numThreads);

	int dummy_shared = 0;
	// TODO: Probably this optimization doesn't work with this function. Need to be tested.
	#if (__COMPUTE__ == 20)
	dummy_shared = 2560;
	#endif

	CUDA_SAFE_CALL(cudaBindTexture(0, boundTex, boundelement, numParticles*sizeof(float4)));
	CUDA_SAFE_CALL(cudaBindTexture(0, infoTex, info, numParticles*sizeof(particleinfo)));

	// execute the kernel
	cuboundaryconditions::saSegmentBoundaryConditions<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
		(oldPos, oldVel, oldTKE, oldEps, oldEulerVel, oldGGam, vertices, vertIDToIndex, vertPos[0], vertPos[1], vertPos[2], particleHash, cellStart, neibsList, particleRangeEnd, deltap, slength, influenceradius, initStep, step, simflags & ENABLE_INLET_OUTLET);

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
			float2*			contupd,
	const	float4*			boundelement,
			vertexinfo*		vertices,
	const	float2			* const vertPos[],
	const	uint*			vertIDToIndex,
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
	const	uint			numDevices)
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
	cuboundaryconditions::saVertexBoundaryConditions<kerneltype><<< numBlocks, numThreads, dummy_shared >>>
		(oldPos, oldVel, oldTKE, oldEps, oldGGam, oldEulerVel, forces, contupd, vertices, vertPos[0], vertPos[1], vertPos[2], vertIDToIndex, info, particleHash, cellStart, neibsList,
		 particleRangeEnd, newNumParticles, dt, step, deltap, slength, influenceradius, initStep, resume, deviceId, numDevices);

	// check if kernel invocation generated an error
	KERNEL_CHECK_ERROR;

	CUDA_SAFE_CALL(cudaUnbindTexture(boundTex));

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
	cuboundaryconditions::saIdentifyCornerVertices<<< numBlocks, numThreads, dummy_shared >>> (
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
