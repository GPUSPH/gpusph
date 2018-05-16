/*
 * engine_boundary_conditions.h
 *
 *  Created on: 10 déc. 2015
 *      Author: alexisherault
 */

#ifndef ENGINE_BOUNDARY_CONDITIONS_H_
#define ENGINE_BOUNDARY_CONDITIONS_H_

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "buffer.h"

/// TODO AbstractBoundaryConditionsEngine is presently just horrible hack to
/// speed up the transition to header-only / engine definitions
class AbstractBoundaryConditionsEngine
{
public:
	virtual ~AbstractBoundaryConditionsEngine() {}

/// Update the ID offset for new particle generation
virtual void
updateNewIDsOffset(const uint &newIDsOffset) = 0;

// Computes the boundary conditions on segments using the information from the fluid (on solid walls used for Neumann boundary conditions).
virtual void
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
	const	uint			step) = 0;

/// Detect particles that cross an open boundary and find the boundary element they have crossed
virtual void
findOutgoingSegment(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint*			cellStart,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius) = 0;

// There is no need to use two velocity arrays (read and write) and swap them after.
// Computes the boundary conditions on vertex particles using the values from the segments associated to it. Also creates particles for inflow boundary conditions.
// Data is only read from fluid and segments and written only on vertices.
virtual void
saVertexBoundaryConditions(
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
	const	uint			step,
	const	bool			resume, // TODO FIXME splitneibs-merge check if still needed
	const	float			dt, // for open boundaries
	// These are the cloning-related members
			uint*			newNumParticles,
	const	uint			deviceId,
	const	uint			numDevices,
	const	uint			totParticles) = 0;

// computes a normal for vertices in the initialization step
virtual void
computeVertexNormal(
	const BufferList&	bufread,
	BufferList&		bufwrite,
	const	uint*			cellStart,
	const	uint			numParticles,
	const	uint			particleRangeEnd) = 0;

// initialize gamma
virtual void
saInitGamma(
	const BufferList&	bufread,
	BufferList&		bufwrite,
	const	uint*			cellStart,
	const	float			slength,
	const	float			influenceradius,
	const	float			deltap,
	const	float			epsilon,
	const	uint			numParticles,
	const	uint			particleRangeEnd) = 0;

// counts vertices that belong to IO and same segment as other IO vertex
virtual
void
initIOmass_vertexCount(
	BufferList& bufwrite,
	const BufferList& bufread,
	const	uint			numParticles,
	const	uint*			cellStart,
	const	uint			particleRangeEnd) = 0;

// modification of initial io vertex mass
virtual
void
initIOmass(
	BufferList& bufwrite,
	const BufferList& bufread,
	const	uint			numParticles,
	const	uint*			cellStart,
	const	uint			particleRangeEnd,
	const	float			deltap) = 0;

// disables particles that went through boundaries when open boundaries are used
virtual void
disableOutgoingParts(		float4*			pos,
							vertexinfo*		vertices,
					const	particleinfo*	info,
					const	uint			numParticles,
					const	uint			particleRangeEnd) = 0;

// downloads the per device waterdepth from the GPU
virtual void
downloadIOwaterdepth(
			uint*	h_IOwaterdepth,
	const	uint*	d_IOwaterdepth,
	const	uint	numObjects) = 0;

// upload the global waterdepth to the GPU
virtual void
uploadIOwaterdepth(
	const	uint*	h_IOwaterdepth,
			uint*	d_IOwaterdepth,
	const	uint	numObjects) = 0;

// identifies vertices at the corners of open boundaries
virtual void
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
	const	float			eps) = 0;

};

#endif /* ENGINE_BOUNDARY_CONDITIONS_H_ */
