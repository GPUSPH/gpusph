/*
 * engine_boundary_conditions.h
 *
 *  Created on: 10 déc. 2015
 *      Author: alexisherault
 */

/*! \file
 * Contains the abstract interface for the BoundaryConditionsEngine
 */

#ifndef ENGINE_BOUNDARY_CONDITIONS_H_
#define ENGINE_BOUNDARY_CONDITIONS_H_

#include "particledefine.h"
#include "physparams.h"
#include "simparams.h"
#include "buffer.h"

/*! Abstract class that defines the interface for BoundaryConditionsEngine.
 * Currently most of its methods are specific to SA_BOUNDARY, so it's a bit
 * of a hack.
 */
class AbstractBoundaryConditionsEngine
{
public:
	virtual ~AbstractBoundaryConditionsEngine() {}

/// Set the number of vertices present in the whole simulation
virtual void
uploadNumOpenVertices(const uint &numOpenVertices) = 0;

/// Computes the boundary conditions on segments using the information from the fluid (on solid walls used for Neumann boundary conditions).
virtual void
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
	const	int			step) = 0;

/// Detect particles that cross an open boundary and find the boundary element they have crossed
virtual void
findOutgoingSegment(
	BufferList &bufwrite,
	BufferList const& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			slength,
	const	float			influenceradius) = 0;

/*! Computes the boundary conditions on vertex particles using the values from
 * the segments associated to it. Also creates particles for inflow boundary
 * conditions. Data is only read from fluid and segments and written only on
 * vertices, so there is no need to use two velocity buffers and swap.
 */
virtual void
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
	const	bool			resume, // TODO FIXME splitneibs-merge check if still needed
	const	float			dt, // for open boundaries
	// These are the cloning-related members
			uint*			newNumParticles,
	const	uint			deviceId,
	const	uint			numDevices,
	const	uint			totParticles) = 0;

//! Computes a normal for vertices in the initialization step
virtual void
computeVertexNormal(
	const BufferList&	bufread,
	BufferList&		bufwrite,
	const	uint			numParticles,
	const	uint			particleRangeEnd) = 0;

//! Initialize gamma
virtual void
saInitGamma(
	const BufferList&	bufread,
	BufferList&		bufwrite,
	const	float			slength,
	const	float			influenceradius,
	const	float			deltap,
	const	float			epsilon,
	const	uint			numParticles,
	const	uint			particleRangeEnd) = 0;

//! Counts vertices that belong to open boundaries and share a segment with other open boundary vertices
virtual
void
initIOmass_vertexCount(
	BufferList& bufwrite,
	const BufferList& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd) = 0;

//! Distribute initial mass for open boundary vertices
virtual
void
initIOmass(
	BufferList& bufwrite,
	const BufferList& bufread,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap) = 0;

//! Disables particles that went through boundaries when open boundaries are used
virtual void
disableOutgoingParts(const	BufferList& bufread,
							BufferList& bufwrite,
					const	uint			numParticles,
					const	uint			particleRangeEnd) = 0;

//! Downloads the per device waterdepth from the GPU
virtual void
downloadIOwaterdepth(
			uint*	h_IOwaterdepth,
	const	uint*	d_IOwaterdepth,
	const	uint	numObjects) = 0;

//! Upload the global waterdepth to the GPU
virtual void
uploadIOwaterdepth(
	const	uint*	h_IOwaterdepth,
			uint*	d_IOwaterdepth,
	const	uint	numObjects) = 0;

//! Identifies vertices at the corners of open boundaries
virtual void
saIdentifyCornerVertices(
	const	BufferList&	bufread,
			BufferList&	bufwrite,
	const	uint			numParticles,
	const	uint			particleRangeEnd,
	const	float			deltap,
	const	float			eps) = 0;

};

#endif /* ENGINE_BOUNDARY_CONDITIONS_H_ */
