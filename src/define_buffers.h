/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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
 * Define one flag for each buffer which is used in a worker
 */

#ifndef DEFINED_BUFFERS
#define DEFINED_BUFFERS

#ifndef FIRST_DEFINED_BUFFER
#include "command_flags.h"
#endif

#ifndef SET_BUFFER_TRAITS
#include "buffer_traits.h"
#endif

// when we want to explicitly specify “no buffer”
#define BUFFER_NONE ((flag_t)0U)

// start from FIRST_DEFINED_BUFFER
// double-precision position buffer (used on host only)
#define BUFFER_POS_GLOBAL	FIRST_DEFINED_BUFFER
SET_BUFFER_TRAITS(BUFFER_POS_GLOBAL, double4, 1, "Position (double precision)");

#define BUFFER_POS			(BUFFER_POS_GLOBAL << 1)
SET_BUFFER_TRAITS(BUFFER_POS, float4, 1, "Position");
#define BUFFER_VEL			(BUFFER_POS << 1)
SET_BUFFER_TRAITS(BUFFER_VEL, float4, 1, "Velocity");
#define BUFFER_INFO			(BUFFER_VEL << 1)
SET_BUFFER_TRAITS(BUFFER_INFO, particleinfo, 1, "Info");
#define BUFFER_HASH			(BUFFER_INFO << 1)
SET_BUFFER_TRAITS(BUFFER_HASH, hashKey, 1, "Hash");

#define BUFFER_PARTINDEX	(BUFFER_HASH << 1)
SET_BUFFER_TRAITS(BUFFER_PARTINDEX, uint, 1, "Particle Index");

/* Cell-related buffers: index of the first and last particle in each cell */
#define BUFFER_CELLSTART	(BUFFER_PARTINDEX << 1)
SET_BUFFER_TRAITS(BUFFER_CELLSTART, uint, 1, "Cell Start");
#define BUFFER_CELLEND		(BUFFER_CELLSTART << 1)
SET_BUFFER_TRAITS(BUFFER_CELLEND, uint, 1, "Cell End");

/* Compact device map
 * (we only use 2 bits per cell, a single uchar might be sufficient)
 */
#define BUFFER_COMPACT_DEV_MAP		(BUFFER_CELLEND << 1)
SET_BUFFER_TRAITS(BUFFER_COMPACT_DEV_MAP, uint, 1, "Compact device map");

#define BUFFER_NEIBSLIST	(BUFFER_COMPACT_DEV_MAP << 1)
SET_BUFFER_TRAITS(BUFFER_NEIBSLIST, neibdata, 1, "Neighbor List");

#define BUFFER_FORCES		(BUFFER_NEIBSLIST << 1)
SET_BUFFER_TRAITS(BUFFER_FORCES, float4, 1, "Force");

/* Forces and torques acting on rigid body particles only
 * Note that these are sized according to the number of object particles,
 * not with the entire particle system
 */
#define BUFFER_RB_FORCES	(BUFFER_FORCES << 1)
SET_BUFFER_TRAITS(BUFFER_RB_FORCES, float4, 1, "Object forces");
#define BUFFER_RB_TORQUES	(BUFFER_RB_FORCES << 1)
SET_BUFFER_TRAITS(BUFFER_RB_TORQUES, float4, 1, "Object torques");

/* Object number for each object particle
 * TODO this is already present in the INFO buffer, rewrite the segmented scan
 * to use that?
 */
#define BUFFER_RB_KEYS	(BUFFER_RB_TORQUES << 1)
SET_BUFFER_TRAITS(BUFFER_RB_KEYS, uint, 1, "Object particle key");

#define BUFFER_INTERNAL_ENERGY (BUFFER_RB_KEYS << 1)
SET_BUFFER_TRAITS(BUFFER_INTERNAL_ENERGY, float, 1, "Internal Energy");

#define BUFFER_INTERNAL_ENERGY_UPD (BUFFER_INTERNAL_ENERGY << 1)
SET_BUFFER_TRAITS(BUFFER_INTERNAL_ENERGY_UPD, float, 1, "Internal Energy derivative");

#define BUFFER_XSPH			(BUFFER_INTERNAL_ENERGY_UPD << 1)
SET_BUFFER_TRAITS(BUFFER_XSPH, float4, 1, "XSPH");

#define BUFFER_TAU			(BUFFER_XSPH << 1)
SET_BUFFER_TRAITS(BUFFER_TAU, float2, 3, "Tau");

#define BUFFER_VORTICITY	(BUFFER_TAU << 1)
SET_BUFFER_TRAITS(BUFFER_VORTICITY, float3, 1, "Vorticity");
#define BUFFER_NORMALS		(BUFFER_VORTICITY << 1)
SET_BUFFER_TRAITS(BUFFER_NORMALS, float4, 1, "Normals");

/** Boundary elements buffer.
 *
 * For each boundary particle, this holds the normal to the corresponding boundary element
 * (in .x, .y, .z) and the boundary element surface (in .w)
 * For each vertex particle, this holds the surface-weighted average of the normals of the adjacent
 * boundary elements of the same IO type (i.e. IO boundary elements for IO vertices, and non-IO
 * boundary elements for non-IO vertices).
 */
#define BUFFER_BOUNDELEMENTS	(BUFFER_NORMALS << 1)
SET_BUFFER_TRAITS(BUFFER_BOUNDELEMENTS, float4, 1, "Boundary Elements");

/** Gradient of gamma (in .x, .y, .z) and gamma itself (in .w);
 *
 * For boundary particles this is averaged from the neighboring vertex particles,
 * for vertex and fluid particles this is computed either via quadrature or via
 * a transport equation (see simflag ENABLE_GAMMA_QUADRATURE and the check USING_DYNAMIC_GAMMA())
 */
#define BUFFER_GRADGAMMA		(BUFFER_BOUNDELEMENTS << 1)
SET_BUFFER_TRAITS(BUFFER_GRADGAMMA, float4, 1, "Gamma Gradient");

/** Connectivity between boundary particles and vertices.
 *
 * For boundary particles this is the list of the IDs of the adjacent vertex particles.
 * For fluid particles, this is only used in the open boundary case, to hold the nearest vertices
 * when a fluid particles moves out of the domain through an open boundary.
 */
#define BUFFER_VERTICES			(BUFFER_GRADGAMMA << 1)
SET_BUFFER_TRAITS(BUFFER_VERTICES, vertexinfo, 1, "Vertices");

/** Relative positions of vertices to boundary elements
 *
 * For each boundary element, this holds the local planar offset (hence float2) of each vertex to the boundary
 * element (hence 3 copies of the buffer, one per vertex).
 */
#define BUFFER_VERTPOS			(BUFFER_VERTICES << 1)
SET_BUFFER_TRAITS(BUFFER_VERTPOS, float2, 3, "Vertex positions relative to s");

#define BUFFER_TKE			(BUFFER_VERTPOS << 1)
SET_BUFFER_TRAITS(BUFFER_TKE, float, 1, "Turbulent Kinetic Energy [k]");
#define BUFFER_EPSILON		(BUFFER_TKE << 1)
SET_BUFFER_TRAITS(BUFFER_EPSILON, float, 1, "Turbulent Dissipation Rate [e]");
#define BUFFER_TURBVISC		(BUFFER_EPSILON << 1)
SET_BUFFER_TRAITS(BUFFER_TURBVISC, float, 1, "Eddy Viscosity");
#define BUFFER_DKDE			(BUFFER_TURBVISC << 1)
SET_BUFFER_TRAITS(BUFFER_DKDE, float3, 1, "[k]-[e] derivatives");

/** Effective viscosity array
 * This is used to hold the per-particle viscosity in models where it's necessary;
 * the value stored here is the dynamic viscosity or kinematic viscosity depending on
 * the \see{ComputationalViscosityType} of the viscous specification
 */
#define BUFFER_EFFVISC		(BUFFER_DKDE << 1)
SET_BUFFER_TRAITS(BUFFER_EFFVISC, float, 1, "Effective viscosity");

#define BUFFER_EULERVEL			(BUFFER_EFFVISC << 1)
SET_BUFFER_TRAITS(BUFFER_EULERVEL, float4, 1, "Eulerian velocity");

/** Next ID of generated particle
 *
 * All open-boundary vertices will have this set to the next ID they can
 * assign to a particle they generate. This is initially set (on host)
 * as N + i where N is the highest ID found in the setup, and i is
 * a sequential open-boundary vertex number.
 *
 * Each time the vertex generates a particle, it will increment this value
 * by the total number of open-boundary vertices present in the simulation.
 */
#define BUFFER_NEXTID			(BUFFER_EULERVEL << 1)
SET_BUFFER_TRAITS(BUFFER_NEXTID, uint, 1, "Next generated ID");

#define BUFFER_CFL			(BUFFER_NEXTID << 1)
SET_BUFFER_TRAITS(BUFFER_CFL, float, 1, "CFL array");
#define BUFFER_CFL_GAMMA		(BUFFER_CFL << 1)
SET_BUFFER_TRAITS(BUFFER_CFL_GAMMA, float, 1, "CFL gamma array");
#define BUFFER_CFL_TEMP		(BUFFER_CFL_GAMMA << 1)
SET_BUFFER_TRAITS(BUFFER_CFL_TEMP, float, 1, "CFL aux array");
#define BUFFER_CFL_KEPS		(BUFFER_CFL_TEMP << 1)
SET_BUFFER_TRAITS(BUFFER_CFL_KEPS, float, 1, "Turbulent Viscosity CFL array");

#define BUFFER_SPS_TURBVISC		(BUFFER_CFL_KEPS << 1)
SET_BUFFER_TRAITS(BUFFER_SPS_TURBVISC, float, 1, "SPS Turbulent viscosity");

// .x: initial volume, .y log (current/initial), .z unused, .w current volume
#define BUFFER_VOLUME		(BUFFER_SPS_TURBVISC << 1)
SET_BUFFER_TRAITS(BUFFER_VOLUME, float4, 1, "Volume");

#define BUFFER_SIGMA		(BUFFER_VOLUME << 1)
SET_BUFFER_TRAITS(BUFFER_SIGMA, float, 1, "Sigma (discrete specific volume)");

// Private buffers are used to save custom post-processing results
// There are three buffers available: a scalar one, one with two components and
// one with four.
#define BUFFER_PRIVATE		(BUFFER_SIGMA << 1)
SET_BUFFER_TRAITS(BUFFER_PRIVATE, float, 1, "Private scalar");

#define BUFFER_PRIVATE2		(BUFFER_PRIVATE << 1)
SET_BUFFER_TRAITS(BUFFER_PRIVATE2, float2, 1, "Private vector2");

#define BUFFER_PRIVATE4		(BUFFER_PRIVATE2 << 1)
SET_BUFFER_TRAITS(BUFFER_PRIVATE4, float4, 1, "Private vector4");

// last defined buffer. if new buffers are defined, remember to update this
#define LAST_DEFINED_BUFFER	BUFFER_PRIVATE4

// common shortcut
#define BUFFERS_POS_VEL_INFO	(BUFFER_POS | BUFFER_VEL | BUFFER_INFO)

// all CFL buffers
#define BUFFERS_CFL			( BUFFER_CFL | BUFFER_CFL_TEMP | BUFFER_CFL_KEPS | BUFFER_CFL_GAMMA)

// all CELL buffers
#define BUFFERS_CELL		( BUFFER_CELLSTART | BUFFER_CELLEND | BUFFER_COMPACT_DEV_MAP)

// elegant way to set to 1 all bits in between the first and the last buffers
// NOTE: READ or WRITE specification must be added for double buffers
#define ALL_DEFINED_BUFFERS		(((FIRST_DEFINED_BUFFER-1) ^ (LAST_DEFINED_BUFFER-1)) | LAST_DEFINED_BUFFER )

// all object particle buffers
#define BUFFERS_RB_PARTICLES (BUFFER_RB_FORCES | BUFFER_RB_TORQUES | BUFFER_RB_KEYS)

// all particle-based buffers
#define ALL_PARTICLE_BUFFERS	(ALL_DEFINED_BUFFERS & \
	~(BUFFERS_RB_PARTICLES | BUFFERS_CFL | BUFFERS_CELL | BUFFER_NEIBSLIST))

// TODO we need a better form of buffer classification, distinguishing:
// * “permanent” buffers for particle properties (which need to be sorted):
// 	** properties that evolve with the system (pos, vel, etc)
// 	** properties that follow the particles, but don't evolve;
// 	** (the distinction sometimes depends on the framework, e.g.
// 	    BOUNDELEMENTS doesn't change if there are no moving objects,
// 	    VERTICES only changes for fluid elements in the open boundary case,
// 	    and the change is ephemeral)
// * “ephemeral” buffers that get used right after compute
//    (forces, CFL, apparenty viscosity, etc) and at most need to be preserved
//    until the next save
/* note: BUFFER_NEXTID wasn't in IMPORT_BUFFERS before the rearranging of the classification,
 * but probably it should be, assuming open boundary vertex particles can end up
 * on a different device from the one they started on
 */

//! Buffers denoting particle properties that (may) evolve with the system
#define PARTICLE_PROPS_BUFFERS \
	(	BUFFER_POS | \
		BUFFER_VEL | \
		BUFFER_INFO | \
		BUFFER_INTERNAL_ENERGY | \
		BUFFER_NEXTID | \
		BUFFER_VERTICES | \
		BUFFER_BOUNDELEMENTS | \
		BUFFER_GRADGAMMA | \
		BUFFER_EULERVEL | \
		BUFFER_TKE | \
		BUFFER_EPSILON | \
		BUFFER_TURBVISC | \
		BUFFER_VOLUME)

//! Auxiliary buffers
/*! These are not physical particle properties, but the buffers are used
 * as support for other things, and generally recomputed or updated only
 * during the sorting/neighbors list construction phase.
 * HASH buffer is needed together with the POS buffer to assemble the global
 * position and to traverse the neighbors list.
 * VERTPOS is needed to speed up the computation of the Gamma gradient contribution.
 */
#define PARTICLE_SUPPORT_BUFFERS (BUFFER_HASH | BUFFER_VERTPOS)

//! Buffers holding temporary data
/*! These are use to store the results of some computation that needs to be reused
 * right after (e.g. forces, CFL etc). They are defined as particle buffers which are
 * not properties or support, plus the CFL buffers
 */
#define EPHEMERAL_BUFFERS \
	((ALL_PARTICLE_BUFFERS & ~(PARTICLE_PROPS_BUFFERS | PARTICLE_SUPPORT_BUFFERS)) | \
	 BUFFERS_CFL | \
	 (BUFFERS_RB_PARTICLES & ~BUFFER_RB_KEYS) \
	)

//! Buffers selectable by CALC_PRIVATE post-processing filter
#define BUFFERS_PRIVATE (BUFFER_PRIVATE | BUFFER_PRIVATE2 | BUFFER_PRIVATE4)

//! Post-processing buffers
/*! These buffers are ephemeral, and only used by post-processing filters */
#define POST_PROCESS_BUFFERS (BUFFER_VORTICITY | BUFFERS_PRIVATE)

//! Buffers that hold data that is useful throughout a simulation step
/*! A typical example is the neighbors list
 */
#define SUPPORT_BUFFERS \
	(ALL_DEFINED_BUFFERS & ~(PARTICLE_PROPS_BUFFERS | EPHEMERAL_BUFFERS))

//! Buffers that get (re)initialized during the neighbors list construction
/*! These are otherwise immutable, shouldn't be sorted, and are shared between states.
 * Note that BUFFER_COMPACT_DEV_MAP is excluded from these buffers because it's
 * generated once at the beginning of the simulation and never updated.
 */
#define NEIBS_SEQUENCE_REFRESH_BUFFERS \
	( (BUFFERS_CELL & ~BUFFER_COMPACT_DEV_MAP) | BUFFER_NEIBSLIST | BUFFER_VERTPOS)

// particle-based buffers to be imported during the APPEND_EXTERNAL command
// These are the particle property buffers plus the hash, from the READ list
#define IMPORT_BUFFERS (PARTICLE_PROPS_BUFFERS | PARTICLE_SUPPORT_BUFFERS)

#define POST_REPACK_SWAP_BUFFERS \
	(	BUFFER_POS | \
		BUFFER_VEL | \
		BUFFER_GRADGAMMA)

#endif

