/*  Copyright (c) 2014-2019 INGV, EDF, UniCT, JHU

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
 * Particle info type and functions
 */

#ifndef _PARTICLEINFO_H
#define _PARTICLEINFO_H

// for memcpy
#include <cstring>
// for NAN
#include <cmath>

// we use CUDA types and host/device specifications here
#include <cuda_runtime.h>

#include "common_types.h" // ushort

/** \addtogroup particleinfo Particles types and flags
 * 	\ingroup datastructs
 *  Anything related to particle's information: type, flags, access functions ....
 *  @{ */

/** \typedef particleinfo
 * 	\brief Particle informations (type, flags, ....)
 *
 * 	Particle related informations such as particle type, flags, id, .... are stored
 * 	in an ushort4 with:
 *		- .x: particle type and flags
 *		- .y: object id and fluid number
 *		- (.z << 16) + .w: global particle id
 *
 *	The particle type is a short integer organized this way:
 * 		- lowest 3 bits: particle type
 * 		- next 13 bits: flags
 *	Particle types are mutually exclusive (e.g. a particle is _either_
 * 	boundary _or_ vertex, but not both)
 *
 *	The maximum number of particle types we can have with 4 is
 *	2^3 = 8 of which 4 are actually used.
 *
 *	The global particle id, is the id assigned to the particle in the problem
 *	copy_to_array method. Considering that the last two fields are unlikely to be
 *	used for actual computations, we use it to store the global particle id. This
 *  allow us to track 2^32 (about 4 billion) particles.
 * 	In the extremely unlikely case that we need more, we can consider the
 *	particle id object-local and use (((.y << 16) + .z) << 16) + .w as a
 * 	_global_ particle id. This would allow us to uniquely identify up to
 * 	2^48 (about 281 trillion) particles.
 *
 * \todo consider making it its own structure
 */
typedef ushort4 particleinfo;

/** \typedef vertexinfo
 * 	\brief Connectivity between boundary particles and vertices for SA
 *
 * 	SA boundary requires connectivity information for the boundary elements:
 * 	we need to know the vertices indices for each boundary element.
 * 	This information is stored in a uint4 with:
 *		- .x: id of vertex number 0
 *		- .y: id of vertex number 1
 *		- .z: id of vertex number 2
 *		- .w: unused
 *
 *	\note the ordering of vertices is not random, when filling the connectivity
 *	table we need to use, without any modification, the data from Crixus
 *
 *	\note uint4 was used instead of uint3 for enforcing coalescing
 */
typedef uint4 vertexinfo;

/// Definition of the shortcut make_vertexinfo
#define make_vertexinfo make_uint4

/// Check if a given ID can be found in the given vertex info
constexpr __host__ __device__ __forceinline__ __attribute__((pure))
bool has_vertex(
	vertexinfo const& verts, ///< vertexinfo holding list of vertices
	uint id			 ///< id of the particles to be checked
	)
{
	return verts.x == id || verts.y == id || verts.z == id;
}

/// Return the index (0, 1 or 2) of the ID in the vertex list
/** -1 is returned if the vertex is not in the list
 */
constexpr __host__ __device__ __forceinline__ __attribute__((pure))
int local_vertex_index(
	vertexinfo const& verts, ///< vertexinfo holding list of vertices
	uint id			 ///< id of the particles to be checked
	)
{
	return	verts.x == id ? 0 :
		verts.y == id ? 1 :
		verts.z == id ? 2 : -1;
}

/// Particle types
/**	Enumeration defining the 4 basic particle types.
 * 	We recall that testpoints are virtual particles used for
 * 	post-processing purpose only and that they don't interact with
 * 	the other one.
 */
enum ParticleType {
	PT_FLUID = 0,	///< particle is a fluid one
	PT_BOUNDARY,	///< particle belongs to a boundary
	PT_VERTEX,		///< particle is a vertex
	PT_TESTPOINT,	///< particle is a testpoint
	PT_NONE			///< ficticious particle type used for optional termination of the neighbors list traversal
};


/** \name Particle flag enumeration
 *  @{ */
/// Number of bits in particleinfo.x after which flags are stored
#define PART_FLAG_SHIFT	3
/// Integer value of the first flag
#define PART_FLAG_START	(1<<PART_FLAG_SHIFT)
/// Particle flags
/**	Enumeration defining the particle flags. */
enum ParticleFlag {
	FG_COMPUTE_FORCE =		(PART_FLAG_START<<0),  ///< particle belongs to a body that needs to compute forces
	FG_MOVING_BOUNDARY =	(PART_FLAG_START<<1),  ///< particle belongs to a body with externally prescribed motion

	FG_INLET =				(PART_FLAG_START<<2),  ///< particle belongs to an inlet
	FG_OUTLET =				(PART_FLAG_START<<3),  ///< particle belongs to an outlet
	FG_VELOCITY_DRIVEN =	(PART_FLAG_START<<4),  ///< particle belongs to an I/O with prescribed velocity
	FG_CORNER =				(PART_FLAG_START<<5),  ///< particle is a corner of an I/O

	FG_SURFACE =			(PART_FLAG_START<<6),  ///< particle is at the free surface
	FG_INTERFACE =			(PART_FLAG_START<<7),  ///< particle is at the interface between two phases
	FG_SEDIMENT =			(PART_FLAG_START<<8),  ///< particle is sediment (rheology + effective pressure)

	FG_DEFORMABLE =			(PART_FLAG_START<<9),  ///< particle belongs to a deformable body
	FG_FEA_NODE =			(PART_FLAG_START<<10), ///< nodes of the mesh associated to a deformable body
	FG_LAST_FLAG = FG_FEA_NODE
};

/** @} */

/** \name Flag getting/setting macros
 *  @{ */
/// Set a specific flag
#define SET_FLAG(info, flag) ((info).x |= (flag))
/// Clear a specific flag
#define CLEAR_FLAG(info, flag) ((info).x &= ~(flag))
/// Query whether all flags are set in the particleinfo struct of a certain particle
#define QUERY_ALL_PART_FLAG(info, flags) ((info).x & (flags) == (flags))
/// Query whether at least one flag (out of flags) is set in the particleinfo struct of a certain particle
#define QUERY_ANY_PART_FLAG(info, flags) ((info).x & (flags))
/** @} */

/** \name Bitmasks for particle type or flags selection
 *  @{ */
/// Bitmasks to select only the particle type
#define PART_TYPE_MASK	((1<<PART_FLAG_SHIFT)-1)
/// Bitmasks to select only the particle dlag
#define PART_FLAGS_MASK	(((1<<16)-1) - PART_TYPE_MASK)
/** @} */

/** \name Subfield selection from particle type
 *  @{ */
/// Extract particle type
#define PART_TYPE(f)		ParticleType(type(f) & PART_TYPE_MASK)
/// Extract particle flags
#define PART_FLAGS(f)		(type(f) & PART_FLAGS_MASK)
/** @} */

/** \name Particle type related macros
 *  @{ */
/// A particle is NOT fluid if its particle type is non-zero
#define NOT_FLUID(f)	(PART_TYPE(f) > PT_FLUID)
/// A particle is a fluid one if tis type is PT_FLUID
#define FLUID(f)		(PART_TYPE(f) == PT_FLUID)

/// Check if particle is a testpoint
#define TESTPOINT(f)	(PART_TYPE(f) == PT_TESTPOINT)
/// Check if particle is a boundary particle
#define BOUNDARY(f)		(PART_TYPE(f) == PT_BOUNDARY)
/// Check if particle is a vertex particle
#define VERTEX(f)		(PART_TYPE(f) == PT_VERTEX)
/** @} */

/** \name Particle flags related macros
 *  @{ */
/// Check if particle is on the free surface
#define SURFACE(f)		(type(f) & FG_SURFACE)

/// Check if particle is at the interface of two phases
#define INTERFACE(f)		(type(f) & FG_INTERFACE)

/// Check if particle is sediment 
#define SEDIMENT(f)		(type(f) & FG_SEDIMENT)

/// Check if a particle belongs to I/O open boundary
/** Particles belonging to an I/O boundary have the FG_INLET
 * or FG_OUTET flag set.
 */
#define IO_BOUNDARY(f)	(type(f) & (FG_INLET | FG_OUTLET))
/// Check if a particle belongs to imposed velocity I/O boundary
/** Particles belonging to an I/O boundary with imposed velocity have the
 *  FG_VELOCITY_DRIVEN flag set.
 */
#define VEL_IO(f)		(type(f) & FG_VELOCITY_DRIVEN)
/// Check if a particle belongs to imposed pressure I/O boundary
/** Particles belonging to an I/O boundary have either imposed velocity or
 *  or imposed pressure. Then if the FG_VELOCITY_DRIVEN flag is not set for
 *  an I/O particle the particle's pressure is imposed.
 */
#define PRES_IO(f)		(!VEL_IO(f))

/// Check if a particle is located in a corner
/** FG_CORNER flag is set for open boundary particles at corners, which behave like
 * 	open boundary particles for all intents and purposes, except that their mass doesn't
 *	vary and they do not produce particles. This avoids having to spawn new particles
 *	very close to the side wall which causes problems
 */
#define CORNER(f)		(type(f) & FG_CORNER)

/// Check if particle belongs to a moving object
/** A moving object is a floating one or boundary with imposed motion
 *  either by the the fluid (floating object) or by the user (moving boundary).
 *  The force on floating objects is automatically computed, while the force on
 *  moving objects is computed only if we set the FG_COMPUTE_FORCE flag.
 */
#define MOVING(f)		(type(f) & FG_MOVING_BOUNDARY)
/// Check if particle belongs to a floating body
/// \see MOVING
#define FLOATING(f)		(type(f) & (FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE))
/// Check if particle belongs to an object on which we want force feedback
/// \see MOVING
#define COMPUTE_FORCE(f)	(type(f) & FG_COMPUTE_FORCE)
/// Check if particle belongs to a deformable body
#define DEFORMABLE(f)	(type(f) & FG_DEFORMABLE)
/// Check if particle represents a FEA node
#define FEA_NODE(f)	(type(f) & FG_FEA_NODE)

/// Check if particle is active
/** Fluid particles can be active or inactive. Particles are marked inactive under appropriate
 *	conditions (e.g. after flowing out through an outlet), and are kept around until the next
 *	buildneibs, that sweeps them away.
 *	Since the inactivity of the particles must be accessible during neighbor list building,
 *	and in particular when computing the particle hash for bucketing, we carry the information
 *	in the position/mass field. Specifically, a particle is marked inactive by setting its
 * mass to Not-a-Number.
 */
#define ACTIVE(p)	(::isfinite((p).w))
/// Check if particle is inactive
/// \see ACTIVE
#define INACTIVE(p)	(!ACTIVE(p))
/** @} */


/** \name Object/fluid number related macros
 *  The fluid (4 bits) and object (12 bits) number share a common location,
 *  so they should be assembled like this: SET_FLUID_NUM(foo) | SET_OBJECT_NUM(bar)
 *  @{ */
/// Bit shift for fluid == bits reserved for object number
#define FLUID_NUM_SHIFT 12
/// Get fluid number
#define GET_FLUID_NUM(y) ((y) >> FLUID_NUM_SHIFT)
/// Set fluid number
#define SET_FLUID_NUM(fnum) ((fnum) << FLUID_NUM_SHIFT)
/// Mask for the object number bits
#define OBJECT_NUM_MASK ((1 << FLUID_NUM_SHIFT) - 1)
/// Get object number
#define GET_OBJECT_NUM(y) ((y) & OBJECT_NUM_MASK)
/// Set object number
#define SET_OBJECT_NUM(fnum) (fnum)

/// Check if a particle needs an object number
#define NEEDS_OBJECT_NUM (FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE | FG_INLET | FG_OUTLET)
/** @} */


/// Disable a particle
/** Disable a particle by setting its mass to Not-A-Number. */
inline __host__ __device__ void
disable_particle(	float4 &pos	///< Particle position and mass
		)
{
	pos.w = NAN;
}


/// RAW particleinfo creator
/** RAW particleinfo creator: the ushorts are loaded directly into the respective fields,
 *  without any check concerning type or other is effected.
 *
 *  \todo make it a shortcut to make_uint4
 */
inline __host__
particleinfo make_particleinfo(
		const ushort &x, 	///< x component
		const ushort &y, 	///< y component
		const ushort &z, 	///< z component
		const ushort &w		///< w component
		)
{
	particleinfo v;
	v.x = x;
	v.y = y;
	v.z = z;
	v.w = w;
	return v;
}

/// Typed particleinfo creator
/*! This constructor sets the type, object/fluid number and id in the following way:
 *		* the type is accepted as-is;
 *		* the id is split into z and w as appropriate;
 *		* the obj_or_fnum is automatically interpreted as an object number or fluid number.
 *	If obj_or_fnum is bigger than OBJECT_NUM_MASK, then it's assumed to be a fluid number,
 *	or a combination of fluid and object number.
 *	If it's smaller than OBJECT_NUM_MASK, then it will be interpreted as an object number
 *	if the particle is not fluid and it NEEDS_OBJECT_NUM, otherwise it's interpreted as
 *	fluid number.
 *
 *	\note particle flags MUST be already set when calling this function. If one needs
 *	change the particle flags after the creation of the particle info, then
 *	must be used.
 *
 *	\todo the only thing that is impossible to do this way is to assign a fluid number of 0
 *	and an object number > 0 to a fluid particle. We'll face the problem if/when it arises.
 */
inline __host__
particleinfo make_particleinfo(
		const ushort &type, 		///< Particle type
		const ushort &obj_or_fnum, 	///< Particle object or fluid number
		const uint &id				///< Particle global id
		)
{
	particleinfo v;
	v.x = type;
	/* Automatic interpretation of obj_or_fnum */
	if (obj_or_fnum > OBJECT_NUM_MASK)
		v.y = obj_or_fnum;
	else if (type & NEEDS_OBJECT_NUM) /* needs an object number */
		v.y = SET_OBJECT_NUM(obj_or_fnum);
	else
		v.y = SET_FLUID_NUM(obj_or_fnum);

	v.z = (id & USHRT_MAX); // low id bits in z
	v.w = (id >> 16); // high id bits in w
	return v;
}

/// Typed particleinfo creator
/*! This constructor works like make_particleinfo() but it allows for explicitly setting
 *	the fluid number *and* the object number. It does not check nor set any particle flag,
 *  so setting them after creating the particleinfo is allowed.
 */
inline __host__ __device__
particleinfo make_particleinfo_by_ids(
		const ushort &type, 			///< Particle type
		const ushort &fluid_number,		///< Particle fluid number
		const ushort &object_number, 	///< Particle object number
		const uint &id					///< Particle global id
		)
{
	particleinfo v;
	// Set type - leave flags as they are
	v.x = type;
	// Set fluid *and* object number
	v.y = SET_FLUID_NUM(fluid_number) | SET_OBJECT_NUM(object_number);
	// Set particle id
	v.z = (id & USHRT_MAX); // low id bits in z
	v.w = (id >> 16); // high id bits in w
	return v;
}

/// Returns particle type
/*! Returns particle type for a given particle info.
 *
 * \return particle type
 */
static __forceinline__ __host__ __device__ __attribute__((pure))
const ushort& type(	const particleinfo &info	///< Particle info
		)
{
	return info.x;
}

/// Returns particle object number
/*! Returns the particle object number for a given particle info.
 *
 * \return particle object number
 */
static __forceinline__ __host__ __device__ __attribute__((pure))
ushort object(const particleinfo &info)
{
	return GET_OBJECT_NUM(info.y);
}

/// Returns particle fluid number
/*! Returns the particle fluid number for a given particle info.
 *
 * \return particle fluid number
 */
static __forceinline__ __host__ __device__ __attribute__((pure))
ushort fluid_num(const particleinfo &info)
{
	return GET_FLUID_NUM(info.y);
}

inline __host__
void set_fluid_num(particleinfo &info, ushort fluid_num)
{
	info.y = object(info) | SET_FLUID_NUM(fluid_num);
}


/// Returns particle global id
/*! Returns the particle global id for a given particle info.
 *
 * \return particle fluid number
 */
static __forceinline__ __host__ __device__ __attribute__((pure))
uint id(const particleinfo &info)
{
	return (uint)(info.z) | ((uint)(info.w) << 16);
}
/** @} */
#endif
