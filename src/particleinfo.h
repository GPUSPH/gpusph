/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/* Particle info type and functions */

#ifndef _PARTICLEINFO_H
#define _PARTICLEINFO_H

// for memcpy
#include <cstring>
// for NAN
#include <cmath>

// we use CUDA types and host/device specifications here
#include <cuda_runtime.h>

#include "common_types.h" // ushort

// vertex info
typedef uint4 vertexinfo;
#define make_vertexinfo make_uint4

/* Particle information. ushort4 with fields:
   .x: particle type and flags
   .y: object id and fluid number
   (.z << 16) + .w: particle id

   The last two fields are unlikely to be used for actual computations, but
   they allow us to track 2^32 (about 4 billion) particles.
   In the extremely unlikely case that we need more, we can consider the
   particle id object-local and use (((.y << 16) + .z) << 16) + .w as a
   _global_ particle id. This would allow us to uniquely identify up to
   2^48 (about 281 trillion) particles.

   TODO: consider making it its own struct
*/
typedef ushort4 particleinfo;

/* The particle type is a short integer organized this way:
   * lowest 3 bits: particle type
   * next 13 bits: flags

  Particle types are mutually exclusive (e.g. a particle is _either_
  boundary _or_ vertex, but not both)

  The maximum number of particle types we can have with 4 is
  2^3 = 8 of which 4 are actually used.
*/

// number of bits after which flags are stored
#define PART_FLAG_SHIFT	3

enum ParticleType {
	PT_FLUID = 0,
	PT_BOUNDARY,
	PT_VERTEX,
	PT_TESTPOINT
};

/* particle flags */
#define PART_FLAG_START	(1<<PART_FLAG_SHIFT)
enum ParticleFlag {
	FG_COMPUTE_FORCE =		(PART_FLAG_START<<0), ///< particle belongs to a body that needs to compute forces
	FG_MOVING_BOUNDARY =	(PART_FLAG_START<<1), ///< particle belongs to a body with externally prescribed motion

	FG_INLET =				(PART_FLAG_START<<2), ///< particle belongs to an inlet
	FG_OUTLET =				(PART_FLAG_START<<3), ///< particle belongs to an outlet
	FG_VELOCITY_DRIVEN =	(PART_FLAG_START<<4), ///< particle belongs to an I/O with prescribed velocity
	FG_CORNER =				(PART_FLAG_START<<5), ///< particle is a corner of an I/O

	FG_SURFACE =			(PART_FLAG_START<<6), ///< particle is at the free surface
};

#define SET_FLAG(info, flag) ((info).x |= (flag))
#define CLEAR_FLAG(info, flag) ((info).x &= ~(flag))
/// Query whether all flags are set in the particleinfo struct of a certain particle
#define QUERY_ALL_PART_FLAG(info, flags) ((info).x & (flags) == (flags))
/// Query whether at least one flag (out of flags) is set in the particleinfo struct of a certain particle
#define QUERY_ANY_PART_FLAG(info, flags) ((info).x & (flags))

/// Bitmasks to select only the particle type or flags
#define PART_TYPE_MASK	((1<<PART_FLAG_SHIFT)-1)
#define PART_FLAGS_MASK	(((1<<16)-1) - PART_TYPE_MASK)

/* Extract a specific subfield from the particle type: */
/// Extract particle type
#define PART_TYPE(f)		(type(f) & PART_TYPE_MASK)
/// Extract particle flags
#define PART_FLAGS(f)		(type(f) & PART_FLAGS_MASK)

/* Tests for particle types */

/// A particle is NOT fluid if its particle type is non-zero
#define NOT_FLUID(f)	(PART_TYPE(f) > PT_FLUID)
/// otherwise it's fluid
#define FLUID(f)		(PART_TYPE(f) == PT_FLUID)

/// Check if particle is a testpoint
#define TESTPOINT(f)	(PART_TYPE(f) == PT_TESTPOINT)
/// Check if particle is a boundary particle
#define BOUNDARY(f)		(PART_TYPE(f) == PT_BOUNDARY)
/// Check if particle is a vertex particle
#define VERTEX(f)		(PART_TYPE(f) == PT_VERTEX)

/* Tests for particle flags */

/// Particle is on the free surface
#define SURFACE(f)		(type(f) & FG_SURFACE)

/// A particle belongs to an open boundary if it has the FG_INLET or FG_OUTET
/// flags set
#define IO_BOUNDARY(f)	(type(f) & (FG_INLET | FG_OUTLET))
/// The FG_VELOCITY_DRIVEN flag is set for open boundaries that impose
/// the normal velocity. Otherwise, pressure is imposed.
#define VEL_IO(f)		(type(f) & FG_VELOCITY_DRIVEN)
#define PRES_IO(f)		(!VEL_IO(f))

/// FG_CORNER flag is set for open boundary particles at corners, which behave like
/// open boundary particles for all intents and purposes, except that their mass doesn't
/// vary and they do not produce particles. This avoids having to spawn new particles
/// very close to the side wall which causes problems
#define CORNER(f)		(type(f) & FG_CORNER)

/// Check if particle belongs to a moving object or boundary (either with imposed motion
/// or free (floating)
#define MOVING(f)		(type(f) & FG_MOVING_BOUNDARY)
/// Check if particle belongs to a floating body
#define FLOATING(f)		(type(f) & (FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE))
/// Check if this particle belongs to a moving body for which we want to compute the
/// force feedback (force exherted by the fluid on the object)
#define COMPUTE_FORCE(f)	(type(f) & FG_COMPUTE_FORCE)

// fluid particles can be active or inactive. Particles are marked inactive under appropriate
// conditions (e.g. after flowing out through an outlet), and are kept around until the next
// buildneibs, that sweeps them away.
// Since the inactivity of the particles must be accessible during neighbor list building,
// and in particular when computing the particle hash for bucketing, we carry the information
// in the position/mass field. Specifically, a particle is marked inactive by setting its
// mass to Not-a-Number.

/// a particle is active if its mass is finite
#define ACTIVE(p)	(isfinite((p).w))
#define INACTIVE(p)	(!ACTIVE(p))

/// disable a particle by setting its mass to Not-A-Number
inline __host__ __device__ void
disable_particle(float4 &pos) {
	pos.w = NAN;
}


//< RAW particleinfo constructor: the ushorts are loaded directly into the respective fields,
//< and no check concerning type or other is effected.
inline __host__ particleinfo make_particleinfo(const ushort &type, const ushort &obj, const ushort &z, const ushort &w)
{
	particleinfo v;
	v.x = type;
	v.y = obj;
	v.z = z;
	v.w = w;
	return v;
}

/* The fluid (4 bits) and object (12 bits) number share a common location,
 * so they should be assembled like this:
 * 	SET_FLUID_NUM(foo) | SET_OBJECT_NUM(bar)
 */

#define FLUID_NUM_SHIFT 12 /* bit shift for fluid == bits reserved for object number */
#define GET_FLUID_NUM(y) ((y) >> FLUID_NUM_SHIFT)
#define SET_FLUID_NUM(fnum) ((fnum) << FLUID_NUM_SHIFT)
#define OBJECT_NUM_MASK ((1 << FLUID_NUM_SHIFT) - 1) /* mask for the object number bits */
#define GET_OBJECT_NUM(y) ((y) & OBJECT_NUM_MASK)
#define SET_OBJECT_NUM(fnum) (fnum)

// Flags that imply that the particle needs an object number.
#define NEEDS_OBJECT_NUM (FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE | FG_INLET | FG_OUTLET)

/// Typed particleinfo creator
/*! This constructor sets the type, object/fluid number and id in the following way:
 *		* the type is accepted as-is;
 *		* the id is split into z and w as appropriate;
 *		* the obj_or_fnum is automatically interpreted as an object number or fluid number.
 *	If obj_or_fnum is bigger than OBJECT_NUM_MASK, then it's assumed to be a fluid number,
 *	or a combination of fluid and object number.
 *	If it's smaller than OBJECT_NUM_MASK, then it will be interpreted as an object number
 *	if the particle is not fuid and it NEEDS_OBJECT_NUM, otherwise it's interpreted as
 *	fluid number.
 *	NOTE: particle flags MUST be already set when calling this function. If one needs
 *	change the particle flags after the creation of the particle info, then
 *	must be used.
 *	TODO the only thing that is impossible to do this way is to assign a fluid number of 0
 *	and an object number > 0 to a fluid particle. We'll face the problem if/when it arises.
 */
inline __host__ particleinfo make_particleinfo(const ushort &type, const ushort &obj_or_fnum, const uint &id)
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
inline __host__ __device__ particleinfo make_particleinfo_by_ids(const ushort &type, const ushort &fluid_number,
	const ushort &object_number, const uint &id)
{
	particleinfo v;
	// set type - leave flags as they are
	v.x = type;
	// set fluid *and* object number
	v.y = SET_FLUID_NUM(fluid_number) | SET_OBJECT_NUM(object_number);
	// set particle id
	v.z = (id & USHRT_MAX); // low id bits in z
	v.w = (id >> 16); // high id bits in w
	return v;
}

static __forceinline__ __host__ __device__ __attribute__((pure)) const ushort& type(const particleinfo &info)
{
	return info.x;
}

static __forceinline__ __host__ __device__ __attribute__((pure)) ushort object(const particleinfo &info)
{
	return GET_OBJECT_NUM(info.y);
}

static __forceinline__ __host__ __device__ __attribute__((pure)) ushort fluid_num(const particleinfo &info)
{
	return GET_FLUID_NUM(info.y);
}

static __forceinline__ __host__ __device__ __attribute__((pure)) uint id(const particleinfo &info)
{
	return (uint)(info.z) | ((uint)(info.w) << 16);
}

#endif
