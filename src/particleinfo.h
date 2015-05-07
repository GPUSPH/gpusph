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
#include "cuda_runtime.h"

// vertex info
typedef uint4 vertexinfo;
#define make_vertexinfo make_uint4

/* Particle information. ushort4 with fields:
   .x: particle type and flags
   .y: object id or fluid number
   (.z << 16) + .w: particle id

   The last two fields are unlikely to be used for actual computations, but
   they allow us to track 2^32 (about 4 billion) particles.
   In the extremely unlikely case that we need more, we can consider the
   particle id object-local and use (((.y << 16) + .z) << 16) + .w as a
   _global_ particle id. This would allow us to uniquely identify up to
   2^48 (about 281 trillion) particles.
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
	PT_TESTPOINT,
	PT_BOUNDARY,
	PT_VERTEX
};

/* particle flags */
#define PART_FLAG_START	(1<<PART_FLAG_SHIFT)
enum ParticleFlag {
	FG_COMPUTE_FORCE =		(PART_FLAG_START<<0), ///< particle belongs to a body that needs to compute forces
	FG_MOVING_BOUNDARY =	(PART_FLAG_START<<1), ///< particle belongs to a body with externally prescriped motion

	FG_INLET =				(PART_FLAG_START<<2), ///< particle belongs to an inlet
	FG_OUTLET =				(PART_FLAG_START<<3), ///< particle belongs to an outlet
	FG_VELOCITY_DRIVEN =	(PART_FLAG_START<<4), ///< particle belongs to an I/O with prescribed velocity
	FG_CORNER =				(PART_FLAG_START<<5), ///< particle is a corner of an I/O

	FG_SURFACE =			(PART_FLAG_START<<6), ///< particle is at the free surface
};

#define SET_FLAG(info, flag) ((info).x |= (flag))
#define CLEAR_FLAG(info, flag) ((info).x &= ~(flag))
#define QUERY_FLAG(info, flag) ((info).x & (flag))

/* A bitmask to select only the particle type */
#define PART_TYPE_MASK	((1<<PART_FLAG_SHIFT)-1)

/* Extract a specific subfield from the particle type: */
// Extract particle type
#define PART_TYPE(f)		(type(f) & PART_TYPE_MASK)
// Extract particle flag
#define PART_FLAGS(f)		(type(f) >> PART_FLAG_SHIFT)

/* Tests for particle types */

/* A particle is NOT fluid if its particle type is non-zero */
#define NOT_FLUID(f)	((type(f) & PART_TYPE_MASK) > PT_FLUID)
/* otherwise it's fluid */
#define FLUID(f)		((type(f) & PART_TYPE_MASK) == PT_FLUID)

// Testpoints
#define TESTPOINT(f)	(PART_TYPE(f) == PT_TESTPOINT)
// Boundary particle
#define BOUNDARY(f)		(PART_TYPE(f) == PT_BOUNDARY)
// Vertex particle
#define VERTEX(f)		(PART_TYPE(f) == PT_VERTEX)

/* Tests for particle flags */

// Free surface detection
#define SURFACE(f)		(type(f) & FG_SURFACE)

// If one of these flag is set the object is and open boundary
#define IO_BOUNDARY(f)	(type(f) & (FG_INLET | FG_OUTLET))
// If this flag is set the normal velocity is imposed at an open boundary
// if it is not set the pressure is imposed instead
#define VEL_IO(f)		(type(f) & FG_VELOCITY_DRIVEN)
// If vel_io is not set then we have a pressure inlet
#define PRES_IO(f)		(!VEL_IO(f))
// If this flag is set then a particle at an open boundary will have a non-varying mass but still
// be treated like an open boundary particle apart from that. This avoids having to span new particles
// very close to the side wall which causes problems
#define CORNER(f)		(type(f) & FG_CORNER)

// This flag is set for moving vertices / segments either forced or free (floating)
#define MOVING(f)		(type(f) & FG_MOVING_BOUNDARY)
// This flag is set for particles belonging to a floating body
#define FLOATING(f)		(type(f) & (FG_MOVING_BOUNDARY | FG_COMPUTE_FORCE))
// This flag is set for particles belonging to a moving body on which we want to compute reaction force
#define COMPUTE_FORCE(f)	(type(f) & FG_COMPUTE_FORCE)

// fluid particles can be active or inactive. Particles are marked inactive under appropriate
// conditions (e.g. after flowing out through an outlet), and are kept around until the next
// buildneibs, that sweeps them away.
// Since the inactivity of the particles must be accessible during neighbor list building,
// and in particular when computing the particle hash for bucketing, we carry the information
// in the position/mass field. Specifically, a particle is marked inactive by setting its
// mass to Not-a-Number.

// a particle is active if its mass is finite
#define ACTIVE(p)	(isfinite((p).w))
#define INACTIVE(p)	(!ACTIVE(p))

// disable a particle by zeroing its mass
inline __host__ __device__ void
disable_particle(float4 &pos) {
	pos.w = NAN;
}


inline __host__ particleinfo make_particleinfo(const ushort &type, const ushort &obj, const ushort &z, const ushort &w)
{
	particleinfo v;
	v.x = type;
	v.y = obj;
	v.z = z;
	v.w = w;
	return v;
}

inline __host__ particleinfo make_particleinfo(const ushort &type, const ushort &obj, const uint &id)
{
	particleinfo v;
	v.x = type;
	v.y = obj;
	// id is in the location of two shorts.
	/* The following line does not work with optimization if the C99
	   standard for strict aliasing holds. Rather than forcing
	   -fno-strict-aliasing (which is GCC only) we resort to
	   memcpy which is the only `portable' way of doing this stuff,
	   allegedly. Note that even this is risky because it might fail
	   in cases of different endianness. So I'll mark this
	   FIXME endianness
	 */
	// *(uint*)&v.z = id;
	memcpy((void *)&v.z, (const void *)&id, 4);
	return v;
}

static __forceinline__ __host__ __device__ const ushort& type(const particleinfo &info)
{
	return info.x;
}

static __forceinline__ __host__ __device__ const ushort& object(const particleinfo &info)
{
	return info.y;
}


static __forceinline__ __host__ __device__ __attribute__((pure)) ushort fluid_num(const particleinfo &info)
{
	// TODO FIXME the fluid_num should never be queried for non-fluid part. In the mean time,
	// retun 0 in such a case, since this is generally used to get information on pressure or
	// viscosity. This effectively makes the object system incompatible with multifluid.
	// TODO FIXME also check multifluid SA
	return BOUNDARY(info) ? 0 : info.y;
}

static __forceinline__ __host__ __device__ const uint & id(const particleinfo &info)
{
	return *(const uint*)&info.z;
}

#endif
