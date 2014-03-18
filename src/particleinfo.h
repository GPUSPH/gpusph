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


/* Particle information. ushort4 with fields:
   .x: particle type (for multifluid)
   .y: object id (which object does this particle belong to?)
   (.z << 16) + .w: particle id

   The last two fields are unlikely to be used for actual computations, but
   they allow us to track 2^32 (about 4 billion) particles.
   In the extremely unlikely case that we need more, we can consider the
   particle id object-local and use (((.y << 16) + .z) << 16) + .w as a
   _global_ particle id. This would allow us to uniquely identify up to
   2^48 (about 281 trillion) particles.
*/

typedef ushort4 particleinfo;

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

static __inline__ __host__ __device__ const ushort& type(const particleinfo &info)
{
	return info.x;
}

static __inline__ __host__ __device__ const ushort& object(const particleinfo &info)
{
	return info.y;
}

static __inline__ __host__ __device__ const uint & id(const particleinfo &info)
{
	return *(const uint*)&info.z;
}

#endif
