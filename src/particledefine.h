/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

/* System-related definitions */

// TODO split non-particle stuff into different header(s)

#ifndef _PARTICLEDEFINE_H
#define	_PARTICLEDEFINE_H

#include <cstring>

#include "vector_math.h"
#include "cuda_call.h"

enum KernelType {
	CUBICSPLINE = 1,
	QUADRATIC,
	WENDLAND,
	INVALID_KERNEL
} ;

#ifndef GPUSPH_MAIN
extern
#endif
const char* KernelName[INVALID_KERNEL+1]
#ifdef GPUSPH_MAIN
= {
	"(null)",
	"Cubic spline",
	"Quadratic",
	"Wendland",
	"(invalid)"
}
#endif
;

enum SPHFormulation {
	SPH_F1 = 1,
	SPH_F2,
	SPH_INVALID
} ;

#ifndef GPUSPH_MAIN
extern
#endif
const char* SPHFormulationName[SPH_INVALID+1]
#ifdef GPUSPH_MAIN
= {
	"(null)",
	"F1",
	"F2",
	"(invalid)"
}
#endif
;


enum BoundaryType {
	LJ_BOUNDARY,
	MK_BOUNDARY,
	MF_BOUNDARY,
	INVALID_BOUNDARY
};

#ifndef GPUSPH_MAIN
extern
#endif
const char* BoundaryName[INVALID_BOUNDARY+1]
#ifdef GPUSPH_MAIN
= {
	"Lennard-Jones",
	"Monaghan-Kajtar",
	"Ferrand et al.",
	"(invalid)"
}
#endif
;

#define EPSDETMLS				0.05f
#define MINCORRNEIBSMLS			4

enum ViscosityType {
	ARTVISC = 1,
	KINEMATICVISC,
	DYNAMICVISC,
	SPSVISC,
	INVALID_VISCOSITY
} ;

#ifndef GPUSPH_MAIN
extern
#endif
const char* ViscosityName[INVALID_VISCOSITY+1]
#ifdef GPUSPH_MAIN
= {
	"(null)",
	"Artificial",
	"Kinematic",
	"Dynamic",
	"SPS + kinematic",
	"(invalid)"
}
#endif
;

#define MAXPLANES			8
#define MAXMOVINGBOUND		16


/* The particle type is a short integer organized this way:
   * lowest 4 bits: fluid number (for multifluid)
   * next 4 bits: non-fluid code (boundary, piston, etc)
   * high 8 bits: flags
*/

// number of bits reserved for the fluid number
#define MAX_FLUID_BITS	4
// number of bits after which flags are stored
#define PART_FLAG_SHIFT	8

/* this can be increased to up to (1<<MAX_FLUID_BITS) */
#define MAX_FLUID_TYPES      4

/* compile-time consistency check */
#if MAX_FLUID_TYPES > (1<<MAX_FLUID_BITS)
#error "Too many fluids"
#endif

/* non-fluid particle types are mutually exclusive (e.g. a particle is _either_
 * boundary _or_ piston, but not both, so they could be increasing from 1 to the
 * maximum number of particle types that we need. But these are encoded in the high
 * bits of the lowest byte, so they are all shifted by MAX_FLUID_BITS.
 *
 * Remember: these are numbers (but encoded in a higher position), not flags.
 * Add to them by increasing the number that gets shifted.
 *
 * The maximum number of particle types we can have with 4 fluid bits is
 * 2^4 -1 = 15 (16 including particle type 0 = fluid).
 *
 * If we ever need more, we can reduce MAX_FLUID_BITS to 2 (and force the limit of
 * 4 fluid types), and we could have up to 2^6 - 1 = 63 non-fluid particle types.
 */

#define FLUIDPART		0
#define BOUNDPART		(1<<MAX_FLUID_BITS)
#define PISTONPART		(2<<MAX_FLUID_BITS)
#define PADDLEPART		(3<<MAX_FLUID_BITS)
#define GATEPART		(4<<MAX_FLUID_BITS)
#define TESTPOINTSPART		(5<<MAX_FLUID_BITS)
#define OBJECTPART		(6<<MAX_FLUID_BITS)
#define VERTEXPART		(7<<MAX_FLUID_BITS)
#define PROBEPART		(8<<MAX_FLUID_BITS)

/* particle flags */
#define PART_FLAG_START	(1<<PART_FLAG_SHIFT)

#define SURFACE_PARTICLE_FLAG	(PART_FLAG_START<<0)


/* A bitmask to select only the fluid number */
#define FLUID_NUM_MASK	((1<<MAX_FLUID_BITS)-1)
/* A bitmask to select only the particle type */
#define FLUID_TYPE_MASK	((1<<PART_FLAG_SHIFT)-(1<<MAX_FLUID_BITS))

/* A particle is NOT fluid if its fluid type is non-zero */
#define NOT_FLUID(f)	((f).x & FLUID_TYPE_MASK)
/* otherwise it's fluid */
#define FLUID(f)		(!(NOT_FLUID(f)))

/* Tests for particle types */
// Testpoints
#define TESTPOINTS(f)	((f).x == TESTPOINTSPART)
// Particle belonging to an object
#define OBJECT(f)		((f).x == OBJECTPART)
// Boundary particle
#define BOUNDARY(f)		((f).x == BOUNDPART)
// Vertex particle
#define VERTEX(f)		((f).x == VERTEXPART)
// Probe particle
#define PROBE(f)		((f).x == PROBEPART)

/* Tests for particle flags */
// Free surface detection
#define SURFACE_PARTICLE(f)	((f).x & SURFACE_PARTICLE_FLAG) // TODO; rename SURFACE_PARTICLE to SURFACE

/* Extract a specific subfield from the particle type, unshifted:
 * this is used when saving data
 */
// Extract particle type
#define PART_TYPE(f)		(((f).x & FLUID_TYPE_MASK) >> MAX_FLUID_BITS)
// Extract particle flag
#define PART_FLAG(f)		((f).x >> PART_FLAG_SHIFT)
// Extract particle fluid number
#define PART_FLUID_NUM(f)	((f).x & FLUID_NUM_MASK)


/* Periodic neighborhood warping */
#define WARPZMINUS				(1U<<31)
#define WARPZPLUS				(1U<<30)
#define WARPYMINUS				(1U<<29)
#define WARPYPLUS				(1U<<28)
#define WARPXMINUS				(1U<<27)
#define WARPXPLUS				(1U<<26)
#define MAXPARTICLES			WARPXPLUS
#define NOWARP					~(WARPXPLUS|WARPXMINUS|WARPYPLUS|WARPYMINUS|WARPZPLUS|WARPZMINUS)


/* Maximum number of floating bodies*/
#define	MAXBODIES				10


#define NEIBINDEX_INTERLEAVE		32

#if (__COMPUTE__ >= 20)
	#define INTMUL(x,y) (x)*(y)
#else
	#define INTMUL(x,y) __mul24(x,y)
#endif

typedef unsigned int uint;

typedef struct TimingInfo {
	float   t;
	float   dt;
	uint	numParticles;
	uint	maxNeibs;
	uint	numInteractions;
	long	iterations;
	long	meanNumInteractions;
	float   timeNeibsList;
	float   meanTimeNeibsList;
	float   timeInteract;
	float   meanTimeInteract;
	float   timeEuler;
	double  meanTimeEuler;
} TimingInfo;


struct SavingInfo {
	float   displayfreq;		// unit time
	uint	screenshotfreq;		// unit displayfreq
	uint	writedatafreq;		// unit displayfreq
};


/* Particle information. short4 with fields:
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

typedef short4 particleinfo;

inline __host__ particleinfo make_particleinfo(const short &type, const short &obj, const short &z, const short &w)
{
	particleinfo v;
	v.x = type;
	v.y = obj;
	v.z = z;
	v.w = w;
	return v;
}

inline __host__ particleinfo make_particleinfo(const short &type, const short &obj, const uint &id)
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
	memcpy((void *)&v.z, (const void *)&id, (unsigned int) 4);
	return v;
}

inline __host__ __device__ const short& type(const particleinfo &info)
{
	return info.x;
}

inline __host__ __device__ const short& object(const particleinfo &info)
{
	return info.y;   /***********NOTE */
}

inline __host__ __device__ const uint & id(const particleinfo &info)
{
	return *(uint*)&info.z;
}

typedef uint4 vertexinfo;

#endif
