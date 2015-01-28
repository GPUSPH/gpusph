/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

/* System-related definitions */

// TODO split non-particle stuff into different header(s)

#ifndef _PARTICLEDEFINE_H
#define	_PARTICLEDEFINE_H

#include "vector_math.h"
#include "cuda_call.h"

#include "ode/ode.h"

#include "common_types.h"

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
	SA_BOUNDARY,
	DYN_BOUNDARY,
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
	"Dynamic",
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
	KEPSVISC,
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
	"k-e model",
	"(invalid)"
}
#endif
;

/* Periodic boundary */
enum Periodicity {
	PERIODIC_NONE = 0,
	PERIODIC_X   = 1,
	PERIODIC_Y   = PERIODIC_X << 1,
	PERIODIC_XY  = PERIODIC_X | PERIODIC_Y,
	PERIODIC_Z   = PERIODIC_Y << 1,
	PERIODIC_XZ  = PERIODIC_X | PERIODIC_Z,
	PERIODIC_YZ  = PERIODIC_Y | PERIODIC_Z,
	PERIODIC_XYZ = PERIODIC_X | PERIODIC_Y | PERIODIC_Z,
};

#ifndef GPUSPH_MAIN
extern
#endif
const char* PeriodicityName[PERIODIC_XYZ+1]
#ifdef GPUSPH_MAIN
= {
	"none",
	"X",
	"Y",
	"X and Y",
	"Z",
	"X and Z",
	"Y and Z",
	"X, Y and Z",
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
 * maximum number of particle types that we need.
 *
 * These will then be encoded in the particle info field by shifting them by
 * MAX_FLUID_BITS.
 *
 * The maximum number of particle types we can have with 4 fluid bits is
 * 2^4 -1 = 15 (16 including particle type 0 = fluid).
 * If we ever need more, we can reduce MAX_FLUID_BITS to 2 (and force the limit of
 * 4 fluid types), and we could have up to 2^6 - 1 = 63 non-fluid particle types.
 */

enum ParticleType {
	PT_FLUID = 0,
	PT_VERTEX,
	PT_BOUNDARY,
	PT_PISTON,
	PT_PADDLE,
	PT_GATE,
	PT_OBJECT,
	PT_TESTPOINT
};

/* The ParticleType enum is rarely used directly, since for storage its value
 * is encoded in the high bits of the lowest byte of the particle info field,
 * so they are all shifted by MAX_FLUID_BITS.
 */

#define FLUIDPART		(PT_FLUID << MAX_FLUID_BITS)
/* non-fluid types start at (1<<MAX_FLUID_BITS) */
#define BOUNDPART		(PT_BOUNDARY << MAX_FLUID_BITS)
#define PISTONPART		(PT_PISTON << MAX_FLUID_BITS)
#define PADDLEPART		(PT_PADDLE << MAX_FLUID_BITS)
#define GATEPART		(PT_GATE << MAX_FLUID_BITS)
#define OBJECTPART		(PT_OBJECT << MAX_FLUID_BITS)
#define TESTPOINTSPART	(PT_TESTPOINT << MAX_FLUID_BITS)
#define VERTEXPART		(PT_VERTEX << MAX_FLUID_BITS)

/* particle flags */
#define PART_FLAG_START	(1<<PART_FLAG_SHIFT)

#define SET_FLAG(info, flag) ((info).x |= (flag))
#define CLEAR_FLAG(info, flag) ((info).x &= ~(flag))

#define SURFACE_PARTICLE_FLAG	(PART_FLAG_START<<0)
#define FIXED_PARTICLE_FLAG		(PART_FLAG_START<<1)
#define IO_PARTICLE_FLAG		(PART_FLAG_START<<2)
#define VEL_IO_PARTICLE_FLAG	(PART_FLAG_START<<3)
#define CORNER_PARTICLE_FLAG	(PART_FLAG_START<<5)
#define MOVING_PARTICLE_FLAG	(PART_FLAG_START<<6)
#define FLOATING_PARTICLE_FLAG	(PART_FLAG_START<<7)

/* A bitmask to select only the fluid number */
#define FLUID_NUM_MASK	((1<<MAX_FLUID_BITS)-1)
/* A bitmask to select only the particle type */
#define PART_TYPE_MASK	((1<<PART_FLAG_SHIFT)-(1<<MAX_FLUID_BITS))

/* A particle is NOT fluid if its particle type is non-zero */
#define NOT_FLUID(f)	(type(f) & PART_TYPE_MASK)
/* otherwise it's fluid */
#define FLUID(f)		(!(NOT_FLUID(f)))

// fluid particles can be active or inactive. Particles are marked inactive under appropriate
// conditions (e.g. after flowing out through an outlet), and are kept around until the next
// buildneibs, that sweeps them away.
// Since the inactivity of the particles must be accessible during neighbor list building,
// and in particular when computing the particle hash for bucketing, we carry the information
// in the position/mass field. Since fluid particles have positive mass, it is sufficient
// to set its mass to zero to mark the particle inactive

// a particle is active if its mass is non-zero
#define ACTIVE(p)	(isfinite((p).w))
#define INACTIVE(p)	(!ACTIVE(p))

// disable a particle by zeroing its mass
inline __host__ __device__ void
disable_particle(float4 &pos) {
	pos.w = NAN;
}

/* Tests for particle types */
// Testpoints
#define TESTPOINTS(f)	((type(f) & PART_TYPE_MASK) == TESTPOINTSPART)
// Particle belonging to an object
#define OBJECT(f)		((type(f) & PART_TYPE_MASK) == OBJECTPART)
// Boundary particle
#define BOUNDARY(f)		((type(f) & PART_TYPE_MASK) == BOUNDPART)
// Vertex particle
#define VERTEX(f)		((type(f) & PART_TYPE_MASK) == VERTEXPART)

/* Tests for particle flags */
// Free surface detection
#define SURFACE(f)		(type(f) & SURFACE_PARTICLE_FLAG)
// Fixed particle (e.g. Dalrymple's dynamic bounary particles)
#define FIXED_PART(f)	(type(f) & FIXED_PARTICLE_FLAG)
// If this flag is set the object is and open boundary
#define IO_BOUNDARY(f)	(type(f) & IO_PARTICLE_FLAG)
// If this flag is set the normal velocity is imposed at an open boundary
// if it is not set the pressure is imposed instead
#define VEL_IO(f)		(type(f) & VEL_IO_PARTICLE_FLAG)
// If vel_io is not set then we have a pressure inlet
#define PRES_IO(f)		(!VEL_IO(f))
// If this flag is set then a particle at an open boundary will have a non-varying mass but still
// be treated like an open boundary particle apart from that. This avoids having to span new particles
// very close to the side wall which causes problems
#define CORNER(f)		(type(f) & CORNER_PARTICLE_FLAG)
// This flag is set for moving vertices / segments either forced or free (floating)
#define MOVING(f)		(type(f) & MOVING_PARTICLE_FLAG)
// If the floating flag is set then the particles 
#define FLOATING(f)		(type(f) & FLOATING_PARTICLE_FLAG)

/* Extract a specific subfield from the particle type, unshifted:
 * this is used when saving data
 */
// Extract particle type
#define PART_TYPE(f)		((type(f) & PART_TYPE_MASK) >> MAX_FLUID_BITS)
// Extract particle flag
#define PART_FLAG(f)		(type(f) >> PART_FLAG_SHIFT)
// Extract particle fluid number
#define PART_FLUID_NUM(f)	(type(f) & FLUID_NUM_MASK)


/* Maximum number of floating bodies*/
#define	MAXBODIES				10

#define MAX_CUDA_LINEAR_TEXTURE_ELEMENTS (1U << 27)

#define NEIBINDEX_INTERLEAVE		32U

#if (__COMPUTE__ >= 20)
	#define INTMUL(x,y) (x)*(y)
#else
	#define INTMUL(x,y) __mul24(x,y)
#endif

#endif
