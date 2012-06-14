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
/*
 * File:   particleTypes.h
 * Author: alexis
 *
 * Created on 27 avril 2008, 15:27
 */
 /*  MODIFICATIONS on July 16, 2010
  *  to include ParticleType and variable density
  *
  */

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


/*
   Particle types. Non-fluid parts are negative.
   When adding new ones make sure they are added before the first,
   increasing the (negative) index so that FLUIDPART always ends at 0
 */

#define MAX_FLUID_TYPES      4

/* The particle type is a short integer organized this way:
   * lowest 4 bits: fluid number (for multifluid)
   * next 4 bits: non-fluid code (boundary, piston, etc)
   * high 8 bits: flags
*/

#define MAX_FLUID_BITS       4

/* this can be increased to up to (1<<MAX_FLUID_BITS) */
#define MAX_FLUID_TYPES      4

/* compile-time consistency check */
#if MAX_FLUID_TYPES > (1<<MAX_FLUID_BITS)
#error "Too many fluids"
#endif

#define FLUIDPART 0

/* non-fluid types start at (1<<MAX_FLUID_BITS) */
#define BOUNDPART  (1<<MAX_FLUID_BITS)
#define PISTONPART (2<<MAX_FLUID_BITS)
#define PADDLEPART (3<<MAX_FLUID_BITS)
#define GATEPART   (4<<MAX_FLUID_BITS)
#define TESTPOINTSPART   (5<<MAX_FLUID_BITS)
#define OBJECTPART (6<<MAX_FLUID_BITS)

/* particle flags */
#define PARTICLE_FLAG_START (1<<8)

#define SURFACE_PARTICLE_FLAG (PARTICLE_FLAG_START<<0)

/* A particle is NOT fluid if it has the high bits of the lowest byte set */
#define NOT_FLUID(f) ((f).x & 0xf0)
/* otherwise it's fluid */
#define FLUID(f) (!(NOT_FLUID(f)))
// Testpoints
#define TESTPOINTS(f) ((f).x == TESTPOINTSPART)
// Particle belonging to an object
#define OBJECT(f) ((f).x == OBJECTPART)
// Free surface detection
#define SURFACE_PARTICLE(f) ((f).x & SURFACE_PARTICLE_FLAG) // TODO; rename SURFACE_PARTICLE to SURFACE
// Boundary particle
#define BOUNDARY(f) ((f).x == BOUNDPART)
// Extract particle type
#define PART_TYPE(f) (((f).x >> MAX_FLUID_BITS) & 0xf)
// Extract particle flag
#define PART_FLAG(f) ((f).x >> 8)
// Extract particle fluid number
#define PART_FLUID_NUM(f) ((f).x & 0xf)

/* compile-time consistency check:
   definition of NOT_FLUID() depends on MAX_FLUID_BITS being 4 */
#if MAX_FLUID_BITS != 4
#error "Adjust NOT_FLUID() macro"
#endif


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


typedef struct PhysParams {
	float	rho0[MAX_FLUID_TYPES]; // density of various particles

	float	partsurf;		// particle area (for surface friction)

	float3	gravity;		// gravity
	float	bcoeff[MAX_FLUID_TYPES];
	float	gammacoeff[MAX_FLUID_TYPES];
	float	sscoeff[MAX_FLUID_TYPES];
	float	sspowercoeff[MAX_FLUID_TYPES];

	// Lennard-Jones boundary coefficients
 	float	r0;		 		// influence radius of boundary repulsive force
	float	dcoeff;
	float	p1coeff;
	float	p2coeff;
	// Monaghan-Kajtar boundary coefficients
	float	MK_K;			// Typically: maximum velocity squared, or gravity times maximum height
	float	MK_d;			// Typically: distance between boundary particles
	float	MK_beta;		// Typically: ratio between h and MK_d

	float	kinematicvisc;	// Kinematic viscosity
	float	artvisccoeff;	// Artificial viscosity coefficient
	// For ARTVSIC: artificial viscosity coefficient
	// For KINEMATICVISC: 4*kinematic viscosity,
	// For DYNAMICVISC: dynamic viscosity
	float	visccoeff;
	float	epsartvisc;
	float	epsxsph;		// XSPH correction coefficient
	float3	dispvect;		// offset vector for periodic boundaries
	float3	maxlimit;
	float3	minlimit;
	float	ewres;			// DEM east-west resolution
	float	nsres;			// DEM north-south resolution
	float	demdx;			// Used for normal compution: displcement in x direction range ]0, exres[
	float	demdy;			// displcement in y direction range ]0, nsres[
	float	demdxdy;
	float	demzmin;		// demdx*demdy
	float	smagfactor;		// Cs*∆p^2
	float	kspsfactor;		// 2/3*Ci*∆^2
	int     numFluids;      // number of fluids in simulation
	float	cosconeanglefluid;	     // cos of cone angle for free surface detection (If the neighboring particle is fluid)
	float	cosconeanglenonfluid;	 // cos of cone angle for free surface detection (If the neighboring particle is non_fluid
	float	objectobjectdf;	// damping factor for object-object interaction 
	float	objectboundarydf;	// damping factor for object-boundary interaction 
	PhysParams(void) :
		partsurf(0),
		p1coeff(12.0f),
		p2coeff(6.0f),
		epsxsph(0.5f),
		numFluids(1),
		cosconeanglefluid(0.86),
		cosconeanglenonfluid (0.5),
		objectobjectdf (1.0),
		objectboundarydf (1.0)

	{};
	/*! Set density parameters
	  @param i	index in the array of materials
	  @param rho	base density
	  @param gamma	gamma coefficient
	  @param c0	sound speed for density at rest
	 */
	void set_density(uint i, float rho, float gamma, float c0) {
		rho0[i] = rho;
		gammacoeff[i] = gamma;
		bcoeff[i] = rho*c0*c0/gamma;
		sscoeff[i] = c0;
		sspowercoeff[i] = (gamma - 1)/2;
	}
} PhysParams;


typedef struct MbCallBack {
	short			type;
	float			tstart;
	float			tend;
	float3			origin;
	float3			vel;
	float3			disp;
	float			sintheta;
	float			costheta;
	float			omega;
	float			amplitude;
	float			phase;
} MbCallBack;


typedef struct SimParams {
	float			slength;			// smoothing length
	KernelType		kerneltype;			// kernel type
	float			kernelradius;		// kernel radius
	float			dt;					// initial timestep
	float			tend;				// simulation end time (0 means run forever)
	bool			xsph;				// true if XSPH correction
	bool			dtadapt;			// true if adaptive timestep
	float			dtadaptfactor;		// safety factor in the adaptive time step formula
	int				buildneibsfreq;		// frequency (in iterations) of neib list rebuilding
	int				shepardfreq;		// frequency (in iterations) of Shepard density filter
	int				mlsfreq;			// frequency (in iterations) of MLS density filter
	ViscosityType	visctype;			// viscosity type (1 artificial, 2 laminar)
	int				displayfreq;		// display update frequence (in seconds)
	int				savedatafreq;		// simulation data saving frequence (in displayfreq)
	int				saveimagefreq;		// screen capture frequence (in displayfreq)
	bool			mbcallback;			// true if moving boundary velocity varies
	bool			gcallback;			// true if using a variable gravity in problem
	bool			periodicbound;		// true in case of periodic boundary
	float			nlexpansionfactor;	// increase influcenradius by nlexpansionfactor for neib list construction
	bool			usedem;				// true if using a DEM
	SPHFormulation	sph_formulation;	// formulation to use for density and pressure computation
	BoundaryType	boundarytype;		// boundary force formulation (Lennard-Jones etc)
	bool			vorticity;			// true if we want to save vorticity
	bool            testpoints;         // true if we want to find velocity at testpoints
	bool            savenormals;        // true if we want to save the normals at free surface
	bool            surfaceparticle;    // true if we want to find surface particles
	//WaveGage
	bool			writeWaveGage;		//true if we want to use a wave gage
	float			xgage;
	float			ygage;
	//Rozita
	float3			gage[10];
	float			WaveGageNum;
	int				numbodies;			// number of floating bodies
	uint			maxneibsnum;		// maximum number of neibs (should be a multiple of NEIBS_INTERLEAVE)
	SimParams(void) :
		kernelradius(2.0),
		dt(0.00013),
		tend(0),
		xsph(false),
		dtadapt(true),
		dtadaptfactor(0.3),
		buildneibsfreq(10),
		shepardfreq(0),
		mlsfreq(15),
		visctype(ARTVISC),
		mbcallback(false),
		gcallback(false),
		periodicbound(false),
		nlexpansionfactor(1.0),
		usedem(false),
		sph_formulation(SPH_F1),
		boundarytype(LJ_BOUNDARY),
		vorticity(false),
		testpoints(false),
		savenormals(false),
		surfaceparticle(false),
		//WaveGage 
		writeWaveGage (false),
		xgage(0),
		ygage(0),
		numbodies(0),
		maxneibsnum(128)
	{};
} SimParams;


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
	memcpy((void *)&v.z, (void *)&id, 4);
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
#endif
