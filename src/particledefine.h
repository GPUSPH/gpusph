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
#define MAXNEIBSNUM			128

/*
   Particle types. Non-fluid parts are negative.
   When adding new ones make sure they are added before the first,
   increasing the (negative) index so that FLUIDPART always ends at 0
 */


#define MAX_FLUID_TYPES      4

enum ParticleType {
    GATEPART = -4,
  	PADDLEPART,
  	PISTONPART,
  	BOUNDPART,
  	FLUIDPART
};

// SAME AS TWO-D code, but have to use ParticleInfo for f as in FLUID(info[i])

#define NOT_FLUID(f) ((f).x != FLUIDPART)
#define FLUID(f) ((f).x == FLUIDPART)


/* Periodic neighborhood warping */
#define WARPZMINUS				(1U<<31)
#define WARPZPLUS				(1U<<30)
#define WARPYMINUS				(1U<<29)
#define WARPYPLUS				(1U<<28)
#define WARPXMINUS				(1U<<27)
#define WARPXPLUS				(1U<<26)
#define MAXPARTICLES			WARPXPLUS
#define NOWARP					~(WARPXPLUS|WARPXMINUS|WARPYPLUS|WARPYMINUS|WARPZPLUS|WARPZMINUS)

/* GPU warp size */
#define WARPSIZE				32

#define BLOCK_SIZE_FORCES		64

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
	float3	dispvect;
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
	PhysParams(void) :
		partsurf(0),
		p1coeff(12.0f),
		p2coeff(6.0f),
		epsxsph(0.5f),
		numFluids(1)
	{};
    /*! Set density parameters
        @param i         index in the array of materials
        @param rho       base density
        @param gamma     gamma coefficient
        @param ssmul     sound speed multiplier: sscoeff will be sqrt(ssmul*gravity)
     */
    void set_density(uint i, float rho, float gamma, float ssmul) {
        rho0[i] = rho;
        gammacoeff[i] = gamma;
        bcoeff[i] = rho*ssmul/gamma;
        sscoeff[i] = sqrt(ssmul*length(gravity));
        sspowercoeff[i] = (gamma - 1)/2;
    }
} PhysParams;


typedef struct MbCallBack {
	ParticleType	type;
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
	bool			periodicbound;		// type of periodic boundary used
	float			nlexpansionfactor;	// increase influcenradius by nlexpansionfactor for neib list construction
	bool			usedem;				// true if using a DEM
	SPHFormulation	sph_formulation;	// formulation to use for density and pressure computation
	BoundaryType	boundarytype;		// boundary force formulation (Lennard-Jones etc)
	bool			vorticity;
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
		vorticity(false)
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
	*(uint*)&v.z = id;   // id is in the location of two shorts.
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
