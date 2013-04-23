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
 * Device code.
 */

#ifndef _FORCES_KERNEL_
#define _FORCES_KERNEL_

#include "particledefine.h"
#include "textures.cuh"

#define GPU_CODE
#include "kahan.h"
#undef GPU_CODE

texture<float, 2, cudaReadModeElementType> demTex;	// DEM

namespace cuforces {
__constant__ uint d_maxneibsnum_time_neibindexinterleave;

__constant__ float	d_wcoeff_cubicspline;			// coeff = 1/(Pi h^3)
__constant__ float	d_wcoeff_quadratic;				// coeff = 15/(16 Pi h^3)
__constant__ float	d_wcoeff_wendland;				// coeff = 21/(16 Pi h^3)

__constant__ float	d_fcoeff_cubicspline;			// coeff = 3/(4Pi h^4)
__constant__ float	d_fcoeff_quadratic;				// coeff = 15/(32Pi h^4)
__constant__ float	d_fcoeff_wendland;				// coeff = 105/(128Pi h^5)

__constant__ int    d_numfluids;					// number of different fluids

__constant__ float	d_rho0[MAX_FLUID_TYPES];		// rest density of fluids

// Speed of sound constants
__constant__ float	d_bcoeff[MAX_FLUID_TYPES];
__constant__ float	d_gammacoeff[MAX_FLUID_TYPES];
__constant__ float	d_sscoeff[MAX_FLUID_TYPES];
__constant__ float	d_sspowercoeff[MAX_FLUID_TYPES];

__constant__ float3	d_gravity;						// gravity (vector)

__constant__ float	d_ferrari;				// coefficient for Ferrari correction

// LJ boundary repusion force comuting
__constant__ float	d_dcoeff;
__constant__ float	d_p1coeff;
__constant__ float	d_p2coeff;
__constant__ float	d_r0;

// Monaghan-Kaijar boundary repulsion force constants
// This is typically the square of the maximum velocity, or gravity times the maximum height
__constant__ float	d_MK_K;
// This is typically the distance between boundary particles
__constant__ float	d_MK_d;
// This is typically the ration between h and the distance between boundary particles
__constant__ float	d_MK_beta;

__constant__ float	d_visccoeff;
__constant__ float	d_epsartvisc;

__constant__ float3	d_dispvect;					// displacment vector for periodic boundaries

// Constants used for DEM
__constant__ float	d_ewres;
__constant__ float	d_nsres;
__constant__ float	d_demdx;
__constant__ float	d_demdy;
__constant__ float	d_demdxdy;
__constant__ float	d_demzmin;

__constant__ float	d_partsurf;						// particle surface

// Definition of planes for geometrical boundaries
__constant__ uint	d_numplanes;
__constant__ float4	d_planes[MAXPLANES];
__constant__ float	d_plane_div[MAXPLANES];

// Sub-Particle Scale (SPS) Turbulence parameters
__constant__ float	d_smagfactor;
__constant__ float	d_kspsfactor;

// Free surface detection
__constant__ float	d_cosconeanglefluid;
__constant__ float	d_cosconeanglenonfluid;

// Rigid body data (test version)
__device__ float3	d_force;
__device__ float3	d_torque;
__constant__ float3 d_rbcg[MAXBODIES];
__constant__ uint	d_rbstartindex[MAXBODIES];
__constant__ float d_objectobjectdf;
__constant__ float d_objectboundarydf;


typedef struct sym33mat {
	float a11;
	float a12;
	float a13;
	float a22;
	float a23;
	float a33;
} sym33mat;


/************************************************************************************************************/
/*							  Functions used by the differents CUDA kernels							   */
/************************************************************************************************************/

/********************************************* SPH kernels **************************************************/
// Return kernel value at distance r, for a given smoothing length
template<KernelType kerneltype>
__device__ __forceinline__ float
W(const float r, const float slength);


// Cubic Spline kernel
template<>
__device__ __forceinline__ float
W<CUBICSPLINE>(const float r, const float slength)
{
	float val = 0.0f;
	const float R = r/slength;

	if (R < 1)
		val = 1.0f - 1.5f*R*R + 0.75f*R*R*R;			// val = 1 - 3/2 R^2 + 3/4 R^3
	else
		val = 0.25f*(2.0f - R)*(2.0f - R)*(2.0f - R);	// val = 1/4 (2 - R)^3

	val *= d_wcoeff_cubicspline;						// coeff = 1/(Pi h^3)

	return val;
}


// Qudratic kernel
template<>
__device__ __forceinline__ float
W<QUADRATIC>(const float r, const float slength)
{
	float val = 0.0f;
	const float R = r/slength;

	val = 0.25f*R*R - R + 1.0f;		// val = 1/4 R^2 -  R + 1
	val *= d_wcoeff_quadratic;		// coeff = 15/(16 Pi h^3)

	return val;
}


// Wendland kernel
template<>
__device__ __forceinline__ float
W<WENDLAND>(float r, float slength)
{
	const float R = r/slength;

	float val = 1.0f - 0.5f*R;
	val *= val;
	val *= val;						// val = (1 - R/2)^4
	val *= 1.0f + 2.0f*R;			// val = (2R + 1)(1 - R/2)^4*
	val *= d_wcoeff_wendland;		// coeff = 21/(16 Pi h^3)
	return val;
}


// Return 1/r dW/dr at distance r, for a given smoothing length
template<KernelType kerneltype>
__device__ __forceinline__ float
F(const float r, const float slength);


template<>
__device__ __forceinline__ float
F<CUBICSPLINE>(const float r, const float slength)
{
	float val = 0.0f;
	const float R = r/slength;

	if (R < 1.0f)
		val = (-4.0f + 3.0f*R)/slength;		// val = (-4 + 3R)/h
	else
		val = -(-2.0f + R)*(-2.0f + R)/r;	// val = -(-2 + R)^2/r
	val *= d_fcoeff_cubicspline;			// coeff = 3/(4Pi h^4)

	return val;
}


template<>
__device__ __forceinline__ float
F<QUADRATIC>(const float r, const float slength)
{
	const float R = r/slength;

	float val = (-2.0f + R)/r;		// val = (-2 + R)/r
	val *= d_fcoeff_quadratic;		// coeff = 15/(32Pi h^4)

	return val;
}


template<> 
__device__ __forceinline__ float
F<WENDLAND>(const float r, const float slength)
{
	const float qm2 = r/slength - 2.0f;	// val = (-2 + R)^3
	float val = qm2*qm2*qm2*d_fcoeff_wendland;
	return val;
}
/************************************************************************************************************/


/********************** Equation of state, speed of sound, repulsive force **********************************/
// Equation of state: pressure from density, where i is the fluid kind, not particle_id
__device__ __forceinline__ float
P(const float rho, const uint i)
{
	return d_bcoeff[i]*(__powf(rho/d_rho0[i], d_gammacoeff[i]) - 1);
}

// Inverted equation of state: density from pressure
__device__ __forceinline__ float
rho(const float P, const uint i)
{
	return d_rho0[i]*__powf(P/d_bcoeff[i] + 1, 1.0f/d_gammacoeff[i]);
}

// Sound speed computed from density
__device__ __forceinline__ float
soundSpeed(const float rho, const uint i)
{
	return d_sscoeff[i]*__powf(rho/d_rho0[i], d_sspowercoeff[i]);
}


// Lennard-Jones boundary repulsion force
__device__ __forceinline__ float
LJForce(const float r)
{
	float force = 0.0f;

	if (r <= d_r0)
		force = d_dcoeff*(__powf(d_r0/r, d_p1coeff) - __powf(d_r0/r, d_p2coeff))/(r*r);

	return force;
}

// Monaghan-Kajtar boundary repulsion force doi:10.1016/j.cpc.2009.05.008
// to be multiplied by r_aj vector
// we allow the fluid particle mass mass_f to be different from the
// boundary particle mass mass_b even though they are typically the same
// (except for multi-phase fluids)
__device__ __forceinline__ float
MKForce(const float r, const float slength,
		const float mass_f, const float mass_b)
{
	// MK always uses the 1D cubic or quintic Wendland spline
	float w = 0.0f;

	float force = 0.0f;

	// Wendland has radius 2
	if (r <= 2*slength) {	//TODO: fixme use influenceradius
		float qq = r/slength;
		w = 1.8f * __powf(1.0f - 0.5f*qq, 4.0f) * (2.0f*qq + 1.0f);  //TODO: optimize
		// float dist = r - d_MK_d;
		float dist = max(d_epsartvisc, r - d_MK_d);
		force = d_MK_K*w*2*mass_b/(d_MK_beta * dist * r * (mass_f+mass_b));
	}

	return force;
}
/************************************************************************************************************/


/***************************************** Viscosities *******************************************************/
// Artificial viscosity s
__device__ __forceinline__ float
artvisc(	const float	vel_dot_pos,
			const float	rho,
			const float	neib_rho,
			const float	sspeed,
			const float	neib_sspeed,
			const float	r,
			const float	slength)
{
	return vel_dot_pos*slength*d_visccoeff*(sspeed + neib_sspeed)/
									((r*r + d_epsartvisc)*(rho + neib_rho));
}


// ATTENTION: for all non artificial viscosity
// µ is the dynamic viscosity (ρν)

// Scalar part of viscosity using Morris 1997
// expression 21 p218 when all particles have the same viscosity
// in this case d_visccoeff = 4 nu
// returns 4.mj.nu/(ρi + ρj) (1/r ∂Wij/∂r)
__device__ __forceinline__ float
laminarvisc_kinematic(	const float	rho,
						const float	neib_rho,
						const float	neib_mass,
						const float	f)
{
	return neib_mass*d_visccoeff*f/(rho + neib_rho);
}


// Same behaviour as laminarvisc but for particle
// dependent viscosity.
// returns mj.(µi + µi)/(ρi.ρj) (1/r ∂Wij/∂r)
__device__ __forceinline__ float
laminarvisc_dynamic(const float	rho,
					const float	neib_rho,
					const float	neib_mass,
					const float	f,
					const float	visc,
					const float	neib_visc)
{
	return neib_mass*(visc + neib_visc)*f/(rho*neib_rho);
}
/************************************************************************************************************/


/*********************************** Adptative time stepping ************************************************/
// Function called at the end of the forces or powerlawVisc function doing
// a per block maximum reduction
__device__ __forceinline__ void
dtadaptBlockReduce(	float*	sm_max,
					float*	cfl)
{
	for(unsigned int s = blockDim.x/2; s > 0; s >>= 1) 
	{
		__syncthreads();
		if (threadIdx.x < s) 
		{
			sm_max[threadIdx.x] = max(sm_max[threadIdx.x + s], sm_max[threadIdx.x]);
		}
	}

	// write result for this block to global mem
	if (!threadIdx.x)
		cfl[blockIdx.x] = sm_max[0];
}
/************************************************************************************************************/


/********************************* Periodic boundary management *********************************************/
// Function returning the neigbor index, position, relative distance and velocity
template<bool periodicbound>
__device__ __forceinline__ void
getNeibData(const float4	pos,
			const float		influenceradius,
			uint&			neib_index,
			float4&			neib_pos,
			float3&			relPos,
			float&			r);


// In case of periodic boundaries we add the displacement
// vector when needed
template<>
__device__ __forceinline__ void
getNeibData<true>(	const float4	pos,
					const float		influenceradius,
					uint&			neib_index,
					float4&			neib_pos,
					float3&			relPos,
					float&			r)
{
	int3 periodic = make_int3(0);
	if (neib_index & WARPXPLUS)
		periodic.x = 1;
	else if (neib_index & WARPXMINUS)
		periodic.x = -1;
	if (neib_index & WARPYPLUS)
		periodic.y = 1;
	else if (neib_index & WARPYMINUS)
		periodic.y = -1;
	if (neib_index & WARPZPLUS)
		periodic.z = 1;
	else if (neib_index & WARPZMINUS)
		periodic.z = -1;

	neib_index &= NOWARP;

	neib_pos = tex1Dfetch(posTex, neib_index);

	relPos.x = pos.x - neib_pos.x;
	relPos.y = pos.y - neib_pos.y;
	relPos.z = pos.z - neib_pos.z;
	r = length(relPos);
	if (periodic.x || periodic.y || periodic.z) {
		if (r > influenceradius) {
			relPos += periodic*d_dispvect;
			r = length(relPos);
		}
	}
}


template<>
__device__ __forceinline__ void
getNeibData<false>(	const float4	pos,
					const float		influenceradius,
					uint&			neib_index,
					float4&			neib_pos,
					float3&			relPos,
					float&			r)
{
	neib_pos = tex1Dfetch(posTex, neib_index);

	relPos.x = pos.x - neib_pos.x;
	relPos.y = pos.y - neib_pos.y;
	relPos.z = pos.z - neib_pos.z;
	r = length(relPos);
}

template<bool periodicbound>
__device__ __forceinline__ void
getNeibData(const float4	pos,
			const float4*	posArray,
			const float		influenceradius,
			uint&			neib_index,
			float4&			neib_pos,
			float3&			relPos,
			float&			r);


// In case of periodic boundaries we add the displacement
// vector when needed
template<>
__device__ __forceinline__ void
getNeibData<true>(	const float4	pos,
					const float4*	posArray,
					const float		influenceradius,
					uint&			neib_index,
					float4&			neib_pos,
					float3&			relPos,
					float&			r)
{
	int3 periodic = make_int3(0);
	if (neib_index & WARPXPLUS)
		periodic.x = 1;
	else if (neib_index & WARPXMINUS)
		periodic.x = -1;
	if (neib_index & WARPYPLUS)
		periodic.y = 1;
	else if (neib_index & WARPYMINUS)
		periodic.y = -1;
	if (neib_index & WARPZPLUS)
		periodic.z = 1;
	else if (neib_index & WARPZMINUS)
		periodic.z = -1;

	neib_index &= NOWARP;

	neib_pos = posArray[neib_index];

	relPos.x = pos.x - neib_pos.x;
	relPos.y = pos.y - neib_pos.y;
	relPos.z = pos.z - neib_pos.z;
	r = length(relPos);
	if (periodic.x || periodic.y || periodic.z) {
		if (r > influenceradius) {
			relPos += periodic*d_dispvect;
			r = length(relPos);
		}
	}
}


template<>
__device__ __forceinline__ void
getNeibData<false>(	const float4	pos,
					const float4*	posArray,
					const float		influenceradius,
					uint&			neib_index,
					float4&			neib_pos,
					float3&			relPos,
					float&			r)
{
	neib_pos = posArray[neib_index];

	relPos.x = pos.x - neib_pos.x;
	relPos.y = pos.y - neib_pos.y;
	relPos.z = pos.z - neib_pos.z;
	r = length(relPos);
}
/************************************************************************************************************/


/******************** Functions for computing repulsive force directly from DEM *****************************/
// TODO: check for the maximum timestep

// Normal and viscous force wrt to solid boundary
__device__ __forceinline__ float
PlaneForce(	const float4	pos,
			const float4	plane,
			const float		l,
			const float3	vel,
			const float		dynvisc,
			float4&			force)
{
	const float r = abs(dot(as_float3(pos), as_float3(plane)) + plane.w)/l;
	if (r < d_r0) {
		const float DvDt = LJForce(r);
		// Unitary normal vector of the surface
		const float3 relPos = make_float3(plane)*r/l;

		force.x += DvDt*relPos.x;
		force.y += DvDt*relPos.y;
		force.z += DvDt*relPos.z;

		// normal velocity component
		const float normal = dot(vel, relPos)/r;
		const float3 v_n = normal*relPos/r;
		// tangential velocity component
		const float3 v_t = vel - v_n;

		// f = -µ u/∆n

		// viscosity
		// float coeff = -dynvisc*M_PI*(d_r0*d_r0-r*r)/(pos.w*r);
		// float coeff = -dynvisc*M_PI*(d_r0*d_r0*3/(M_PI*2)-r*r)/(pos.w*r);
		const float coeff = -dynvisc*d_partsurf/(pos.w*r);

		// coeff should not be higher than needed to nil v_t in the maximum allowed dt
		// coefficients are negative, so the smallest in absolute value is the biggest

		/*
		float fmag = length(as_float3(force));
		float coeff2 = -sqrt(fmag/slength)/(d_dtadaptfactor*d_dtadaptfactor);
		if (coeff2 < -d_epsartvisc)
			coeff = max(coeff, coeff2);
			*/

		force.x += coeff*v_t.x;
		force.y += coeff*v_t.y;
		force.z += coeff*v_t.z;

		return -coeff;
	}

	return 0.0f;
}

__device__ __forceinline__ float
GeometryForce(	const float4	pos,
				const float3	vel,
				const float		dynvisc,
				float4&			force)
{
	float coeff_max = 0.0f;
	for (uint i = 0; i < d_numplanes; ++i) {
		float coeff = PlaneForce(pos, d_planes[i], d_plane_div[i], vel, dynvisc, force);
		if (coeff > coeff_max)
			coeff_max = coeff;
	}

	return coeff_max;
}


__device__ __forceinline__ float
DemInterpol(const texture<float, 2, cudaReadModeElementType> texref, 
			const float x, 
			const float y)
{
	return tex2D(texref, x/d_ewres + 0.5f, y/d_nsres + 0.5f);
}


__device__ __forceinline__ float
DemLJForce(	const texture<float, 2, cudaReadModeElementType> texref,
			const float4	pos,
			const float3	vel,
			const float		dynvisc,
			float4&			force)
{
	const float z0 = DemInterpol(texref, pos.x, pos.y);
	if (pos.z - z0 < d_demzmin) {
		const float z1 = DemInterpol(texref, pos.x + d_demdx, pos.y);
		const float z2 = DemInterpol(texref, pos.x, pos.y + d_demdy);
		const float a = d_demdy*(z0 - z1);
		const float b = d_demdx*(z0 - z2);
		const float c = d_demdxdy;	// demdx*demdy
		const float d = -a*pos.x - b*pos.y - c*z0;
		const float l = sqrt(a*a+b*b+c*c);
		return PlaneForce(pos, make_float4(a, b, c, d), l, vel, dynvisc, force);
	}
	return 0;
}

/************************************************************************************************************/

/************************************************************************************************************/
/*		   Kernels for computing SPS tensor and SPS viscosity												*/
/************************************************************************************************************/

// Compute the Sub-Particle-Stress (SPS) Tensor matrix for all Particles
// WITHOUT Kernel correction
// Procedure:
// (1) compute velocity gradients
// (2) compute turbulent eddy viscosity (non-dynamic)
// (3) compute turbulent shear stresses
// (4) return SPS tensor matrix (tau) divided by rho^2
template<KernelType kerneltype, bool periodicbound>
__global__ void
__launch_bounds__(BLOCK_SIZE_SPS, MIN_BLOCKS_SPS)
SPSstressMatrixDevice(	const float4* posArray,
						float2*		tau0,
						float2*		tau1,
						float2*		tau2,
						const uint*	neibsList,
						const uint	numParticles,
						const float	slength,
						const float	influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);
	
	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// compute SPS matrix only for fluid particles
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (NOT_FLUID(info))
		return;

	// read particle data from sorted arrays
	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif
	const float4 vel = tex1Dfetch(velTex, index);

	// SPS stress matrix elements
	sym33mat tau;
//	tau.a11 = 0.0f;   // tau11 = tau_xx
//	tau.a12 = 0.0f;   // tau12 = tau_xy
//	tau.a13 = 0.0f;   // tau13 = tau_xz
//	tau.a22 = 0.0f;   // tau22 = tau_yy
//	tau.a23 = 0.0f;   // tau23 = tau_yz
//	tau.a33 = 0.0f;   // tau33 = tau_zz

	// Gradients of the the velocity components
	float3 dvx = make_float3(0.0f);
	float3 dvy = make_float3(0.0f);
	float3 dvz = make_float3(0.0f);

	// first loop over all the neighbors for the Velocity Gradients
	for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave; i += NEIBINDEX_INTERLEAVE) {
		uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		#if( __COMPUTE__ >= 20)							
		getNeibData<periodicbound>(pos, posArray, influenceradius, neib_index, neib_pos, relPos, r);
		#else
		getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
		#endif
		const float4 neib_vel = tex1Dfetch(velTex, neib_index);
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && FLUID(neib_info)) {
			const float f = F<kerneltype>(r, slength)*neib_pos.w/neib_vel.w;	// 1/r ∂Wij/∂r Vj

			float3 relVel;
			relVel.x = vel.x - neib_vel.x;
			relVel.y = vel.y - neib_vel.y;
			relVel.z = vel.z - neib_vel.z;

			// Velocity Gradients
			dvx -= relVel.x*relPos*f;	// dvx = -∑mj/ρj vxij (ri - rj)/r ∂Wij/∂r
			dvy -= relVel.y*relPos*f;	// dvy = -∑mj/ρj vyij (ri - rj)/r ∂Wij/∂r
			dvz -= relVel.y*relPos*f;	// dvy = -∑mj/ρj vzij (ri - rj)/r ∂Wij/∂r
			}
		} // end of loop through neighbors

	// Calculate Sub-Particle Scale viscosity
	// and special turbulent terms
	float SijSij_bytwo = 2.0f*(dvx.x*dvx.x + dvy.y*dvy.y + dvz.z*dvz.z);	// 2*SijSij = 2.0((∂vx/∂x)^2 + (∂vy/∂yx)^2 + (∂vz/∂z)^2)
	float temp = dvx.y + dvy.x;		// 2*SijSij += (∂vx/∂y + ∂vy/∂x)^2
	tau.a12 = temp;
	SijSij_bytwo += temp*temp;
	temp = dvx.z + dvz.x;			// 2*SijSij += (∂vx/∂z + ∂vz/∂x)^2
	tau.a13 = temp;
	SijSij_bytwo += temp*temp;
	temp = dvy.z + dvz.y;			// 2*SijSij += (∂vy/∂z + ∂vz/∂y)^2
	tau.a23 = temp;
	SijSij_bytwo += temp*temp;
	float S = sqrtf(SijSij_bytwo);
	float nu_SPS = d_smagfactor*S;		// Dalrymple & Rogers (2006): eq. (12)
	float divu_SPS = 0.6666666666f*nu_SPS*(dvx.x + dvy.y + dvz.z);
	float Blinetal_SPS = d_kspsfactor*SijSij_bytwo;

	// Shear Stress matrix = TAU (pronounced taf)
	// Dalrymple & Rogers (2006): eq. (10)
	tau.a11 = nu_SPS*(dvx.x + dvx.x) - divu_SPS - Blinetal_SPS;	// tau11 = tau_xx/ρ^2
	tau.a11 /= vel.w;
	tau.a12 *= nu_SPS/vel.w;								// tau12 = tau_xy/ρ^2
	tau.a13 *= nu_SPS/vel.w;								// tau13 = tau_xz/ρ^2
	tau.a22 = nu_SPS*(dvy.y + dvy.y) - divu_SPS - Blinetal_SPS;	// tau22 = tau_yy/ρ^2
	tau.a22 /= vel.w;
	tau.a23 *= nu_SPS/vel.w;								// tau23 = tau_yz/ρ^2
	tau.a33 = nu_SPS*(dvz.z + dvz.z) - divu_SPS - Blinetal_SPS;	// tau33 = tau_zz/ρ^2
	tau.a33 /= vel.w;

	tau0[index] = make_float2(tau.a11, tau.a12);
	tau1[index] = make_float2(tau.a13, tau.a22);
	tau2[index] = make_float2(tau.a23, tau.a33);
}
/************************************************************************************************************/

/************************************************************************************************************/
/*					   Gamma calculations						    */
/************************************************************************************************************/
template<KernelType kerneltype>
__device__ __forceinline__ float4
gradGamma(	const float slength,
		const float r,
		const float4 boundElement)
{
	float4 retval = W<kerneltype>(r, slength) * boundElement.w * boundElement;
	retval.w = 0;
	return retval;
}


template<KernelType kerneltype, bool periodicbound>
__global__ void
initGradGammaDevice(	float4*		newPos,
			float4*		virtualVel,
			float4*		gradGam,
			const uint*	neibsList,
			const uint	numParticles,
			const float	slength,
			const float	inflRadius)
{
	const uint index = INTMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);
	
	if(index < numParticles) {
		float4 pos = tex1Dfetch(posTex, index);
		const particleinfo info = tex1Dfetch(infoTex, index);
		
		// Taking info account self contribution in summation
		float4 gGam = make_float4(0.0f);
		float4 virtVel = make_float4(0.0f);
		
		// Compute gradient of gamma for fluid only
		if(FLUID(info)) {
			//uint counter = 0; //DEBUG

			// Loop over all neighbors
			for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave; i += NEIBINDEX_INTERLEAVE) {
				uint neibIndex = neibsList[d_maxneibsnum_time_neibindexinterleave * lane + offset + i];
				
				if(neibIndex == 0xffffffff) break;
				
				float4 neibPos;
				float3 relPos;
				float r;
				
				getNeibData<periodicbound>(pos, inflRadius, neibIndex, neibPos, relPos, r);
				
				const particleinfo neibInfo = tex1Dfetch(infoTex, neibIndex);
				
				if(r < inflRadius && BOUNDARY(neibInfo)) {
					const float4 boundElement = tex1Dfetch(boundTex, neibIndex);
					gGam += gradGamma<kerneltype>(slength, r, boundElement);
					//counter++; //DEBUG
				}
			}
			//DEBUG output
			//if(counter && ((pos.x < 0.1 && pos.y < 0.1) || (pos.x > 1.35 && pos.y > 1.35)) )
			//	printf("X: %g\tY: %g\tZ: %g\tnumBound: %d\n", pos.x, pos.y, pos.z, counter);
			
			//Set the virtual displacement
			float magnitude = length(make_float3(gGam));
			if (magnitude > 1.e-10) {
				virtVel = -1.0 * inflRadius * gGam / magnitude;
				virtVel.w = 0.0;
			}
		}
		
		// Set gamma to 1
		gGam.w = 1.0;
		
		gradGam[index] = gGam;
		virtualVel[index].x = virtVel.x;
		virtualVel[index].y = virtVel.y;
		virtualVel[index].z = virtVel.z;
		newPos[index] = pos - virtVel;
	}
}


template<KernelType kerneltype, bool periodicbound>
__global__ void
updateGammaDevice(	float4*		newGam,
			const uint*	neibsList,
			const uint	numParticles,
			const float	slength,
			const float	inflRadius,
			const float	virtDt)
{
	const uint index = INTMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);

	if(index < numParticles) {
		float4 pos = tex1Dfetch(posTex, index);
		const particleinfo info = tex1Dfetch(infoTex, index);
		float3 vel = make_float3(tex1Dfetch(velTex, index));
		float4 oldGam = tex1Dfetch(gamTex, index);

		float4 gGam = make_float4(0.0f);
		float deltaGam = 0.0;

		// Compute gradient of gamma for fluid only
		if(FLUID(info)) {
			// Loop over all neighbors
			for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave; i += NEIBINDEX_INTERLEAVE) {
				uint neibIndex = neibsList[d_maxneibsnum_time_neibindexinterleave * lane + offset + i];

				if(neibIndex == 0xffffffff) break;

				float4 neibPos;
				float3 relPos;
				float r;
				
				getNeibData<periodicbound>(pos, inflRadius, neibIndex, neibPos, relPos, r);

				const particleinfo neibInfo = tex1Dfetch(infoTex, neibIndex);

				if(r < inflRadius && BOUNDARY(neibInfo)) {
					const float4 boundElement = tex1Dfetch(boundTex, neibIndex);
					const float4 gradGamma_as = gradGamma<kerneltype>(slength, r, boundElement);
					gGam += gradGamma_as;
					deltaGam += dot(make_float3(gradGamma_as), vel);
				}
			}

			//Update gamma value
			float magnitude = length(make_float3(gGam));
			if (magnitude > 1.e-10) {
				gGam.w = oldGam.w + deltaGam * 0.5*virtDt;
			}
			else
				gGam.w = 1.0;
		}

		newGam[index] = gGam;
	}
}


template<KernelType kerneltype, bool periodicbound>
__global__ void
updateGammaPrCorDevice( float4*		newPos,
			float4*		newGam,
			const uint*	neibsList,
			const uint	numParticles,
			const float	slength,
			const float	inflRadius,
			const float	virtDt)
{
	const uint index = INTMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);

	if(index < numParticles) {
		float4 oldpos = tex1Dfetch(posTex, index);
		float4 newpos = newPos[index];
		const particleinfo info = tex1Dfetch(infoTex, index);
		float3 vel = make_float3(tex1Dfetch(velTex, index));
		float4 oldGam = tex1Dfetch(gamTex, index);

		float4 gGam = make_float4(0.0f);
		float deltaGam = 0.0;
		deltaGam += dot(make_float3(oldGam), vel); //FIXME: It is incorrect for moving boundaries

		// Compute gradient of gamma for fluid only
		if(FLUID(info)) {

			// Loop over all neighbors
			for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave; i += NEIBINDEX_INTERLEAVE) {
				uint neibIndex = neibsList[d_maxneibsnum_time_neibindexinterleave * lane + offset + i];

				if(neibIndex == 0xffffffff) break;

				float4 neibPos;
				float3 relPos;
				float r;

				getNeibData<periodicbound>(newpos, newPos, inflRadius, neibIndex, neibPos, relPos, r);

				const particleinfo neibInfo = tex1Dfetch(infoTex, neibIndex);

				if(r < inflRadius && BOUNDARY(neibInfo)) {
					const float4 boundElement = tex1Dfetch(boundTex, neibIndex);
					const float4 gradGamma_as = gradGamma<kerneltype>(slength, r, boundElement);
					gGam += gradGamma_as;
					deltaGam += dot(make_float3(gradGamma_as), vel);
				}
			}

			//Update gamma value
			float magnitude = length(make_float3(gGam));
			if (magnitude > 1.e-10) {
				gGam.w = oldGam.w + deltaGam * 0.25*virtDt;
			}
			else
				gGam.w = 1.0;
		}

		newGam[index] = gGam;
	}
}


//FIXME: Modify this kernel taking into account periodic boundary
//template<KernelType kerneltype, bool periodicbound>
__global__ void
updatePositionsDevice(	float4*	newPos,
			float	virtDt,
			uint	numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if(index < numParticles) {
		float4 pos = tex1Dfetch(posTex, index);
		const particleinfo info = tex1Dfetch(infoTex, index);
		float4 vel = tex1Dfetch(velTex, index);

		if(FLUID(info)) {
			pos.x += virtDt * vel.x;
			pos.y += virtDt * vel.y;
			pos.z += virtDt * vel.z;
		}

//		if (periodicbound) {
//			if (d_dispvect.x) {
//				if (pos.x >= d_maxlimit.x)
//					pos.x -= d_dispvect.x;
//				else if (pos.x < d_minlimit.x)
//					pos.x += d_dispvect.x;
//			}
//			if (d_dispvect.y) {
//				if (pos.y >= d_maxlimit.y)
//					pos.y -= d_dispvect.y;
//				else if (pos.y < d_minlimit.y)
//					pos.y += d_dispvect.y;
//			}
//			if (d_dispvect.z) {
//				if (pos.z >= d_maxlimit.z)
//					pos.z -= d_dispvect.z;
//				else if (pos.z < d_minlimit.z)
//					pos.z += d_dispvect.z;
//			}
//		}

		newPos[index] = pos;
	}
}

__global__ void
updateBoundValuesDevice(	float4*		oldVel,
				float*		oldPressure,
				const uint	numParticles,
				bool		initStep)
{
	const uint index = INTMUL(blockIdx.x, blockDim.x) + threadIdx.x;
	
	if(index < numParticles) {
		const particleinfo info = tex1Dfetch(infoTex, index);
		const vertexinfo vertices = tex1Dfetch(vertTex, index);
		const float ro1 = oldVel[vertices.x].w;
		const float ro2 = oldVel[vertices.y].w;
		const float ro3 = oldVel[vertices.z].w;
		const float pres1 = oldPressure[vertices.x];
		const float pres2 = oldPressure[vertices.y];
		const float pres3 = oldPressure[vertices.z];

		if (BOUNDARY(info)) {
			oldVel[index].w = (ro1 + ro2 + ro3)/3.f;
			oldPressure[index] = (pres1 + pres2 + pres3)/3.f;
		}
		//FIXME: it should be implemented somewhere in initializeGammaAndGradGamma
		//FIXME: keeping initial velocity values, if given
		if (initStep && FLUID(info)) {
			oldVel[index].x = 0;
			oldVel[index].y = 0;
			oldVel[index].z = 0;
		}
	}
}

template<KernelType kerneltype, bool periodicbound >
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
dynamicBoundConditionsDevice(	const float4*	oldPos,
				float4*		oldVel,
				float*		oldPressure,
				const uint*	neibsList,
				const uint	numParticles,
				const float	slength,
				const float	influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (!VERTEX(info))
		return;

	#if( __COMPUTE__ >= 20)
	const float4 pos = oldPos[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	const float vel = length(make_float3(oldVel[index]));

	// in contrast to Shepard filter particle itself doesn't contribute into summation
	float temp1 = 0;
	float temp2 = 0;
	float alpha = 0;

	// loop over all the neighbors
	for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
		uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		#if( __COMPUTE__ >= 20)
		getNeibData<periodicbound>(pos, oldPos, influenceradius, neib_index, neib_pos, relPos, r);
		#else
		getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
		#endif

//		const float neib_rho = tex1Dfetch(velTex, neib_index).w;
		const float neib_rho = oldVel[neib_index].w;
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);
		const float neib_pres = P(neib_rho, PART_FLUID_NUM(neib_info));
		const float neib_vel = length(make_float3(oldVel[neib_index]));

		if (r < influenceradius && FLUID(neib_info)) {
			const float w = W<kerneltype>(r, slength)*neib_pos.w;
			temp1 += w;
			temp2 += w/neib_rho*(neib_pres/neib_rho + dot(d_gravity,relPos) + 0.5*(neib_vel*neib_vel-vel*vel));
			alpha += w/neib_rho;
		}
	}

	if(alpha)
	{
		oldVel[index].w = temp1/alpha; //FIXME: this can be included directly in the next line
		oldPressure[index] = temp2*oldVel[index].w/alpha;
		oldVel[index].w = rho(oldPressure[index], PART_FLUID_NUM(info));
	}
}

template<KernelType kerneltype, bool periodicbound >
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
calcProbeDevice(	float4*		oldPos,
			float4*		oldVel,
			float*		oldPressure,
			const uint*	neibsList,
			const uint	numParticles,
			const float	slength,
			const float	influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	const particleinfo info = tex1Dfetch(infoTex, index);
	// kernel is only run for probe particles
	if (!PROBE(info))
		return;

	#if( __COMPUTE__ >= 20)
	const float4 pos = oldPos[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	// in contrast to Shepard filter particle itself doesn't contribute into summation
	float pressure = 0;
	float alpha = 0;
	uint num_neib = 0;

	// loop over all the neighbors
	for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
		uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		#if( __COMPUTE__ >= 20)
		getNeibData<periodicbound>(pos, oldPos, influenceradius, neib_index, neib_pos, relPos, r);
		#else
		getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
		#endif

		const float neib_rho = oldVel[neib_index].w;
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && FLUID(neib_info)) {
			const float w = W<kerneltype>(r, slength)*neib_pos.w/neib_rho;
			pressure += w*P(neib_rho, PART_FLUID_NUM(neib_info));
			alpha += w;
			num_neib++;
		}
	}

	if(alpha)
	{
		oldPos[index].w = alpha;
	}
	if(num_neib > 10)
		oldPressure[index] = pressure/alpha;
	else
		oldPressure[index] = 0;
}
/************************************************************************************************************/

/************************************************************************************************************/
/*					   Kernels for computing acceleration without gradient correction					 */
/************************************************************************************************************/

/* Normal kernels */
#include "forces_kernel.xsphdt.inc"

/************************************************************************************************************/


/************************************************************************************************************/
/*					   Kernels for XSPH, Shepard and MLS corrections									   */
/************************************************************************************************************/

// This kernel computes the Shepard correction
template<KernelType kerneltype, bool periodicbound >
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
shepardDevice(	const float4*	posArray,
				float4*			newVel,
				const uint*		neibsList,
				const uint		numParticles,
				const float		slength,
				const float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);
	
	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// normalize kernel only if the given particle is a fluid one
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (NOT_FLUID(info) && !VERTEX(info))
		return;

	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif
	float4 vel = tex1Dfetch(velTex, index);

	// taking into account self contribution in summation
	float temp1 = pos.w*W<kerneltype>(0, slength);
	float temp2 = temp1/vel.w ;

	// loop over all the neighbors
	for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
		uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		#if( __COMPUTE__ >= 20)							
		getNeibData<periodicbound>(pos, posArray, influenceradius, neib_index, neib_pos, relPos, r);
		#else
		getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
		#endif

		const float neib_rho = tex1Dfetch(velTex, neib_index).w;
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && (FLUID(neib_info)/* || VERTEX(neib_info)*/)) {
			const float w = W<kerneltype>(r, slength)*neib_pos.w;
			temp1 += w;
			temp2 += w/neib_rho;
		}
	}

	vel.w = temp1/temp2;
	newVel[index] = vel;
}


// This kernel computes the MLS correction
template<KernelType kerneltype, bool periodicbound>
__global__ void
__launch_bounds__(BLOCK_SIZE_MLS, MIN_BLOCKS_MLS)
MlsDevice(	const float4*	posArray,
			float4*			newVel,
			const uint*		neibsList,
			const uint		numParticles,
			const float		slength,
			const float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);
	
	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// computing MLS matrix only for fluid particles
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (NOT_FLUID(info))
		return;

	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif
	float4 vel = tex1Dfetch(velTex, index);

	// MLS matrix elements
	float a11 = 0.0f, a12 = 0.0f, a13 = 0.0f, a14 = 0.0f;
	float a22 = 0.0f, a23 = 0.0f, a24 = 0.0f;
	float a33 = 0.0f, a34 = 0.0f;
	float a44 = 0.0f;

	// number of neighbors
	int neibs_num = 0;

	// taking into account self contribution in MLS matrix construction
	a11 = W<kerneltype>(0, slength)*pos.w/vel.w;

	// first loop over all the neighbors for the MLS matrix
	for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
		uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		#if( __COMPUTE__ >= 20)							
		getNeibData<periodicbound>(pos, posArray, influenceradius, neib_index, neib_pos, relPos, r);
		#else
		getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
		#endif

		const float neib_rho = tex1Dfetch(velTex, neib_index).w;
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// interaction between two particles
		if (r < influenceradius && FLUID(neib_info)) {
			neibs_num ++;
			const float w = W<kerneltype>(r, slength)*neib_pos.w/neib_rho;	// Wij*Vj
			a11 += w;						// a11 = ∑Wij*Vj
			a12 += relPos.x*w;				// a12 = ∑(xi - xj)*Wij*Vj
			a13 += relPos.y*w;				// a13 = ∑(yi - yj)*Wij*Vj
			a14 += relPos.z*w;				// a14 = ∑(zi - zj)*Wij*Vj
			a22 += relPos.x*relPos.x*w;		// a22 = ∑(xi - xj)^2*Wij*Vj
			a23 += relPos.x*relPos.y*w;		// a23 = ∑(xi - xj)(yi - yj)*Wij*Vj
			a24 += relPos.x*relPos.z*w;		// a23 = ∑(xi - xj)(zi - zj)*Wij*Vj
			a33 += relPos.y*relPos.y*w;		// a33 = ∑(yi - yj)^2*Wij*Vj
			a34 += relPos.y*relPos.z*w;		// a33 = ∑(yi - yj)(zi - zj)*Wij*Vj
			a44 += relPos.z*relPos.z*w;		// a33 = ∑(yi - yj)^2*Wij*Vj
		}
	} // end of first loop trough neighbors

	// safe inverse of MLS matrix
	// the matrix is inverted only if |det|/max|aij|^4 > EPSDET
	// and if the number of fluids neighbors if above a minimum
	// value, otherwise no correction is applied
	float maxa = fmaxf(fabsf(a11), fabsf(a12));
	maxa = fmaxf(maxa, fabsf(a13));
	maxa = fmaxf(maxa, fabsf(a14));
	maxa = fmaxf(maxa, fabsf(a22));
	maxa = fmaxf(maxa, fabsf(a23));
	maxa = fmaxf(maxa, fabsf(a24));
	maxa = fmaxf(maxa, fabsf(a33));
	maxa = fmaxf(maxa, fabsf(a34));
	maxa = fmaxf(maxa, fabsf(a44));
	maxa *= maxa;
	maxa *= maxa;
	float det = a11*(a22*a33*a44 + a23*a34*a24 + a24*a23*a34 - a22*a34*a34 - a23*a23*a44 - a24*a33*a24)
			  + a12*(a12*a34*a34 + a23*a13*a44 + a24*a33*a14 - a12*a33*a44 - a23*a34*a14 - a24*a13*a34)
			  + a13*(a12*a23*a44 + a22*a34*a14 + a24*a13*a24 - a12*a34*a24 - a22*a13*a44 - a24*a23*a14)
			  + a14*(a12*a33*a24 + a22*a13*a34 + a23*a23*a14 - a12*a23*a34 - a22*a33*a14 - a23*a13*a24);
	if (det > maxa*EPSDETMLS && neibs_num > MINCORRNEIBSMLS) {  // FIXME: should be |det| ?????
		// first row of inverse matrix
		det = 1/det;
		const float b11 = (a22*a33*a44 + a23*a34*a24 + a24*a23*a34 - a22*a34*a34 - a23*a23*a44 - a24*a33*a24)*det;
		const float b21 = (a12*a34*a34 + a23*a13*a44 + a24*a33*a14 - a12*a33*a44 - a23*a34*a14 - a24*a13*a34)*det;
		const float b31 = (a12*a23*a44 + a22*a34*a14 + a24*a13*a24 - a12*a34*a24 - a22*a13*a44 - a24*a23*a14)*det;
		const float b41 = (a12*a33*a24 + a22*a13*a34 + a23*a23*a14 - a12*a23*a34 - a22*a33*a14 - a23*a13*a24)*det;

		// taking into account self contribution in density summation
		vel.w = b11*W<kerneltype>(0, slength)*pos.w;

		// second loop over all the neighbors for correction
		for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
			uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

			if (neib_index == 0xffffffff) break;

			float4 neib_pos;
			float3 relPos;
			float r;

			#if( __COMPUTE__ >= 20)							
			getNeibData<periodicbound>(pos, posArray, influenceradius, neib_index, neib_pos, relPos, r);
			#else
			getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
			#endif
			const float neib_rho = tex1Dfetch(velTex, neib_index).w;
			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			// interaction between two particles
			if (r < influenceradius && FLUID(neib_info)) {
				const float w = W<kerneltype>(r, slength)*neib_pos.w;	 // ρj*Wij*Vj = mj*Wij
				vel.w += (b11 + b21*relPos.x + b31*relPos.y
							+ b41*relPos.z)*w;	 // ρ = ∑(ß0 + ß1(xi - xj) + ß2(yi - yj))*Wij*Vj
			}
		}  // end of second loop trough neighbors
	} else {
		// Resort to Shepard filter in absence of invertible matrix
		// see also shepardDevice. TODO: share the code
		// we use a11 and a12 for temp1, temp2
		a11 = pos.w*W<kerneltype>(0, slength);
		a12 = a11/vel.w;

			// loop over all neighbors
		for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
			uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

				if (neib_index == 0xffffffff) break;

				float4 neib_pos;
				float3 relPos;
				float r;

				#if( __COMPUTE__ >= 20)							
				getNeibData<periodicbound>(pos, posArray, influenceradius, neib_index, neib_pos, relPos, r);
				#else
				getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
				#endif
				const float neib_rho = tex1Dfetch(velTex, neib_index).w;
				const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

				// interaction between two particles
				if (r < influenceradius && FLUID(neib_info)) {
						// ρj*Wij*Vj = mj*Wij
						const float w = W<kerneltype>(r, slength)*neib_pos.w;
						// ρ = ∑(ß0 + ß1(xi - xj) + ß2(yi - yj))*Wij*Vj
						a11 += w;
						a12 +=w/neib_rho;
				}
		}  // end of second loop through neighbors

		vel.w = a11/a12;
	}

	newVel[index] = vel;
}
/************************************************************************************************************/

/************************************************************************************************************/
/*					   CFL max kernel																		*/
/************************************************************************************************************/
template <unsigned int blockSize>
__global__ void
fmaxDevice(float *g_idata, float *g_odata, const uint n)
{
	extern __shared__ float sdata[];

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float myMax = 0;

	// we reduce multiple elements per thread.  The number is determined by the 
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{         
		myMax = max(myMax, g_idata[i]);
		// ensure we don't read out of bounds
		if (i + blockSize < n) 
			myMax = max(myMax, g_idata[i + blockSize]);
		i += gridSize;
	} 

	// each thread puts its local sum into shared memory 
	sdata[tid] = myMax;
	__syncthreads();

	// do reduction in shared mem
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] = myMax = max(myMax,sdata[tid + 256]); } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] = myMax = max(myMax,sdata[tid + 128]); } __syncthreads(); }
	if (blockSize >= 128) { if (tid <  64) { sdata[tid] = myMax = max(myMax,sdata[tid +  64]); } __syncthreads(); }

	// now that we are using warp-synchronous programming (below)
	// we need to declare our shared memory volatile so that the compiler
	// doesn't reorder stores to it and induce incorrect behavior.
	if (tid < 32)
	{
		volatile float* smem = sdata;
		if (blockSize >=  64) { smem[tid] = myMax = max(myMax, smem[tid + 32]); }
		if (blockSize >=  32) { smem[tid] = myMax = max(myMax, smem[tid + 16]); }
		if (blockSize >=  16) { smem[tid] = myMax = max(myMax, smem[tid +  8]); }
		if (blockSize >=   8) { smem[tid] = myMax = max(myMax, smem[tid +  4]); }
		if (blockSize >=   4) { smem[tid] = myMax = max(myMax, smem[tid +  2]); }
		if (blockSize >=   2) { smem[tid] = myMax = max(myMax, smem[tid +  1]); }
	}

	// write result for this block to global mem 
	if (tid == 0) 
        g_odata[blockIdx.x] = sdata[0];
}
/************************************************************************************************************/

/************************************************************************************************************/
/*					   Parallel reduction kernels															*/
/************************************************************************************************************/

extern __shared__ float4 shmem4[];

extern "C" __global__
void calcEnergies(
		const float4* pPos,
		const float4* pVel,
		const particleinfo* pInfo,
		uint	numParticles,
		uint	numFluids,
		float4* output
		)
{
	// shared memory for this kernel should be sized to
	// blockDim.x*numFluids*sizeof(float4)*2

	uint gid = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	uint stride = INTMUL(gridDim.x,blockDim.x);
	// .x kinetic, .y potential, .z internal
	float4 energy[MAX_FLUID_TYPES], E_k[MAX_FLUID_TYPES];

#pragma unroll
	for (uint i = 0; i < MAX_FLUID_TYPES; ++i)
		energy[i] = E_k[i] = make_float4(0.0f);

	while (gid < numParticles) {
		float4 pos = pPos[gid];
		float4 vel = pVel[gid];
		particleinfo pinfo = pInfo[gid];
		if (FLUID(pinfo)) {
			uint fluid_num = PART_FLUID_NUM(pinfo);
			float v2 = kahan_sqlength(as_float3(vel));
			float gh = kahan_dot(d_gravity, as_float3(pos));
			kahan_add(energy[fluid_num].x, pos.w*v2/2, E_k[fluid_num].x);
			kahan_add(energy[fluid_num].y, -pos.w*gh, E_k[fluid_num].y);
			// internal elastic energy
			float gamma = d_gammacoeff[fluid_num];
			float gm1 = d_gammacoeff[fluid_num]-1;
			float rho0 = d_rho0[fluid_num];
			float elen = __powf(vel.w/rho0, gm1)/gm1 + rho0/vel.w - gamma/gm1;
			float ssp = soundSpeed(vel.w, fluid_num);
			elen *= ssp*ssp/gamma;
			kahan_add(energy[fluid_num].z, pos.w*elen, E_k[fluid_num].z);
		}
		gid += stride;
	}

	uint lid = threadIdx.x;
	for (uint offset = blockDim.x/2; offset; offset >>= 1) {
		stride = offset*numFluids; // stride between fields in shmem4 memory
		if (lid >= offset && lid < 2*offset) {
			for (uint i = 0; i < numFluids; ++i) {
				uint idx = lid + offset*i;
				shmem4[idx] = energy[i];
				idx += stride;
				shmem4[idx] = E_k[i];
			}
		}
		__syncthreads();
		if (lid < offset) {
			for (uint i = 0; i < numFluids; ++i) {
				uint idx = lid + offset*(i+1);
				float4 other = shmem4[idx];
				idx += stride;
				float4 oth_k = shmem4[idx];
				kahan_add(energy[i].x, oth_k.x, E_k[i].x);
				kahan_add(energy[i].x, other.x, E_k[i].x);
				kahan_add(energy[i].y, oth_k.y, E_k[i].y);
				kahan_add(energy[i].y, other.y, E_k[i].y);
				kahan_add(energy[i].z, oth_k.z, E_k[i].z);
				kahan_add(energy[i].z, other.z, E_k[i].z);
			}
		}
	}

	if (lid == 0) {
		for (uint i = 0; i < numFluids; ++i) {
			output[blockIdx.x + INTMUL(gridDim.x,i)] = energy[i];
			output[blockIdx.x + INTMUL(gridDim.x,numFluids+i)] = E_k[i];
		}
	}
}

// final reduction stage
extern "C" __global__
void calcEnergies2(
		float4* buffer,
		uint	prev_blocks,
		uint	numFluids)
{
	// shared memory for this kernel should be sized to
	// blockDim.x*numFluids*sizeof(float4)*2

	uint gid = threadIdx.x;
	float4 energy[MAX_FLUID_TYPES];
	float4 E_k[MAX_FLUID_TYPES];
	for (uint i = 0; i < numFluids; ++i) {
		if (gid < prev_blocks) {
			energy[i] = buffer[gid + prev_blocks*i];
			E_k[i] = buffer[gid + prev_blocks*(numFluids+i)];
		} else {
			energy[i] = E_k[i] = make_float4(0.0f);
		}
	}

	uint stride;
	for (uint offset = blockDim.x/2; offset; offset >>= 1) {
		stride = offset*numFluids; // stride between fields in shmem4 memory
		if (gid >= offset && gid < 2*offset) {
			for (uint i = 0; i < numFluids; ++i) {
				uint idx = gid + offset*i;
				shmem4[idx] = energy[i];
				idx += stride;
				shmem4[idx] = E_k[i];
			}
		}
		__syncthreads();
		if (gid < offset) {
			for (uint i = 0; i < numFluids; ++i) {
				uint idx = gid + offset*(i+1);
				float4 other = shmem4[idx];
				idx += stride;
				float4 oth_k = shmem4[idx];
				kahan_add(energy[i].x, oth_k.x, E_k[i].x);
				kahan_add(energy[i].x, other.x, E_k[i].x);
				kahan_add(energy[i].y, oth_k.y, E_k[i].y);
				kahan_add(energy[i].y, other.y, E_k[i].y);
				kahan_add(energy[i].z, oth_k.z, E_k[i].z);
				kahan_add(energy[i].z, other.z, E_k[i].z);
			}
		}
	}

	if (gid == 0) {
		for (uint i = 0; i < numFluids; ++i)
			buffer[i] = energy[i] + E_k[i];
	}
}


/************************************************************************************************************/
/*					   Auxiliary kernels used for post processing										    */
/************************************************************************************************************/

// This kernel compute the vorticity field
template<KernelType kerneltype, bool periodicbound>
__global__ void
calcVortDevice(	float3*		vorticity,
				const uint*	neibsList,
				const uint	numParticles,
				const float	slength,
				const float	influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);
	
	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// computing vorticity only for fluid particles
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (NOT_FLUID(info))
		return;

	float4 pos = tex1Dfetch(posTex, index);
	float4 vel = tex1Dfetch(velTex, index);

	// MLS matrix elements
	float3 vort = make_float3(0.0f);

	// loop over all the neighbors
	for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
		uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
		const float4 neib_vel = tex1Dfetch(velTex, neib_index);
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// interaction between two particles
		if (r < influenceradius && FLUID(neib_info)) {
			float3 relVel;
			relVel.x = vel.x - neib_vel.x;
			relVel.y = vel.y - neib_vel.y;
			relVel.z = vel.z - neib_vel.z;
			const float f = F<kerneltype>(r, slength)*neib_pos.w/neib_vel.w;	// ∂Wij/∂r*Vj
			// vxij = vxi - vxj and same for vyij and vzij
			vort.x += f*(relVel.y*relPos.z - relVel.z*relPos.y);		// vort.x = ∑(vyij(zi - zj) - vzij*(yi - yj))*∂Wij/∂r*Vj
			vort.y += f*(relVel.z*relPos.x - relVel.x*relPos.z);		// vort.y = ∑(vzij(xi - xj) - vxij*(zi - zj))*∂Wij/∂r*Vj
			vort.z += f*(relVel.x*relPos.y - relVel.y*relPos.x);		// vort.x = ∑(vxij(yi - yj) - vyij*(xi - xj))*∂Wij/∂r*Vj
		}
	} // end of loop trough neighbors

	vorticity[index] = vort;
}


// Testpoints
// This kernel compute the velocity at testpoints
template<KernelType kerneltype, bool periodicbound >
__global__ void
calcTestpointsVelocityDevice(	float4*		newVel,
								const uint*	neibsList,
								const uint	numParticles,
								const float	slength,
								const float	influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);
	
	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	const particleinfo info = tex1Dfetch(infoTex, index);
	if(type(info) != TESTPOINTSPART)
		return;
	
	float4 pos = tex1Dfetch(posTex, index);
	float4 vel = tex1Dfetch(velTex, index);
	
	float4 temp = make_float4(0.0f);

	// loop over all the neighbors
	for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
		uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
		const float4 neib_vel = tex1Dfetch(velTex, neib_index);
        const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && FLUID(neib_info)) {
			const float w = W<kerneltype>(r, slength)*neib_pos.w/neib_vel.w;	// Wij*mj
			temp.x += w*neib_vel.x;
			temp.y += w*neib_vel.y;
			temp.z += w*neib_vel.z;
			//Pressure
			temp.w += w*P(neib_vel.w, object(neib_info));

		}
	}

	vel = temp;

	newVel[index] = vel;
}


// Free surface detection
// This kernel detects the surface particles
template<KernelType kerneltype, bool periodicbound, bool savenormals>
__global__ void
calcSurfaceparticleDevice(	float4*			normals,
							particleinfo*	newInfo,
							const uint*		neibsList,
							const uint		numParticles,
							const float		slength,
							const float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;
	const uint lane = index/NEIBINDEX_INTERLEAVE;
	const uint offset = threadIdx.x & (NEIBINDEX_INTERLEAVE - 1);
	
	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	particleinfo info = tex1Dfetch(infoTex, index);

	if (NOT_FLUID(info)) {
		newInfo[index] = info;		
		return;
	}

	float4 pos = tex1Dfetch(posTex, index);
	float4 normal = make_float4(0.0f);
	
	info.x &= ~SURFACE_PARTICLE_FLAG;
	normal.w = W<kerneltype>(0.0f, slength)*pos.w;

	// loop over all the neighbors (First loop)
	for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
		uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
		const float neib_density = tex1Dfetch(velTex, neib_index).w;

		if (r < influenceradius) {
			const float f = F<kerneltype>(r, slength)* neib_pos.w /neib_density; // 1/r ∂Wij/∂r Vj
			normal.x -= f * relPos.x;
			normal.y -= f * relPos.y;
			normal.z -= f * relPos.z;
			normal.w += W<kerneltype>(r, slength)*neib_pos.w;	// Wij*mj ;

		}
	}

	float normal_length = length(as_float3(normal));

	//Checking the planes
	for (uint i = 0; i < d_numplanes; ++i) {
		float r = abs(dot(as_float3(pos), as_float3(d_planes[i])) + d_planes[i].w)/d_plane_div[i];
		if (r < influenceradius) {
			as_float3(normal) += as_float3(d_planes[i])* normal_length;
			normal_length = length(as_float3(normal));
		}
	}

	// loop over all the neighbors (Second loop)
	int nc = 0;
	for(uint i = 0; i < d_maxneibsnum_time_neibindexinterleave ; i += NEIBINDEX_INTERLEAVE) {
		uint neib_index = neibsList[d_maxneibsnum_time_neibindexinterleave*lane + i + offset];
		
		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		float cosconeangle;

		getNeibData<periodicbound>(pos, influenceradius, neib_index, neib_pos, relPos, r);
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius) {
			float criteria = -(normal.x * relPos.x + normal.y * relPos.y + normal.z * relPos.z);
			if (FLUID(neib_info))
				cosconeangle = d_cosconeanglefluid;
			else
				cosconeangle = d_cosconeanglenonfluid;

			if (criteria > r*normal_length*cosconeangle)
				nc++;
		}

	}

	if (!nc)
		info.x |= SURFACE_PARTICLE_FLAG;

	newInfo[index] = info;

	if (savenormals) {
		normal.x /= normal_length;
		normal.y /= normal_length;
		normal.z /= normal_length;
		normals[index] = normal;
		}

}
/************************************************************************************************************/

} //namespace cuforces
#endif
