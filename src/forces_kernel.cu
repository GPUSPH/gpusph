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

texture<float, 2, cudaReadModeElementType> demTex;	// DEM

__constant__ float	d_wcoeff_cubicspline;			// coeff = 1/(Pi h^3)
__constant__ float	d_wcoeff_quadratic;				// coeff = 15/(16 Pi h^3)
__constant__ float	d_wcoeff_wendland;				// coeff = 21/(16 Pi h^3)

__constant__ float	d_fcoeff_cubicspline;			// coeff = 3/(4Pi h^4)
__constant__ float	d_fcoeff_quadratic;				// coeff = 15/(32Pi h^4)
__constant__ float	d_fcoeff_wendland;				// coeff = 105/(128Pi h^5)

__constant__ int    d_numFluids;					// number of different fluids

__constant__ float	d_rho0[MAX_FLUID_TYPES];		// rest density of fluids

// Speed of sound constants
__constant__ float	d_bcoeff[MAX_FLUID_TYPES];
__constant__ float	d_gammacoeff[MAX_FLUID_TYPES];
__constant__ float	d_sscoeff[MAX_FLUID_TYPES];
__constant__ float	d_sspowercoeff[MAX_FLUID_TYPES];

__constant__ float3	d_gravity;						// gravity (vector)

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

__constant__ float3	d_dispvect2;					// displacment vector for periodic boundaries

// Constants used for DEM
__constant__ float	d_ewres;
__constant__ float	d_nsres;
__constant__ float	d_demdx;
__constant__ float	d_demdy;
__constant__ float	d_demdxdy;
__constant__ float	d_demzmin;

__constant__ float	d_partsurf;						// particle surface

// Definition of planes for geometrical boundaries
__constant__ uint	d_numPlanes;
__constant__ float4	d_planes[MAXPLANES];
__constant__ float	d_plane_div[MAXPLANES];

// Sub-Particle Scale (SPS) Turbulence parameters
__constant__ float	d_smagfactor;
__constant__ float	d_kspsfactor;

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
__device__ float
W(float r, float slength);


// Cubic Spline kernel
template<>
__device__ float
W<CUBICSPLINE>(float r, float slength)
{
	float val = 0.0f;
	float R = r/slength;

	if (R < 1)
		val = 1.0f - 1.5f*R*R + 0.75f*R*R*R;			// val = 1 - 3/2 R^2 + 3/4 R^3
	else
		val = 0.25f*(2.0f - R)*(2.0f - R)*(2.0f - R);	// val = 1/4 (2 - R)^3

	val *= d_wcoeff_cubicspline;						// coeff = 1/(Pi h^3)

	return val;
}


// Qudratic kernel
template<>
__device__ float
W<QUADRATIC>(float r, float slength)
{
	float val = 0.0f;
	float R = r/slength;

	val = 0.25f*R*R - R + 1.0f;		// val = 1/4 R^2 -  R + 1
	val *= d_wcoeff_quadratic;		// coeff = 15/(16 Pi h^3)

	return val;
}


// Wendland kernel
template<>
__device__ float
W<WENDLAND>(float r, float slength)
{
	float R = r/slength;

	float val = 1 - 0.5f*R;
	val *= val;
	val *= val;						// val = (1 - R/2)^4
	val *= 1 + 2.0f*R;				// val = (2R + 1)(1 - R/2)^4*
	val *= d_wcoeff_wendland;		// coeff = 21/(16 Pi h^3)
	return val;
}


// Return 1/r dW/dr at distance r, for a given smoothing length
template<KernelType kerneltype>
__device__ float
F(float r, float slength);


template<>
__device__ float
F<CUBICSPLINE>(float r, float slength)
{
	float val = 0.0f;
	float R = r/slength;

	if (R < 1.0f)
		val = (-4.0f + 3.0f*R)/slength;		// val = (-4 + 3R)/h
	else
		val = -(-2.0f + R)*(-2.0f + R)/r;	// val = -(-2 + R)^2/r
	val *= d_fcoeff_cubicspline;			// coeff = 3/(4Pi h^4)

	return val;
}


template<>
__device__ float
F<QUADRATIC>(float r, float slength)
{
	float R = r/slength;

	float val = (-2.0f + R)/r;		// val = (-2 + R)/r
	val *= d_fcoeff_quadratic;		// coeff = 15/(32Pi h^4)

	return val;
}


template<> __device__ float
F<WENDLAND>(float r, float slength)
{
	float qm2 = r/slength - 2.0f;	// val = (-2 + R)^3
	float val = qm2*qm2*qm2*d_fcoeff_wendland;
	return val;
}
/************************************************************************************************************/


/********************** Equation of state, speed of sound, repulsive force **********************************/
// Equation of state: pressure from density, where i is the fluid kind, not particle_id
__device__ float
P(float rho, uint i)
{
	return d_bcoeff[i]*(__powf(rho/d_rho0[i], d_gammacoeff[i]) - 1);
}


// Sound speed computed from density
__device__ float
soundSpeed(float rho, uint i)
{
	return d_sscoeff[i]*__powf(rho/d_rho0[i], d_sspowercoeff[i]);
}


// Lennard-Jones boundary repulsion force
__device__ float
LJForce(float r)
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
__device__ float
MKForce(const float &r, const float &slength,
		const float &mass_f, const float &mass_b)
{
	// MK always uses the 1D cubic or quintic Wendland spline
	float w = 0;

	float force = 0;

	// Wendland has radius 2
	if (r <= 2*slength) {
		float qq = r/slength;
		w = 1.8f * __powf(1.0f - 0.5f*qq, 4.0f) * (2.0f*qq + 1.0f);
		// float dist = r - d_MK_d;
		float dist = max(d_epsartvisc, r - d_MK_d);
		force = d_MK_K*w*2*mass_b/(d_MK_beta * dist * r * (mass_f+mass_b));
	}

	return force;
}
/************************************************************************************************************/


/***************************************** Viscosities *******************************************************/
// Artificial viscosity scalar part when the projection
// of the velocity (hx.u/r) is not precomputed
__device__ float
artvisc(	float	vel_dot_pos,
			float	rho,
			float	neib_rho,
			float	sspeed,
			float	neib_sspeed,
			float	r,
			float	slength)
{
	return vel_dot_pos*slength*d_visccoeff*(sspeed + neib_sspeed)/
									((r*r + d_epsartvisc)*(rho + neib_rho));
}


// Artificial viscosity scalar part when the projection
// of the relative velocity (hx.u/r) is needed for the
// adaptaive time step control and so precomputed
__device__ float
artviscdt(	float	prelvel_by_slength,
			float	rho,
			float	neib_rho,
			float	sspeed,
			float	neib_sspeed)
{
	return prelvel_by_slength*d_visccoeff*(sspeed + neib_sspeed)/(rho + neib_rho);
}

// ATTENTION: for all non artificial viscosity
// µ is the dynamic viscosity (ρν)

// Scalar part of viscosity using Morris 1997
// expression 21 p218 when all particles have the same viscosity
// in this case d_visccoeff = 4 nu
// returns 4.mj.nu/(ρi + ρj) (1/r ∂Wij/∂r)
__device__ float
laminarvisc_kinematic(	float	rho,
						float	neib_rho,
						float	neib_mass,
						float	f)
{
	return neib_mass*d_visccoeff*f/(rho + neib_rho);
}


// Same behaviour as laminarvisc but for particle
// dependent viscosity.
// returns mj.(µi + µi)/(ρi.ρj) (1/r ∂Wij/∂r)
__device__ float
laminarvisc_dynamic(float	rho,
					float	neib_rho,
					float	neib_mass,
					float	f,
					float	visc,
					float	neib_visc)
{
	return neib_mass*(visc + neib_visc)*f/(rho*neib_rho);
}
/************************************************************************************************************/


/*********************************** Adptative time stepping ************************************************/
// Function called at the end of the forces or powerlawVisc function doing
// a per block maximum reduction
__device__ void
dtadaptBlockReduce(	float	*s_cfl,
					float	*cfl)
{
   __syncthreads();

   if (threadIdx.x % WARPSIZE == 0) {
		int offset = threadIdx.x/WARPSIZE;
		for (int i = 1; i < WARPSIZE; i++) {
			if (s_cfl[i + offset*WARPSIZE] > s_cfl[offset*WARPSIZE])
				s_cfl[offset*WARPSIZE] = s_cfl[i + offset*WARPSIZE];
		}
	}
	__syncthreads();

	if(threadIdx.x == 0) {
		for (int i = 1; i < BLOCK_SIZE_FORCES/WARPSIZE; i++) {
			if (s_cfl[i*WARPSIZE] > s_cfl[0])
				s_cfl[0] = s_cfl[i*WARPSIZE];
		}
		cfl[blockIdx.x] = s_cfl[0];
	}
}
/************************************************************************************************************/


/********************************* Periodic boundary management *********************************************/
// Function returning the neigbor index, position, relative distance and velocity
template<bool periodicbound>
__device__ void
getNeibData(float4	pos,
			uint*	neibsList,
			float	influenceradius,
			uint&	neib_index,
			float4&	neib_pos,
			float3&	relPos,
			float&	r);


// In case of periodic boundaries we add the displacement
// vector when needed
template<>
__device__ void
getNeibData<true>(	float4	pos,
					uint*	neibsList,
					float	influenceradius,
					uint&	neib_index,
					float4&	neib_pos,
					float3&	relPos,
					float&	r)
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
			relPos += periodic*d_dispvect2;
			r = length(relPos);
		}
	}
}


template<>
__device__ void
getNeibData<false>(	float4	pos,
					uint*	neibsList,
					float	influenceradius,
					uint&	neib_index,
					float4&	neib_pos,
					float3&	relPos,
					float&	r)
{
		neib_pos = tex1Dfetch(posTex, neib_index);

		relPos.x = pos.x - neib_pos.x;
		relPos.y = pos.y - neib_pos.y;
		relPos.z = pos.z - neib_pos.z;
		r = length(relPos);
}
/************************************************************************************************************/


/******************** Functions for computing repulsive force directly from DEM *****************************/
// TODO: check for the maximum timestep

// Normal and viscous force wrt to solid boundary
__device__ float
PlaneForce(	float4	pos,
			float4	plane,
			float	l,
			float3	vel,
			float	dynvisc,
			float	slength,
			float	influenceradius,
			float4&	force)
{
	float r = abs(dot(as_float3(pos), as_float3(plane)) + plane.w)/l;
	if (r < d_r0) {
		float DvDt = LJForce(r);
		// Unitary normal vector of the surface
		float3 relPos = make_float3(plane)*r/l;

		force.x += DvDt*relPos.x;
		force.y += DvDt*relPos.y;
		force.z += DvDt*relPos.z;

		// normal velocity component
		float normal = dot(vel, relPos)/r;
		float3 v_n = normal*relPos/r;
		// tangential velocity component
		float3 v_t = vel - v_n;

		// f = -µ u/∆n

		// viscosity
		// float coeff = -dynvisc*M_PI*(d_r0*d_r0-r*r)/(pos.w*r);
		// float coeff = -dynvisc*M_PI*(d_r0*d_r0*3/(M_PI*2)-r*r)/(pos.w*r);
		float coeff = -dynvisc*d_partsurf/(pos.w*r);

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

__device__ float
GeometryForce(	float4	pos,
				float3	vel,
				float	dynvisc,
				float	slength,
				float	influenceradius,
				float4&	force)
{
	float coeff_max = 0.0f;
	for (uint i = 0; i < d_numPlanes; ++i) {
		float coeff = PlaneForce(pos, d_planes[i], d_plane_div[i], vel, dynvisc,
				slength, influenceradius, force);
		if (coeff > coeff_max)
			coeff_max = coeff;
	}

	return coeff_max;
}


__device__ float
DemInterpol(const texture<float, 2, cudaReadModeElementType> texref, float x, float y)
{
	return tex2D(texref, x/d_ewres + 0.5f, y/d_nsres + 0.5f);
}


__device__ float
DemLJForce(	const texture<float, 2, cudaReadModeElementType> texref,
			float4	pos,
			float3	vel,
			float	dynvisc,
			float	slength,
			float	influenceradius,
			float4&	force)
{
	float z0 = DemInterpol(texref, pos.x, pos.y);
	if (pos.z - z0 < d_demzmin) {
		float z1 = DemInterpol(texref, pos.x + d_demdx, pos.y);
		float z2 = DemInterpol(texref, pos.x, pos.y + d_demdy);
		float a = d_demdy*(z0 - z1);
		float b = d_demdx*(z0 - z2);
		float c = d_demdxdy;	// demdx*demdy
		float d = -a*pos.x - b*pos.y - c*z0;
		float l = sqrt(a*a+b*b+c*c);
		return PlaneForce(pos, make_float4(a, b, c, d), l, vel, dynvisc,
				slength, influenceradius, force);
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
SPSstressMatrixDevice(	float2*	tau0,
						float2*	tau1,
						float2*	tau2,
						uint*	neibsList,
						uint	numParticles,
						float	slength,
						float	influenceradius)
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// compute SPS matrix only for fluid particles
	particleinfo info = tex1Dfetch(infoTex, index);
	if (!FLUID(info))
		return;

	// read particle data from sorted arrays
	float4 pos = tex1Dfetch(posTex, index);
	float4 vel = tex1Dfetch(velTex, index);

	// SPS stress matrix elements
	sym33mat tau;
	tau.a11 = 0.0f;   // tau11 = tau_xx
	tau.a12 = 0.0f;   // tau12 = tau_xy
	tau.a13 = 0.0f;   // tau13 = tau_xz
	tau.a22 = 0.0f;   // tau22 = tau_yy
	tau.a23 = 0.0f;   // tau23 = tau_yz
	tau.a33 = 0.0f;   // tau33 = tau_zz

	// Gradients of the the velocity components
	float3 dvx = make_float3(0.0f);
	float3 dvy = make_float3(0.0f);
	float3 dvz = make_float3(0.0f);

	// first loop over all the neighbors for the Velocity Gradients
	for(uint i = index*MAXNEIBSNUM; i < index*MAXNEIBSNUM + MAXNEIBSNUM; i++) {
		uint neib_index = neibsList[i];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		getNeibData<periodicbound>(pos, neibsList, influenceradius, neib_index, neib_pos, relPos, r);
		float4 neib_vel = tex1Dfetch(velTex, neib_index);
		particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && FLUID(neib_info)) {
			float f = F<kerneltype>(r, slength)*neib_pos.w/neib_vel.w;	// 1/r ∂Wij/∂r Vj

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
/*					   Kernels for computing acceleration without gradient correction					 */
/************************************************************************************************************/

/* Normal kernels */
#include "forces_kernel.xsphdt.inc"

/************************************************************************************************************/


/************************************************************************************************************/
/*					   Kernels for XSPH, Shepard and MLS corrections									   */
/************************************************************************************************************/

// This kernel computes only the XSPH correction
template<KernelType kernel_type, bool periodicbound>
__global__ void
xsphDevice(	float4*	xsph,
			uint*	neibsList,
			uint	numParticles,
			float	slength,
			float	influenceradius)
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// normalize kernel only if the given particle is a fluid one
	particleinfo info = tex1Dfetch(infoTex, index);
	if (!FLUID(info))
		return;

	// read particle data from sorted arrays
	float4 pos = tex1Dfetch(posTex, index);
	float4 vel = tex1Dfetch(velTex, index);

	// force and density derivative
	float3 mean_vel = make_float3(0.0f);

	// loop over all neighbors
	for(uint i = index*MAXNEIBSNUM; i < index*MAXNEIBSNUM + MAXNEIBSNUM; i++) {
		uint neib_index = neibsList[i];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		getNeibData<periodicbound>(pos, neibsList, influenceradius, neib_index, neib_pos, relPos, r);
		float4 neib_vel = tex1Dfetch(velTex, neib_index);

		if (r < influenceradius && FLUID(neib_pos)) {
			float3 relVel;
			relVel.x = vel.x - neib_vel.x;
			relVel.y = vel.y - neib_vel.y;
			relVel.z = vel.z - neib_vel.z;

			mean_vel -= 2.0f*neib_pos.w*W<kernel_type>(r, slength)*relVel/(vel.w + neib_vel.w);
		}
	}

	xsph[index] = make_float4(mean_vel, 0.0f);
}


// This kernel computes the Shepard correction
template<KernelType kerneltype, bool periodicbound >
__global__ void
shepardDevice(	float4*	newVel,
				uint*	neibsList,
				uint	numParticles,
				float	slength,
				float	influenceradius)
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// normalize kernel only if the given particle is a fluid one
	particleinfo info = tex1Dfetch(infoTex, index);
	if (!FLUID(info))
		return;

	float4 pos = tex1Dfetch(posTex, index);
	float4 vel = tex1Dfetch(velTex, index);

	// taking into account self contribution in summation
	float temp1 = pos.w*W<kerneltype>(0, slength);
	float temp2 = temp1/vel.w ;

	// loop over all the neighbors
	for(uint i = index*MAXNEIBSNUM; i < index*MAXNEIBSNUM + MAXNEIBSNUM; i++) {
		uint neib_index = neibsList[i];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		getNeibData<periodicbound>(pos, neibsList, influenceradius, neib_index, neib_pos, relPos, r);
		float neib_rho = tex1Dfetch(velTex, neib_index).w;
		particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && FLUID(neib_info)) {
			float w = W<kerneltype>(r, slength)*neib_pos.w;
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
MlsDevice(	float4*	newVel,
			uint*	neibsList,
			uint	numParticles,
			float	slength,
			float	influenceradius)
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// computing MLS matrix only for fluid particles
	particleinfo info = tex1Dfetch(infoTex, index);
	if (!FLUID(info))
		return;

	float4 pos = tex1Dfetch(posTex, index);
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
	for(uint i = index*MAXNEIBSNUM; i < index*MAXNEIBSNUM + MAXNEIBSNUM; i++) {
		uint neib_index = neibsList[i];

		if (neib_index == 0xffffffff) break;

		float4 neib_pos;
		float3 relPos;
		float r;

		getNeibData<periodicbound>(pos, neibsList, influenceradius, neib_index, neib_pos, relPos, r);
		float neib_rho = tex1Dfetch(velTex, neib_index).w;
		particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// interaction between two particles
		if (r < influenceradius && FLUID(neib_info)) {
			neibs_num ++;
			float w = W<kerneltype>(r, slength)*neib_pos.w/neib_rho;	// Wij*Vj
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
	float maxa = fmax(fabs(a11), fabs(a12));
	maxa = fmax(maxa, fabs(a13));
	maxa = fmax(maxa, fabs(a14));
	maxa = fmax(maxa, fabs(a22));
	maxa = fmax(maxa, fabs(a23));
	maxa = fmax(maxa, fabs(a24));
	maxa = fmax(maxa, fabs(a33));
	maxa = fmax(maxa, fabs(a34));
	maxa = fmax(maxa, fabs(a44));
	maxa = __powf(maxa, 4);
	float det = a11*(a22*a33*a44 + a23*a34*a24 + a24*a23*a34 - a22*a34*a34 - a23*a23*a44 - a24*a33*a24)
			  + a12*(a12*a34*a34 + a23*a13*a44 + a24*a33*a14 - a12*a33*a44 - a23*a34*a14 - a24*a13*a34)
			  + a13*(a12*a23*a44 + a22*a34*a14 + a24*a13*a24 - a12*a34*a24 - a22*a13*a44 - a24*a23*a14)
			  + a14*(a12*a33*a24 + a22*a13*a34 + a23*a23*a14 - a12*a23*a34 - a22*a33*a14 - a23*a13*a24);
	if (det > maxa*EPSDETMLS && neibs_num > MINCORRNEIBSMLS) {
		// first row of inverse matrix
		det = 1/det;
		float b11 = (a22*a33*a44 + a23*a34*a24 + a24*a23*a34 - a22*a34*a34 - a23*a23*a44 - a24*a33*a24)*det;
		float b21 = (a12*a34*a34 + a23*a13*a44 + a24*a33*a14 - a12*a33*a44 - a23*a34*a14 - a24*a13*a34)*det;
		float b31 = (a12*a23*a44 + a22*a34*a14 + a24*a13*a24 - a12*a34*a24 - a22*a13*a44 - a24*a23*a14)*det;
		float b41 = (a12*a33*a24 + a22*a13*a34 + a23*a23*a14 - a12*a23*a34 - a22*a33*a14 - a23*a13*a24)*det;

		// taking into account self contribution in density summation
		vel.w = b11*W<kerneltype>(0, slength)*pos.w;

		// second loop over all the neighbors for correction
		for(uint i = index*MAXNEIBSNUM; i < index*MAXNEIBSNUM + MAXNEIBSNUM; i++) {
			uint neib_index = neibsList[i];

			if (neib_index == 0xffffffff) break;

			float4 neib_pos;
			float3 relPos;
			float r;

			getNeibData<periodicbound>(pos, neibsList, influenceradius, neib_index, neib_pos, relPos, r);
			float neib_rho = tex1Dfetch(velTex, neib_index).w;
			particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			// interaction between two particles
			if (r < influenceradius && FLUID(neib_info)) {
				float w = W<kerneltype>(r, slength)*neib_pos.w;	 // ρj*Wij*Vj = mj*Wij
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
			for(uint i = index*MAXNEIBSNUM; i < index*MAXNEIBSNUM + MAXNEIBSNUM; i++) {
					uint neib_index = neibsList[i];

					if (neib_index == 0xffffffff) break;

					float4 neib_pos;
					float3 relPos;
					float r;

					getNeibData<periodicbound>(pos, neibsList, influenceradius, neib_index, neib_pos, relPos, r);
					float neib_rho = tex1Dfetch(velTex, neib_index).w;
					particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

					// interaction between two particles
					if (r < influenceradius && FLUID(neib_info)) {
							// ρj*Wij*Vj = mj*Wij
							float w = W<kerneltype>(r, slength)*neib_pos.w;
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
/*					   Auxiliary kernels used for post processing										 */
/************************************************************************************************************/

// This kernel compute the vorticity field
template<KernelType kerneltype, bool periodicbound>
__global__ void
calcVortDevice(	float3*	vorticity,
				uint*	neibsList,
				uint	numParticles,
				float	slength,
				float	influenceradius)
{
	int index = __mul24(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	float4 pos = tex1Dfetch(posTex, index);
	float4 vel = tex1Dfetch(velTex, index);
	particleinfo info = tex1Dfetch(infoTex, index);

	// MLS matrix elements
	float3 vort = make_float3(0.0f);

	// computing vorticity only for fluid particles
	if (FLUID(info)) {

		// loop over all the neighbors
		for(uint i = index*MAXNEIBSNUM; i < index*MAXNEIBSNUM + MAXNEIBSNUM; i++) {
			uint neib_index = neibsList[i];

			if (neib_index == 0xffffffff) break;

			float4 neib_pos;
			float3 relPos;
			float r;

			getNeibData<periodicbound>(pos, neibsList, influenceradius, neib_index, neib_pos, relPos, r);
			float4 neib_vel = tex1Dfetch(velTex, neib_index);
			particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			// interaction between two particles
			if (r < influenceradius && FLUID(neib_info)) {
				float3 relVel;
				relVel.x = vel.x - neib_vel.x;
				relVel.y = vel.y - neib_vel.y;
				relVel.z = vel.z - neib_vel.z;
				float f = F<kerneltype>(r, slength)*neib_pos.w/neib_vel.w;	// ∂Wij/∂r*Vj
				// vxij = vxi - vxj and same for vyij and vzij
				vort.x += f*(relVel.y*relPos.z - relVel.z*relPos.y);		// vort.x = ∑(vyij(zi - zj) - vzij*(yi - yj))*∂Wij/∂r*Vj
				vort.y += f*(relVel.z*relPos.x - relVel.x*relPos.z);		// vort.y = ∑(vzij(xi - xj) - vxij*(zi - zj))*∂Wij/∂r*Vj
				vort.z += f*(relVel.x*relPos.y - relVel.y*relPos.x);		// vort.x = ∑(vxij(yi - yj) - vyij*(xi - xj))*∂Wij/∂r*Vj
			}
		} // end of loop trough neighbors
	} // if fluid part

	vorticity[index] = vort;
}
/************************************************************************************************************/

#endif
