/*  Copyright 2013 Alexis Herault, Giuseppe Bilotta, Robert A.
 	Dalrymple, Eugenio Rustico, Ciro Del Negro

	Conservatoire National des Arts et Metiers, Paris, France

	Istituto Nazionale di Geofisica e Vulcanologia,
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

#include <cfloat> //FLT_EPSILON

#include "particledefine.h"
#include "textures.cuh"
#include "vector_math.h"
#include "multi_gpu_defines.h"
#include "GlobalData.h"

#include "kahan.h"
#include "tensor.cu"

#if __COMPUTE__ < 20
#define printf(...) /* eliminate printf from 1.x */
#endif

// Single-precision M_PI
// FIXME : ah, ah ! Single precision with 976896587958795795 decimals ....
#define M_PIf 3.141592653589793238462643383279502884197169399375105820974944f

#define MAXKASINDEX 10

/** \namespace cuforces
 *  \brief Contains all device functions/kernels/variables used force computations, filters and boundary conditions
 *
 *  The namespace cuforces contains all the device part of force computations, filters and boundary conditions :
 *  	- device constants/variables
 *  	- device functions
 *  	- kernels
 *
 *  \ingroup forces
 */
namespace cuforces {

using namespace cubounds;

// Core SPH functions
#include "sph_core_utils.cuh"
#include "gamma.cuh"

/** \name Device constants
 *  @{ */
__constant__ idx_t	d_neiblist_end;			///< maximum number of neighbors * number of allocated particles
__constant__ idx_t	d_neiblist_stride;		///< stride between neighbors of the same particle

__constant__ int	d_numfluids;			///< number of different fluids

__constant__ float	d_sqC0[MAX_FLUID_TYPES];	///< square of sound speed for at-rest density for each fluid

__constant__ float	d_ferrari;				///< coefficient for Ferrari correction
__constant__ float	d_rhodiffcoeff;			///< coefficient for density diffusion

__constant__ float	d_epsinterface;			///< interface epsilon for simplified surface tension in Grenier

// LJ boundary repusion force comuting
__constant__ float	d_dcoeff;
__constant__ float	d_p1coeff;
__constant__ float	d_p2coeff;
__constant__ float	d_r0;

// Monaghan-Kaijar boundary repulsion force constants
__constant__ float	d_MK_K;		///< This is typically the square of the maximum velocity, or gravity times the maximum height
__constant__ float	d_MK_d;		///< This is typically the distance between boundary particles
__constant__ float	d_MK_beta;	///< This is typically the ration between h and the distance between boundary particles

__constant__ float	d_visccoeff[MAX_FLUID_TYPES];	///< viscous coefficient
__constant__ float	d_epsartvisc;					///< epsilon of artificial viscosity

__constant__ float	d_partsurf;		///< particle surface (typically particle spacing suared)

// Sub-Particle Scale (SPS) Turbulence parameters
__constant__ float	d_smagfactor;
__constant__ float	d_kspsfactor;

// Rigid body data
__constant__ int3	d_rbcgGridPos[MAX_BODIES]; //< cell of the center of gravity
__constant__ float3	d_rbcgPos[MAX_BODIES]; //< in-cell coordinate of the center of gravity
__constant__ int	d_rbstartindex[MAX_BODIES];
__constant__ float	d_objectobjectdf;
__constant__ float	d_objectboundarydf;


/*  @} */

/** \name Device functions
 *  @{ */

/************************************************************************************************************/
/*							  Functions used by the different CUDA kernels							        */
/************************************************************************************************************/

//! Lennard-Jones boundary repulsion force
__device__ __forceinline__ float
LJForce(const float r)
{
	float force = 0.0f;

	if (r <= d_r0)
		force = d_dcoeff*(__powf(d_r0/r, d_p1coeff) - __powf(d_r0/r, d_p2coeff))/(r*r);

	return force;
}

//! Monaghan-Kajtar boundary repulsion force
/*!
 Monaghan-Kajtar boundary repulsion force doi:10.1016/j.cpc.2009.05.008
 to be multiplied by r_aj vector
 we allow the fluid particle mass mass_f to be different from the
 boundary particle mass mass_b even though they are typically the same
 (except for multi-phase fluids)
*/
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
//! Artificial viscosity
__device__ __forceinline__ float
artvisc(	const float	vel_dot_pos,
			const float	rho,
			const float	neib_rho,
			const float	sspeed,
			const float	neib_sspeed,
			const float	r,
			const float	slength)
{
	// TODO check if it makes sense to support different artificial viscosity coefficients
	// for different fluids
	return vel_dot_pos*slength*d_visccoeff[0]*(sspeed + neib_sspeed)/
									((r*r + d_epsartvisc)*(rho + neib_rho));
}


// ATTENTION: for all non artificial viscosity
// µ is the dynamic viscosity (ρν)

//! Morris laminar viscous term
/*!
 Scalar part of viscosity using Morris 1997
 expression 21 p218 when all particles have the same viscosity
 in this case d_visccoeff = 4 nu
 returns 4.mj.nu/(ρi + ρj) (1/r ∂Wij/∂r)
*/
__device__ __forceinline__ float
laminarvisc_kinematic(	const float	rho,
						const float	neib_rho,
						const float	neib_mass,
						const float	f)
{
	// NOTE: this won't work in multi-fluid!
	// TODO FIXME kinematic viscosity should probably be marked as incompatible
	// with multi-fluid (or at least if fluids don't have the same, constant
	// viscosity
	return neib_mass*d_visccoeff[0]*f/(rho + neib_rho);
}


//! Morris laminar viscous term for variable viscosity
/*!
 Same behaviour as laminarvisc_kinematic but for particle
 dependent viscosity.
 returns mj.(µi + µi)/(ρi.ρj) (1/r ∂Wij/∂r)
*/
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
// Computes dt across different GPU blocks
/*!
 Function called at the end of the forces or powerlawVisc function doing
 a per block maximum reduction
 cflOffset is used in case the forces kernel was partitioned (striping)
*/
__device__ __forceinline__ void
dtadaptBlockReduce(	float*	sm_max,
					float*	cfl,
					uint	cflOffset)
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
		cfl[cflOffset + blockIdx.x] = sm_max[0];
}
/************************************************************************************************************/

/******************** Functions for computing repulsive force directly from DEM *****************************/

// TODO: check for the maximum timestep

//! Computes normal and viscous force wrt to solid planar boundary
__device__ __forceinline__ float
PlaneForce(	const int3&		gridPos,
			const float3&	pos,
			const float		mass,
			const plane_t&	plane,
			const float3&	vel,
			const float		dynvisc,
			float4&			force)
{
	// relative position of our particle from the reference point of the plane
	const float r = PlaneDistance(gridPos, pos, plane);
	if (r < d_r0) {
		const float DvDt = LJForce(r);
		// Unitary normal vector of the surface
		const float3 relPos = plane.normal*r;

		as_float3(force) += DvDt*relPos;

		// tangential velocity component
		const float3 v_t = vel - dot(vel, relPos)/r*relPos/r; //TODO: check

		// f = -µ u/∆n

		// viscosity
		// float coeff = -dynvisc*M_PI*(d_r0*d_r0-r*r)/(pos.w*r);
		// float coeff = -dynvisc*M_PI*(d_r0*d_r0*3/(M_PI*2)-r*r)/(pos.w*r);
		const float coeff = -dynvisc*d_partsurf/(mass*r);

		// coeff should not be higher than needed to nil v_t in the maximum allowed dt
		// coefficients are negative, so the smallest in absolute value is the biggest

		/*
		float fmag = length(as_float3(force));
		float coeff2 = -sqrt(fmag/slength)/(d_dtadaptfactor*d_dtadaptfactor);
		if (coeff2 < -d_epsartvisc)
			coeff = max(coeff, coeff2);
			*/

		as_float3(force) += coeff*v_t;

		return -coeff;
	}

	return 0.0f;
}

//! DOC-TODO Describe function
__device__ __forceinline__ float
GeometryForce(	const int3&		gridPos,
				const float3&	pos,
				const float		mass,
				const float3&	vel,
				const float		dynvisc,
				float4&			force)
{
	float coeff_max = 0.0f;
	for (uint i = 0; i < d_numplanes; ++i) {
		float coeff = PlaneForce(gridPos, pos, mass, d_plane[i], vel, dynvisc, force);
		if (coeff > coeff_max)
			coeff_max = coeff;
	}

	return coeff_max;
}

//! DOC-TODO describe function
__device__ __forceinline__ float
DemLJForce(	const texture<float, 2, cudaReadModeElementType> texref,
			const int3&	gridPos,
			const float3&	pos,
			const float		mass,
			const float3&	vel,
			const float		dynvisc,
			float4&			force)
{
	const float2 demPos = DemPos(gridPos, pos);

	const float globalZ = d_worldOrigin.z + (gridPos.z + 0.5f)*d_cellSize.z + pos.z;
	const float globalZ0 = DemInterpol(texref, demPos);

	if (globalZ - globalZ0 < d_demzmin) {
		const plane_t demPlane(DemTangentPlane(texref, gridPos, pos, demPos, globalZ0));

		return PlaneForce(gridPos, pos, mass, demPlane, vel, dynvisc, force);
	}
	return 0;
}

/************************************************************************************************************/

/************************************************************************************************************/
/*		   Kernels for computing SPS tensor and SPS viscosity												*/
/************************************************************************************************************/

//! A functor that writes out turbvisc for SPS visc
template<bool>
struct write_sps_turbvisc
{
	template<typename FP>
	__device__ __forceinline__
	static void
	with(FP const& params, const uint index, const float turbvisc)
	{ /* do nothing */ }
};

template<>
template<typename FP>
__device__ __forceinline__ void
write_sps_turbvisc<true>::with(FP const& params, const uint index, const float turbvisc)
{ params.turbvisc[index] = turbvisc; }

//! A functor that writes out tau for SPS visc
template<bool>
struct write_sps_tau
{
	template<typename FP>
	__device__ __forceinline__
	static void
	with(FP const& params, const uint index, const float2& tau0, const float2& tau1, const float2& tau2)
	{ /* do nothing */ }
};

template<>
template<typename FP>
__device__ __forceinline__ void
write_sps_tau<true>::with(FP const& params, const uint index, const float2& tau0,
							const float2& tau1, const float2& tau2)
{
	params.tau0[index] = tau0;
	params.tau1[index] = tau1;
	params.tau2[index] = tau2;
}

/************************************************************************************************************/

/************************************************************************************************************/
/*		Device functions used in kernels other than the main forces kernel									*/
/************************************************************************************************************/

//! Computes boundary conditions at open boundaries
/*!
 Depending on whether velocity or pressure is prescribed at a boundary the respective other component
 is computed using the appropriate Riemann invariant.
*/
__device__ __forceinline__ void
calculateIOboundaryCondition(
			float4			&eulerVel,
	const	particleinfo	info,
	const	float			rhoInt,
	const	float			rhoExt,
	const	float3			uInt,
	const	float			unInt,
	const	float			unExt,
	const	float3			normal)
{
	const int a = fluid_num(info);
	const float rInt = R(rhoInt, a);

	// impose velocity (and k,eps) => compute density
	if (VEL_IO(info)) {
		float riemannR = 0.0f;
		if (unExt <= unInt) // Expansion wave
			riemannR = rInt + (unExt - unInt);
		else { // Shock wave
			float riemannRho = RHO(P(rhoInt, a) + rhoInt * unInt * (unInt - unExt), a);
			riemannR = R(riemannRho, a);
			float riemannC = soundSpeed(riemannRho, a);
			float lambda = unExt + riemannC;
			const float cInt = soundSpeed(rhoInt, a);
			float lambdaInt = unInt + cInt;
			if (lambda <= lambdaInt) // must be a contact discontinuity then (which would actually mean lambda == lambdaInt
				riemannR = rInt;
		}
		eulerVel.w = RHOR(riemannR, a);
	}
	// impose pressure => compute velocity (normal & tangential; k and eps are already interpolated)
	else {
		float flux = 0.0f;
		// Rankine-Hugoniot is not properly working
		const float cExt = soundSpeed(rhoExt, a);
		const float cInt = soundSpeed(rhoInt, a);
		const float lambdaInt = unInt + cInt;
		const float rExt = R(rhoExt, a);
		if (rhoExt <= rhoInt) { // Expansion wave
			flux = unInt + (rExt - rInt);
			float lambda = flux + cExt;
			if (lambda > lambdaInt) { // shock wave
				flux = (P(rhoInt, a) - P(rhoExt, a))/(rhoInt*fmaxf(unInt,1e-5f*d_sscoeff[a])) + unInt;
				// check that unInt was not too small
				if (fabsf(flux) > d_sscoeff[a] * 0.1f)
					flux = unInt;
				lambda = flux + cExt;
				if (lambda <= lambdaInt) // contact discontinuity
					flux = unInt;
			}
		}
		else { // shock wave
			flux = (P(rhoInt, a) - P(rhoExt, a))/(rhoInt*fmaxf(unInt,1e-5f*d_sscoeff[a])) + unInt;
			// check that unInt was not too small
			if (fabsf(flux) > d_sscoeff[a] * 0.1f)
				flux = unInt;
			float lambda = flux + cExt;
			if (lambda <= lambdaInt) { // expansion wave
				flux = unInt + (rExt - rInt);
				lambda = flux + cExt;
				if (lambda > lambdaInt) // contact discontinuity
					flux = unInt;
			}
		}
		// remove normal component of imposed Eulerian velocity
		//as_float3(eulerVel) = as_float3(eulerVel) - dot(as_float3(eulerVel), normal)*normal;
		as_float3(eulerVel) = make_float3(0.0f);
		// if the imposed pressure on the boundary is negative make sure that the flux is negative
		// as well (outflow)
		if (rhoExt < d_rho0[a])
			flux = fminf(flux, 0.0f);
		// Outflow
		if (flux < 0.0f)
			// impose eulerVel according to dv/dn = 0
			// and remove normal component of velocity
			as_float3(eulerVel) = uInt - dot(uInt, normal)*normal;
		// add calculated normal velocity
		as_float3(eulerVel) += normal*flux;
		// set density to the imposed one
		eulerVel.w = rhoExt;
	}
}

//! Determines the distribution of mass based on a position on a segment
/*!
 A position inside a segment is used to split the segment area into three parts. The respective
 size of these parts are used to determine how much the mass is redistributed that is associated
 with this position. This is used in two cases:

 1.) A mass flux is given or computed for a certain segment, then the position for the function
     is equivalent to the segement position. This determines the mass flux for the vertices

 2.) A fluid particle traverses a segment. Then the position is equal to the fluid position and
     the function determines how much mass of the fluid particle is distributed to each vertex
*/
__device__ __forceinline__ void
getMassRepartitionFactor(	const	float3	*vertexRelPos,
							const	float3	normal,
									float3	&beta)
{
	float3 v01 = vertexRelPos[0]-vertexRelPos[1];
	float3 v02 = vertexRelPos[0]-vertexRelPos[2];
	float3 p0  = vertexRelPos[0]-dot(vertexRelPos[0], normal)*normal;
	float3 p1  = vertexRelPos[1]-dot(vertexRelPos[1], normal)*normal;
	float3 p2  = vertexRelPos[2]-dot(vertexRelPos[2], normal)*normal;

	float refSurface = 0.5*dot(cross(v01, v02), normal);

	float3 v21 = vertexRelPos[2]-vertexRelPos[1];

	float surface0 = 0.5*dot(cross(p2, v21), normal);
	float surface1 = 0.5*dot(cross(p0, v02), normal);
	// Warning v10 = - v01
	float surface2 = - 0.5*dot(cross(p1, v01), normal);
	if (surface0 < 0. && surface2 < 0.) {
		// the projected point is clipped to v1
		surface0 = 0.;
		surface1 = refSurface;
		surface2 = 0.;
	} else if (surface0 < 0. && surface1 < 0.) {
		// the projected point is clipped to v2
		surface0 = 0.;
		surface1 = 0.;
		surface2 = refSurface;
	} else if (surface1 < 0. && surface2 < 0.) {
		// the projected point is clipped to v0
		surface0 = refSurface;
		surface1 = 0.;
		surface2 = 0.;
	} else if (surface0 < 0.) {
		// We project p2 into the v21 line, parallel to p0
		// then surface0 is 0
		// we also modify p0 an p1 accordingly
		float coef = surface0/(0.5*dot(cross(p0, v21), normal));

		p1 -= coef*p0;
		p0 *= (1.-coef);

		surface0 = 0.;
		surface1 = 0.5*dot(cross(p0, v02), normal);
		surface2 = - 0.5*dot(cross(p1, v01), normal);
	} else if (surface1 < 0.) {
		// We project p0 into the v02 line, parallel to p1
		// then surface1 is 0
		// we also modify p1 an p2 accordingly
		float coef = surface1/(0.5*dot(cross(p1, v02), normal));
		p2 -= coef*p1;
		p1 *= (1.-coef);

		surface0 = 0.5*dot(cross(p2, v21), normal);
		surface1 = 0.;
		surface2 = - 0.5*dot(cross(p1, v01), normal);
	} else if (surface2 < 0.) {
		// We project p1 into the v01 line, parallel to p2
		// then surface2 is 0
		// we also modify p0 an p2 accordingly
		float coef = -surface2/(0.5*dot(cross(p2, v01), normal));
		p0 -= coef*p2;
		p2 *= (1.-coef);

		surface0 = 0.5*dot(cross(p2, v21), normal);
		surface1 = 0.5*dot(cross(p0, v02), normal);
		surface2 = 0.;
	}

	beta.x = surface0/refSurface;
	beta.y = surface1/refSurface;
	beta.z = surface2/refSurface;
}

//! contribution of neighbor at relative position relPos with weight w to the MLS matrix mls
__device__ __forceinline__ void
MlsMatrixContrib(symtensor4 &mls, float4 const& relPos, float w)
{
	mls.xx += w;						// xx = ∑Wij*Vj
	mls.xy += relPos.x*w;				// xy = ∑(xi - xj)*Wij*Vj
	mls.xz += relPos.y*w;				// xz = ∑(yi - yj)*Wij*Vj
	mls.xw += relPos.z*w;				// xw = ∑(zi - zj)*Wij*Vj
	mls.yy += relPos.x*relPos.x*w;		// yy = ∑(xi - xj)^2*Wij*Vj
	mls.yz += relPos.x*relPos.y*w;		// yz = ∑(xi - xj)(yi - yj)*Wij*Vj
	mls.yw += relPos.x*relPos.z*w;		// yz = ∑(xi - xj)(zi - zj)*Wij*Vj
	mls.zz += relPos.y*relPos.y*w;		// zz = ∑(yi - yj)^2*Wij*Vj
	mls.zw += relPos.y*relPos.z*w;		// zz = ∑(yi - yj)(zi - zj)*Wij*Vj
	mls.ww += relPos.z*relPos.z*w;		// zz = ∑(yi - yj)^2*Wij*Vj

}

//! MLS contribution
/*!
 contribution of neighbor at relative position relPos with weight w to the
 MLS correction when B is the first row of the inverse MLS matrix
*/
__device__ __forceinline__ float
MlsCorrContrib(float4 const& B, float4 const& relPos, float w)
{
	return (B.x + B.y*relPos.x + B.z*relPos.y + B.w*relPos.z)*w;
	// ρ = ∑(ß0 + ß1(xi - xj) + ß2(yi - yj))*Wij*Vj
}

//! Fetch tau tensor from texture
/*!
 an auxiliary function that fetches the tau tensor
 for particle i from the textures where it's stored
*/
__device__
symtensor3 fetchTau(uint i)
{
	symtensor3 tau;
	float2 temp = tex1Dfetch(tau0Tex, i);
	tau.xx = temp.x;
	tau.xy = temp.y;
	temp = tex1Dfetch(tau1Tex, i);
	tau.xz = temp.x;
	tau.yy = temp.y;
	temp = tex1Dfetch(tau2Tex, i);
	tau.yz = temp.x;
	tau.zz = temp.y;
	return tau;
}

/*  @} */

/** \name Kernels
 *  @{ */

//! Compute SPS matrix
/*!
 Compute the Sub-Particle-Stress (SPS) Tensor matrix for all Particles
 WITHOUT Kernel correction

 Procedure:

 (1) compute velocity gradients

 (2) compute turbulent eddy viscosity (non-dynamic)

 (3) compute turbulent shear stresses

 (4) return SPS tensor matrix (tau) divided by rho^2
*/
template<KernelType kerneltype,
	BoundaryType boundarytype,
	uint simflags>
__global__ void
__launch_bounds__(BLOCK_SIZE_SPS, MIN_BLOCKS_SPS)
SPSstressMatrixDevice(sps_params<kerneltype, boundarytype, simflags> params)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	// Compute SPS matrix only for any kind of particles
	// TODO testpoints should also compute SPS, it'd be useful
	// when we will enable SPS saving to disk
	const particleinfo info = tex1Dfetch(infoTex, index);

	// read particle data from sorted arrays
	#if PREFER_L1
	const float4 pos = params.pos[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	// skip inactive particles
	if (INACTIVE(pos))
		return;

	const float4 vel = tex1Dfetch(velTex, index);

	// Gradients of the the velocity components
	float3 dvx = make_float3(0.0f);
	float3 dvy = make_float3(0.0f);
	float3 dvz = make_float3(0.0f);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( params.particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = -1;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = params.neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, params.cellStart,
				neib_data, gridPos, neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if PREFER_L1
		const float4 relPos = pos_corr - params.pos[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length3(relPos);

		// Compute relative velocity
		// Now relVel is a float4 and neib density is stored in relVel.w
		const float4 relVel = as_float3(vel) - tex1Dfetch(velTex, neib_index);
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// Velocity gradient is contributed by all particles
		// TODO: fix SA case
		if ( r < params.influenceradius ) {
			const float f = F<kerneltype>(r, params.slength)*relPos.w/relVel.w;	// 1/r ∂Wij/∂r Vj

			// Velocity Gradients
			dvx -= relVel.x*as_float3(relPos)*f;	// dvx = -∑mj/ρj vxij (ri - rj)/r ∂Wij/∂r
			dvy -= relVel.y*as_float3(relPos)*f;	// dvy = -∑mj/ρj vyij (ri - rj)/r ∂Wij/∂r
			dvz -= relVel.z*as_float3(relPos)*f;	// dvz = -∑mj/ρj vzij (ri - rj)/r ∂Wij/∂r
			}
		} // end of loop through neighbors


	// SPS stress matrix elements
	symtensor3 tau;

	// Calculate Sub-Particle Scale viscosity
	// and special turbulent terms
	float SijSij_bytwo = 2.0f*(dvx.x*dvx.x + dvy.y*dvy.y + dvz.z*dvz.z);	// 2*SijSij = 2.0((∂vx/∂x)^2 + (∂vy/∂yx)^2 + (∂vz/∂z)^2)
	float temp = dvx.y + dvy.x;		// 2*SijSij += (∂vx/∂y + ∂vy/∂x)^2
	tau.xy = temp;
	SijSij_bytwo += temp*temp;
	temp = dvx.z + dvz.x;			// 2*SijSij += (∂vx/∂z + ∂vz/∂x)^2
	tau.xz = temp;
	SijSij_bytwo += temp*temp;
	temp = dvy.z + dvz.y;			// 2*SijSij += (∂vy/∂z + ∂vz/∂y)^2
	tau.yz = temp;
	SijSij_bytwo += temp*temp;
	const float S = sqrtf(SijSij_bytwo);
	const float nu_SPS = d_smagfactor*S;		// Dalrymple & Rogers (2006): eq. (12)
	const float divu_SPS = 0.6666666666f*nu_SPS*(dvx.x + dvy.y + dvz.z);
	const float Blinetal_SPS = d_kspsfactor*SijSij_bytwo;

	// Storing the turbulent viscosity for each particle
	write_sps_turbvisc<simflags & SPSK_STORE_TURBVISC>::with(params, index, nu_SPS);

	// Shear Stress matrix = TAU (pronounced taf)
	// Dalrymple & Rogers (2006): eq. (10)
	if (simflags & SPSK_STORE_TAU) {

		tau.xx = nu_SPS*(dvx.x + dvx.x) - divu_SPS - Blinetal_SPS;	// tau11 = tau_xx/ρ^2
		tau.xx /= vel.w;
		tau.xy *= nu_SPS/vel.w;								// tau12 = tau_xy/ρ^2
		tau.xz *= nu_SPS/vel.w;								// tau13 = tau_xz/ρ^2
		tau.yy = nu_SPS*(dvy.y + dvy.y) - divu_SPS - Blinetal_SPS;	// tau22 = tau_yy/ρ^2
		tau.yy /= vel.w;
		tau.yz *= nu_SPS/vel.w;								// tau23 = tau_yz/ρ^2
		tau.zz = nu_SPS*(dvz.z + dvz.z) - divu_SPS - Blinetal_SPS;	// tau33 = tau_zz/ρ^2
		tau.zz /= vel.w;

		write_sps_tau<simflags & SPSK_STORE_TAU>::with(params, index, make_float2(tau.xx, tau.xy),
				make_float2(tau.xz, tau.yy), make_float2(tau.yz, tau.zz));
	}
}
/************************************************************************************************************/

/************************************************************************************************************/
/*										Density computation							*/
/************************************************************************************************************/

//! Continuity equation with the Grenier formulation
/*!
 When using the Grenier formulation, density is reinitialized at each timestep from
 a Shepard-corrected mass distribution limited to same-fluid particles M and volumes ω computed
 from a continuity equation, with ρ = M/ω.
 During the same run, we also compute σ, the discrete specific volume
 (see e.g. Hu & Adams 2005), obtained by summing the kernel computed over
 _all_ neighbors (not just the same-fluid ones) which is used in the continuity
 equation as well as the Navier-Stokes equation
*/
template<KernelType kerneltype, BoundaryType boundarytype>
__global__ void
densityGrenierDevice(
			float* __restrict__		sigmaArray,
	const	float4* __restrict__	posArray,
			float4* __restrict__	velArray,
	const	particleinfo* __restrict__	infoArray,
	const	hashKey* __restrict__	particleHash,
	const	float4* __restrict__	volArray,
	const	uint* __restrict__		cellStart,
	const	neibdata* __restrict__	neibsList,
	const	uint	numParticles,
	const	float	slength,
	const	float	influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	const particleinfo info = infoArray[index];

	/* We only process FLUID particles normally,
	   except with DYN_BOUNDARY, where we also process boundary particles
	   */
	if (boundarytype != DYN_BOUNDARY && NOT_FLUID(info))
		return;

	const float4 pos = posArray[index];

	if (INACTIVE(pos))
		return;

	const ushort fnum = fluid_num(info);
	const float vol = volArray[index].w;
	float4 vel = velArray[index];

	// self contribution
	float corr = W<kerneltype>(0, slength);
	float sigma = corr;
	float mass_corr = pos.w*corr;

	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );
	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// For DYN_BOUNDARY particles, we compute sigma in the same way as fluid particles,
	// except that if the boundary particle has no fluid neighbors we set its
	// sigma to a default value which is the 'typical' specific volume, given by
	// the typical number of neighbors divided by the volume of the influence sphere
	bool has_fluid_neibs = false;

	// Loop over all neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
			neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance

		const particleinfo neib_info = infoArray[neib_index];
		const float4 relPos = pos_corr - posArray[neib_index];
		float r = length(as_float3(relPos));

		/* Contributions only come from active particles within the influence radius
		   that are fluid particles (or also non-fluid in DYN_BOUNDARY case).
		   TODO check what to do with SA
		   Sigma calculations uses all such particles, whereas smoothed mass
		   only uses same-fluid particles.
		   Note that this requires PT_BOUNDARY neighbors to be in the list for
		   PT_BOUNDARY particles, lest the boundary particles end up assuming
		   they are always on the free surface.
		   TODO an alternative approach for DYN_BOUNDARY would be to assign
		   the sigma from the closest fluid particle, but that would require
		   two runs, one for fluid and one for neighbor particles.
		 */
		if (INACTIVE(relPos) || r >= influenceradius ||
			((boundarytype != DYN_BOUNDARY) && NOT_FLUID(neib_info)))
			continue;

		const float w = W<kerneltype>(r, slength);
		sigma += w;
		if (FLUID(neib_info))
			has_fluid_neibs = true;

		/* For smoothed mass, fluid particles only consider fluid particles,
		   and non-fluid (only present for DYN_BOUNDARY) only consider non-fluid
		   */
		if ((boundarytype != DYN_BOUNDARY || (PART_TYPE(neib_info) == PART_TYPE(info)))
			&& fluid_num(neib_info) == fnum) {
			mass_corr += relPos.w*w;
			corr += w;
		}
	}

	if (boundarytype == DYN_BOUNDARY && NOT_FLUID(info) && !has_fluid_neibs) {
		// TODO OPTIMIZE
		const float typical_sigma = 3*cuneibs::d_maxNeibs/
			(4*M_PIf*influenceradius*influenceradius*influenceradius);
		sigma = typical_sigma;
	}

	// M = mass_corr/corr, ρ = M/ω
	// this could be optimized to pos.w/vol assuming all same-fluid particles
	// have the same mass
	vel.w = mass_corr/(corr*vol);
	velArray[index] = vel;
	sigmaArray[index] = sigma;
}

/************************************************************************************************************/

// flags for the vertexinfo .w coordinate which specifies how many vertex particles of one segment
// is associated to an open boundary
#define VERTEX1 ((flag_t)1)
#define VERTEX2 (VERTEX1 << 1)
#define VERTEX3 (VERTEX2 << 1)
#define ALLVERTICES ((flag_t)(VERTEX1 | VERTEX2 | VERTEX3))

//! Computes the boundary condition on segments for SA boundaries
/*!
 This function computes the boundary condition for density/pressure on segments if the SA boundary type
 is selected. It does this not only for solid wall boundaries but also open boundaries. Additionally,
 this function detects when a fluid particle crosses the open boundary and it identifies which segment it
 crossed. The vertices of this segment are then used to identify how the mass of this fluid particle is
 split.
*/
template<KernelType kerneltype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
saSegmentBoundaryConditions(			float4*		oldPos,
										float4*		oldVel,
										float*		oldTKE,
										float*		oldEps,
										float4*		oldEulerVel,
										float4*		oldGGam,
										vertexinfo*	vertices,
								const	float2*		vertPos0,
								const	float2*		vertPos1,
								const	float2*		vertPos2,
								const	hashKey*	particleHash,
								const	uint*		cellStart,
								const	neibdata*	neibsList,
								const	uint		numParticles,
								const	float		deltap,
								const	float		slength,
								const	float		influenceradius,
								const	bool		initStep,
								const	uint		step,
								const	bool		inoutBoundaries)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	const particleinfo info = tex1Dfetch(infoTex, index);

	// For boundary segments this kernel computes the boundary conditions
	if (BOUNDARY(info)) {

		float4 eulerVel = make_float4(0.0f);
		const vertexinfo verts = vertices[index];
		float tke = 0.0f;
		float eps = 0.0f;

		// get the imposed quantities from the arrays which were set in the problem specific routines
		if (IO_BOUNDARY(info)) {
			// for imposed velocity the velocity, tke and eps are required and only rho will be calculated
			if (VEL_IO(info)) {
				eulerVel = oldEulerVel[index];
				eulerVel.w = 0.0f;
				if (oldTKE)
					tke = oldTKE[index];
				if (oldEps)
					eps = oldEps[index];
			}
			// for imposed density only eulerVel.w will be required, the rest will be computed
			else
				eulerVel = oldEulerVel[index];
		}
		// velocity for segment (for moving objects) taken as average from the vertices
		float3 vel = make_float3(0.0f);
		// gamma of segment (if not set) taken as average from the vertices
		float4 gGam = make_float4(0.0f, 0.0f, 0.0f, oldGGam[index].w);
		bool calcGam = oldGGam[index].w < 1e-5f;

		const float4 pos = oldPos[index];

		// note that all sums below run only over fluid particles (including the Shepard filter)
		float sumpWall = 0.0f; // summation for computing the density
		float sump = 0.0f; // summation for computing the pressure
		float3 sumvel = make_float3(0.0f); // summation to compute the internal velocity for open boundaries
		float sumtke = 0.0f; // summation for computing tke (k-epsilon model)
		float sumeps = 0.0f; // summation for computing epsilon (k-epsilon model)
		float alpha  = 0.0f;  // the shepard filter

		// Compute grid position of current particle
		const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

		// Persistent variables across getNeibData calls
		char neib_cellnum = 0;
		uint neib_cell_base_index = 0;
		float3 pos_corr;

		// Square of sound speed. Would need modification for multifluid
		const float sqC0 = d_sqC0[fluid_num(info)];

		const float4 normal = tex1Dfetch(boundTex, index);

		// Loop over all the neighbors
		for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
			neibdata neib_data = neibsList[i + index];

			if (neib_data == NEIBS_END) break;

			const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
						neib_cellnum, neib_cell_base_index);

			// Compute relative position vector and distance
			// Now relPos is a float4 and neib mass is stored in relPos.w
			const float4 relPos = pos_corr - oldPos[neib_index];

			// skip inactive particles
			if (INACTIVE(relPos))
				continue;

			const float r = length(as_float3(relPos));
			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			// averages taken from associated vertices:
			// - velocity (for moving objects)
			// - gGam if not yet computed
			// - Eulerian velocity (if TKE is enabled and only for solid walls)
			if (verts.x == id(neib_info) || verts.y == id(neib_info) || verts.z == id(neib_info)) {
				if (MOVING(info))
					vel += as_float3(oldVel[neib_index]);
				if (calcGam)
					gGam += oldGGam[neib_index];
				if (!IO_BOUNDARY(info) && oldTKE)
					eulerVel += oldEulerVel[neib_index];
			}

			if (dot3(normal, relPos) < 0.0f &&
				r < influenceradius &&
				FLUID(neib_info)
				//(FLUID(neib_info) || (!IO_BOUNDARY(info) && VERTEX(neib_info) && IO_BOUNDARY(neib_info) && !CORNER(neib_info)))
				//(FLUID(neib_info) || (VERTEX(neib_info) && !IO_BOUNDARY(neib_info) && IO_BOUNDARY(info)))
			   ){
				const float neib_rho = oldVel[neib_index].w;

				const float neib_pres = P(neib_rho, fluid_num(neib_info));
				const float neib_vel = length(make_float3(oldVel[neib_index]));
				const float neib_k = oldTKE ? oldTKE[neib_index] : NAN;
				const float neib_eps = oldEps ? oldEps[neib_index] : NAN;

				// kernel value times volume
				const float w = W<kerneltype>(r, slength)*relPos.w/neib_rho;
				// normal distance based on grad Gamma which approximates the normal of the domain
				const float normDist = fmaxf(fabsf(dot3(normal,relPos)), deltap);
				sumpWall += fmaxf(neib_pres + neib_rho*dot(d_gravity, as_float3(relPos)), 0.0f)*w;
				// for all boundaries we have dk/dn = 0
				sumtke += w*neib_k;
				if (IO_BOUNDARY(info)) {
					sumvel += w*as_float3(oldVel[neib_index] + oldEulerVel[neib_index]);
					// for open boundaries compute pressure interior state
					//sump += w*fmaxf(0.0f, neib_pres+dot(d_gravity, as_float3(relPos)*d_rho0[fluid_num(neib_info)]));
					sump += w*fmaxf(0.0f, neib_pres);
					// and de/dn = 0
					sumeps += w*neib_eps;
				}
				else
					// for solid boundaries we have de/dn = c_mu^(3/4)*4*k^(3/2)/(\kappa r)
					// the constant is coming from 4*powf(0.09,0.75)/0.41
					sumeps += w*(neib_eps + 1.603090412f*powf(neib_k,1.5f)/normDist);
				alpha += w;
			}
		}

		// set variables that have been obtained as average from the associated vertices
		// we write only into array positions that are associated with segments here
		// all the reads above are only on fluid or vertex particles
		as_float3(oldVel[index]) = vel/3.0f;
		if (calcGam) {
			gGam /= 3.0f;
			oldGGam[index] = gGam;
			gGam.w = fmaxf(gGam.w, 1e-5f);
		}

		if (IO_BOUNDARY(info)) {
			if (alpha > 0.1f*gGam.w) { // note: defaults are set in the place where bcs are imposed
				sumvel /= alpha;
				sump /= alpha;
				oldVel[index].w = RHO(sump, fluid_num(info));
				if (VEL_IO(info)) {
					// for velocity imposed boundaries we impose k and epsilon
					if (oldTKE)
						oldTKE[index] = tke;
					if (oldEps)
						oldEps[index] = eps;
				}
				else {
					oldEulerVel[index] = make_float4(0.0f);
					// for pressure imposed boundaries we take dk/dn = 0
					if (oldTKE)
						oldTKE[index] = sumtke/alpha;
					// for pressure imposed boundaries we have de/dn = 0
					if (oldEps)
						oldEps[index] = sumeps/alpha;
				}

			}
			else {
				sump = 0.0f;
				if (VEL_IO(info)) {
					sumvel = as_float3(eulerVel);
					oldVel[index].w = d_rho0[fluid_num(info)];
				}
				else {
					sumvel = make_float3(0.0f);
					oldVel[index].w = oldEulerVel[index].w;
					oldEulerVel[index] = make_float4(0.0f, 0.0f, 0.0f, oldEulerVel[index].w);
				}
				if (oldTKE)
					oldTKE[index] = 1e-6f;
				if (oldEps)
					oldEps[index] = 1e-6f;
			}

			// compute Riemann invariants for open boundaries
			const float unInt = dot(sumvel, as_float3(normal));
			const float unExt = dot3(eulerVel, normal);
			const float rhoInt = oldVel[index].w;
			const float rhoExt = eulerVel.w;

			calculateIOboundaryCondition(eulerVel, info, rhoInt, rhoExt, sumvel, unInt, unExt, as_float3(normal));

			oldEulerVel[index] = eulerVel;
			// the density of the particle is equal to the "eulerian density"
			oldVel[index].w = eulerVel.w;

		}
		// non-open boundaries
		else {
			alpha = fmaxf(alpha, 0.1f*gGam.w); // avoid division by 0
			// density condition
			oldVel[index].w = RHO(sumpWall/alpha,fluid_num(info));
			// k-epsilon boundary conditions
			if (oldTKE) {
				// k condition
				oldTKE[index] = sumtke/alpha;
				// average eulerian velocity on the wall (from associated vertices)
				eulerVel /= 3.0f;
				// ensure that velocity is normal to segment normal
				eulerVel -= dot3(eulerVel,normal)*normal;
				oldEulerVel[index] = eulerVel;
			}
			// if k-epsilon is not used but oldEulerVel is present (for open boundaries) set it to 0
			else if (oldEulerVel)
				oldEulerVel[index] = make_float4(0.0f);
			// epsilon condition
			if (oldEps)
				// for solid boundaries we have de/dn = 4 0.09^0.075 k^1.5/(0.41 r)
				oldEps[index] = fmaxf(sumeps/alpha,1e-5f); // eps should never be 0
		}

	}
	// for fluid particles this kernel checks whether they have crossed the boundary at open boundaries
	else if (inoutBoundaries && step==2 && FLUID(info)) {

		float4 pos = oldPos[index];

		// don't check inactive particles and those that have already found their segment
		if (INACTIVE(pos) || vertices[index].x | vertices[index].y != 0)
			return;

		// Compute grid position of current particle
		const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

		// Persistent variables across getNeibData calls
		char neib_cellnum = 0;
		uint neib_cell_base_index = 0;
		float3 pos_corr;

		const float4 vel = oldVel[index];

		float rSqMin = influenceradius*influenceradius;
		uint neib_indexMin = UINT_MAX;
		float4 relPosMin = make_float4(0.0f);

		// Loop over all the neighbors
		for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
			neibdata neib_data = neibsList[i + index];

			if (neib_data == NEIBS_END) break;

			const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
						neib_cellnum, neib_cell_base_index);
			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			// for open boundary segments check whether this fluid particle has crossed the boundary
			// in order to do so we try to identify the closest segment which the particle has passed
			if (BOUNDARY(neib_info) && IO_BOUNDARY(neib_info)) {

				// Compute relative position vector and distance
				// Now relPos is a float4 and neib mass is stored in relPos.w
				const float4 relPos = pos_corr - oldPos[neib_index];

				const float4 normal = tex1Dfetch(boundTex, neib_index);

				const float3 relVel = as_float3(vel - oldVel[neib_index]);

				const float rSq = sqlength3(relPos);

				// check if we are behind a segment
				// additionally check if the velocity vector is pointing outwards
				if (dot3(normal, relPos) <= 0.0f &&
					rSq < rSqMin &&
					dot(relVel, as_float3(normal)) < 0.0f) {
					// this can only be reached if the segment is closer than all those before, so we save its distance
					rSqMin = rSq;
					// its relative position
					relPosMin = relPos;
					// and also its index
					neib_indexMin = neib_index;
				}
			}
		} // end neighbour loop

		// if we have found a segment that was crossed and that is close by
		if (neib_indexMin != UINT_MAX) {
			const float4 normal = tex1Dfetch(boundTex, neib_indexMin);
			// first get the position of the vertices local coordinate system for relative positions to vertices
			uint j = 0;
			// Get index j for which n_s is minimal
			if (fabsf(normal.x) > fabsf(normal.y))
				j = 1;
			if ((1-j)*fabsf(normal.x) + j*fabsf(normal.y) > fabsf(normal.z))
				j = 2;

			// compute the first coordinate which is a 2-D rotated version of the normal
			const float4 coord1 = normalize(make_float4(
				// switch over j to give: 0 -> (0, z, -y); 1 -> (-z, 0, x); 2 -> (y, -x, 0)
				-((j==1)*normal.z) +  (j == 2)*normal.y , // -z if j == 1, y if j == 2
				  (j==0)*normal.z  - ((j == 2)*normal.x), // z if j == 0, -x if j == 2
				-((j==0)*normal.y) +  (j == 1)*normal.x , // -y if j == 0, x if j == 1
				0));
			// the second coordinate is the cross product between the normal and the first coordinate
			const float4 coord2 = cross3(normal, coord1);

			const float2 vPos0 = vertPos0[neib_indexMin];
			const float2 vPos1 = vertPos1[neib_indexMin];
			const float2 vPos2 = vertPos2[neib_indexMin];

			// relative positions of vertices with respect to the segment, normalized by h
			float4 v0 = -(vPos0.x*coord1 + vPos0.y*coord2); // e.g. v0 = r_{v0} - r_s
			float4 v1 = -(vPos1.x*coord1 + vPos1.y*coord2);
			float4 v2 = -(vPos2.x*coord1 + vPos2.y*coord2);

			// the fluid particle found a segment so let's save it
			// note normally vertices is empty for fluid particles so this will indicate
			// from now on that it has to be destroyed
			vertexinfo verts = vertices[neib_indexMin];

			// furthermore we need to save the weights beta_{a,v} to avoid using
			// neighbours of neighbours. As the particle will be deleted anyways we
			// just use the velocity array which we don't need anymore. The beta_{a,v}
			// in the 3-D case are based on surface areas based on the triangle partition
			// governed by the position of the fluid particle
			float4 vertexWeights = make_float4(0.0f);
			const float3 vx[3] = {as_float3(relPosMin - v0), as_float3(relPosMin - v1), as_float3(relPosMin - v2)};
			getMassRepartitionFactor(vx, as_float3(normal), as_float3(vertexWeights));
			// transfer mass to .w index as it is overwritten with the disable below
			vertexWeights.w = pos.w;
			oldGGam[index] = vertexWeights;
			vertices[index] = verts;
		}
	}
}

/// Compute boundary conditions for vertex particles in the semi-analytical boundary case
/*! This function determines the physical properties of vertex particles in the semi-analytical boundary case. The properties of fluid particles are used to compute the properties of the vertices. Due to this most arrays are read from (the fluid info) and written to (the vertex info) simultaneously inside this function. In the case of open boundaries the vertex mass is updated in this routine and new fluid particles are created on demand. Additionally, the mass of outgoing fluid particles is redistributed to vertex particles herein.
 *	\param[in,out] oldPos : pointer to positions and masses; masses of vertex particles are updated
 *	\param[in,out] oldVel : pointer to velocities and density; densities of vertex particles are updated
 *	\param[in,out] oldTKE : pointer to turbulent kinetic energy
 *	\param[in,out] oldEps : pointer to turbulent dissipation
 *	\param[in,out] oldGGam : pointer to (grad) gamma; used only for cloning (i.e. creating a new particle)
 *	\param[in,out] oldEulerVel : pointer to Eulerian velocity & density; imposed values are set and the other is computed here
 *	\param[in,out] forces : pointer to forces; used only for cloning
 *	\param[in,out] contupd : pointer to contudp; used only for cloning
 *	\param[in,out] vertices : pointer to associated vertices; fluid particles have this information if they are passing through a boundary and are going to be deleted
 *	\param[in] vertPos[0] : relative position of the vertex 0 with respect to the segment center
 *	\param[in] vertPos[1] : relative position of the vertex 1 with respect to the segment center
 *	\param[in] vertPos[2] : relative position of the vertex 2 with respect to the segment center
 *	\param[in,out] pinfo : pointer to particle info; written only when cloning
 *	\param[in,out] particleHash : pointer to particle hash; written only when cloning
 *	\param[in] cellStart : pointer to indices of first particle in cells
 *	\param[in] neibsList : neighbour list
 *	\param[in] numParticles : number of particles
 *	\param[out] newNumParticles : number of particles after creation of new fluid particles due to open boundaries
 *	\param[in] dt : time-step size
 *	\param[in] step : the step in the time integrator
 *	\param[in] deltap : the particle size
 *	\param[in] slength : the smoothing length
 *	\param[in] influenceradius : the kernel radius
 *	\param[in] deviceId : current device identifier
 *	\param[in] numDevices : total number of devices; used for id generation of new fluid particles
 */
template<KernelType kerneltype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
saVertexBoundaryConditions(
						float4*			oldPos,
						float4*			oldVel,
						float*			oldTKE,
						float*			oldEps,
						float4*			oldGGam,
						float4*			oldEulerVel,
						float4*			forces,
						float2*			contupd,
						vertexinfo*		vertices,
				const	float2*			vertPos0,
				const	float2*			vertPos1,
				const	float2*			vertPos2,
						particleinfo*	pinfo,
						hashKey*		particleHash,
				const	uint*			cellStart,
				const	neibdata*		neibsList,
				const	uint			numParticles,
				const	uint			oldNumParticles,
						uint*			newNumParticles,
				const	float			dt,
				const	int				step,
				const	float			deltap,
				const	float			slength,
				const	float			influenceradius,
				const	bool			initStep,
				const	bool			resume,
				const	uint			deviceId,
				const	uint			numDevices,
				const	uint			totParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles
	const particleinfo info = pinfo[index];
	if (!VERTEX(info))
		return;

	float4 pos = oldPos[index];

	const float vel = length(make_float3(oldVel[index]));

	// these are taken as the sum over all adjacent segments
	float sumpWall = 0.0f; // summation for computing the density
	float sumtke = 0.0f; // summation for computing tke (k-epsilon model)
	float sumeps = 0.0f; // summation for computing epsilon (k-epsilon model)
	float sumMdot = 0.0f; // summation for computing the mass variance based on in/outflow
	float massFluid = 0.0f; // mass obtained from a outgoing - mass of a new fluid
	float sump = 0.0f; // summation for the pressure on IO boundaries
	float3 sumvel = make_float3(0.0f); // summation for the velocity on IO boundaries
	float alpha = 0.0f; // summation of normalization for IO boundaries
	bool foundFluid = false; // check if a vertex particle has a fluid particle in its support
	float numseg = 0.0f;

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;
	const float gam = oldGGam[index].w;
	// normal:
	// for solid walls this normal only takes the associated normals of segments into account that are solid as well
	// for io walls this normal only takes the associated normals of segments into account that themeselves are io
	const float3 normal = as_float3(tex1Dfetch(boundTex, index));
	// wall normal:
	// for corner vertices the wall normal is equal to the normal of the associated segments that belong to a solid wall
	// at the initialization step the wall normal is computed for all vertices in order to get an approximate normal
	// which is then used to compute grad gamma and gamma
	float3 wallNormal = make_float3(0.0f);
	const float sqC0 = d_sqC0[fluid_num(info)];

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		const particleinfo neib_info = pinfo[neib_index];

		if (BOUNDARY(neib_info) || FLUID(neib_info)) {

			// prepare indices of neib vertices
			const vertexinfo neibVerts = vertices[neib_index];

			if (FLUID(neib_info)) {
			//if (FLUID(neib_info) || (VERTEX(neib_info) && !IO_BOUNDARY(neib_info) && IO_BOUNDARY(info))) {
			//if (FLUID(neib_info) || (!IO_BOUNDARY(info) && VERTEX(neib_info) && IO_BOUNDARY(neib_info) && !CORNER(neib_info))) {
				const float4 relPos = pos_corr - oldPos[neib_index];
				//if (INACTIVE(relPos) || dot(normal, as_float3(relPos)) > 0.0f)
				if (INACTIVE(relPos))
					continue;
				const float r = length(as_float3(relPos));

				if (r < influenceradius){
					const float neib_rho = oldVel[neib_index].w;
					const float neib_pres = P(neib_rho, fluid_num(neib_info));
					const float neib_vel = length(make_float3(oldVel[neib_index]));

					// kernel value times volume
					const float w = W<kerneltype>(r, slength)*relPos.w/neib_rho;
					// normal distance based on grad Gamma which approximates the normal of the domain
					sumpWall += fmaxf(neib_pres + neib_rho*dot(d_gravity, as_float3(relPos)), 0.0f)*w;
					// for all boundaries we have dk/dn = 0
					if (IO_BOUNDARY(info) && !CORNER(info)) {
						// for open boundaries compute dv/dn = 0
						sumvel += w*as_float3(oldVel[neib_index] + oldEulerVel[neib_index]);
						// for open boundaries compute pressure interior state
						//sump += w*fmaxf(0.0f, neib_pres+dot(d_gravity, as_float3(relPos)*d_rho0[fluid_num(neib_info)]));
						sump += w*fmaxf(0.0f, neib_pres);
						// and de/dn = 0
					}
					alpha += w;
				}
			}

			if (BOUNDARY(neib_info)) {
				const float4 boundElement = tex1Dfetch(boundTex, neib_index);

				// check if vertex is associated with this segment
				if (neibVerts.x == id(info) || neibVerts.y == id(info) || neibVerts.z == id(info)) {
					// in the initial step we need to compute an approximate grad gamma direction
					// for the computation of gamma, in general we need a sort of normal as well
					// for corner vertices this wallNormal takes only solid walls into account so
					// that the eulerian velocity in the k-eps case is only normal to the solid wall
					if (initStep || (CORNER(info) && !IO_BOUNDARY(neib_info)))
						wallNormal += as_float3(boundElement)*boundElement.w;
					// k and eps are taken directly from the associated segments
					const float neib_k = oldTKE ? oldTKE[neib_index] : NAN;
					const float neib_eps = oldEps ? oldEps[neib_index] : NAN;
					sumtke += neib_k;
					sumeps += neib_eps;
					numseg += 1.0f;
					// corner vertices only take solid wall segments into account
					if (CORNER(info) && IO_BOUNDARY(neib_info))
						continue;
					// boundary conditions on rho, k, eps
					const float neibRho = oldVel[neib_index].w;
					if (!CORNER(info) && IO_BOUNDARY(neib_info)){
						/* The following would increase the output of particles close to an edge
						 * But it is not used for the following reason: If only 1/3 of each segment is taken into account
						 * it lowers the effective inflow area. This is ok, as part of the area of a segment that is associated
						 * with a corner "belongs" to a corner vertex.
						// number of vertices associated to a segment that are of the same object type
						float numOutVerts = 2.0f;
						if (neibVerts.w == ALLVERTICES) // all vertices are of the same object type
							numOutVerts = 3.0f;
						else if (neibVerts.w & ~VERTEX1 == 0 || neibVerts.w & ~VERTEX2 == 0 || neibVerts.w & ~VERTEX3 == 0) // only one vertex
							numOutVerts = 1.0f;
						*/
						/*
						// Distribute mass flux evenly among vertex particles of a segment
						float numOutVerts = 3.0f;
						*/

						// first get the position of the vertices local coordinate system for relative positions to vertices
						uint j = 0;
						// Get index j for which n_s is minimal
						if (fabsf(boundElement.x) > fabsf(boundElement.y))
							j = 1;
						if ((1-j)*fabsf(boundElement.x) + j*fabsf(boundElement.y) > fabsf(boundElement.z))
							j = 2;

						// compute the first coordinate which is a 2-D rotated version of the normal
						const float4 coord1 = normalize(make_float4(
							// switch over j to give: 0 -> (0, z, -y); 1 -> (-z, 0, x); 2 -> (y, -x, 0)
							-((j==1)*boundElement.z) +  (j == 2)*boundElement.y , // -z if j == 1, y if j == 2
							  (j==0)*boundElement.z  - ((j == 2)*boundElement.x), // z if j == 0, -x if j == 2
							-((j==0)*boundElement.y) +  (j == 1)*boundElement.x , // -y if j == 0, x if j == 1
							0));
						// the second coordinate is the cross product between the normal and the first coordinate
						const float4 coord2 = cross3(boundElement, coord1);

						const float2 vPos0 = vertPos0[neib_index];
						const float2 vPos1 = vertPos1[neib_index];
						const float2 vPos2 = vertPos2[neib_index];

						// relative positions of vertices with respect to the segment, normalized by h
						float4 v0 = (vPos0.x*coord1 + vPos0.y*coord2); // e.g. v0 = r_{v0} - r_s
						float4 v1 = (vPos1.x*coord1 + vPos1.y*coord2);
						float4 v2 = (vPos2.x*coord1 + vPos2.y*coord2);
						float3 vertexWeights;
						const float3 vx[3] = {as_float3(v0), as_float3(v1), as_float3(v2)};
						getMassRepartitionFactor(vx, as_float3(boundElement), vertexWeights);
						float beta = 0.0f;
						if (neibVerts.x == id(info))
							beta = vertexWeights.x;
						else if (neibVerts.y == id(info))
							beta = vertexWeights.y;
						else if (neibVerts.z == id(info))
							beta = vertexWeights.z;

						sumMdot += neibRho*beta*boundElement.w*
									dot3(oldEulerVel[neib_index],boundElement); // the euler vel should be subtracted by the lagrangian vel which is assumed to be 0 now.
					}
				}
			}
			else if (IO_BOUNDARY(info) && FLUID(neib_info)){
				const float4 relPos = pos_corr - oldPos[neib_index];
				const float r = length3(relPos);
				if(!foundFluid && r < influenceradius)
					foundFluid = true;

				// check if this fluid particles is marked for deletion (i.e. vertices != 0)
				if (step == 2 && neibVerts.x | neibVerts.y != 0 && ACTIVE(relPos)) {
					// betaAV is the weight in barycentric coordinates
					float betaAV = 0.0f;
					const float4 vertexWeights = oldGGam[neib_index];
					// check if one of the vertices is equal to the present one
					if (neibVerts.x == id(info))
						betaAV = vertexWeights.x;
					else if (neibVerts.y == id(info))
						betaAV = vertexWeights.y;
					else if (neibVerts.z == id(info))
						betaAV = vertexWeights.z;
					if(betaAV > 0.0f){
						// add mass from fluid particle to vertex particle
						// note that the mass was transfered from pos to gam
						massFluid += betaAV*vertexWeights.w;
					}
				}

			}
		} // BOUNDARY(neib_info) || FLUID(neib_info)
	}

	// normalize wall normal
	if (CORNER(info) || initStep)
		wallNormal = normalize(wallNormal);

	// update boundary conditions on array
	if (!initStep)
		alpha = fmaxf(alpha, 0.1f*gam); // avoid division by 0
	else
		alpha = fmaxf(alpha, 1e-5f);
	oldVel[index].w = RHO(sumpWall/alpha,fluid_num(info));
	if (oldTKE)
		oldTKE[index] = fmax(sumtke/numseg, 1e-6f);
	if (oldEps)
		oldEps[index] = fmax(sumeps/numseg, 1e-6f);
	if (!initStep && oldTKE && (!IO_BOUNDARY(info) || CORNER(info) || PRES_IO(info))) {
		// adjust Eulerian velocity so that it is tangential to the fixed wall
		if (CORNER(info)) {
			// normal for corners is normal to the IO it belongs, so we use wallNormal which is normal
			// to the solid wall it is adjacent to
			as_float3(oldEulerVel[index]) -= dot(as_float3(oldEulerVel[index]), wallNormal)*wallNormal;
		}
		else if (!IO_BOUNDARY(info))
			as_float3(oldEulerVel[index]) -= dot(as_float3(oldEulerVel[index]), normal)*normal;
	}
	// open boundaries
	if (IO_BOUNDARY(info) && !CORNER(info)) {
		float4 eulerVel = oldEulerVel[index];
		if (alpha > 0.1f*oldGGam[index].w) { // note: defaults are set in the place where bcs are imposed
			sumvel /= alpha;
			sump /= alpha;
			const float unInt = dot(sumvel, normal);
			const float unExt = dot(as_float3(eulerVel), normal);
			const float rhoInt = RHO(sump, fluid_num(info));
			const float rhoExt = eulerVel.w;

			calculateIOboundaryCondition(eulerVel, info, rhoInt, rhoExt, sumvel, unInt, unExt, normal);
		}
		else {
			if (VEL_IO(info))
				eulerVel.w = d_rho0[fluid_num(info)];
			else
				eulerVel = make_float4(0.0f, 0.0f, 0.0f, eulerVel.w);
		}
		oldEulerVel[index] = eulerVel;
		// the density of the particle is equal to the "eulerian density"
		oldVel[index].w = eulerVel.w;

		// finalize mass computation
		// reference mass:
		const float rho0 = d_rho0[fluid_num(info)];
		const float refMass = deltap*deltap*deltap*rho0;

		// Update vertex mass
		if (!initStep) {
			// time stepping
			pos.w += dt*sumMdot;
			// if a vertex has no fluid particles around and its mass flux is negative then set its mass to 0
			if (alpha < 0.1*gam && sumMdot < 0.0f) // sphynx version
			//if (!foundFluid && sumMdot < 0.0f)
				pos.w = 0.0f;

			// clip to +/- 2 refMass all the time
			pos.w = fmaxf(-2.0f*refMass, fminf(2.0f*refMass, pos.w));

			// clip to +/- originalVertexMass if we have outflow
			// or if the normal eulerian velocity is less or equal to 0
			if (sumMdot < 0.0f || dot(normal,as_float3(eulerVel)) < 1e-5f*d_sscoeff[fluid_num(info)]) {
				const float4 boundElement = tex1Dfetch(boundTex, index);
				pos.w = fmaxf(-refMass*boundElement.w, fminf(refMass*boundElement.w, pos.w));
			}

		}
		// particles that have an initial density less than the reference density have their mass set to 0
		// or if their velocity is initially 0
		else if (!resume &&
			( (PRES_IO(info) && eulerVel.w - rho0 <= 1e-10f*rho0) ||
			  (VEL_IO(info) && length3(eulerVel) < 1e-10f*d_sscoeff[fluid_num(info)])) )
			pos.w = 0.0f;

		// check whether new particles need to be created
			// only create new particles in the second part of the time step
		if (step == 2 &&
			// create new particle if the mass of the vertex is large enough
			pos.w > refMass*0.5f &&
			// if mass flux > 0
			sumMdot > 0 &&
			// if imposed velocity is greater 0
			dot(normal,as_float3(eulerVel)) > 1e-5f &&
			// pressure inlets need p > 0 to create particles
			(VEL_IO(info) || eulerVel.w-rho0 > rho0*1e-5f) &&
			// corner vertices are not allowed to create new particles
			!CORNER(info))
		{
			massFluid -= refMass;
			// Create new particle
			particleinfo clone_info;
			uint clone_idx = createNewFluidParticle(clone_info, info, oldNumParticles, numDevices, newNumParticles, totParticles);

			// Problem has already checked that there is enough memory for new particles
			float4 clone_pos = pos; // new position is position of vertex particle
			clone_pos.w = refMass; // new fluid particle has reference mass
			int3 clone_gridPos = gridPos; // as the position is the same so is the grid position

			// assign new values to array
			oldPos[clone_idx] = clone_pos;
			pinfo[clone_idx] = clone_info;
			particleHash[clone_idx] = calcGridHash(clone_gridPos);
			// the new velocity of the fluid particle is the eulerian velocity of the vertex
			oldVel[clone_idx] = oldEulerVel[index];
			forces[clone_idx] = make_float4(0.0f);

			// the eulerian velocity of fluid particles is always 0
			oldEulerVel[clone_idx] = make_float4(0.0f);
			contupd[clone_idx] = make_float2(0.0f);
			oldGGam[clone_idx] = oldGGam[index];
			vertices[clone_idx] = make_vertexinfo(0, 0, 0, 0);
			if (oldTKE)
				oldTKE[clone_idx] = oldTKE[index];
			if (oldEps)
				oldEps[clone_idx] = oldEps[index];
		}

		// add contribution from newly created fluid or outgoing fluid particles
		pos.w += massFluid;
		oldPos[index].w = pos.w;
	}
	// corners in pressure boundaries have imposed pressures
	//else if (IO_BOUNDARY(info) && CORNER(info) && PRES_IO(info)) {
	//	oldVel[index].w = oldEulerVel[index].w;
	//}

	// finalize computation of average norm for gamma calculation in the initial step
	if (initStep && !resume) {
		oldGGam[index].x = wallNormal.x;
		oldGGam[index].y = wallNormal.y;
		oldGGam[index].z = wallNormal.z;
		oldGGam[index].w = 0.0f;
	}
}

/// Compute the initial value of gamma in the semi-analytical boundary case
/*! This function computes the initial value of \f[\gamma\f] in the semi-analytical boundary case, using a Gauss quadrature formula.
 *	\param[out] newGGam : pointer to the new value of (grad) gamma
 *	\param[in,out] boundelement : normal of segments and of vertices (the latter is computed in this routine)
 *	\param[in] oldPos : pointer to positions and masses; masses of vertex particles are updated
 *	\param[in] oldGGam : pointer to (grad) gamma; used as an approximate normal to the boundary in the computation of gamma
 *	\param[in] vertPos[0] : relative position of the vertex 0 with respect to the segment center
 *	\param[in] vertPos[1] : relative position of the vertex 1 with respect to the segment center
 *	\param[in] vertPos[2] : relative position of the vertex 2 with respect to the segment center
 *	\param[in] pinfo : pointer to particle info; written only when cloning
 *	\param[in] particleHash : pointer to particle hash; written only when cloning
 *	\param[in] cellStart : pointer to indices of first particle in cells
 *	\param[in] neibsList : neighbour list
 *	\param[in] numParticles : number of particles
 *	\param[in] slength : the smoothing length
 *	\param[in] influenceradius : the kernel radius
 */
template<KernelType kerneltype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
initGamma(
						float4*			newGGam,
						float4*			boundelement,
				const	float4*			oldPos,
				const	float4*			oldGGam,
				const	vertexinfo*		vertices,
				const	float2*			vertPos0,
				const	float2*			vertPos1,
				const	float2*			vertPos2,
				const	hashKey*		particleHash,
				const	particleinfo*	pinfo,
				const	uint*			cellStart,
				const	neibdata*		neibsList,
				const	uint			numParticles,
				const	float			slength,
				const	float			deltap,
				const	float			influenceradius,
				const	float			epsilon)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles
	const particleinfo info = pinfo[index];
	if (BOUNDARY(info))
		return;

	float4 pos = oldPos[index];

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;
	float gam = 1;
	float4 gGam = make_float4(0.f);
	const float3 normal = as_float3(oldGGam[index]);
	float4 newNormal = make_float4(0.0f);

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		const particleinfo neib_info = pinfo[neib_index];

		if (BOUNDARY(neib_info)) {

			const float4 ns = boundelement[neib_index];
			const float4 relPos = pos_corr - oldPos[neib_index];
			if (INACTIVE(relPos))
				continue;

			// compute new normal for vertices
			if (VERTEX(info)) {
				// prepare ids of neib vertices
				const vertexinfo neibVerts = vertices[neib_index];

				if (neibVerts.x == id(info) || neibVerts.y == id(info) || neibVerts.z == id(info)) {
					if ((IO_BOUNDARY(info) && IO_BOUNDARY(neib_info)) || (!IO_BOUNDARY(info) && !IO_BOUNDARY(neib_info)))
						newNormal += ns;
				}
			}

			// compute gamma for all particles
			// local coordinate system for relative positions to vertices
			uint j = 0;
			// Get index j for which n_s is minimal
			if (fabsf(ns.x) > fabsf(ns.y))
				j = 1;
			if ((1-j)*fabsf(ns.x) + j*fabsf(ns.y) > fabsf(ns.z))
				j = 2;

			// compute the first coordinate which is a 2-D rotated version of the normal
			const float4 coord1 = normalize(make_float4(
						// switch over j to give: 0 -> (0, z, -y); 1 -> (-z, 0, x); 2 -> (y, -x, 0)
						-((j==1)*ns.z) +  (j == 2)*ns.y , // -z if j == 1, y if j == 2
						(j==0)*ns.z  - ((j == 2)*ns.x), // z if j == 0, -x if j == 2
						-((j==0)*ns.y) +  (j == 1)*ns.x , // -y if j == 0, x if j == 1
						0));
			// the second coordinate is the cross product between the normal and the first coordinate
			const float4 coord2 = cross3(ns, coord1);

			// relative positions of vertices with respect to the segment
			float4 v0 = -(vertPos0[neib_index].x*coord1 + vertPos0[neib_index].y*coord2); // e.g. v0 = r_{v0} - r_s
			float4 v1 = -(vertPos1[neib_index].x*coord1 + vertPos1[neib_index].y*coord2);
			float4 v2 = -(vertPos2[neib_index].x*coord1 + vertPos2[neib_index].y*coord2);
			float4 vertexRelPos[3] = {v0, v1, v2};

			float ggamAS = gradGamma<kerneltype>(slength, as_float3(relPos), vertexRelPos, as_float3(ns));
			float minlRas = 0;
			const float gamAS = Gamma<kerneltype>(slength, as_float3(relPos), vertexRelPos, as_float3(ns), 
					normal, epsilon, deltap, true, minlRas);
			gGam.x += ggamAS*ns.x;
			gGam.y += ggamAS*ns.y;
			gGam.z += ggamAS*ns.z;

			// general formula (also used if particle is on 
			// vertex / edge to compute remaining edges)
			const float x = fminf(dot3(ns, relPos)/slength, 0.25f);
			const float sx = fmaxf(x*8.0f - 1.0f,0.0f);
			// smootherstep function
			const float smooth = VERTEX(info) ? 1.0f : ((2.0f*sx-5.0f)*3.0f*sx+10.0f)*sx*sx*sx;
			gam -= (smooth > epsilon ? gamAS : 0.0f)*smooth;
		}
	}
	newGGam[index] = make_float4(gGam.x, gGam.y, gGam.z, gam);
	newNormal = normalize3(newNormal);
	boundelement[index] = make_float4(newNormal.x, newNormal.y, newNormal.z, boundelement[index].w);
}

#define MAXNEIBVERTS 30

/// Modifies the initial mass of vertices on open boundaries
/*! This function computes the initial value of \f[\gamma\f] in the semi-analytical boundary case, using a Gauss quadrature formula.
 *	\param[out] newGGam : pointer to the new value of (grad) gamma
 *	\param[in,out] boundelement : normal of segments and of vertices (the latter is computed in this routine)
 *	\param[in] oldPos : pointer to positions and masses; masses of vertex particles are updated
 *	\param[in] oldGGam : pointer to (grad) gamma; used as an approximate normal to the boundary in the computation of gamma
 *	\param[in] vertPos[0] : relative position of the vertex 0 with respect to the segment center
 *	\param[in] vertPos[1] : relative position of the vertex 1 with respect to the segment center
 *	\param[in] vertPos[2] : relative position of the vertex 2 with respect to the segment center
 *	\param[in] pinfo : pointer to particle info; written only when cloning
 *	\param[in] particleHash : pointer to particle hash; written only when cloning
 *	\param[in] cellStart : pointer to indices of first particle in cells
 *	\param[in] neibsList : neighbour list
 *	\param[in] numParticles : number of particles
 *	\param[in] slength : the smoothing length
 *	\param[in] influenceradius : the kernel radius
 */
template<KernelType kerneltype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
initIOmass_vertexCount(
				const	vertexinfo*		vertices,
				const	hashKey*		particleHash,
				const	particleinfo*	pinfo,
				const	uint*			cellStart,
				const	neibdata*		neibsList,
						float4*			forces,
				const	uint			numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles
	const particleinfo info = pinfo[index];
	if (!(VERTEX(info) && IO_BOUNDARY(info) && !CORNER(info)))
		return;

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	uint vertexCount = 0;

	const float4 pos = make_float4(0.0f); // we don't need pos, so let's just set it to 0
	float3 pos_corr;
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	uint neibVertIds[MAXNEIBVERTS];
	uint neibVertIdsCount=0;

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		const particleinfo neib_info = pinfo[neib_index];

		// only boundary neighbours as we need to count the vertices that belong to the same segment as our vertex particle
		if (BOUNDARY(neib_info) && IO_BOUNDARY(neib_info)) {

			// prepare ids of neib vertices
			const vertexinfo neibVerts = vertices[neib_index];

			// only check adjacent boundaries
			if (neibVerts.x == id(info) || neibVerts.y == id(info) || neibVerts.z == id(info)) {
				// check if we don't have the current vertex
				if (id(info) != neibVerts.x) {
					neibVertIds[neibVertIdsCount] = neibVerts.x;
					neibVertIdsCount+=1;
				}
				if (id(info) != neibVerts.y) {
					neibVertIds[neibVertIdsCount] = neibVerts.y;
					neibVertIdsCount+=1;
				}
				if (id(info) != neibVerts.z) {
					neibVertIds[neibVertIdsCount] = neibVerts.z;
					neibVertIdsCount+=1;
				}
			}

		}
	}

	neib_cellnum = 0;
	neib_cell_base_index = 0;

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		const particleinfo neib_info = pinfo[neib_index];

		if (!VERTEX(neib_info))
			continue;

		for (uint j = 0; j<neibVertIdsCount; j++) {
			if (id(neib_info) == neibVertIds[j] && !CORNER(neib_info))
				vertexCount += 1;
		}
	}

	forces[index].w = (float)(vertexCount);
}

template<KernelType kerneltype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
initIOmass(
				const	float4*			oldPos,
				const	float4*			forces,
				const	vertexinfo*		vertices,
				const	hashKey*		particleHash,
				const	particleinfo*	pinfo,
				const	uint*			cellStart,
				const	neibdata*		neibsList,
						float4*			newPos,
				const	uint			numParticles,
				const	float			deltap)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	const particleinfo info = pinfo[index];
	const float4 pos = oldPos[index];
	newPos[index] = pos;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles
	//const particleinfo info = pinfo[index];
	if (!(VERTEX(info) && IO_BOUNDARY(info) && !CORNER(info)))
		return;

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// does this vertex get or donate mass; decided by the id of a vertex particle
	const bool getMass = id(info)%2;
	float massChange = 0.0f;

	const float refMass = 0.5f*deltap*deltap*deltap*d_rho0[fluid_num(info)]; // half of the fluid mass

	// difference between reference mass and actual mass of particle
	const float massDiff = refMass - pos.w;
	// number of vertices associated with the same boundary segment as this vertex (that are also IO)
	const float vertexCount = forces[index].w;

	uint neibVertIds[MAXNEIBVERTS];
	uint neibVertIdsCount=0;

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		const particleinfo neib_info = pinfo[neib_index];

		// only boundary neighbours as we need to count the vertices that belong to the same segment as our vertex particle
		if (BOUNDARY(neib_info) && IO_BOUNDARY(neib_info)) {

			// prepare ids of neib vertices
			const vertexinfo neibVerts = vertices[neib_index];

			// only check adjacent boundaries
			if (neibVerts.x == id(info) || neibVerts.y == id(info) || neibVerts.z == id(info)) {
				// check if we don't have the current vertex
				if (id(info) != neibVerts.x) {
					neibVertIds[neibVertIdsCount] = neibVerts.x;
					neibVertIdsCount+=1;
				}
				if (id(info) != neibVerts.y) {
					neibVertIds[neibVertIdsCount] = neibVerts.y;
					neibVertIdsCount+=1;
				}
				if (id(info) != neibVerts.z) {
					neibVertIds[neibVertIdsCount] = neibVerts.z;
					neibVertIdsCount+=1;
				}
			}

		}
	}

	neib_cellnum = 0;
	neib_cell_base_index = 0;

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		const particleinfo neib_info = pinfo[neib_index];

		if (!VERTEX(neib_info))
			continue;

		for (uint j = 0; j<neibVertIdsCount; j++) {
			if (id(neib_info) == neibVertIds[j]) {
				const bool neib_getMass = id(neib_info)%2;
				if (getMass != neib_getMass && !CORNER(neib_info)) { // if not both vertices get or donate mass
					if (getMass) {// original vertex gets mass
						if (massDiff > 0.0f)
							massChange += massDiff/vertexCount; // get mass from all adjacent vertices equally
					}
					else {
						const float neib_massDiff = refMass - oldPos[neib_index].w;
						if (neib_massDiff > 0.0f) {
							const float neib_vertexCount = forces[neib_index].w;
							massChange -= neib_massDiff/neib_vertexCount; // get mass from this vertex
						}
					}
				}
			}
		}
	}

	newPos[index].w += massChange;
}

/************************************************************************************************************/
/*					   Kernels for computing acceleration without gradient correction					 */
/************************************************************************************************************/

/* forcesDevice kernel and auxiliary types and functions */
#include "forces_kernel.def"

/************************************************************************************************************/


/************************************************************************************************************/
/*					   Kernels for XSPH, Shepard and MLS corrections									   */
/************************************************************************************************************/

//! This kernel computes the Sheppard correction
template<KernelType kerneltype,
	BoundaryType boundarytype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
shepardDevice(	const float4*	posArray,
				float4*			newVel,
				const hashKey*		particleHash,
				const uint*		cellStart,
				const neibdata*	neibsList,
				const uint		numParticles,
				const float		slength,
				const float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	const particleinfo info = tex1Dfetch(infoTex, index);

	#if PREFER_L1
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	// If particle is inactive there is absolutely nothing to do
	if (INACTIVE(pos))
		return;

	float4 vel = tex1Dfetch(velTex, index);

	// We apply Shepard normalization :
	//	* with LJ or DYN boundary only on fluid particles
	//TODO 	* with SA boundary ???
	// in any other case we have to copy the vel vector in the new velocity array
	if (NOT_FLUID(info)) {
		newVel[index] = vel;
		return;
	}


	// Taking into account self contribution in summation
	float temp1 = pos.w*W<kerneltype>(0, slength);
	float temp2 = temp1/vel.w ;

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if PREFER_L1
		const float4 relPos = pos_corr - posArray[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif


		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// Skip inactive neighbors
		if (INACTIVE(relPos)) {
			continue;
		}

		const float r = length(as_float3(relPos));

		const float neib_rho = tex1Dfetch(velTex, neib_index).w;

		// Add neib contribution only if it's a fluid one
		// TODO: check with SA
		if ((boundarytype == DYN_BOUNDARY || (boundarytype != DYN_BOUNDARY && FLUID(neib_info)))
				&& r < influenceradius ) {
			const float w = W<kerneltype>(r, slength)*relPos.w;
			temp1 += w;
			temp2 += w/neib_rho;
		}
	}

	// Normalize the density and write in global memory
	vel.w = temp1/temp2;
	newVel[index] = vel;
}

//! This kernel computes the MLS correction
template<KernelType kerneltype,
	BoundaryType boundarytype>
__global__ void
__launch_bounds__(BLOCK_SIZE_MLS, MIN_BLOCKS_MLS)
MlsDevice(	const float4*	posArray,
			float4*			newVel,
			const hashKey*		particleHash,
			const uint*		cellStart,
			const neibdata*	neibsList,
			const uint		numParticles,
			const float		slength,
			const float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	const particleinfo info = tex1Dfetch(infoTex, index);

	#if PREFER_L1
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	// If particle is inactive there is absolutely nothing to do
	if (INACTIVE(pos))
		return;

	float4 vel = tex1Dfetch(velTex, index);

	// We apply MLS normalization :
	//	* with LJ or DYN boundary only on fluid particles
	//TODO 	* with SA boundary ???
	// in any other case we have to copy the vel vector in the new velocity array
	//if (NOT_FLUID(info)) {
	//	newVel[index] = vel;
	//	return;
	//}

	// MLS matrix elements
	symtensor4 mls;
	mls.xx = mls.xy = mls.xz = mls.xw =
		mls.yy = mls.yz = mls.yw =
		mls.zz = mls.zw = mls.ww = 0;

	// Number of neighbors
	int neibs_num = 0;

	// Taking into account self contribution in MLS matrix construction
	mls.xx = W<kerneltype>(0, slength)*pos.w/vel.w;

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// First loop over all neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if PREFER_L1
		const float4 relPos = pos_corr - posArray[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// Skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		const float neib_rho = tex1Dfetch(velTex, neib_index).w;
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// Add neib contribution only if it's a fluid one
		// TODO: check with SA
		if (r < influenceradius && (boundarytype == DYN_BOUNDARY || FLUID(neib_info))) {
			neibs_num ++;
			const float w = W<kerneltype>(r, slength)*relPos.w/neib_rho;	// Wij*Vj

			/* Scale relPos by slength for stability and resolution independence */
			MlsMatrixContrib(mls, relPos/slength, w);
		}
	} // end of first loop trough neighbors

	// Resetting persistent variables across getNeibData
	neib_cellnum = 0;
	neib_cell_base_index = 0;

	// We want to compute B solution of M B = E where E =(1, 0, 0, 0) and
	// M is our MLS matrix. M is symmetric, positive (semi)definite. Since we
	// cannot guarantee that the matrix is invertible (it won't be in cases
	// such as thin sheets of particles or structures of even lower topological
	// dimension), we rely on the iterative conjugate residual method to
	// find a solution, with E itself as initial guess.

	// known term
	const float4 E = make_float4(1, 0, 0, 0);

	const float D = det(mls);

	// solution
	float4 B;
	if (fabsf(D) < FLT_EPSILON) {
		symtensor4 mls_eps = mls;
		const float eps = fabsf(D) + FLT_EPSILON;
		mls_eps.xx += eps;
		mls_eps.yy += eps;
		mls_eps.zz += eps;
		mls_eps.ww += eps;
		const float D_eps = det(mls_eps);
		B = adjugate_row1(mls_eps)/D_eps;
	} else {
		B = adjugate_row1(mls)/D;
	}

#define MAX_CR_STEPS 32
	uint steps = 0;
	for (; steps < MAX_CR_STEPS; ++steps) {
		float lenB = hypot(B);

		float4 MdotB = dot(mls, B);
		float4 residual = E - MdotB;

		// r.M.r
		float num = ddot(mls, residual);

		// (M.r).(M.r)
		float4 Mp = dot(mls, residual);
		float den = dot(Mp, Mp);

		float4 corr = (num/den)*residual;
		float lencorr = hypot(corr);

		if (hypot(residual) < lenB*FLT_EPSILON)
			break;

		if (lencorr < 2*lenB*FLT_EPSILON)
			break;

		B += corr;
	}

	/* Scale for resolution independence, again */
	B.y /= slength;
	B.z /= slength;
	B.w /= slength;

	// Taking into account self contribution in density summation
	vel.w = B.x*W<kerneltype>(0, slength)*pos.w;

	// Loop over all the neighbors (Second loop)
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
			neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
#if PREFER_L1
		const float4 relPos = pos_corr - posArray[neib_index];
#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
#endif

		// Skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// Interaction between two particles
		if (r < influenceradius && (boundarytype == DYN_BOUNDARY || FLUID(neib_info))) {
			const float w = W<kerneltype>(r, slength)*relPos.w;	 // ρj*Wij*Vj = mj*Wij
			vel.w += MlsCorrContrib(B, relPos, w);
		}
	}  // end of second loop trough neighbors

	// If MLS starts misbehaving, define DEBUG_PARTICLE: this will
	// print the MLS-corrected density for the particles statisfying
	// the DEBUG_PARTICLE condition. Some examples:

//#define DEBUG_PARTICLE (index == numParticles - 1)
//#define DEBUG_PARTICLE (id(info) == numParticles - 1)
//#define DEBUG_PARTICLE (fabsf(err) > 64*FLT_EPSILON)

#ifdef DEBUG_PARTICLE
	{
		const float old = tex1Dfetch(velTex, index).w;
		const float err = 1 - vel.w/old;
		if (DEBUG_PARTICLE) {
			printf("MLS %d %d %22.16g => %22.16g (%6.2e)\n",
				index, id(info),
				old, vel.w, err*100);
		}
	}
#endif

	newVel[index] = vel;
}
/************************************************************************************************************/

/************************************************************************************************************/
/*					   CFL max kernel																		*/
/************************************************************************************************************/
//! Computes the max of a float across several threads
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

//! Disables particles that have exited through an open boundary
/*!
 This kernel is only used for SA boundaries in combination with the outgoing particle identification
 in saSegmentBoundaryConditions(). If a particle crosses a segment then the vertexinfo array is set
 for this fluid particle. This is used here to identify such particles. In turn the vertexinfo array
 is reset and the particle is disabled.
*/
__global__ void
disableOutgoingPartsDevice(			float4*		oldPos,
									vertexinfo*	oldVertices,
							const	uint		numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if(index < numParticles) {
		const particleinfo info = tex1Dfetch(infoTex, index);
		if (FLUID(info)) {
			float4 pos = oldPos[index];
			if (ACTIVE(pos)) {
				vertexinfo vertices = oldVertices[index];
				if (vertices.x | vertices.y != 0) {
					disable_particle(pos);
					vertices.x = 0;
					vertices.y = 0;
					vertices.z = 0;
					vertices.w = 0;
					oldPos[index] = pos;
					oldVertices[index] = vertices;
				}
			}
		}
	}
}

//! Identify corner vertices on open boundaries
/*!
 Corner vertices are vertices that have segments that are not part of an open boundary. These
 vertices are treated slightly different when imposing the boundary conditions during the
 computation in saVertexBoundaryConditions.
*/
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
saIdentifyCornerVertices(
				const	float4*			oldPos,
						particleinfo*	pinfo,
				const	hashKey*		particleHash,
				const	vertexinfo*		vertices,
				const	uint*			cellStart,
				const	neibdata*		neibsList,
				const	uint			numParticles,
				const	float			deltap,
				const	float			eps)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles which are associated to an open boundary
	particleinfo info = pinfo[index];
	const uint obj = object(info);
	if (!(VERTEX(info) && IO_BOUNDARY(info)))
		return;

	float4 pos = oldPos[index];

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	const uint vid = id(info);

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == NEIBS_END) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		const particleinfo neib_info = pinfo[neib_index];
		const uint neib_obj = object(neib_info);

		// loop only over boundary elements that are not of the same open boundary
		if (BOUNDARY(neib_info) && !(obj == neib_obj && IO_BOUNDARY(neib_info))) {
			// check if the current vertex is part of the vertices of the segment
			if (vertices[neib_index].x == vid ||
				vertices[neib_index].y == vid ||
				vertices[neib_index].z == vid) {
				SET_FLAG(info, FG_CORNER);
				pinfo[index] = info;
				break;
			}
		}
	}
}

/** @} */

/************************************************************************************************************/

} //namespace cuforces
#endif
