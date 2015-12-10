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

using namespace cugeom;
using namespace cusph;
using namespace cuphys;
using namespace cuneibs;
using namespace cuboundaryconditions; // TODO: remove this once saVertexBoundaryConditions is transfered to boundary_conditions_kernel.cu

// Core SPH functions
/** \name Device constants
 *  @{ */
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
	#if( __COMPUTE__ >= 20)
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

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, params.cellStart,
				neib_data, gridPos, neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
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

		if (neib_data == 0xffff) break;

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
		const float typical_sigma = 3*(cuneibs::d_maxNeibs[PT_FLUID] + cuneibs::d_maxNeibs[PT_BOUNDARY])/
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

//! Compute a private variable
/*!
 This function computes an arbitrary passive array. It can be used for debugging purposes or passive scalars
*/
__global__ void
calcPrivateDevice(	const	float4*		pos_array,
							float*		priv,
					const	hashKey*	particleHash,
					const	uint*		cellStart,
					const	neibdata*	neibsList,
					const	float		slength,
					const	float		inflRadius,
							uint		numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if(index < numParticles) {
		#if( __COMPUTE__ >= 20)
		float4 pos = pos_array[index];
		#else
		float4 pos = tex1Dfetch(posTex, index);
		#endif
		const particleinfo info = tex1Dfetch(infoTex, index);
		float4 vel = tex1Dfetch(velTex, index);

		// Compute grid position of current particle
		const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

		// Persistent variables across getNeibData calls
		char neib_cellnum = 0;
		uint neib_cell_base_index = 0;
		float3 pos_corr;

		priv[index] = 0;

		// Loop over all the neighbors
		for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
			neibdata neib_data = neibsList[i + index];

			if (neib_data == 0xffff) break;

			const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
						neib_cellnum, neib_cell_base_index);

			// Compute relative position vector and distance

			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);
			#if( __COMPUTE__ >= 20)
			const float4 relPos = pos_corr - pos_array[neib_index];
			#else
			const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
			#endif
			float r = length(as_float3(relPos));
			if (r < inflRadius)
				priv[index] += 1;
		}

	}
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

	#if( __COMPUTE__ >= 20)
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

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
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

	#if( __COMPUTE__ >= 20)
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

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
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
	if (fabs(D) < FLT_EPSILON) {
		symtensor4 mls_eps = mls;
		const float eps = fabs(D) + FLT_EPSILON;
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

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
			neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
#if( __COMPUTE__ >= 20)
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
//#define DEBUG_PARTICLE (fabs(err) > 64*FLT_EPSILON)

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

/************************************************************************************************************/
/*					   Parallel reduction kernels															*/
/************************************************************************************************************/

extern __shared__ float4 shmem4[];

//! Computes the energy of all particles
extern "C" __global__
void calcEnergiesDevice(
	const		float4	*pPos,
	const		float4	*pVel,
	const	particleinfo	*pInfo,
	const		hashKey	*particleHash,
		uint	numParticles,
		uint	numFluids,
		float4	*output
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
		const float4 pos = pPos[gid];
		const float4 vel = pVel[gid];
		const int3 gridPos = calcGridPosFromParticleHash( particleHash[gid] );
		particleinfo pinfo = pInfo[gid];
		if (FLUID(pinfo)) {
			uint fnum = fluid_num(pinfo);
			float v2 = kahan_sqlength(as_float3(vel));
			// TODO improve precision by splitting the float part from the grid part
			float gh = kahan_dot(d_gravity, as_float3(pos) + (make_float3(gridPos) + 0.5f)*d_cellSize);
			kahan_add(energy[fnum].x, pos.w*v2/2, E_k[fnum].x);
			kahan_add(energy[fnum].y, -pos.w*gh, E_k[fnum].y);
			// internal elastic energy
			float gamma = d_gammacoeff[fnum];
			float gm1 = d_gammacoeff[fnum]-1;
			float rho0 = d_rho0[fnum];
			float elen = __powf(vel.w/rho0, gm1)/gm1 + rho0/vel.w - gamma/gm1;
			float ssp = soundSpeed(vel.w, fnum);
			elen *= ssp*ssp/gamma;
			kahan_add(energy[fnum].z, pos.w*elen, E_k[fnum].z);
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

//! Sum the previously computed energy up (across threads)
extern "C" __global__
void calcEnergies2Device(
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

//! Computes the vorticity field
template<KernelType kerneltype>
__global__ void
calcVortDevice(	const	float4*		posArray,
						float3*		vorticity,
				const	hashKey*		particleHash,
				const	uint*		cellStart,
				const	neibdata*	neibsList,
				const	uint		numParticles,
				const	float		slength,
				const	float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// computing vorticity only for fluid particles
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (NOT_FLUID(info))
		return;

	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif
	const float4 vel = tex1Dfetch(velTex, index);

	float3 vort = make_float3(0.0f);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// First loop over all neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
		const float4 relPos = pos_corr - posArray[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		// Compute relative velocity
		// Now relVel is a float4 and neib density is stored in relVel.w
		const float4 relVel = as_float3(vel) - tex1Dfetch(velTex, neib_index);
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// Compute vorticity
		if (r < influenceradius && FLUID(neib_info)) {
			const float f = F<kerneltype>(r, slength)*relPos.w/relVel.w;	// ∂Wij/∂r*Vj
			// vxij = vxi - vxj and same for vyij and vzij
			vort.x += f*(relVel.y*relPos.z - relVel.z*relPos.y);		// vort.x = ∑(vyij(zi - zj) - vzij*(yi - yj))*∂Wij/∂r*Vj
			vort.y += f*(relVel.z*relPos.x - relVel.x*relPos.z);		// vort.y = ∑(vzij(xi - xj) - vxij*(zi - zj))*∂Wij/∂r*Vj
			vort.z += f*(relVel.x*relPos.y - relVel.y*relPos.x);		// vort.x = ∑(vxij(yi - yj) - vyij*(xi - xj))*∂Wij/∂r*Vj
		}
	} // end of loop trough neighbors

	vorticity[index] = vort;
}


//! Compute the values of velocity, density, k and epsilon at test points
template<KernelType kerneltype>
__global__ void
calcTestpointsVelocityDevice(	const float4*	oldPos,
								float4*			newVel,
								float*			newTke,
								float*			newEpsilon,
								const hashKey*	particleHash,
								const uint*		cellStart,
								const neibdata*	neibsList,
								const uint		numParticles,
								const float		slength,
								const float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	const particleinfo info = tex1Dfetch(infoTex, index);
	if(!TESTPOINT(info))
		return;

	#if (__COMPUTE__ >= 20)
	const float4 pos = oldPos[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	// this is the velocity (x,y,z) and pressure (w)
	float4 velavg = make_float4(0.0f);
	// this is for k/epsilon
	float tkeavg = 0.0f;
	float epsavg = 0.0f;
	// this is the shepard filter sum(w_b w_{ab})
	float alpha = 0.0f;

	// Compute grid position of current particle
	int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// First loop over all neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if (__COMPUTE__ >= 20)
		const float4 relPos = pos_corr - oldPos[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && (FLUID(neib_info) || VERTEX(neib_info))) {
			const float4 neib_vel = tex1Dfetch(velTex, neib_index);
			const float w = W<kerneltype>(r, slength)*relPos.w/neib_vel.w;	// Wij*mj
			//Velocity
			velavg.x += w*neib_vel.x;
			velavg.y += w*neib_vel.y;
			velavg.z += w*neib_vel.z;
			//Pressure
			velavg.w += w*P(neib_vel.w, fluid_num(neib_info));
			// Turbulent kinetic energy
			if(newTke){
				const float neib_tke = tex1Dfetch(keps_kTex, neib_index);
				tkeavg += w*neib_tke;
			}
			if(newEpsilon){
				const float neib_eps = tex1Dfetch(keps_eTex, neib_index);
				epsavg += w*neib_eps;
			}
			//Shepard filter
			alpha += w;
		}
	}

	// Renormalization by the Shepard filter
	if(alpha>1e-5f) {
		velavg /= alpha;
		if(newTke)
			tkeavg /= alpha;
		if(newEpsilon)
			epsavg /= alpha;
	}
	else {
		velavg = make_float4(0.0f);
		if(newTke)
			tkeavg = 0.0f;
		if(newEpsilon)
			epsavg = 0.0f;
	}

	newVel[index] = velavg;
	if(newTke)
		newTke[index] = tkeavg;
	if(newEpsilon)
		newEpsilon[index] = epsavg;
}


//! Identifies particles which form the free-surface
template<KernelType kerneltype, flag_t simflags, bool savenormals>
__global__ void
calcSurfaceparticleDevice(	const	float4*			posArray,
									float4*			normals,
									particleinfo*	newInfo,
							const	hashKey*		particleHash,
							const	uint*			cellStart,
							const	neibdata*		neibsList,
							const	uint			numParticles,
							const	float			slength,
							const	float			influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	particleinfo info = tex1Dfetch(infoTex, index);

	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif
	float4 normal = make_float4(0.0f);

	if (NOT_FLUID(info) || INACTIVE(pos)) {
		// NOTE: inactive particles will keep their last surface flag status
		newInfo[index] = info;
		return;
	}

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	CLEAR_FLAG(info, FG_SURFACE);
	normal.w = W<kerneltype>(0.0f, slength)*pos.w;

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// First loop over all neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
		const float4 relPos = pos_corr - posArray[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		const float neib_density = tex1Dfetch(velTex, neib_index).w;

		if (r < influenceradius) {
			const float f = F<kerneltype>(r, slength)*relPos.w /neib_density; // 1/r ∂Wij/∂r Vj
			normal.x -= f * relPos.x;
			normal.y -= f * relPos.y;
			normal.z -= f * relPos.z;
			normal.w += W<kerneltype>(r, slength)*relPos.w;	// Wij*mj ;

		}
	}

	float normal_length = length(as_float3(normal));

	// Checking the planes
	if (simflags & ENABLE_PLANES)
		for (uint i = 0; i < d_numplanes; ++i) {
			const float r = PlaneDistance(gridPos, as_float3(pos), d_plane[i]);
			if (r < influenceradius) {
				as_float3(normal) += d_plane[i].normal;
				normal_length = length(as_float3(normal));
			}
		}

	// Second loop over all neighbors

	// Resetting persistent variables across getNeibData
	neib_cellnum = 0;
	neib_cell_base_index = 0;

	// loop over all the neighbors (Second loop)
	int nc = 0;
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
		const float4 relPos = pos_corr - posArray[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		float cosconeangle;

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
		SET_FLAG(info, FG_SURFACE);

	newInfo[index] = info;

	if (savenormals) {
		normal.x /= normal_length;
		normal.y /= normal_length;
		normal.z /= normal_length;
		normals[index] = normal;
		}

}

//! Find closest vertex to a segment which has only corner vertices
/*!
 This function also determines which of the vertices of a segment are not corners.
*/
// TODO this function can probably be removed if the current mass repartitioning is final
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
saFindClosestVertex(
				const	float4*			oldPos,
						particleinfo*	pinfo,
						vertexinfo*		vertices,
				const	uint*			vertIDToIndex,
				const	hashKey*		particleHash,
				const	uint*			cellStart,
				const	neibdata*		neibsList,
				const	uint			numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for boundary particles which are associated to an open boundary
	particleinfo info = pinfo[index];
	const uint obj = object(info);
	if (!(BOUNDARY(info) && IO_BOUNDARY(info)))
		return;

	const vertexinfo verts = vertices[index];
	// load the indices of the vertices only once
	const uint vertXidx = vertIDToIndex[verts.x];
	const uint vertYidx = vertIDToIndex[verts.y];
	const uint vertZidx = vertIDToIndex[verts.z];
	// get the info of those vertices
	particleinfo infoX = pinfo[vertXidx];
	particleinfo infoY = pinfo[vertYidx];
	particleinfo infoZ = pinfo[vertZidx];
	// check if at least one of vertex particles is part of same IO object and not a corner vertex
	if ((object(infoX) == obj && IO_BOUNDARY(infoX) && !CORNER(infoX)) ||
		(object(infoY) == obj && IO_BOUNDARY(infoY) && !CORNER(infoY)) ||
		(object(infoZ) == obj && IO_BOUNDARY(infoZ) && !CORNER(infoZ))   ) {
		// in this case set vertices.w which identifies how many vertex particles are associated to the same
		// IO object
		uint vertCount = 0;
		// if i-th vertex is part of an open boundary and not a corner set i-th bit of vertCount to 1
		if(IO_BOUNDARY(infoX) && !CORNER(infoX))
			vertCount |= VERTEX1;
		if(IO_BOUNDARY(infoY) && !CORNER(infoY))
			vertCount |= VERTEX2;
		if(IO_BOUNDARY(infoZ) && !CORNER(infoZ))
			vertCount |= VERTEX3;
		vertices[index].w = vertCount;
		if (vertCount != 0) { // AAA
			return;
		}
		// nothing left to do here in this routine
		//return;
	}

	// otherwise identify the closest vertex particle that belongs to the same IO object and is not a corner vertex
	float4 pos = oldPos[index];

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	float minDist = 1e10;
	uint minVertId = UINT_MAX;

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		const particleinfo neib_info = pinfo[neib_index];
		const uint neib_obj = object(neib_info);

		if (VERTEX(neib_info) && obj == neib_obj && IO_BOUNDARY(neib_info) && !CORNER(neib_info)) {
			const float4 relPos = pos_corr - oldPos[neib_index];
			const float r = length3(relPos);
			if (minDist > r) {
				minDist = r;
				minVertId = id(neib_info);
			}
		}
	}
	if (minVertId == UINT_MAX) {
		// TODO FIXME MERGE
		SET_FLAG(info, FG_CORNER);
		pinfo[index] = info;
		//// make sure we get a nice crash here
		//printf("-- ERROR -- Could not find a non-corner vertex for segment id: %d with object type: %d\n", id(info), obj);
		//return;
	} else {
		vertices[index].w = minVertId;
		SET_FLAG(info, FG_CORNER);
		pinfo[index] = info;
	}
}
/** @} */

/************************************************************************************************************/

} //namespace cuforces
#endif
