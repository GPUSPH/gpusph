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

// an auxiliary function that fetches the tau tensor
// for particle i from the textures where it's stored
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

#define MAXKASINDEX 10

texture<float, 2, cudaReadModeElementType> demTex;	// DEM

namespace cuforces {

__constant__ idx_t	d_neiblist_end;			///< maximum number of neighbors * number of allocated particles
__constant__ idx_t	d_neiblist_stride;		///< stride between neighbors of the same particle

__constant__ int	d_numfluids;			///< number of different fluids

__constant__ float	d_sqC0[MAX_FLUID_TYPES];	///< square of sound speed for at-rest density for each fluid

__constant__ float	d_ferrari;				///< coefficient for Ferrari correction

// interface epsilon for simplified surface tension in Grenier
__constant__ float	d_epsinterface;

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

__constant__ float	d_visccoeff[MAX_FLUID_TYPES];
__constant__ float	d_epsartvisc;

// Constants used for DEM
// TODO switch to float2s
__constant__ float	d_ewres;		///< east-west resolution (x)
__constant__ float	d_nsres;		///< north-south resolution (y)
__constant__ float	d_demdx;		///< ∆x increment of particle position for normal computation
__constant__ float	d_demdy;		///< ∆y increment of particle position for normal computation
__constant__ float	d_demdxdy;		///< ∆x*∆y
__constant__ float	d_demzmin;		///< minimum distance from DEM for normal computation

__constant__ float	d_partsurf;		/// particle surface (typically particle spacing suared)

// Definition of planes for geometrical boundaries
__constant__ uint	d_numplanes;
__constant__ float3	d_planeNormal[MAX_PLANES];
__constant__ int3	d_planePointGridPos[MAX_PLANES];
__constant__ float3	d_planePointLocalPos[MAX_PLANES];

// Sub-Particle Scale (SPS) Turbulence parameters
__constant__ float	d_smagfactor;
__constant__ float	d_kspsfactor;

// Free surface detection
__constant__ float	d_cosconeanglefluid;
__constant__ float	d_cosconeanglenonfluid;

// Rigid body data
__constant__ int3	d_rbcgGridPos[MAX_BODIES]; //< cell of the center of gravity
__constant__ float3	d_rbcgPos[MAX_BODIES]; //< in-cell coordinate of the center of gravity
__constant__ int	d_rbstartindex[MAX_BODIES];
__constant__ float	d_objectobjectdf;
__constant__ float	d_objectboundarydf;

// Grid data
#include "cellgrid.cuh"
// Core SPH functions
#include "sph_core_utils.cuh"

// Neibdata cell number to offset
__constant__ char3	d_cell_to_offset[27];

// host-computed id offset used for id generation
__constant__ uint	d_newIDsOffset;

/************************************************************************************************************/
/*							  Functions used by the different CUDA kernels							        */
/************************************************************************************************************/

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

/************************************************************************************************************/
/*					   Reflect position or velocity with respect to a plane									*/
/************************************************************************************************************/

#if 0
// TODO FIXME update for homogeneous precision

// opposite of a point wrt to a plane specified as p.x * x + p.y * y + p.z * z + p.w, with
// normal vector norm div
__device__ __forceinline__ float4
reflectPoint(const float4 &pos, const float4 &plane, float pdiv)
{
	// we only care about the 4th component of pos in the dot product to get
	// a*x_0 + b*y_0 + c*z_0 + d*1, so:
	float4 ret = make_float4(pos.x, pos.y, pos.z, 1);
	ret = ret - 2*plane*dot(ret,plane)/(pdiv*pdiv);
	// the fourth component will be whatever, we don't care
	return ret;
}

// opposite of a point wrt to the nplane-th plane; the content of the 4th component is
// undefined
__device__ __forceinline__ float4
reflectPoint(const float4 &pos, uint nplane)
{
	float4 plane = d_planes[nplane];
	float pdiv = d_plane_div[nplane];

	return reflectPoint(pos, plane, pdiv);
}
#endif


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
	// TODO check if it makes sense to support different artificial viscosity coefficients
	// for different fluids
	return vel_dot_pos*slength*d_visccoeff[0]*(sspeed + neib_sspeed)/
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
	// NOTE: this won't work in multi-fluid!
	// TODO FIXME kinematic viscosity should probably be marked as incompatible
	// with multi-fluid (or at least if fluids don't have the same, constant
	// viscosity
	return neib_mass*d_visccoeff[0]*f/(rho + neib_rho);
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
// cflOffset is used in case the forces kernel was partitioned (striping)
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


/********************************* Neighbor data access management ******************************************/

/// Compute hash value from grid position
/*! Compute the hash value corresponding to the given position. If the position
 * 	is not in the range [0, gridSize.x - 1]x[0, gridSize.y - 1]x[0, gridSize.z - 1]
 * 	we have periodic boundary and the grid position is updated according to the
 * 	chosen periodicity.
 *
 *	\param[in] gridPos : grid position
 *
 *	\return hash value
 *
 *	Note : no test is done by this function to ensure that grid position is within the
 *	range and no clamping is done
 */
// TODO: verify periodicity along multiple axis and templatize
__device__ __forceinline__ uint
calcGridHashPeriodic(int3 gridPos)
{
	if (gridPos.x < 0) gridPos.x = d_gridSize.x - 1;
	if (gridPos.x >= d_gridSize.x) gridPos.x = 0;
	if (gridPos.y < 0) gridPos.y = d_gridSize.y - 1;
	if (gridPos.y >= d_gridSize.y) gridPos.y = 0;
	if (gridPos.z < 0) gridPos.z = d_gridSize.z - 1;
	if (gridPos.z >= d_gridSize.z) gridPos.z = 0;
	return calcGridHash(gridPos);
}

/// Return neighbor index and add cell offset vector to current position
/*! For given neighbor data this function compute the neighbor index
 *  and subtract, if necessary, the neighbor cell offset vector to the
 *  current particle position. This last operation is done only
 *  when the neighbor cell change and result is stored in pos_corr.
 *
 *	\param[in] pos : current particle's positions
 *	\param[out] pos_corr : pos - current neighbor cell offset
 *	\param[in] cellStart : cells first particle index
 *	\param[in] neibdata : neighbor data
 *	\param[in,out] neib_cellnum : current neighbor cell number (0...27)
 *	\param[in,out] neib_cell_base_index : index of first particle of the current cell
 *
 * 	\return neighbor index
 *
 * Note: neib_cell_num and neib_cell_base_index must be persistent along
 * getNeibIndex calls.
 */
__device__ __forceinline__ uint
getNeibIndex(float4 const&	pos,
			float3&			pos_corr,
			const uint*		cellStart,
			neibdata		neib_data,
			int3 const&		gridPos,
			char&			neib_cellnum,
			uint&			neib_cell_base_index)
{
	if (neib_data >= CELLNUM_ENCODED) {
		// Update current neib cell number
		neib_cellnum = DECODE_CELL(neib_data);

		// Compute neighbor index relative to belonging cell
		neib_data &= NEIBINDEX_MASK;

		// Substract current cell offset vector to pos
		pos_corr = as_float3(pos) - d_cell_to_offset[neib_cellnum]*d_cellSize;

		// Compute index of the first particle in the current cell
		// use calcGridHashPeriodic because we can only have an out-of-grid cell with neighbors
		// only in the periodic case.
		neib_cell_base_index = cellStart[calcGridHashPeriodic(gridPos + d_cell_to_offset[neib_cellnum])];
	}

	// Compute and return neighbor index
	return neib_cell_base_index + neib_data;
}
/************************************************************************************************************/

/******************** Functions for computing repulsive force directly from DEM *****************************/

__device__ __forceinline__ float
PlaneDistance(	const int3&		gridPos,
				const float3&	pos,
				const float3&	planeNormal,
				const int3&		planePointGridPos,
				const float3&	planePointLocalPos)
{
	// relative position of our particle from the reference point of the plane
	const float3 refRelPos = (gridPos - planePointGridPos)*d_cellSize + (pos - planePointLocalPos);
	return abs(dot(planeNormal, refRelPos));
}

// TODO: check for the maximum timestep

// Normal and viscous force wrt to solid boundary
__device__ __forceinline__ float
PlaneForce(	const int3&		gridPos,
			const float3&	pos,
			const float		mass,
			const float3&	planeNormal,
			const int3&		planePointGridPos,
			const float3&	planePointLocalPos,
			const float3&	vel,
			const float		dynvisc,
			float4&			force)
{
	// relative position of our particle from the reference point of the plane
	const float r = PlaneDistance(gridPos, pos, planeNormal, planePointGridPos, planePointLocalPos);
	if (r < d_r0) {
		const float DvDt = LJForce(r);
		// Unitary normal vector of the surface
		const float3 relPos = planeNormal*r;

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
		float coeff = PlaneForce(gridPos, pos, mass,
			d_planeNormal[i], d_planePointGridPos[i], d_planePointLocalPos[i],
			vel, dynvisc, force);
		if (coeff > coeff_max)
			coeff_max = coeff;
	}

	return coeff_max;
}

/**! Convert a grid + local position into a DEM cell position
 * This is done assuming that the worldOrigin is at DEM coordinates (0, 0).
 */
__device__ __forceinline__ float2
DemPos(const int2& gridPos, const float2 &pos)
{
	// note that we separate the grid conversion part from the pos conversion part,
	// for improved accuracy. The final 0.5f is because texture values are assumed to be
	// at the center of the DEM cell.
	return make_float2(
		(gridPos.x + 0.5f)*(d_cellSize.x/d_ewres) + pos.x/d_ewres + 0.5f,
		(gridPos.y + 0.5f)*(d_cellSize.y/d_nsres) + pos.y/d_nsres + 0.5f);
}

/**! Interpolate DEM texref for a point at DEM cell pos demPos,
  plus an optional multiple of (∆x, ∆y).
  NOTE: the returned z coordinate is GLOBAL, not LOCAL!
  TODO for improved homogeneous accuracy, maybe have a texture for grid cells and a
  texture for local z coordinates?
 */

__device__ __forceinline__ float
DemInterpol(const texture<float, 2, cudaReadModeElementType> texref,
	const float2& demPos, int dx=0, int dy=0)
{
	return tex2D(texref, demPos.x + dx*d_demdx/d_ewres, demPos.y + dy*d_demdy/d_nsres);
}


__device__ __forceinline__ float
DemLJForce(	const texture<float, 2, cudaReadModeElementType> texref,
			const int3&	gridPos,
			const float3&	pos,
			const float		mass,
			const float3&	vel,
			const float		dynvisc,
			float4&			force)
{
	const float2 demPos = DemPos(as_int2(gridPos), as_float2(pos));

	const float globalZ = d_worldOrigin.z + (gridPos.z + 0.5f)*d_cellSize.z + pos.z;
	const float globalZ0 = DemInterpol(texref, demPos);

	if (globalZ - globalZ0 < d_demzmin) {
		// TODO this method to generate the interpolating plane is suboptimal, as it
		// breaks any possible symmetry in the original DEM. A better (but more expensive)
		// approach would be to sample four points, one on each side of our point (in both
		// directions)
		const float globalZ1 = DemInterpol(texref, demPos, 1, 0);
		const float globalZ2 = DemInterpol(texref, demPos, 0, 1);

		// TODO find a more accurate way to compute the normal
		const float a = d_demdy*(globalZ0 - globalZ1);
		const float b = d_demdx*(globalZ0 - globalZ2);
		const float c = d_demdxdy;
		const float l = sqrt(a*a+b*b+c*c);

		const float3 planeNormal = make_float3(a, b, c)/l;

		// our plane point is the one at globalZ0: this has the same (x, y) grid and local
		// position as our particle, and the z grid and local position to be computed
		// from globalZ0
		const int3 planePointGridPos = make_int3(gridPos.x, gridPos.y,
			(int)floor((globalZ0 - d_worldOrigin.z)/d_cellSize.z));
		const float3 planePointLocalPos = make_float3(pos.x, pos.y,
			globalZ0 - d_worldOrigin.z - (planePointGridPos.z + 0.5f)*d_cellSize.z);

		return PlaneForce(gridPos, pos, mass,
			planeNormal, planePointGridPos, planePointLocalPos,
			vel, dynvisc, force);
	}
	return 0;
}

/************************************************************************************************************/

/************************************************************************************************************/
/*		   Kernels for computing SPS tensor and SPS viscosity												*/
/************************************************************************************************************/

/// A functor that writes out turbvisc for SPS visc
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

/// A functor that writes out tau for SPS visc
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

// Compute the Sub-Particle-Stress (SPS) Tensor matrix for all Particles
// WITHOUT Kernel correction
// Procedure:
// (1) compute velocity gradients
// (2) compute turbulent eddy viscosity (non-dynamic)
// (3) compute turbulent shear stresses
// (4) return SPS tensor matrix (tau) divided by rho^2
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

// When using the Grenier formulation, density is reinitialized at each timestep from
// a Shepard-corrected mass distribution limited to same-fluid particles M and volumes ω computed
// from a continuity equation, with ϱ = M/ω.
// During the same run, we also compute σ, the approximation of the inverse volume obtained by summing
// the kernel computed over _all_ neighbors (not just the same-fluid ones) which is used in the continuity
// equation as well as the Navier-Stokes equation
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
	// sigma to 1.
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

	if (boundarytype == DYN_BOUNDARY && NOT_FLUID(info) && !has_fluid_neibs)
		sigma = 1;

	// M = mass_corr/corr, ϱ = M/ω
	// this could be optimized to pos.w/vol assuming all same-fluid particles
	// have the same mass
	vel.w = mass_corr/(corr*vol);
	velArray[index] = vel;
	sigmaArray[index] = sigma;
}

/************************************************************************************************************/

/************************************************************************************************************/
/*					   Gamma calculations						    */
/************************************************************************************************************/

// Load old gamma value.
// If computeGamma was false, it means the caller wants us to check gam.w against epsilon
// to see if the new gamma is to be computed
__device__ __forceinline__
float4
fetchOldGamma(const uint index, const float epsilon, bool &computeGamma)
{
	float4 gam = tex1Dfetch(gamTex, index);
	if (!computeGamma)
		computeGamma = (gam.w < epsilon);
	return gam;
}

// This function returns the function value of the wendland kernel and of the integrated wendland kernel
__device__ __forceinline__ float2
wendlandOnSegment(const float q)
{
	float kernel = 0.0f;
	float intKernel = 0.0f;

	if (q < 2.0f) {
		float tmp = (1.0f-q/2.0f);
		float tmp4 = tmp*tmp;
		tmp4 *= tmp4;

// Wendland coefficient: 21/(16 π)
#define WENDLAND_K_COEFF 0.417781725616225256393319878852850200340456570068698177962626f
// Integrated Wendland coefficient: 1/(32 π)
#define WENDLAND_I_COEFF 0.009947183943243458485555235210782147627153727858778528046729f

		// Wendland kernel
		kernel = WENDLAND_K_COEFF*tmp4*(1.0f+2.0f*q);

		// integrated Wendland kernel
		const float uq = 1.0f/q;
		intKernel = WENDLAND_I_COEFF*tmp4*tmp*((((8.0f*uq + 20.0f)*uq + 30.0f)*uq) + 21.0f);
	}

	return make_float2(kernel, intKernel);
}

/*
 * Gaussian quadrature
 */

// Function that computes the surface integral of a function on a triangle using a 1st order Gaussian quadrature rule
__device__ __forceinline__ float2
gaussQuadratureO1(	const	float3	vPos0,
					const	float3	vPos1,
					const	float3	vPos2,
					const	float3	relPos)
{
	float2 val = make_float2(0.0f);
	// perform the summation
	float3 pa =	vPos0/3.0f +
				vPos1/3.0f +
				vPos2/3.0f  ;
	pa -= relPos;
	val += 1.0f*wendlandOnSegment(length(pa));
	// compute the triangle volume
	const float vol = length(cross(vPos1-vPos0,vPos2-vPos0))/2.0f;
	// return the summed values times the volume
	return val*vol;
}

// 5th order: weights
__constant__ float GQ_O5_weights[3] = {0.225f, 0.132394152788506f, 0.125939180544827f};

// 5th order: points, in barycentric coordinates
__constant__ float GQ_O5_points[3][3] = {
	{0.333333333333333f, 0.333333333333333f, 0.333333333333333f},
	{0.059715871789770f, 0.470142064105115f, 0.470142064105115f},
	{0.797426985353087f, 0.101286507323456f, 0.101286507323456f}
};

// 5th order: multiplicity of each quadrature point
__constant__ int GQ_O5_mult[3] = {1, 3, 3};

// Function that computes the surface integral of a function on a triangle using a 5th order Gaussian quadrature rule
__device__ __forceinline__ float2
gaussQuadratureO5(	const	float3	vPos0,
					const	float3	vPos1,
					const	float3	vPos2,
					const	float3	relPos)
{
	float2 val = make_float2(0.0f);
	// perform the summation
#pragma unroll
	for (int i=0; i<3; i++) {
#pragma unroll
		for (int j=0; j<3; j++) {
			float3 pa =	vPos0*GQ_O5_points[i][j]       +
						vPos1*GQ_O5_points[i][(j+1)%3] +
						vPos2*GQ_O5_points[i][(j+2)%3]  ;
			pa -= relPos;
			val += GQ_O5_weights[i]*wendlandOnSegment(length(pa));
			if (j >= GQ_O5_mult[i])
				break;
		}
	}
	// compute the triangle volume
	const float vol = length(cross(vPos1-vPos0,vPos2-vPos0))/2.0f;
	// return the summed values times the volume
	return val*vol;
}

// 14th order: weights
__constant__ float GQ_O14_weights[10] = {
	0.021883581369429f,
	0.032788353544125f,
	0.051774104507292f,
	0.042162588736993f,
	0.014433699669777f,
	0.004923403602400f,
	0.024665753212564f,
	0.038571510787061f,
	0.014436308113534f,
	0.005010228838501f
};


// 14th order: points, in barycentric coordinates
__constant__ float GQ_O14_points[10][3] = {
	{0.022072179275643f,0.488963910362179f,0.488963910362179f},
	{0.164710561319092f,0.417644719340454f,0.417644719340454f},
	{0.453044943382323f,0.273477528308839f,0.273477528308839f},
	{0.645588935174913f,0.177205532412543f,0.177205532412543f},
	{0.876400233818255f,0.061799883090873f,0.061799883090873f},
	{0.961218077502598f,0.019390961248701f,0.019390961248701f},
	{0.057124757403648f,0.172266687821356f,0.770608554774996f},
	{0.092916249356972f,0.336861459796345f,0.570222290846683f},
	{0.014646950055654f,0.298372882136258f,0.686980167808088f},
	{0.001268330932872f,0.118974497696957f,0.879757171370171f}
};

// 14th order: multiplicity of each quadrature point
__constant__ int GQ_O14_mult[10] = {1,3,3,3,3,3,6,6,6,6};


// Function that computes the surface integral of a function on a triangle using a 14th order Gaussian quadrature rule
__device__ __forceinline__ float2
gaussQuadratureO14(	const	float3	vPos0,
					const	float3	vPos1,
					const	float3	vPos2,
					const	float3	relPos)
{
	float2 val = make_float2(0.0f);
	// perform the summation
#pragma unroll
	for (int i=0; i<10; i++) {
#pragma unroll
		for (int j=0; j<6; j++) {
			float3 pa =	vPos0*GQ_O14_points[i][j%3]       +
						vPos1*GQ_O14_points[i][(j+1+j/3)%3] +
						vPos2*GQ_O14_points[i][(j+2-j/3)%3]  ;
			pa -= relPos;
			val += GQ_O14_weights[i]*wendlandOnSegment(length(pa));
			if (j >= GQ_O14_mult[i])
				break;
		}
	}
	// compute the triangle volume
	const float vol = length(cross(vPos1-vPos0,vPos2-vPos0))/2.0f;
	// return the summed values times the volume
	return val*vol;
}

// returns grad gamma_{as} as x coordinate, gamma_{as} as y coordinate
template<KernelType kerneltype>
__device__ __forceinline__ float2
Gamma(	const	float		&slength,
				float4		relPos,
		const	float2		&vPos0,
		const	float2		&vPos1,
		const	float2		&vPos2,
		const	float4		&boundElement,
				float4		oldGGam,
		const	float		&epsilon,
		const	float		&deltap,
		const	bool		&computeGamma,
		const	uint		&nIndex,
				float		&minlRas)
{
	// normalize the distance r_{as} with h
	relPos.x /= slength;
	relPos.y /= slength;
	relPos.z /= slength;
	// Sigma is the point a projected onto the plane spanned by the edge
	// q_aSigma is the non-dimensionalized distance between this plane and the particle
	float4 q_aSigma = boundElement*dot3(boundElement,relPos);
	q_aSigma.w = fmin(length3(q_aSigma),2.0f);
	// local coordinate system for relative positions to vertices
	uint j = 0;
	// Get index j for which n_s is minimal
	if (fabs(boundElement.x) > fabs(boundElement.y))
		j = 1;
	if ((1-j)*fabs(boundElement.x) + j*fabs(boundElement.y) > fabs(boundElement.z))
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

	// relative positions of vertices with respect to the segment, normalized by h
	float4 v0 = -(vPos0.x*coord1 + vPos0.y*coord2)/slength; // e.g. v0 = r_{v0} - r_s
	float4 v1 = -(vPos1.x*coord1 + vPos1.y*coord2)/slength;
	float4 v2 = -(vPos2.x*coord1 + vPos2.y*coord2)/slength;
	// calculate if the projection of a (with respect to n) is inside the segment
	const float4 ba = v1 - v0; // vector from v0 to v1
	const float4 ca = v2 - v0; // vector from v0 to v2
	const float4 pa = relPos - v0; // vector from v0 to the particle
	const float uu = sqlength3(ba);
	const float uv = dot3(ba,ca);
	const float vv = sqlength3(ca);
	const float wu = dot3(ba,pa);
	const float wv = dot3(ca,pa);
	const float invdet = 1.0f/(uv*uv-uu*vv);
	const float u = (uv*wv-vv*wu)*invdet;
	const float v = (uv*wu-uu*wv)*invdet;
	//const float w = 1.0f - u - v;
	// set minlRas only if the projection is close enough to the triangle and if the normal
	// distance is close
	if (q_aSigma.w < 0.5f && (u > -1.0f && v > -1.0f && 1.0f - u - v > -1.0f && u < 2.0f && v < 2.0f && 1.0f - u - v < 2.0f)) {
		minlRas = min(minlRas, q_aSigma.w);
	}
	float gradGamma_as = 0.0f;
	float gamma_as = 0.0f;
	float gamma_vs = 0.0f;
	// check if the particle is on a vertex
	if ((	(fabs(u-1.0f) < epsilon && fabs(v) < epsilon) ||
			(fabs(v-1.0f) < epsilon && fabs(u) < epsilon) ||
			(     fabs(u) < epsilon && fabs(v) < epsilon)   ) && q_aSigma.w < epsilon) {
		// set touching vertex to v0
		if (fabs(u-1.0f) < epsilon && fabs(v) < epsilon) {
			const float4 tmp = v1;
			v1 = v2;
			v2 = v0;
			v0 = tmp;
		}
		else if (fabs(v-1.0f) < epsilon && fabs(u) < epsilon) {
			const float4 tmp = v2;
			v2 = v1;
			v1 = v0;
			v0 = tmp;
		}
		// additional value of grad gamma
		const float openingAngle = acos(dot3((v1-v0),(v2-v0))/sqrt(sqlength3(v1-v0)*sqlength3(v2-v0)));
		gradGamma_as = openingAngle*0.1193662073189215018266628225293857715258447343053423f; // 3/(8π)

		// compute the sum of all solid angles of the tetrahedron spanned by v1-v0, v2-v0 and -gradgamma
		// the minus is due to the fact that initially gamma is equal to one, so we want to subtract the outside
		oldGGam /= -fmax(length3(oldGGam),slength*1e-3f);
		float l1 = length3(v1-v0);
		float l2 = length3(v2-v0);
		float abc = dot3((v1-v0),oldGGam)/l1 + dot3((v2-v0),oldGGam)/l2 + dot3((v1-v0),(v2-v0))/l1/l2;
		float d = dot3(oldGGam,cross3((v1-v0),(v2-v0)))/l1/l2;

		// formula by A. Van Oosterom and J. Strackee “The Solid Angle of a Plane Triangle”, IEEE Trans. Biomed. Eng. BME-30(2), 125-126 (1983)
		float SolidAngle = fabs(2.0f*atan2(d,(1.0f+abc)));
		gamma_vs = SolidAngle*0.079577471545947667884441881686257181017229822870228224373833f; // 1/(4π)
	}
	// check if particle is on an edge
	else if ((	(fabs(u) < epsilon && v > -epsilon && v < 1.0f+epsilon) ||
				(fabs(v) < epsilon && u > -epsilon && u < 1.0f+epsilon) ||
				(fabs(u+v-1.0f) < epsilon && u > -epsilon && u < 1.0f+epsilon && v > -epsilon && v < 1.0f+epsilon)
			 ) && q_aSigma.w < epsilon) {
		oldGGam /= -length3(oldGGam);
		// grad gamma for a half-plane
		gradGamma_as = 0.375f; // 3.0f/4.0f/2.0f;

		// compute the angle between a segment and -gradgamma
		const float theta0 = acos(dot3(boundElement,oldGGam)); // angle of the norms between 0 and pi
		const float4 refDir = cross3(boundElement, relPos); // this defines a reference direction
		const float4 normDir = cross3(boundElement, oldGGam); // this is the sin between the two norms
		const float theta = M_PIf + copysign(theta0, dot3(refDir, normDir)); // determine the actual angle based on the orientation of the sin

		// this is actually two times gamma_as:
		gamma_vs = theta*0.1591549430918953357688837633725143620344596457404564f; // 1/(2π)
	}
	// general formula (also used if particle is on vertex / edge to compute remaining edges)
	if (q_aSigma.w < 2.0f && q_aSigma.w > epsilon) {
		// Gaussian quadrature of 14th order
		//float2 intVal = gaussQuadratureO1(-as_float3(v0), -as_float3(v1), -as_float3(v2), as_float3(relPos));
		// Gaussian quadrature of 14th order
		//float2 intVal = gaussQuadratureO14(-as_float3(v0), -as_float3(v1), -as_float3(v2), as_float3(relPos));
		// Gaussian quadrature of 5th order
		const float2 intVal = gaussQuadratureO5(-as_float3(v0), -as_float3(v1), -as_float3(v2), as_float3(relPos));
		gradGamma_as += intVal.x;
		gamma_as += intVal.y*dot3(boundElement,q_aSigma);
	}
	gamma_as = gamma_vs + gamma_as;
	return make_float2(gradGamma_as/slength, gamma_as);
}

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

// flags for the vertexinfo .w coordinate which specifies how many vertex particles of one segment
// is associated to an open boundary
#define VERTEX1 ((flag_t)1)
#define VERTEX2 (VERTEX1 << 1)
#define VERTEX3 (VERTEX2 << 1)
#define ALLVERTICES ((flag_t)(VERTEX1 | VERTEX2 | VERTEX3))

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
								const	uint*		vertIDToIndex,
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
								const	bool		inoutBoundaries)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	const particleinfo info = tex1Dfetch(infoTex, index);

	// For boundary segments this kernel computes the boundary conditions
	if (BOUNDARY(info)) {

		// if we are on an in/outflow boundary get the imposed velocity / pressure and average
		float4 eulerVel = make_float4(0.0f);
		float tke = 0.0f;
		float eps = 0.0f;
		const vertexinfo verts = vertices[index];

		// load the indices of the vertices only once
		const uint vertXidx = vertIDToIndex[verts.x];
		const uint vertYidx = vertIDToIndex[verts.y];
		const uint vertZidx = vertIDToIndex[verts.z];

		// get the imposed quantities from the vertices
		if (IO_BOUNDARY(info)) {
			// for imposed velocity the velocity, tke and eps are required and only rho will be calculated
			if (VEL_IO(info)) {
				eulerVel.x =   (oldEulerVel[vertXidx].x +
								oldEulerVel[vertYidx].x +
								oldEulerVel[vertZidx].x )/3.0f;
				eulerVel.y =   (oldEulerVel[vertXidx].y +
								oldEulerVel[vertYidx].y +
								oldEulerVel[vertZidx].y )/3.0f;
				eulerVel.z =   (oldEulerVel[vertXidx].z +
								oldEulerVel[vertYidx].z +
								oldEulerVel[vertZidx].z )/3.0f;
				if (oldTKE)
					tke =  (oldTKE[vertXidx] +
							oldTKE[vertYidx] +
							oldTKE[vertZidx] )/3.0f;
				if (oldEps)
					eps =  (oldEps[vertXidx] +
							oldEps[vertYidx] +
							oldEps[vertZidx] )/3.0f;
			}
			// for imposed density only eulerVel.w will be required, the rest will be computed
			else {
				eulerVel.w =   (oldEulerVel[vertXidx].w +
								oldEulerVel[vertYidx].w +
								oldEulerVel[vertZidx].w )/3.0f;
			}
		}

		// velocity for moving objects transferred from vertices
		float3 vel = make_float3(0.0f);
		if (MOVING(info)) {
			vel += as_float3(oldVel[vertXidx]);
			vel += as_float3(oldVel[vertYidx]);
			vel += as_float3(oldVel[vertZidx]);
			vel /= 3.0f;
		}
		as_float3(oldVel[index]) = vel;

		const float4 pos = oldPos[index];

		// note that all sums below run only over fluid particles (including the Shepard filter)
		float sumrho = 0.0f; // summation for computing the density
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

			if (neib_data == 0xffff) break;

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

			if (r < influenceradius && (FLUID(neib_info) || (VERTEX(neib_info) && !IO_BOUNDARY(neib_info) && IO_BOUNDARY(info)))) {
				const float neib_rho = oldVel[neib_index].w;

				const float neib_pres = P(neib_rho, fluid_num(neib_info));
				const float neib_vel = length(make_float3(oldVel[neib_index]));
				const float neib_k = oldTKE ? oldTKE[neib_index] : NAN;
				const float neib_eps = oldEps ? oldEps[neib_index] : NAN;

				// kernel value times volume
				const float w = W<kerneltype>(r, slength)*relPos.w/neib_rho;
				// normal distance based on grad Gamma which approximates the normal of the domain
				const float normDist = fmax(fabs(dot3(normal,relPos)), deltap);
				sumrho += (1.0f + dot(d_gravity,as_float3(relPos))/sqC0)*w*neib_rho;
				// for all boundaries we have dk/dn = 0
				sumtke += w*neib_k;
				if (IO_BOUNDARY(info)) {
					// for open boundaries compute dv/dn = 0
					sumvel += w*as_float3(oldVel[neib_index] + oldEulerVel[neib_index]);
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

		if (alpha > 1e-5f) {
			// for the k-epsilon model we also need to determine the velocity of the wall.
			// This is an average of the velocities of the vertices
			if (!(IO_BOUNDARY(info) && VEL_IO(info))) {
				if (oldTKE){
					// for solid boundaries we want to get the eulerian velocity (based on viscous forces)
					// from all associated vertices
					if (!IO_BOUNDARY(info)) {
						eulerVel = (	oldEulerVel[vertXidx] +
										oldEulerVel[vertYidx] +
										oldEulerVel[vertZidx] )/3.0f;
						// ensure that velocity is normal to segment normal
						eulerVel -= dot3(eulerVel,normal)*normal;
						oldEulerVel[index] = eulerVel;
					}
					// for solid boundaries and pressure imposed boundaries we take dk/dn = 0
					oldTKE[index] = sumtke/alpha;
				}
				else if (oldEulerVel)
					oldEulerVel[index] = make_float4(0.0f);
				if (oldEps)
					// for solid boundaries we have de/dn = 4 0.09^0.075 k^1.5/(0.41 r)
					// for open boundaries we have dk/dn = 0
					oldEps[index] = sumeps/alpha; // eps should never be 0
			}
			// velocity imposition
			else {
				if (oldTKE)
					oldTKE[index] = tke;
				if (oldEps)
					oldEps[index] = eps;
			}
			if (IO_BOUNDARY(info))
				sumvel /= alpha;
			oldVel[index].w = fmax(sumrho/alpha,d_rho0[fluid_num(info)]);
		}
		else {
			oldVel[index].w = d_rho0[fluid_num(info)];
			if (oldEulerVel)
				oldEulerVel[index] = make_float4(0.0f);
			if (oldTKE)
				oldTKE[index] = 1e-5f;
			if (oldEps)
				oldEps[index] = 1e-5f;
			if (IO_BOUNDARY(info)) {
				if (VEL_IO(info)) {
					sumvel = as_float3(eulerVel);
				}
				else {
					sumvel = make_float3(0.0f);
				}
			}
		}

		// Compute the Riemann Invariants for I/O conditions
		if (IO_BOUNDARY(info) && !CORNER(info)) {
			const float unInt = dot(sumvel, as_float3(normal));
			const float unExt = dot3(eulerVel, normal);
			const float rhoInt = oldVel[index].w;
			const float rhoExt = eulerVel.w;
			const int a = fluid_num(info);

			// impose velocity (and k,eps) => compute density
			if (VEL_IO(info)) {
				// Rankine-Hugoniot is not properly working
				/*if (unExt > unInt) { // Shock wave (Rankine-Hugoniot)
					eulerVel.w = RHO(P(rhoInt, a) + rhoInt*unInt*(unInt - unExt), a);
					// with the new rho check if this is actually a shock wave
					const float c = d_sscoeff[a]*powf(eulerVel.w/d_rho0[a], (d_gammacoeff[a]-1.0f)/2.0f);
					const float lambda = unExt + c;
					const float cInt = d_sscoeff[a]*powf(rhoInt/d_rho0[a], (d_gammacoeff[a]-1.0f)/2.0f);
					const float lambdaInt = unInt + cInt;
					if (lambda < lambdaInt) // It is in reality an expansion wave
						eulerVel.w = RHOR(R(rhoInt, a) + (unExt - unInt), a);
				} else { // expansion wave
					eulerVel.w = RHOR(R(rhoInt, a) + (unExt - unInt), a);
				}*/
				eulerVel.w = RHOR(R(rhoInt, a) + (unExt - unInt), a);
			}
			// impose pressure => compute velocity (normal & tangential; k and eps are already interpolated)
			else {
				float flux = 0.0f;
				// Rankine-Hugoniot is not properly working
				/*if (rhoExt > rhoInt && false) { // Shock wave
					if (fabs(unInt) > d_sscoeff[a]*1e-5f)
						flux = (P(rhoInt, a) - P(rhoExt, a))/(rhoInt*unInt) + unInt;
					// Check whether it is really a shock wave
					const float c = d_sscoeff[a]*powf(rhoExt/d_rho0[a], (d_gammacoeff[a]-1.0f)/2.0f);
					const float lambda = flux + c;
					const float cInt = d_sscoeff[a]*powf(rhoInt/d_rho0[a], (d_gammacoeff[a]-1.0f)/2.0f);
					const float lambdaInt = unInt + cInt;
					if (lambda < lambdaInt) // It is in reality an expansion wave
						flux = unInt + (R(rhoExt, a) - R(rhoInt, a));
				} else { // Expansion wave
					flux = unInt + (R(rhoExt, a) - R(rhoInt, a));
				}*/
				flux = unInt + (R(rhoExt, a) - R(rhoInt, a));
				// if p <= 0 is imposed then only outgoing flux is allowed
				if (rhoExt < d_rho0[fluid_num(info)]*(1.0f+1e-5f))
					flux = fmin(0.0f, flux);
				// impose eulerVel according to dv/dn = 0
				as_float3(eulerVel) = sumvel;
				// remove normal component of velocity
				eulerVel = eulerVel - dot3(eulerVel, normal)*normal;
				// if a pressure boundary has a positive flux set the tangential velocity to 0
				// otherwise the tangential velocity is taken from the interior of the fluid
				if (flux > 0)
					eulerVel = make_float4(0.0f);
				// add calculated normal velocity
				eulerVel += normal*flux;
				// set density to the imposed one
				eulerVel.w = rhoExt;
			}
			oldEulerVel[index] = eulerVel;
			// the density of the particle is equal to the "eulerian density"
			oldVel[index].w = eulerVel.w;

		}

	}
	// for fluid particles this kernel checks whether they have crossed the boundary at open boundaries
	else if (inoutBoundaries && FLUID(info)) {

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

		// Loop over all the neighbors
		for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
			neibdata neib_data = neibsList[i + index];

			if (neib_data == 0xffff) break;

			const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
						neib_cellnum, neib_cell_base_index);
			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			// for open boundary segments check whether this fluid particle has crossed the boundary
			if (BOUNDARY(neib_info) && IO_BOUNDARY(neib_info)) {

				// Compute relative position vector and distance
				// Now relPos is a float4 and neib mass is stored in relPos.w
				const float4 relPos = pos_corr - oldPos[neib_index];

				const float4 normal = tex1Dfetch(boundTex, neib_index);

				const float3 relVel = as_float3(vel - oldVel[neib_index]);

				// quick check if we are behind a segment and if the segment is reasonably close by
				// (max distance vertex to segment is deltap/2)
				if (dot3(normal, relPos) <= 0.0f &&
					sqlength3(relPos) < deltap*deltap &&
					dot(relVel, as_float3(normal)) < 0.0f) {
					// now check whether the normal projection is inside the triangle
					// first get the position of the vertices local coordinate system for relative positions to vertices
					uint j = 0;
					// Get index j for which n_s is minimal
					if (fabs(normal.x) > fabs(normal.y))
						j = 1;
					if ((1-j)*fabs(normal.x) + j*fabs(normal.y) > fabs(normal.z))
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

					const float2 vPos0 = vertPos0[neib_index];
					const float2 vPos1 = vertPos1[neib_index];
					const float2 vPos2 = vertPos2[neib_index];

					// relative positions of vertices with respect to the segment, normalized by h
					float4 v0 = -(vPos0.x*coord1 + vPos0.y*coord2); // e.g. v0 = r_{v0} - r_s
					float4 v1 = -(vPos1.x*coord1 + vPos1.y*coord2);
					float4 v2 = -(vPos2.x*coord1 + vPos2.y*coord2);

					const float4 relPosV0 = relPos - v0;
					const float4 relPosV10 = v1 - v0;
					const float4 relPosV20 = v2 - v0;

					const float dot00 = sqlength3(relPosV10);
					const float dot01 = dot3(relPosV10, relPosV20);
					const float dot02 = dot3(relPosV10, relPosV0);
					const float dot11 = sqlength3(relPosV20);
					const float dot12 = dot3(relPosV20, relPosV0);

					const float invdet = 1.0/(dot00*dot11-dot01*dot01);
					const float u = (dot11*dot02-dot01*dot12)*invdet;
					const float v = (dot00*dot12-dot01*dot02)*invdet;

					// error measure
					const float eps = 1e-3f*deltap;
					// u, v are the barycentric coordinates
					if ( u < -eps || v < -eps || u+v > 1.0f+eps)
						continue;

					// the fluid particle found a segment so let's save it
					// note normally vertices is empty for fluid particles so this will indicate
					// from now on that it has to be destroyed
					vertexinfo verts = vertices[neib_index];

					// furthermore we need to save the weights beta_{a,v} to avoid using
					// neighbours of neighbours. As the particle will be deleted anyways we
					// just use the velocity array which we don't need anymore. The beta_{a,v}
					// in the 3-D case are the barycentric coordinates which we have already
					// computed.
					float4 vertexWeights = make_float4(0.0f);
					if (CORNER(neib_info)) {
						vertexWeights.x = 1.0f;
						verts.x = verts.w;
					}
					else {
						// Check if all vertices are associated to an open boundary
						// in this case we can use the barycentric coordinates
						if (verts.w == ALLVERTICES) {
							vertexWeights.x = 1.0f - (u+v);
							vertexWeights.y = u;
							vertexWeights.z = v;
						}
						// If there are two vertices then use the remaining two and split accordingly
						else if (verts.w & (VERTEX1 | VERTEX2)) {
							vertexWeights.x = 1.0f - (u+v);
							vertexWeights.y = u;
							vertexWeights.z = 0.0f;
						}
						else if (verts.w & (VERTEX2 | VERTEX3)) {
							vertexWeights.x = 1.0f - (u+v);
							vertexWeights.y = 0.0f;
							vertexWeights.z = v;
						}
						else if (verts.w & (VERTEX3 | VERTEX1)) {
							vertexWeights.x = 0.0f;
							vertexWeights.y = u;
							vertexWeights.z = v;
						}
						// if only one vertex is associated to the open boundary use only that one
						else if (verts.w & VERTEX1) {
							vertexWeights.x = 1.0f;
							vertexWeights.y = 0.0f;
							vertexWeights.z = 0.0f;
						}
						else if (verts.w & VERTEX2) {
							vertexWeights.x = 0.0f;
							vertexWeights.y = 1.0f;
							vertexWeights.z = 0.0f;
						}
						else if (verts.w & VERTEX3) {
							vertexWeights.x = 0.0f;
							vertexWeights.y = 0.0f;
							vertexWeights.z = 1.0f;
						}
					}
					// normalize to make sure that all the weight is split up
					vertexWeights = normalize3(vertexWeights);
					// transfer mass to .w index as it is overwritten with the disable below
					vertexWeights.w = pos.w;
					oldGGam[index] = vertexWeights;
					vertices[index] = verts;

					// one segment is enough so jump out of the neighbour loop
					break;
				}

			}
		}
	}
}

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
				const	uint*			vertIDToIndex,
						particleinfo*	pinfo,
						hashKey*		particleHash,
				const	uint*			cellStart,
				const	neibdata*		neibsList,
				const	uint			numParticles,
						uint*			newNumParticles,
				const	float			dt,
				const	int				step,
				const	float			deltap,
				const	float			slength,
				const	float			influenceradius,
				const	bool			initStep,
				const	uint			deviceId,
				const	uint			numDevices)
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
	float sumrho = 0.0f; // summation for computing the density
	float sumtke = 0.0f; // summation for computing tke (k-epsilon model)
	float sumeps = 0.0f; // summation for computing epsilon (k-epsilon model)
	float sumMdot = 0.0f; // summation for computing the mass variance based on in/outflow
	float massFluid = 0.0f; // mass obtained from a outgoing - mass of a new fluid
	float4 sumEulerVel = make_float4(0.0f); // summation for computing the averages of the Euler velocities
	float numseg  = 0.0f;  // number of adjacent segments
	bool foundFluid = false; // check if a vertex particle has a fluid particle in its support
	// Average norm used in the intial step to compute grad gamma for vertex particles
	// During the simulation this is used for open boundaries to determine whether particles are created
	// For all other boundaries in the keps case this is the average normal of all non-open boundaries used to ensure that the
	// Eulerian velocity is only normal to the fixed wall
	float3 avgNorm = make_float3(0.0f);

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

		const particleinfo neib_info = pinfo[neib_index];

		if (BOUNDARY(neib_info) || FLUID(neib_info)) {

			// prepare indices of neib vertices
			const vertexinfo neibVerts = vertices[neib_index];

			// load the indices of the vertices
			const uint neibVertXidx = vertIDToIndex[neibVerts.x];
			const uint neibVertYidx = vertIDToIndex[neibVerts.y];
			const uint neibVertZidx = vertIDToIndex[neibVerts.z];

			if (BOUNDARY(neib_info)) {
				const float4 boundElement = tex1Dfetch(boundTex, neib_index);

				// check if vertex is associated with this segment
				if (neibVertXidx == index || neibVertYidx == index || neibVertZidx == index) {
					// boundary conditions on rho, k, eps
					const float neibRho = oldVel[neib_index].w;
					sumrho += neibRho;
					if (!CORNER(info) && IO_BOUNDARY(neib_info)){
						// number of vertices associated to a segment that are of the same object type
						float numOutVerts = 2.0f;
						if (neibVerts.w == ALLVERTICES) // all vertices are of the same object type
							numOutVerts = 3.0f;
						else if (neibVerts.w & ~VERTEX1 == 0 || neibVerts.w & ~VERTEX2 == 0 || neibVerts.w & ~VERTEX3 == 0) // only one vertex
							numOutVerts = 1.0f;
						// TODO we can have a switch here to decide on whether we want to impose a velocity
						// or a flux. If as now we multiply the whole thing with the density of the segment
						// then the flux will vary.
						numOutVerts = 3.0f;
						sumMdot += neibRho/numOutVerts*boundElement.w*
									dot3(oldEulerVel[neib_index],boundElement); // the euler vel should be subtracted by the lagrangian vel which is assumed to be 0 now.
						sumEulerVel += oldEulerVel[neib_index];
					}
					sumtke += oldTKE ? oldTKE[neib_index] : NAN;
					sumeps += oldEps ? oldEps[neib_index] : NAN;
					numseg += 1.0f;
					// in the initial step we need to compute an approximate grad gamma direction
					// for the computation of gamma, in general we need a sort of normal as well
					// for open boundaries to decide whether or not particles are created at a
					// vertex or not
					if ((IO_BOUNDARY(info) && !CORNER(info)) || initStep || (oldTKE && !initStep && !IO_BOUNDARY(neib_info) && !CORNER(info))) {
						avgNorm += as_float3(boundElement);
					}
				}
				if (oldTKE && !initStep && !IO_BOUNDARY(neib_info) && CORNER(info)) {
					avgNorm += as_float3(boundElement)*boundElement.w;
				}
				// AM TODO FIXME the following code should work, but doesn't for some obscure reason
				//if (CORNER(neib_info) && neibVerts.w == id(info)) {
				//	const float neibRho = oldVel[neib_index].w;
				//	sumMdot += neibRho*boundElement.w*
				//				dot3(oldEulerVel[neib_index],boundElement); // the euler vel should be subtracted by the lagrangian vel which is assumed to be 0 now.
				//}
			}
			else if (IO_BOUNDARY(info) && FLUID(neib_info)){
				const float4 relPos = pos_corr - oldPos[neib_index];
				if(!foundFluid && length3(relPos) < influenceradius)
					foundFluid = true;

				// check if this fluid particles is marked for deletion (i.e. vertices != 0)
				if (neibVerts.x | neibVerts.y != 0 && ACTIVE(relPos)) {
					// betaAV is the weight in barycentric coordinates
					float betaAV = 0.0f;
					const float4 vertexWeights = oldGGam[neib_index];
					// check if one of the vertices is equal to the present one
					if (neibVertXidx == index)
						betaAV = vertexWeights.x;
					else if (neibVertYidx == index)
						betaAV = vertexWeights.y;
					else if (neibVertZidx == index)
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

	// normalize average norm
	if (IO_BOUNDARY(info) || initStep || oldTKE)
		avgNorm = normalize(avgNorm);

	// update boundary conditions on array
	// note that numseg should never be zero otherwise you found a bug
	oldVel[index].w = sumrho/numseg;
	if (oldTKE) {
		oldTKE[index] = sumtke/numseg;
		// adjust Eulerian velocity so that it is tangential to the fixed wall
		if ((!IO_BOUNDARY(info) || CORNER(info)) && !initStep)
			as_float3(oldEulerVel[index]) -= dot(as_float3(oldEulerVel[index]), avgNorm)*avgNorm;
	}
	if (oldEps)
		oldEps[index] = sumeps/numseg;
	// open boundaries
	if (IO_BOUNDARY(info) && !CORNER(info)) {
		float4 eulerVel = oldEulerVel[index];
		// imposing velocities => density needs to be averaged from segments
		if (VEL_IO(info))
			eulerVel.w = sumEulerVel.w/numseg;
		// imposing pressure => velocity needs to be averaged from segments
		else {
			eulerVel.x = sumEulerVel.x/numseg;
			eulerVel.y = sumEulerVel.y/numseg;
			eulerVel.z = sumEulerVel.z/numseg;
		}
		oldEulerVel[index] = eulerVel;
		// the density of the particle is equal to the "eulerian density"
		oldVel[index].w = eulerVel.w;

		// finalize mass computation
		// reference mass:
		const float rho0 = d_rho0[fluid_num(info)];
		const float refMass = deltap*deltap*deltap*rho0;
			// only create new particles in the second part of the time step
		if (step == 2 &&
			// create new particle if the mass of the vertex is large enough
			pos.w > refMass*0.5f &&
			// check that the flow vector points into the domain
			dot(as_float3(eulerVel),avgNorm) > 1e-4f*d_sscoeff[fluid_num(info)] &&
			// pressure inlets need p > 0 to create particles
			(VEL_IO(info) || eulerVel.w-rho0 > rho0*1e-5f) &&
			// corner vertices are not allowed to create new particles
			!CORNER(info))
		{
			massFluid -= refMass;
			// Create new particle
			// TODO of course make_particleinfo doesn't work on GPU due to the memcpy(),
			// so we need a GPU-safe way to do this. The current code is little-endian
			// only, so it's bound to break on other archs. I'm seriously starting to think
			// that we can drop the stupid particleinfo ushort4 typedef and we should just
			// define particleinfo as a ushort ushort uint struct, with proper alignment.

			const uint clone_idx = atomicAdd(newNumParticles, 1);
			// number of new particles that were created on this device in this
			// time step
			const uint newNumPartsOnDevice = clone_idx + 1 - numParticles;
			// the i-th device can only allocate an id that satisfies id%n == i, where
			// n = number of total devices
			const uint nextId = newNumPartsOnDevice*numDevices;

			// FIXME endianness
			uint clone_id = nextId + d_newIDsOffset;
			particleinfo clone_info = info;
			clone_info.x = PT_FLUID; // clear all flags and set it to fluid particle
			clone_info.y = 0; // reset object to 0
			clone_info.z = (clone_id & 0xffff); // set the id of the object
			clone_info.w = ((clone_id >> 16) & 0xffff);

			// Problem has already checked that there is enough memory for new particles
			float4 clone_pos = pos; // new position is position of vertex particle
			clone_pos.w = refMass; // new fluid particle has reference mass
			int3 clone_gridPos = gridPos; // as the position is the same so is the grid position

			// assign new values to array
			oldPos[clone_idx] = clone_pos;
			// the new velocity of the fluid particle is the eulerian velocity of the vertex
			oldVel[clone_idx] = oldEulerVel[index];
			// the eulerian velocity of fluid particles is always 0
			oldEulerVel[clone_idx] = make_float4(0.0f);
			pinfo[clone_idx] = clone_info;
			particleHash[clone_idx] = makeParticleHash( calcGridHash(clone_gridPos), clone_info);
			forces[clone_idx] = make_float4(0.0f);
			contupd[clone_idx] = make_float2(0.0f);
			oldGGam[clone_idx] = oldGGam[index];
			vertices[clone_idx] = make_vertexinfo(0, 0, 0, 0);
			if (oldTKE)
				oldTKE[clone_idx] = oldTKE[index];
			if (oldEps)
				oldEps[clone_idx] = oldEps[index];
		}
		if (!VEL_IO(info) && sumMdot > 0.0f && !(eulerVel.w-rho0 > rho0*1e-5f))
			sumMdot = 0.0f;
		// time stepping
		pos.w += dt*sumMdot;
		pos.w = fmax(-2.0f*refMass, fmin(2.0f*refMass, pos.w));
		if (sumMdot < 0.0f)
			pos.w = fmax(-0.5f*refMass, fmin(0.5f*refMass, pos.w));
		// add contribution from newly created fluid or outgoing fluid particles
		pos.w += massFluid;
		// if a vertex has no fluid particles around and its mass flux is negative then set its mass to 0
		if (!foundFluid && sumMdot < 1e-5f*dt*refMass)
			pos.w = 0.0f;
		oldPos[index].w = pos.w;
	}

	// finalize computation of average norm for gamma calculation in the initial step
	if (initStep) {
		oldGGam[index].x = avgNorm.x;
		oldGGam[index].y = avgNorm.y;
		oldGGam[index].z = avgNorm.z;
		oldGGam[index].w = 0.0f;
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

// This kernel computes the Sheppard correction
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

// contribution of neighbor at relative position relPos with weight w to the
// MLS matrix mls
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

// contribution of neighbor at relative position relPos with weight w to the
// MLS correction when B is the first row of the inverse MLS matrix
__device__ __forceinline__ float
MlsCorrContrib(float4 const& B, float4 const& relPos, float w)
{
	return (B.x + B.y*relPos.x + B.z*relPos.y + B.w*relPos.z)*w;
	// ρ = ∑(ß0 + ß1(xi - xj) + ß2(yi - yj))*Wij*Vj
}


// This kernel computes the MLS correction
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
	if (NOT_FLUID(info)) {
		newVel[index] = vel;
		return;
	}

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
		if (r < influenceradius && FLUID(neib_info)) {
			neibs_num ++;
			const float w = W<kerneltype>(r, slength)*relPos.w/neib_rho;	// Wij*Vj
			MlsMatrixContrib(mls, relPos, w);
		}
	} // end of first loop trough neighbors

	// Resetting persistent variables across getNeibData
	neib_cellnum = 0;
	neib_cell_base_index = 0;

	// Safe inverse of MLS matrix :
	// the matrix is inverted only if |det|/max|aij|^4 > EPSDET
	// and if the number of fluids neighbors if above a minimum
	// value, otherwise no correction is applied
	float maxa = norm_inf(mls);
	maxa *= maxa;
	maxa *= maxa;
	float D = det(mls);
	if (fabs(D) > maxa*EPSDETMLS && neibs_num > MINCORRNEIBSMLS) {
		D = 1/D;
		float4 B;
		B.x = (mls.yy*mls.zz*mls.ww + mls.yz*mls.zw*mls.yw + mls.yw*mls.yz*mls.zw - mls.yy*mls.zw*mls.zw - mls.yz*mls.yz*mls.ww - mls.yw*mls.zz*mls.yw)*D;
		B.y = (mls.xy*mls.zw*mls.zw + mls.yz*mls.xz*mls.ww + mls.yw*mls.zz*mls.xw - mls.xy*mls.zz*mls.ww - mls.yz*mls.zw*mls.xw - mls.yw*mls.xz*mls.zw)*D;
		B.z = (mls.xy*mls.yz*mls.ww + mls.yy*mls.zw*mls.xw + mls.yw*mls.xz*mls.yw - mls.xy*mls.zw*mls.yw - mls.yy*mls.xz*mls.ww - mls.yw*mls.yz*mls.xw)*D;
		B.w = (mls.xy*mls.zz*mls.yw + mls.yy*mls.xz*mls.zw + mls.yz*mls.yz*mls.xw - mls.xy*mls.yz*mls.zw - mls.yy*mls.zz*mls.xw - mls.yz*mls.xz*mls.yw)*D;

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

			const float neib_rho = tex1Dfetch(velTex, neib_index).w;
			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			// Interaction between two particles
			if ((boundarytype == DYN_BOUNDARY || (boundarytype != DYN_BOUNDARY && FLUID(neib_info)))
					&& r < influenceradius ) {
				const float w = W<kerneltype>(r, slength)*relPos.w;	 // ρj*Wij*Vj = mj*Wij
				vel.w += MlsCorrContrib(B, relPos, w);
			}
		}  // end of second loop trough neighbors
	} else {
		// Resort to Sheppard filter in absence of invertible matrix
		// see also shepardDevice. TODO: share the code
		float temp1 = pos.w*W<kerneltype>(0, slength);
		float temp2 = temp1/vel.w;

		// loop over all the neighbors (Second loop)
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
			if ((boundarytype == DYN_BOUNDARY || (boundarytype != DYN_BOUNDARY && FLUID(neib_info)))
					&& r < influenceradius ) {
					// ρj*Wij*Vj = mj*Wij
					const float w = W<kerneltype>(r, slength)*relPos.w;
					// ρ = ∑(ß0 + ß1(xi - xj) + ß2(yi - yj))*Wij*Vj
					temp1 += w;
					temp2 += w/neib_rho;
			}
		}  // end of second loop through neighbors

		vel.w = temp1/temp2;
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
			float gh = kahan_dot(d_gravity, as_float3(pos) + gridPos*d_cellSize + 0.5f*d_cellSize);
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

// final reduction stage
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

// This kernel compute the vorticity field
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


// Testpoints
// This kernel compute the velocity at testpoints
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


// Free surface detection
// This kernel detects the surface particles
template<KernelType kerneltype, bool savenormals>
__global__ void
calcSurfaceparticleDevice(	const	float4*			posArray,
									float4*			normals,
									particleinfo*	newInfo,
							const	hashKey*			particleHash,
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

	//Checking the planes
	for (uint i = 0; i < d_numplanes; ++i) {
		const float3 planeNormal = d_planeNormal[i];
		const float r = PlaneDistance(gridPos, as_float3(pos), planeNormal,
			d_planePointGridPos[i], d_planePointLocalPos[i]);
		if (r < influenceradius) {
			as_float3(normal) += planeNormal*normal_length;
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

__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
saIdentifyCornerVertices(
				const	float4*			oldPos,
						particleinfo*	pinfo,
				const	hashKey*		particleHash,
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

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		const particleinfo neib_info = pinfo[neib_index];
		const uint neib_obj = object(neib_info);

		// loop only over boundary elements that are not of the same open boundary
		if (BOUNDARY(neib_info) && !(obj == neib_obj && IO_BOUNDARY(neib_info))) {
			const float4 relPos = pos_corr - oldPos[neib_index];
			const float r = length3(relPos);
			// if the position is greater than 1.5 dr then the segment is too far away
			if (r > deltap*1.5f)
				continue;

			// check normal distance to segment
			const float4 boundElement = tex1Dfetch(boundTex, neib_index);
			const float normDist = fabs(dot3(boundElement,relPos));

			// if normal distance is less than dr then the particle is a corner particle
			// this implies that the mass of this particle won't vary but it is still treated
			// like an open boundary in every other aspect
			if (normDist < deltap - eps){
				SET_FLAG(info, FG_CORNER);
				pinfo[index] = info;
				break;
			}
		}
	}
}

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

/************************************************************************************************************/

} //namespace cuforces
#endif
