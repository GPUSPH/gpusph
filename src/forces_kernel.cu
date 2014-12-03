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

/*
 * Device code.
 */

#ifndef _FORCES_KERNEL_
#define _FORCES_KERNEL_

#include "particledefine.h"
#include "textures.cuh"
#include "vector_math.h"
#include "multi_gpu_defines.h"

#include "kahan.h"
#include "tensor.cu"

// single-precision M_PI
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

__constant__ idx_t	d_neiblist_end; // maxneibsnum * number of allocated particles
__constant__ idx_t	d_neiblist_stride; // stride between neighbors of the same particle

__constant__ float	d_wcoeff_cubicspline;			// coeff = 1/(Pi h^3)
__constant__ float	d_wcoeff_quadratic;				// coeff = 15/(16 Pi h^3)
__constant__ float	d_wcoeff_wendland;				// coeff = 21/(16 Pi h^3)

__constant__ float	d_fcoeff_cubicspline;			// coeff = 3/(4Pi h^4)
__constant__ float	d_fcoeff_quadratic;				// coeff = 15/(32Pi h^4)
__constant__ float	d_fcoeff_wendland;				// coeff = 105/(128Pi h^5)

__constant__ int	d_numfluids;					// number of different fluids

__constant__ float	d_rho0[MAX_FLUID_TYPES];		// rest density of fluids

// Speed of sound constants
__constant__ float	d_bcoeff[MAX_FLUID_TYPES];
__constant__ float	d_gammacoeff[MAX_FLUID_TYPES];
__constant__ float	d_sscoeff[MAX_FLUID_TYPES];
__constant__ float	d_sspowercoeff[MAX_FLUID_TYPES];
// square of sound speed for at-rest density
__constant__ float	d_sqC0[MAX_FLUID_TYPES];

__constant__ float3	d_gravity;						// gravity (vector)

__constant__ float	d_ferrari;						// coefficient for Ferrari correction

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
__constant__ float3	d_rbcg[MAXBODIES];
__constant__ uint	d_rbstartindex[MAXBODIES];
__constant__ float	d_objectobjectdf;
__constant__ float	d_objectboundarydf;

// Grid data
#include "cellgrid.h"

// Neibdata cell number to offset
__constant__ char3	d_cell_to_offset[27];

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

/************************************************************************************************************/
/*					   Reflect position or velocity wrt to a plane										    */
/************************************************************************************************************/

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
// TODO: check for the maximum timestep

// Normal and viscous force wrt to solid boundary
__device__ __forceinline__ float
PlaneForce(	const float3 &	pos,
			const float 	mass,
			const float4 &	plane,
			const float		l,
			const float3&	vel,
			const float		dynvisc,
			float4&			force)
{
	const float r = abs(dot(pos, as_float3(plane)) + plane.w)/l;
	if (r < d_r0) {
		const float DvDt = LJForce(r);
		// Unitary normal vector of the surface
		const float3 relPos = make_float3(plane)*r/l;

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
GeometryForce(	const float3&	pos,
				const float		mass,
				const float3&	vel,
				const float		dynvisc,
				float4&			force)
{
	float coeff_max = 0.0f;
	for (uint i = 0; i < d_numplanes; ++i) {
		float coeff = PlaneForce(pos, mass, d_planes[i], d_plane_div[i], vel, dynvisc, force);
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
			const float3&	pos,
			const float		mass,
			const float3&	vel,
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
		return PlaneForce(pos, mass, make_float4(a, b, c, d), l, vel, dynvisc, force);
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
template<KernelType kerneltype, bool dynbounds>
__global__ void
__launch_bounds__(BLOCK_SIZE_SPS, MIN_BLOCKS_SPS)
SPSstressMatrixDevice(	const float4* posArray,
						float2*		tau0,
						float2*		tau1,
						float2*		tau2,
						const hashKey*	particleHash,
						const uint*	cellStart,
						const neibdata*	neibsList,
						const uint	numParticles,
						const float	slength,
						const float	influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// compute SPS matrix only for fluid particles, except in the
	// DYN_BOUNDARY case (dynbounds = true), in which case we compute
	// it for everything
	// TODO testpoints should also compute SPS, it'd be useful
	// when we will enable SPS saving to disk
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (NOT_FLUID(info) && !dynbounds)
		return;

	// read particle data from sorted arrays
	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	// skip inactive particles
	if (INACTIVE(pos))
		return;

	const float4 vel = tex1Dfetch(velTex, index);

	// SPS stress matrix elements
	symtensor3 tau;

	// Gradients of the the velocity components
	float3 dvx = make_float3(0.0f);
	float3 dvy = make_float3(0.0f);
	float3 dvz = make_float3(0.0f);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = -1;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// loop over all the neighbors
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

		// velocity gradient is contributed by fluid particles only,
		// except in the DYN_BOUNDARY case where all particles (except TESTPOINTS,
		// of course) contribute
		const bool valid_neib_type = FLUID(neib_info) || (dynbounds && !TESTPOINTS(neib_info));

		if (r < influenceradius && valid_neib_type) {
			const float f = F<kerneltype>(r, slength)*relPos.w/relVel.w;	// 1/r ∂Wij/∂r Vj

			// Velocity Gradients
			dvx -= relVel.x*as_float3(relPos)*f;	// dvx = -∑mj/ρj vxij (ri - rj)/r ∂Wij/∂r
			dvy -= relVel.y*as_float3(relPos)*f;	// dvy = -∑mj/ρj vyij (ri - rj)/r ∂Wij/∂r
			dvz -= relVel.z*as_float3(relPos)*f;	// dvz = -∑mj/ρj vzij (ri - rj)/r ∂Wij/∂r
			}
		} // end of loop through neighbors

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
	float S = sqrtf(SijSij_bytwo);
	float nu_SPS = d_smagfactor*S;		// Dalrymple & Rogers (2006): eq. (12)
	float divu_SPS = 0.6666666666f*nu_SPS*(dvx.x + dvy.y + dvz.z);
	float Blinetal_SPS = d_kspsfactor*SijSij_bytwo;

	// Shear Stress matrix = TAU (pronounced taf)
	// Dalrymple & Rogers (2006): eq. (10)
	tau.xx = nu_SPS*(dvx.x + dvx.x) - divu_SPS - Blinetal_SPS;	// tau11 = tau_xx/ρ^2
	tau.xx /= vel.w;
	tau.xy *= nu_SPS/vel.w;								// tau12 = tau_xy/ρ^2
	tau.xz *= nu_SPS/vel.w;								// tau13 = tau_xz/ρ^2
	tau.yy = nu_SPS*(dvy.y + dvy.y) - divu_SPS - Blinetal_SPS;	// tau22 = tau_yy/ρ^2
	tau.yy /= vel.w;
	tau.yz *= nu_SPS/vel.w;								// tau23 = tau_yz/ρ^2
	tau.zz = nu_SPS*(dvz.z + dvz.z) - divu_SPS - Blinetal_SPS;	// tau33 = tau_zz/ρ^2
	tau.zz /= vel.w;

	tau0[index] = make_float2(tau.xx, tau.xy);
	tau1[index] = make_float2(tau.xz, tau.yy);
	tau2[index] = make_float2(tau.yz, tau.zz);
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
		const float uq = 1/q;
		intKernel = WENDLAND_I_COEFF*tmp4*tmp*((((8*uq + 20)*uq + 30)*uq) + 21);
	}

	return make_float2(kernel, intKernel);
}

/*
 * Gaussian quadrature
 */

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
	float4 coord1 = make_float4(0.0f);
	float4 coord2 = make_float4(0.0f);
	// Get index j for which n_s is minimal
	if (fabs(boundElement.x) > fabs(boundElement.y))
		j = 1;
	if (((float)1-j)*fabs(boundElement.x) + ((float)j)*fabs(boundElement.y) > fabs(boundElement.z))
		j = 2;
	// compute second coordinate which is equal to n_s x e_j
	if (j==0) {
		coord1 = make_float4(1.0f, 0.0f, 0.0f, 0.0f);
		coord2 = make_float4(0.0f, boundElement.z, -boundElement.y, 0.0f);
	}
	else if (j==1) {
		coord1 = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
		coord2 = make_float4(-boundElement.z, 0.0f, boundElement.x, 0.0f);
	}
	else {
		coord1 = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
		coord2 = make_float4(boundElement.y, -boundElement.x, 0.0f, 0.0f);
	}
	// relative positions of vertices with respect to the segment, normalized by h
	float4 v0 = -(vPos0.x*coord1 + vPos0.y*coord2)/slength; // e.g. v0 = r_{v0} - r_s
	float4 v1 = -(vPos1.x*coord1 + vPos1.y*coord2)/slength;
	float4 v2 = -(vPos2.x*coord1 + vPos2.y*coord2)/slength;
	// check if we are close to a wall
	if (q_aSigma.w < 0.5f) {
		minlRas = min(minlRas, q_aSigma.w);
	}
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
		oldGGam /= -length3(oldGGam);
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
//		float2 intVal = gaussQuadratureO14(-as_float3(v0), -as_float3(v1), -as_float3(v2), as_float3(relPos));
		// Gaussian quadrature of 5th order
		float2 intVal = gaussQuadratureO5(-as_float3(v0), -as_float3(v1), -as_float3(v2), as_float3(relPos));
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

__global__ void
updateBoundValuesDevice(	float4*		oldVel,
				float*		oldTKE,
				float*		oldEps,
				uint*		vertIDToIndex,
				const uint	numParticles,
				bool		initStep)
{
	const uint index = INTMUL(blockIdx.x, blockDim.x) + threadIdx.x;

	if(index < numParticles) {
		const particleinfo info = tex1Dfetch(infoTex, index);
		if (BOUNDARY(info)) {
			// get vertex indices associated to the boundary segment
			const vertexinfo vertices = tex1Dfetch(vertTex, index);

			// load the indices of the vertices only once
			const uint vertXidx = vertIDToIndex[vertices.x];
			const uint vertYidx = vertIDToIndex[vertices.y];
			const uint vertZidx = vertIDToIndex[vertices.z];

			// segment values are equal to the averaged vertex values
			// density
			//const float ro1 = oldVel[vertXidx].w;
			//const float ro2 = oldVel[vertYidx].w;
			//const float ro3 = oldVel[vertZidx].w;
			//oldVel[index].w = (ro1 + ro2 + ro3)/3.f;
			// velocity
			const float4 vel1 = oldVel[vertXidx];
			const float4 vel2 = oldVel[vertYidx];
			const float4 vel3 = oldVel[vertZidx];
			oldVel[index] = (vel1 + vel2 + vel3)/3.f;
			// turbulent kinetic energy
			if (oldTKE) {
				const float k1 = oldTKE[vertXidx];
				const float k2 = oldTKE[vertYidx];
				const float k3 = oldTKE[vertZidx];
				oldTKE[index] = (k1 + k2 + k3)/3.f;
			}
			// epsilon
			if (oldEps) {
				const float eps1 = oldEps[vertXidx];
				const float eps2 = oldEps[vertYidx];
				const float eps3 = oldEps[vertZidx];
				oldEps[index] = (eps1 + eps2 + eps3)/3.f;
			}
		}
	}
}

template<KernelType kerneltype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
dynamicBoundConditionsDevice(	const float4*	oldPos,
				float4*		oldVel,
				float*		oldTKE,
				float*		oldEps,
				float4*		gradGamma,
				const hashKey*	particleHash,
				const uint*	cellStart,
				const neibdata*	neibsList,
				const uint	numParticles,
				const float deltap,
				const float	slength,
				const float	influenceradius,
				const bool	initStep)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

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

	// note that all sums below run only over fluid particles (including the Shepard filter)
	float sumrho = 0.0f; // summation for computing the density
	float sumtke = 0.0f; // summation for computing tke (k-epsilon model)
	float sumeps = 0.0f; // summation for computing epsilon (k-epsilon model)
	float alpha  = 0.0f;  // the shepard filter

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;
	float3 avgNorm = make_float3(0.0f);

	// Square of sound speed. Would need modification for multifluid
	const float sqC0 = d_sqC0[PART_FLUID_NUM(info)];

	const float4 normal = normalize3(tex1Dfetch(gamTex, index));

	// Loop over all the neighbors
	for (idx_t i = 0; i < d_neiblist_end; i += d_neiblist_stride) {
		neibdata neib_data = neibsList[i + index];

		if (neib_data == 0xffff) break;

		const uint neib_index = getNeibIndex(pos, pos_corr, cellStart, neib_data, gridPos,
					neib_cellnum, neib_cell_base_index);

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		#if( __COMPUTE__ >= 20)
		const float4 relPos = pos_corr - oldPos[neib_index];
		#else
		const float4 relPos = pos_corr - tex1Dfetch(posTex, neib_index);
		#endif

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

//		const float neib_rho = tex1Dfetch(velTex, neib_index).w;
		const float neib_rho = oldVel[neib_index].w;
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);
		const float neib_pres = P(neib_rho, PART_FLUID_NUM(neib_info));
		const float neib_vel = length(make_float3(oldVel[neib_index]));
		const float neib_k = oldTKE ? oldTKE[neib_index] : NAN;
		const float neib_eps = oldEps ? oldEps[neib_index] : NAN;

		if (r < influenceradius && FLUID(neib_info)) {
			// kernel value times volume
			const float w = W<kerneltype>(r, slength)*relPos.w/neib_rho;
			// normal distance based on grad Gamma which approximates the normal of the domain
			const float normDist = fmax(fabs(dot3(normal,relPos)), deltap);
			sumrho += (1.0f + dot(d_gravity,as_float3(relPos))/sqC0)*w*neib_rho;
			sumtke += w*neib_k;
			// the constant is coming from 4*powf(0.09,0.75)/0.41
			sumeps += w*(neib_eps + 1.603090412f*powf(neib_k,1.5f)/normDist);
			alpha += w;
		}
		// in the initial step we need to compute an approximate grad gamma direction for the computation of gamma
		else if (initStep && r < influenceradius && BOUNDARY(neib_info)) {
			const float3 normal = as_float3(tex1Dfetch(boundTex, neib_index));
			avgNorm += normal;
		}
	}

	if (alpha > 1e-5f) {
		oldVel[index].w = fmax(sumrho/alpha,d_rho0[PART_FLUID_NUM(info)]);
		if (oldTKE)
			oldTKE[index] = sumtke/alpha;
		if (oldEps)
			oldEps[index] = sumeps/alpha;
	}
	else {
		oldVel[index].w = d_rho0[PART_FLUID_NUM(info)];
		if (oldTKE)
			oldTKE[index] = 0.0f;
		if (oldEps)
			oldEps[index] = 0.0f;
	}

	if (initStep) {
		avgNorm /= length(avgNorm);
		gradGamma[index].x = avgNorm.x;
		gradGamma[index].y = avgNorm.y;
		gradGamma[index].z = avgNorm.z;
		gradGamma[index].w = 0.0f;
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
template<KernelType kerneltype>
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

	// read particle data from sorted arrays
	// normalize kernel only if the given particle is a fluid one
	const particleinfo info = tex1Dfetch(infoTex, index);

	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	float4 vel = tex1Dfetch(velTex, index);

	if (NOT_FLUID(info) && !VERTEX(info)) {
		newVel[index] = vel;
		return;
	}

	// taking into account self contribution in summation
	float temp1 = pos.w*W<kerneltype>(0, slength);
	float temp2 = temp1/vel.w ;

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Persistent variables across getNeibData calls
	char neib_cellnum = 0;
	uint neib_cell_base_index = 0;
	float3 pos_corr;

	// loop over all the neighbors
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

		const float neib_rho = tex1Dfetch(velTex, neib_index).w;
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && (FLUID(neib_info)/* || VERTEX(neib_info)*/)) {
			const float w = W<kerneltype>(r, slength)*relPos.w;
			temp1 += w;
			temp2 += w/neib_rho;
		}
	}

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
template<KernelType kerneltype>
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

	// read particle data from sorted arrays
	// computing MLS matrix only for fluid particles
	const particleinfo info = tex1Dfetch(infoTex, index);

	#if( __COMPUTE__ >= 20)
	const float4 pos = posArray[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif

	float4 vel = tex1Dfetch(velTex, index);

	if (NOT_FLUID(info)) {
		newVel[index] = vel;
		return;
	}

	// MLS matrix elements
	symtensor4 mls;
	mls.xx = mls.xy = mls.xz = mls.xw =
		mls.yy = mls.yz = mls.yw =
		mls.zz = mls.zw = mls.ww = 0;

	// number of neighbors
	int neibs_num = 0;

	// taking into account self contribution in MLS matrix construction
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

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const float r = length(as_float3(relPos));

		const float neib_rho = tex1Dfetch(velTex, neib_index).w;
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		// interaction between two particles
		if (r < influenceradius && FLUID(neib_info)) {
			neibs_num ++;
			float w = W<kerneltype>(r, slength)*relPos.w/neib_rho;	// Wij*Vj
			MlsMatrixContrib(mls, relPos, w);
		}
	} // end of first loop trough neighbors

	// Resetting persistent variables across getNeibData
	neib_cellnum = 0;
	neib_cell_base_index = 0;

	// safe inverse of MLS matrix
	// the matrix is inverted only if |det|/max|aij|^4 > EPSDET
	// and if the number of fluids neighbors if above a minimum
	// value, otherwise no correction is applied
	float maxa = norm_inf(mls);
	maxa *= maxa;
	maxa *= maxa;
	float D = det(mls);
	if (D > maxa*EPSDETMLS && neibs_num > MINCORRNEIBSMLS) {  // FIXME: should be |det| ?????
		// first row of inverse matrix
		D = 1/D;
		float4 B;
		B.x = (mls.yy*mls.zz*mls.ww + mls.yz*mls.zw*mls.yw + mls.yw*mls.yz*mls.zw - mls.yy*mls.zw*mls.zw - mls.yz*mls.yz*mls.ww - mls.yw*mls.zz*mls.yw)*D;
		B.y = (mls.xy*mls.zw*mls.zw + mls.yz*mls.xz*mls.ww + mls.yw*mls.zz*mls.xw - mls.xy*mls.zz*mls.ww - mls.yz*mls.zw*mls.xw - mls.yw*mls.xz*mls.zw)*D;
		B.z = (mls.xy*mls.yz*mls.ww + mls.yy*mls.zw*mls.xw + mls.yw*mls.xz*mls.yw - mls.xy*mls.zw*mls.yw - mls.yy*mls.xz*mls.ww - mls.yw*mls.yz*mls.xw)*D;
		B.w = (mls.xy*mls.zz*mls.yw + mls.yy*mls.xz*mls.zw + mls.yz*mls.yz*mls.xw - mls.xy*mls.yz*mls.zw - mls.yy*mls.zz*mls.xw - mls.yz*mls.xz*mls.yw)*D;

		// taking into account self contribution in density summation
		vel.w = B.x*W<kerneltype>(0, slength)*pos.w;

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

			// skip inactive particles
			if (INACTIVE(relPos))
				continue;

			const float r = length(as_float3(relPos));

			const float neib_rho = tex1Dfetch(velTex, neib_index).w;
			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			// interaction between two particles
			if (r < influenceradius && FLUID(neib_info)) {
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

			// skip inactive particles
			if (INACTIVE(relPos))
				continue;

			const float r = length(as_float3(relPos));

			const float neib_rho = tex1Dfetch(velTex, neib_index).w;
			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			// interaction between two particles
			if (r < influenceradius && FLUID(neib_info)) {
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
			uint fluid_num = PART_FLUID_NUM(pinfo);
			float v2 = kahan_sqlength(as_float3(vel));
			// TODO improve precision by splitting the float part from the grid part
			float gh = kahan_dot(d_gravity, as_float3(pos) + gridPos*d_cellSize + 0.5f*d_cellSize);
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

	// read particle data from sorted arrays
	const particleinfo info = tex1Dfetch(infoTex, index);
	if(type(info) != TESTPOINTSPART)
		return;

	#if (__COMPUTE__ >= 20)
	const float4 pos = oldPos[index];
	#else
	const float4 pos = tex1Dfetch(posTex, index);
	#endif
	float4 vel = tex1Dfetch(velTex, index);

	// this is the velocity (x,y,z) and pressure (w)
	float4 temp = make_float4(0.0f);
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

		const float4 neib_vel = tex1Dfetch(velTex, neib_index);
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

		if (r < influenceradius && FLUID(neib_info)) {
			const float w = W<kerneltype>(r, slength)*relPos.w/neib_vel.w;	// Wij*mj
			//Velocity
			temp.x += w*neib_vel.x;
			temp.y += w*neib_vel.y;
			temp.z += w*neib_vel.z;
			//Pressure
			temp.w += w*P(neib_vel.w, object(neib_info));
			//Shepard filter
			alpha += w;
		}
	}

	// Renormalization by the Shepard filter
	if(alpha>1e-5f) {
		vel = temp/alpha;
	}
	else {
		vel = make_float4(0.0f);
	}

	newVel[index] = vel;
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

	CLEAR_FLAG(info, SURFACE_PARTICLE_FLAG);
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
	const float3 globalpos = d_worldOrigin + as_float3(pos) + gridPos*d_cellSize + 0.5f*d_cellSize;

	for (uint i = 0; i < d_numplanes; ++i) {
		float r = abs(dot(globalpos, as_float3(d_planes[i])) + d_planes[i].w)/d_plane_div[i];
		if (r < influenceradius) {
			as_float3(normal) += as_float3(d_planes[i])* normal_length;
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
		SET_FLAG(info, SURFACE_PARTICLE_FLAG);

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
