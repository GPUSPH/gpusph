/*  Copyright (c) 2015-2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

/* Device functions and constants pertaining open boundaries */

#ifndef _BOUNDS_KERNEL_
#define _BOUNDS_KERNEL_

#include "particledefine.h"

/*!
 * \namespace cubounds
 * \brief Contains all device functions/kernels/constants related to open boundaries and domain geometry.
 *
 * The namespace contains the device side of boundary handling
 *	- domain size, origin and cell grid properties and related functions
 *	- open boundaries properties and related functions
 */
namespace cubounds {

using namespace cuneibs;
using namespace cuphys;
using namespace cusph;

/// \name Device constants
/// @{

/// Number of vertices in the whole simulation
/*! When a new particle is generated, its ID will be taken from
 * the generating vertex' newID, which then gets incremented by
 * d_numVertices to prepare the ID of the next generation
 */
__constant__ uint	d_numOpenVertices;

/// @}

/** \name Device functions
 *  @{ */

/*!
 * Create a new particle, cloning an existing particle
 * This returns the index of the generated particle, initializing new_info
 * for a FLUID particle of the same fluid as the generator, no associated
 * object or inlet, and a new id generated in a way which is multi-GPU
 * compatible.
 *
 * All other particle properties (position, velocity, etc) should be
 * set by the caller.
 */
__device__ __forceinline__
uint
createNewFluidParticle(
	/// [out] particle info of the generated particle
			particleinfo	&new_info,
	/// [in] particle info of the generator particle
	const	particleinfo	&info,
	/// [in] number of particles at the start of the current timestep
	const	uint			numParticles,
	/// [in,out] number of particles including all the ones already created in this timestep
			uint			*newNumParticles,
	/// [in,out] next ID for generated particles
			uint			*nextID,
	const	uint			totParticles)
{
	const uint new_index = atomicAdd(newNumParticles, 1);

	const uint new_id = *nextID;
	if (new_id > UINT_MAX - d_numOpenVertices)
		printf( " WARNING: generator %u running out of IDs: %u + %u > %u\n",
			id(info), new_id, d_numOpenVertices, UINT_MAX);

	*nextID = new_id + d_numOpenVertices;

	new_info = make_particleinfo_by_ids(
		PT_FLUID,
		fluid_num(info), 0, // copy the fluid number, not the object number
		new_id);

	return new_index;
}

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
		       // TODO Check case of multifluid for a = fluid_num(info)
			float riemannRho = RHO(P(rhoInt, a) + physical_density(rhoInt,a) * unInt * (unInt - unExt), a); // returns relative
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
		// TODO Check case of multifluid for a = fluid_num(info)
		if (rhoExt <= rhoInt) { // Expansion wave
			flux = unInt + (rExt - rInt);
			float lambda = flux + cExt;
			if (lambda > lambdaInt) { // shock wave
				flux = (P(rhoInt, a) - P(rhoExt, a))/(physical_density(rhoInt,a)*fmaxf(unInt,1e-5f*d_sscoeff[a])) + unInt;
				// check that unInt was not too small
				if (fabsf(flux) > d_sscoeff[a] * 0.1f)
					flux = unInt;
				lambda = flux + cExt;
				if (lambda <= lambdaInt) // contact discontinuity
					flux = unInt;
			}
		}
		else { // shock wave
			flux = (P(rhoInt, a) - P(rhoExt, a))/(physical_density(rhoInt,a)*fmaxf(unInt,1e-5f*d_sscoeff[a])) + unInt;
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
		// AM-TODO allow imposed tangential velocity (make sure normal component is zero)
		// currently for inflow we assume that the tangential velocity is zero
		// GB-TODO FIXME splitneibs merge
        // remove normal component of imposed Eulerian velocity
		//as_float3(eulerVel) = as_float3(eulerVel) - dot(as_float3(eulerVel), normal)*normal;
		as_float3(eulerVel) = make_float3(0.0f);
		// if the imposed pressure on the boundary is negative make sure that the flux is negative
		// as well (outflow)
		if (rhoExt < 0.0f)
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
template<typename WeightsType> // float3 or float4
__device__ __forceinline__ void
getMassRepartitionFactor(
	const	float3	*vertexRelPos,
	const	float3	normal,
	WeightsType	&beta)
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

// flags for the vertexinfo .w coordinate which specifies how many vertex particles of one segment
// is associated to an open boundary
#define VERTEX1 ((flag_t)1)
#define VERTEX2 (VERTEX1 << 1)
#define VERTEX3 (VERTEX2 << 1)
#define ALLVERTICES ((flag_t)(VERTEX1 | VERTEX2 | VERTEX3))

//! Auxiliary structures and functions for SA_BOUNDARY boundary conditions kernels
/** \ref saSegmentBoundaryConditionsDevice, \ref saVertexBoundaryConditionsDevice
 */
namespace sa_bc
{

//! Particle data used by both
//! \ref saSegmentBoundaryConditionsDevice and \ref saVertexBoundaryConditionsDevice
struct common_pdata
{
	uint index;
	particleinfo info;
	float4 pos;
	int3 gridPos;

	// Square of sound speed. Would need modification for multifluid
	float sqC0;

	template<typename Params>
	__device__ __forceinline__
	common_pdata(Params const& params, uint _index, particleinfo const& _info) :
		index(_index),
		info(_info),
		pos(params.pos[index]),
		gridPos(calcGridPosFromParticleHash( params.particleHash[index] )),
		sqC0(d_sqC0[fluid_num(info)])
	{}
};

//! Particle data used by \ref saSegmentBoundaryConditionsDevice
struct segment_pdata :
	common_pdata
{
	vertexinfo verts;
	float4 normal;

	template<typename Params>
	__device__ __forceinline__
	segment_pdata(Params const& params, uint _index, particleinfo const& _info) :
		common_pdata(params, _index, _info),
		verts(params.vertices[index]),
		normal(tex1Dfetch(boundTex, index))
	{}
};


//! Particle data used for vertices with open boundaries
struct vertex_io_pdata
{
	bool corner; // is this a corner vertex?
	const float4 normal;
	const float  refMass; // reference mass ∆p³*ρ_0

	template<typename Params>
	__device__ __forceinline__
	vertex_io_pdata(Params const& params, uint _index, particleinfo const& _info) :
		corner(CORNER(_info)),
		normal(tex1Dfetch(boundTex, _index)),
		refMass(params.deltap*params.deltap*params.deltap*d_rho0[fluid_num(_info)])
	{}
};

//! Particle data used by \ref saVertexBoundaryConditionsDevice
template<typename Params,
	bool has_io = Params::has_io,
	typename io_struct =
		typename COND_STRUCT(has_io, vertex_io_pdata)
	>
struct vertex_pdata :
	common_pdata,
	io_struct
{
	float gam;

	__device__ __forceinline__
	vertex_pdata(Params const& params, uint _index, particleinfo const& _info) :
		common_pdata(params, _index, _info),
		io_struct(params, _index, _info),
		gam(params.gGam[index].w)
	{}
};

/*! \struct pout
    \brief Particle output for \ref saSegmentBoundaryConditionsDevice

  All sum* members in the substructures compute the sum over
  fluid particles, with Shepard filtering
 */

//! Common components for \ref pout structures
/**! This structure holds the component needed by both segment and vertex
 *   boundary conditions kernels, regardless of template specialization
 */
struct common_pout
{
	float sumpWall; // summation to compute the density
	float shepard_div; // Shepard filter divisor

	template<typename Params>
	__device__ __forceinline__
	common_pout(Params const& params, uint index, particleinfo const& info) :
		sumpWall(0), shepard_div(0)
	{}

};

//! \ref pout components needed by all segment (but not vertex) pout specializations
struct common_segment_pout
{
	float4 gGam; // new gamma and its gradient
	float4 vel; // velocity to impose Neumann conditions

	const bool calcGam;

	template<typename Params>
	__device__ __forceinline__
	common_segment_pout(Params const& params, uint index, particleinfo const& info) :
		gGam(make_float4(0, 0, 0, params.gGam[index].w)),
		vel(make_float4(0)),
		// Gamma always needs to be recomputed when moving bodies are enabled.
		// If not, we only need to compute if we are at initialisation stage
		calcGam((Params::simflags & ENABLE_MOVING_BODIES) || !isfinite(gGam.w)
				|| Params::step == 0)
	{
		if (calcGam)
			gGam.w = 0;
#if 0
		else if (gGam.w < 1.e-5f)
			printf("%d (%d %d %d) gamma %g too low\n",
				index, id(info), PART_TYPE(info), PART_FLAGS(info),
				gGam.w);
#endif
	}

};

//! \ref pout components which are only needed for I/O (open boundaries), for both
//! segments and vertices
struct common_io_pout
{
	float sump; // summation to compute the pressure
	float3 sumvel; // summation to compute internal velocity for open boundaries

	__device__ __forceinline__
	common_io_pout() :
		sump(0),
		sumvel(make_float3(0.0f))
	{}
};

//! Open boundaries \ref pout components only needed by vertices
struct vertex_io_pout :
	common_io_pout
{
	float sumMdot; // summation for computing the mass variance based on in/outflow
	float massFluid; // mass obtained from a outgoing - mass of a new fluid
	bool foundFluid; // check if a vertex particle has a fluid particle in its support

	// wall normal:
	// for corner vertices the wall normal is equal to the normal of the associated segments that belong to a solid wall
	float3 wallNormal;

	__device__ __forceinline__
	vertex_io_pout() :
		sumMdot(0.0f),
		massFluid(0.0f),
		foundFluid(false),
		wallNormal(make_float3(0.0f))
	{}
};

//! Eulerian velocity \ref pout component
/** This is needed for open boundaries and k-epsilon viscosity
 */
struct eulervel_pout
{
	float4 eulerVel;

	//! Constructor for the non-IO case
	/** Start with null eulerVel */
	template<typename Params>
	__device__ __forceinline__
	eulervel_pout(Params const& params, uint index, particleinfo const& info,
		// dummy argument to only select this in the non-IO case
		enable_if_t<!Params::has_io, int> _sfinae = 0)
	:
		eulerVel(make_float4(0))
	{}

	//! Constructor in the IO case
	/** In this case we fetch eulerVel for open boundary segments */
	template<typename Params, typename = enable_if_t<Params::has_io> >
	__device__ __forceinline__
	eulervel_pout(Params const& params, uint index, particleinfo const& info,
		// dummy argument to only select this in the IO case
		enable_if_t<Params::has_io, int> _sfinae = 0)
	:
		eulerVel(make_float4(0))
	{
		// For IO boundary, fetch the data that was set in the problem-specific routine
		if (!IO_BOUNDARY(info))
			return;

		// If pressure is imposed, we only care about .w, otherwise we only care
		// about .xyz, and .w should be 0
		eulerVel = params.eulerVel[index];
		if (VEL_IO(info))
			eulerVel.w = 0;
	}
};

//! \ref pout components for k-epsilon viscosity
struct common_keps_pout
{
	// summation to compute TKE and epsilon (k-epsilon model)
	float sumtke;
	float sumeps;

	__device__ __forceinline__
	common_keps_pout() :
		sumtke(0),
		sumeps(0)
	{}
};

struct segment_keps_pout :
	common_keps_pout
{
	float tke;
	float eps;

	template<typename Params>
	__device__ __forceinline__
	segment_keps_pout(Params const& params, uint index, particleinfo const& info) :
		common_keps_pout(),
		tke(0), eps(0)
	{
		// For IO boundary with imposed velocity,
		// fetch the data that was set in the problem-specific routine
		if (IO_BOUNDARY(info) && VEL_IO(info)) {
			tke = params.tke[index];
			eps = params.eps[index];
		}
	}
};

struct vertex_keps_pout :
	common_keps_pout
{
	int numseg; // number of segments adjacent this vertex

	template<typename Params>
	__device__ __forceinline__
	vertex_keps_pout(Params const& params, uint index, particleinfo const& info) :
		common_keps_pout(),
		numseg(0)
	{}
};

/**
 * The segment_pout structure is assembled from \ref common_pout, \ref common_segment_pout
 * and a combination of \ref eulervel_pout, \ref keps_pout, \ref io_pout
 * depending on simulation flags and viscous model (which are derived from
 * Params, which is a specialization of the \ref sa_segment_bc_params
 * parameters structure.
 */
template<typename Params,
	bool has_io = (Params::has_io),
	bool has_keps = (Params::has_keps),
	bool has_eulerVel = (has_io || has_keps),
	typename eulervel_struct =
		typename COND_STRUCT(has_eulerVel, eulervel_pout),
	typename keps_struct =
		typename COND_STRUCT(has_keps, segment_keps_pout),
	typename io_struct =
		typename COND_STRUCT(has_io, common_io_pout)
	>
struct segment_pout :
	common_pout,
	common_segment_pout,
	eulervel_struct,
	keps_struct,
	io_struct
{
	__device__ __forceinline__
	segment_pout(Params const& params, uint index, particleinfo const& info) :
		common_pout(params, index, info),
		common_segment_pout(params, index, info),
		eulervel_struct(params, index,info),
		keps_struct(params, index, info),
		io_struct()
	{}
};

/**
 * The vertex_pout structure is assembled from \ref common_pout,
 * and a combination of \ref eulervel_pout, \ref keps_pout, \ref io_pout
 * depending on simulation flags and viscous model (which are derived from
 * Params, which is a specialization of the \ref sa_segment_bc_params
 * parameters structure.
 * TODO FIXME splitneibs-merge we're only handling non-IO, non-keps
 * in saVertexBoundaryConditionsDevice, but we're more or less aware
 * of what we'll need.
 */
template<typename Params,
	bool has_io = (Params::has_io),
	bool has_keps = (Params::has_keps),
	// note that we don't have the eulerVel in pout with keps,
	// in contrast to most other circumstances
	// (we will adjust the eulerVel to be tangential to the wall,
	// but we'll do that “in place” getting the array directly
	// in the relevant function
	bool has_eulerVel = has_io,
	typename eulervel_struct =
		typename COND_STRUCT(has_eulerVel, eulervel_pout),
	typename keps_struct =
		typename COND_STRUCT(has_keps, vertex_keps_pout),
	typename io_struct =
		typename COND_STRUCT(has_io, vertex_io_pout)
	>
struct vertex_pout :
	common_pout,
	eulervel_struct,
	keps_struct,
	io_struct
{
	__device__ __forceinline__
	vertex_pout(Params const& params, uint index, particleinfo const& info) :
		common_pout(params, index, info),
		eulervel_struct(params, index,info),
		keps_struct(params, index, info),
		io_struct()
	{}
};


/*! \struct ndata
  \brief Neighbor data for saSegmentBoundaryConditionsDevice
 */


//! Common fluid neighbor data
struct common_ndata
{
	uint index;
	particleinfo info;
	float4 relPos;
	float4 vel;
	float r;
	float w; // kernel value times volume
	float press;

	template<typename Params>
	__device__ __forceinline__
	common_ndata(Params const& params, uint _index, float4 const& _relPos)
	:
		index(_index),
		info(tex1Dfetch(infoTex, index)),
		relPos(_relPos),
		vel(params.vel[index]),
		r(length3(relPos)),
		w(W<Params::kerneltype>(r, params.slength)*relPos.w/physical_density(vel.w,fluid_num(info))),
		press(P(vel.w, fluid_num(info)))
	{}
};

//! fluid neighbor data used for k-epsilon
struct common_keps_ndata
{
	float tke;
	float eps;

	template<typename Params>
	__device__ __forceinline__
	common_keps_ndata(Params const& params, uint neib_index)
	:
		tke(params.tke[neib_index]),
		eps(params.eps[neib_index])
	{}
};

struct segment_keps_ndata :
	common_keps_ndata
{
	// normal distance based on grad Gamma which approximates the normal of the element
	float norm_dist;

	template<typename Params, typename PData>
	__device__ __forceinline__
	segment_keps_ndata(Params const& params, PData const& pdata,
		uint neib_index, float4 const& relPos)
	:
		common_keps_ndata(params, neib_index),
		norm_dist(fmax(fabs(dot3(pdata.normal, relPos)), params.deltap))
	{}
};

template<typename Params, //!< \ref sa_segment_bc_params specialization
	bool has_keps = Params::has_keps, //!< is k-epsilon enabled?
	typename keps_struct = //!< optional components if \ref has_keps
		typename COND_STRUCT(has_keps, segment_keps_ndata)
	>
struct segment_ndata :
	common_ndata,
	keps_struct
{
	template<typename PData>
	__device__ __forceinline__
	segment_ndata(Params const& params, PData const& pdata,
		uint neib_index, float4 const& relPos) :
		common_ndata(params, neib_index, relPos),
		keps_struct(params, pdata, neib_index, relPos)
	{}
};

struct vertex_fluid_ndata :
	common_ndata
{
	template<typename Params>
	__device__ __forceinline__
	vertex_fluid_ndata(Params const& params, uint neib_index, float4 const& relPos) :
		common_ndata(params, neib_index, relPos)
	{}
};

template<typename Params, //!< \ref sa_segment_bc_params specialization
	bool has_keps = Params::has_keps, //!< is k-epsilon enabled?
	typename keps_struct = //!< optional components if \ref has_keps
		typename COND_STRUCT(has_keps, common_keps_ndata)
	>
struct vertex_boundary_ndata :
	keps_struct
{
	// TODO further split common_ndata and use index and info from there
	uint index;
	particleinfo info;

	const vertexinfo vertices;
	const float4 normal;

	__device__ __forceinline__
	vertex_boundary_ndata(Params const& params, int neib_index)
	:
		keps_struct(params, neib_index),
		index(neib_index),
		info(tex1Dfetch(infoTex, neib_index)),
		vertices(params.vertices[neib_index]),
		normal(tex1Dfetch(boundTex, neib_index))
	{}
};

//! keps_euler_contrib: vertex neighbor contribution to eulerVel in KEPS case

// KEPS && !IO case
template<typename Params, typename PData, typename POut,
	bool contrib = Params::has_keps && !Params::has_io>
__device__ __forceinline__
enable_if_t<contrib>
keps_vertex_contrib(Params const& params, PData const& pdata, POut &pout, uint neib_index)
{
	pout.eulerVel += params.eulerVel[neib_index];
}

// KEPS && IO case
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_keps && Params::has_io>
keps_vertex_contrib(Params const& params, PData const& pdata, POut &pout, uint neib_index)
{
	if (!IO_BOUNDARY(pdata.info))
		keps_vertex_contrib<Params, PData, POut, true>(params, pdata, pout, neib_index);
}

// !KEPS case
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_keps>
keps_vertex_contrib(Params const& params, PData const& pdata, POut &pout, uint neib_index)
{ /* do nothing */ }


//! moving_vel_contrib : velocity contribution from moving body
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_moving>
moving_vertex_contrib(Params const& params, PData const& pdata, POut &pout, uint neib_index)
{
	if (MOVING(pdata.info)) {
		const float4 neib_vel = params.vel[neib_index];
		pout.vel.x += neib_vel.x;
		pout.vel.y += neib_vel.y;
		pout.vel.z += neib_vel.z;
	}
}

template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_moving>
moving_vertex_contrib(Params const& params, PData const& pdata, POut &pout, uint neib_index)
{ /* do nothing */ }

//! keps_fluid_contrib : keps contribution from fluid neighbors
template<typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_keps>
keps_fluid_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{ /* do nothing */ }

template<typename NData>
__device__ __forceinline__
float de_dn_solid(NData const& ndata)
{
	// for solid boundaries we have de/dn = c_mu^(3/4)*4*k^(3/2)/(\kappa r)
	// the constant is coming from 4*powf(0.09,0.75)/0.41
	return ndata.w*(ndata.eps + 1.603090412f*powf(ndata.tke,1.5f)/ndata.norm_dist);
}

// Standard contribution
template<typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_keps && !Params::has_io>
keps_fluid_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{
	// for all boundaries we have dk/dn = 0
	pout.sumtke += ndata.w*ndata.tke;
	pout.sumeps += de_dn_solid(ndata);
}

template<typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<Params:: has_keps && Params::has_io>
keps_fluid_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{
	// for all boundaries we have dk/dn = 0
	pout.sumtke += ndata.w*ndata.tke;
	if (IO_BOUNDARY(pdata.info))
		// and de/dn = 0
		pout.sumeps += ndata.w*ndata.eps;
	else
		pout.sumeps += de_dn_solid(ndata);
}

//! io_fluid_contrib: contribution from fluid neighbors to open boundaries
template<int step,
	typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_io>
io_fluid_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{ /* do nothing */ }

//! specialization for saSegmentBoundaryConditionsDevice
template<int step,
	typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_io && Params::cptype == PT_BOUNDARY>
io_fluid_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{
	if (IO_BOUNDARY(pdata.info)) {
		pout.sumvel += ndata.w*as_float3(ndata.vel + params.eulerVel[ndata.index]);
		// for open boundaries compute pressure interior state
		//sump += w*fmaxf(0.0f, neib_pres+dot(d_gravity, as_float3(relPos)*d_rho0[fluid_num(neib_info)]));
		pout.sump += ndata.w*fmaxf(0.0f, ndata.press);
	}
}

//! specialization for saVertexBoundaryConditionsDevice
template<int step,
	typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_io && Params::cptype == PT_VERTEX>
io_fluid_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{
	if (!IO_BOUNDARY(pdata.info))
		return;

	// TODO this is most likely not needed anymore
	pout.foundFluid = true;

	// contributions to the velocity and pressure sums are the same
	// as for segments (see other specialization)
	if (!pdata.corner) {
		pout.sumvel += ndata.w*as_float3(ndata.vel + params.eulerVel[ndata.index]);
		// for open boundaries compute pressure interior state
		//sump += w*fmaxf(0.0f, neib_pres+dot(d_gravity, as_float3(relPos)*d_rho0[fluid_num(neib_info)]));
		pout.sump += ndata.w*fmaxf(0.0f, ndata.press);
	}

	if (step == 2) {
		// check if this fluid paricle is marked for deletion
		// (which happens if any vertex is non-zero)
		const vertexinfo neibVerts = params.vertices[ndata.index];
		if ((neibVerts.x | neibVerts.y) != 0) {
			// gradient gamma was abused to store the vertex weights
			// and the original particle mass
			const float4 vertexWeights = params.gGam[ndata.index];
			const int my_id = id(pdata.info);
			const float weight = (
				neibVerts.x == my_id ? vertexWeights.x :
				neibVerts.y == my_id ? vertexWeights.y :
				neibVerts.z == my_id ? vertexWeights.z :
				0.0f);
			if (weight > 0)
				pout.massFluid += weight*vertexWeights.w;
		}
	}
}

//! k-epsilon contribution from boundary element to adjacent vertex
template<typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_keps>
keps_boundary_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{ /* do nothing */ }

template<typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_keps>
keps_boundary_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{
	pout.sumtke += ndata.tke;
	pout.sumeps += ndata.eps;
	pout.numseg += 1;
}

//! open boundary contribution from boundary element to adjacent vertex
template<typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_io>
io_boundary_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{ /* do nothing */ }

template<typename Params, typename PData, typename NData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_io>
io_boundary_contrib(Params const& params, PData const& pdata,
	NData const& ndata, POut &pout)
{
	if (pdata.corner) {
		// corner vertices only interact with solid wall segments,
		// to compute the wallNormal
		if (!IO_BOUNDARY(ndata.info))
			pout.wallNormal += make_float3(ndata.normal)*ndata.normal.w;
		// nothing else to do
		return;
	}
	// non-corner vertex: compute mass evolution contribution from
	// open boundary segments
	if (!IO_BOUNDARY(ndata.info))
		return;

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

	// Relative position of vertices with respect to the segment, normalized
	float3 vx[3];
	calcVertexRelPos(vx, ndata.normal,
		params.vertPos0[ndata.index], params.vertPos1[ndata.index], params.vertPos2[ndata.index], -1);

	float3 vertexWeights;
	getMassRepartitionFactor(vx, make_float3(ndata.normal), vertexWeights);
	const int my_id = id(pdata.info);
	const float weight =
		ndata.vertices.x == my_id ? vertexWeights.x :
		ndata.vertices.y == my_id ? vertexWeights.y :
		ndata.vertices.z == my_id ? vertexWeights.z :
		0.0f;
	pout.sumMdot += physical_density(params.vel[ndata.index].w,fluid_num(ndata.info))*ndata.normal.w*weight*
		dot3(params.eulerVel[ndata.index], ndata.normal); // the euler vel should be subtracted by the lagrangian vel which is assumed to be 0 now.
}




//! Compute contribution from adjacent BOUNDARY particles to a VERTEX boundary conditions
/**! This is the loop over PT_BOUNDARY particles for saVertexBoundaryConditionsDevice,
 * which is only needed with k-epsilon viscous model or with open boundaries
 */
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!(Params::has_keps || Params::has_io)>
vertex_boundary_loop(Params const& params, PData const& pdata, POut &pout)
{ /* do nothing */ }

template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_keps || Params::has_io>
vertex_boundary_loop(Params const& params, PData const& pdata, POut &pout)
{
	for_each_neib(PT_BOUNDARY, pdata.index, pdata.pos, pdata.gridPos,
		params.cellStart, params.neibsList)
	{
		const uint neib_index = neib_iter.neib_index();
		const sa_bc::vertex_boundary_ndata<Params> ndata(params, neib_index);

		// skip non-adjacent boundaries
		if (!has_vertex(ndata.vertices, id(pdata.info)))
			continue;

		keps_boundary_contrib(params, pdata, ndata, pout);
		io_boundary_contrib(params, pdata, ndata, pout);
	}
}

//! Determines the wall normal to use
/**! For anything but corner vertices, this is just the normal;
 * for corner vertices it's the wallNormal computed during the loop
 */
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_io, float3>
wall_normal(Params const& params, PData const& pdata, POut const& pout)
{
	return make_float3(tex1Dfetch(boundTex, pdata.index));
}
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_io, float3>
wall_normal(Params const& params, PData const& pdata, POut const& pout)
{
	if (pdata.corner)
		return normalize(pout.wallNormal);
	else
		return pdata.normal;
}

//! impose k-epsilon boundary conditions on vertices
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_keps>
impose_vertex_keps_bc(Params const& params, PData const& pdata, POut &pout)
{ /* do nothing */ }

template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_keps>
impose_vertex_keps_bc(Params const& params, PData const& pdata, POut &pout)
{
	params.tke[pdata.index] = fmax(pout.sumtke/pout.numseg, 1e-6f);
	params.eps[pdata.index] = fmax(pout.sumeps/pout.numseg, 1e-6f);

	// adjust Eulerian velocity so that it is tangential to the fixed wall
	// This is only done for vertices that do NOT belong to an open boundary
	// with imposed velocity
	// TODO FIXME in master this is only done in the k-epsilon case (if oldTKE)
	// --why?
	if (VEL_IO(pdata.info))
		return;
	const float3 normal = wall_normal(params, pdata, pout);
	float4 eulerVel = params.eulerVel[pdata.index];
	eulerVel -= make_float4(
		dot3(eulerVel, normal)*normal,
		0.0f);
	params.eulerVel[pdata.index] = eulerVel;
}

//! Copy keps data from a vertex to the fluid particle it generated
/**! Only if k-epsilon is enabled */
template<typename Params, typename PData>
__device__ __forceinline__
enable_if_t<!Params::has_keps>
clone_vertex_keps(Params const& params, PData const& pdata, int clone_idx)
{ /* do nothing */ }

template<typename Params, typename PData>
__device__ __forceinline__
enable_if_t<Params::has_keps>
clone_vertex_keps(Params const& params, PData const& pdata, int clone_idx)
{
	params.tke[clone_idx] = params.tke[pdata.index];
	params.eps[clone_idx] = params.eps[pdata.index];
}

//! generate new particles
/**! This is only done in the open boundary case and only if this is
 * the last integration step
 */
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<not (Params::has_io && Params::step == 2)>
generate_new_particles(Params const& params, PData const& pdata, float4 const& pos, POut &pout)
{ /* do nothing */ }

template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_io && Params::step == 2>
generate_new_particles(Params const& params, PData const& pdata, float4 const& pos, POut &pout)
{
	// during the second step, check whether new particles need to be created

	if (
		// create new particle if the mass of the vertex is large enough
		pos.w > pdata.refMass*0.5f &&
		// if mass flux > 0
		pout.sumMdot > 0 &&
		// if imposed velocity is greater than 0
		dot3(pdata.normal, pout.eulerVel) > 1e-5f &&
		// pressure inlets need p > 0 to create particles
		(VEL_IO(pdata.info) || pout.eulerVel.w > 1e-5f)
	   ) {
		// Create new particle
		particleinfo clone_info;
		uint clone_idx = createNewFluidParticle(
			clone_info, pdata.info,
			params.numParticles, params.newNumParticles,
			params.nextIDs + pdata.index,
			params.totParticles);

		// Problem has already checked that there is enough memory for new particles
		float4 clone_pos = pos; // new position is position of vertex particle
		clone_pos.w = pdata.refMass; // new fluid particle has reference mass
		int3 clone_gridPos = pdata.gridPos; // as the position is the same so is the grid position
		pout.massFluid -= clone_pos.w;

		// assign new values to array
		params.clonePos[clone_idx] = clone_pos;
		params.cloneInfo[clone_idx] = clone_info;
		params.cloneParticleHash[clone_idx] = calcGridHash(clone_gridPos);
		// the new velocity of the fluid particle is the eulerian velocity of the vertex
		params.vel[clone_idx] = pout.eulerVel;
		params.gGam[clone_idx] = params.gGam[pdata.index];
		// copy k-eps information,if present
		clone_vertex_keps(params, pdata, clone_idx);

		// reset everything else
		// the eulerian velocity of fluid particles is always 0
		params.eulerVel[clone_idx] = make_float4(0.0f);
		params.cloneForces[clone_idx] = make_float4(0.0f);
		params.cloneVertices[clone_idx] = make_vertexinfo(0, 0, 0, 0);
		params.nextIDs[clone_idx] = UINT_MAX;
		params.cloneBoundElems[clone_idx] = make_float4(-NAN);
		// TODO missing from the reset at the moment:
		// INTERNAL_ENERGY,
		// TURBVISC,
		// VOLUME
	}
}


//! impose boundary conditions on open boundary vertex
/**! This includs solving the Riemann problem to compute the appropriate
 * boundary conditions for velocity and pressure, as well as generating
 * new particles and absorb the ones that have gone through an open boundary
 */
template<typename Params, typename PData, typename POut, uint step = Params::step>
__device__ __forceinline__
enable_if_t<!Params::has_io>
impose_vertex_io_bc(Params const& params, PData const& pdata, POut &pout)
{ /* do nothing */ }

template<typename Params, typename PData, typename POut, uint step = Params::step>
__device__ __forceinline__
enable_if_t<Params::has_io>
impose_vertex_io_bc(Params const& params, PData const& pdata, POut &pout)
{
	/* only open boundary vertex that are not corners participate */
	if (pdata.corner || !IO_BOUNDARY(pdata.info))
		return;

	// TODO in the k-epsilon case wel'll be re-fetching what we wrote
	// during impose_vertex_keps_bc, see if it's possible to avoid this
	// (or if the compiler/hardware are smart enough to keep everything
	// cached as appropriate)
	float4 eulerVel = params.eulerVel[pdata.index];

	// note: defaults are set in the place where bcs are imposed
	if (pout.shepard_div > 0.1f*pdata.gam) {
		pout.sumvel /= pout.shepard_div;
		pout.sump /= pout.shepard_div;
		const float unInt = dot3(pout.sumvel, pdata.normal);
		const float unExt = dot3(eulerVel, pdata.normal);
		const float rhoInt = RHO(pout.sump, fluid_num(pdata.info));
		const float rhoExt = eulerVel.w;

		calculateIOboundaryCondition(eulerVel, pdata.info,
			rhoInt, rhoExt, pout.sumvel,
			unInt, unExt, make_float3(pdata.normal));
	} else {
		if (VEL_IO(pdata.info))
			eulerVel.w = 0.0f;
		else
			eulerVel = make_float4(0.0f, 0.0f, 0.0f, eulerVel.w);
	}
	params.eulerVel[pdata.index] = eulerVel;
	// the density of the particle is equal to the "eulerian density"
	params.vel[pdata.index].w = eulerVel.w;

	float4 pos = pdata.pos;

	if (step != 0) {
		// time stepping
		pos.w += params.dt*pout.sumMdot;
		// if a vertex has no fluid particles around and its mass flux is negative then set its mass to 0
		if (pout.shepard_div < 0.1f*pdata.gam && pout.sumMdot < 0.0f) // sphynx version
			//if (!pout.foundFluid && pout.sumMdot < 0.0f)
			pos.w = 0.0f;

		// clip to +/- 2 refMass all the time
		pos.w = fmaxf(-2.0f*pdata.refMass, fminf(2.0f*pdata.refMass, pos.w));

		// clip to +/- originalVertexMass if we have outflow
		// or if the normal eulerian velocity is less or equal to 0
		if (pout.sumMdot < 0.0f ||
			dot3(pdata.normal, eulerVel) < 1e-5f*d_sscoeff[fluid_num(pdata.info)])
		{
			const float weightedMass = pdata.refMass*pdata.normal.w;
			pos.w = fmaxf(-weightedMass, fminf(weightedMass, pos.w));
		}
	}
	// TODO FIXME splitneibs-merge: in master the following was here
	// as an else branch with a conditiona if (!initStep) that began with
	// the time stepping comment above. Check if it needs to go in the
	// initIOmass kernels or have a different condition
#if 0
	// particles that have an initial density less than the reference density have their mass set to 0
	// or if their velocity is initially 0
	else if (!resume &&
		( (PRES_IO(info) && eulerVel.w <= 1e-10f) ||
		  (VEL_IO(info) && length3(eulerVel) < 1e-10f*d_sscoeff[fluid_num(info)])) )
		pos.w = 0.0f;
#endif

	pout.eulerVel = eulerVel;
	generate_new_particles(params, pdata, pos, pout);

	pos.w += pout.massFluid;
	params.clonePos[pdata.index] = pos;
}



//! impose_solid_keps_bc: impose k-epsilon boundary conditions on solid walls
/** These will only be called for non-open boundaries, so they don't need
 * to check for that
 */

template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_keps || Params::repacking>
impose_solid_keps_bc(Params const& params, PData const& pdata, POut &pout)
{ /* do nothing */ }

template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_keps && !Params::repacking>
impose_solid_keps_bc(Params const& params, PData const& pdata, POut &pout)
{
	// k condition
	params.tke[pdata.index] = pout.sumtke/pout.shepard_div;
	// epsilon condition
	// for solid boundaries we have de/dn = 4 0.09^0.075 k^1.5/(0.41 r)
	params.eps[pdata.index] =
		fmaxf(pout.sumeps/pout.shepard_div,1e-5f); // eps should never be 0

	// average eulerian velocity on the wall (from associated vertices)
	pout.eulerVel /= 3;
	// ensure that velocity is normal to segment normal
	pout.eulerVel -= dot3(pout.eulerVel,pdata.normal)*pdata.normal;
	params.eulerVel[pdata.index] = pout.eulerVel;
}

//! impose_solid_eulerVel: set eulerVel to zero for solid walls
/** This is necessary only if open boundaries are enabled */
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_io || Params::repacking>
impose_solid_eulerVel(Params const& params, PData const& pdata, POut &pout)
{ /* do nothing */ }

template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_io && !Params::repacking>
impose_solid_eulerVel(Params const& params, PData const& pdata, POut &pout)
{
	params.eulerVel[pdata.index] = make_float4(0.0f);
}

//! impose_solid_bc: impose standard boundary conditions on solid walls
/** These may be called both in case where open boundaries are not enabled,
 * and for solid walls when open boundaries are enabled
 */
template<bool enabled,
	typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!enabled>
impose_solid_bc(Params const& params, PData const& pdata, POut &pout)
{ /* do nothing */ }

template<bool enabled,
	typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<enabled>
impose_solid_bc(Params const& params, PData const& pdata, POut &pout)
{
	pout.shepard_div = fmaxf(pout.shepard_div, 0.1f*pout.gGam.w); // avoid division by 0
	// density condition
	pout.vel.w = RHO(pout.sumpWall/pout.shepard_div,fluid_num(pdata.info));
	impose_solid_keps_bc(params, pdata, pout);
	impose_solid_eulerVel(params, pdata, pout);
}


//! impose_io_keps_bc: impose k-epsilon boundary conditions on open boundary
/** Note that these will only be called for open boundaries, so they don't need
 * to check for that
 */

template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_keps>
impose_io_keps_bc(Params const& params, PData const& pdata, POut &pout)
{ /* do nothing */ }

template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_keps>
impose_io_keps_bc(Params const& params, PData const& pdata, POut &pout)
{
	if (pout.shepard_div > 0.1f*pout.gGam.w) {
		if (VEL_IO(pdata.info)) {
			// for velocity imposed boundaries we impose k and epsilon
			// TODO my impression is that these are read, and then
			// written back as-is
			params.tke[pdata.index] = pout.tke;
			params.eps[pdata.index] = pout.eps;
		} else {
			// for pressure imposed boundaries we take dk/dn = de/dn = 0
			params.tke[pdata.index] = pout.sumtke/pout.shepard_div;
			params.eps[pdata.index] = pout.sumeps/pout.shepard_div;
		}
	} else {
		params.tke[pdata.index] = params.eps[pdata.index] = 1e-6f;
	}
}

//! impose_io_bc: Impose boundary conditions on open boundary

/* Simple case: no IO */
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<!Params::has_io>
impose_io_bc(Params const& params, PData const& pdata, POut &pout)
{ /* do nothing */ }

/* IO case */
template<typename Params, typename PData, typename POut>
__device__ __forceinline__
enable_if_t<Params::has_io>
impose_io_bc(Params const& params, PData const& pdata, POut &pout)
{
	if (IO_BOUNDARY(pdata.info)) {
		impose_io_keps_bc(params, pdata, pout);

		if (pout.shepard_div > 0.1f*pout.gGam.w) { // note: defaults are set in the place where bcs are imposed
			pout.sumvel /= pout.shepard_div;
			pout.sump /= pout.shepard_div;
			pout.vel.w = RHO(pout.sump, fluid_num(pdata.info));
			// TODO simplify branching
			if (!VEL_IO(pdata.info)) {
				params.eulerVel[pdata.index] = make_float4(0.0f);
			}
		} else {
			pout.sump = 0.0f;
			if (VEL_IO(pdata.info)) {
				pout.sumvel = as_float3(pout.eulerVel);
				pout.vel.w = 0.0f;
			} else {
				pout.sumvel = make_float3(0.0f);
				// TODO FIXME this is the logic in master, but there's something odd about this,
				// cfr assignments below [*]
				pout.vel.w = params.eulerVel[pdata.index].w;
				params.eulerVel[pdata.index] = make_float4(0.0f, 0.0f, 0.0f, pout.vel.w);
			}
		}

		// compute Riemann invariants for open boundaries
		const float unInt = dot3(pout.sumvel, pdata.normal);
		const float unExt = dot3(pout.eulerVel, pdata.normal);
		const float rhoInt = pout.vel.w;
		const float rhoExt = pout.eulerVel.w;

		calculateIOboundaryCondition(pout.eulerVel, pdata.info, rhoInt, rhoExt, pout.sumvel, unInt, unExt, as_float3(pdata.normal));

		// TODO FIXME cfr assignes above [*]
		params.eulerVel[pdata.index] = pout.eulerVel;
		// the density of the particle is equal to the "eulerian density"
		pout.vel.w = pout.eulerVel.w;
	} else {
		impose_solid_bc<true>(params, pdata, pout);
		return;
	}

}


} // namespace sa_bc

//! Computes the boundary condition on segments for SA boundaries
/*!
 This function computes the boundary condition for density/pressure on segments if the SA boundary type
 is selected. It does this not only for solid wall boundaries but also open boundaries.
 \note updates are made in-place because we only read from fluids and vertex particles and only write
 boundary particles data, and no conflict can thus occurr.
*/
template<typename Params,
	int step = Params::step,
	bool initStep = (step <= 0) // handle both step 0 (initialization) and -1 (reinit after repacking)
>
__global__ void
saSegmentBoundaryConditionsDevice(Params params)
{

	using namespace sa_bc;

	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	const particleinfo info = tex1Dfetch(infoTex, index);

	if (!BOUNDARY(info))
		return;

	const sa_bc::segment_pdata	pdata(params, index, info);
	sa_bc::segment_pout<Params>	pout(params, index, info);

	// Loop over VERTEX neighbors.
	// TODO this is only needed
	// (1) to compute gamma (i.e. init step or moving objects if they get too close)
	// (2) to compute the velocity for boundary of moving objects
	// (3) to compute the eulerian velocity for non-IO boundaries in the KEPS case
	// TODO skip this whole block (loop + calcGam update) when not needed, if possible
	// at compile time
	{
		for_each_neib(PT_VERTEX, index, pdata.pos, pdata.gridPos, params.cellStart, params.neibsList) {
			const uint neib_index = neib_iter.neib_index();

			// Compute relative position vector and distance
			const float4 relPos = neib_iter.relPos(params.pos[neib_index]);

			// skip inactive particles
			if (INACTIVE(relPos))
				continue;

			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			if (has_vertex(pdata.verts, id(neib_info))) {
				moving_vertex_contrib(params, pdata, pout, neib_index);

				if (pout.calcGam)
					pout.gGam += params.gGam[neib_index];

				keps_vertex_contrib(params, pdata, pout, neib_index);
			}
		}

		// finalize gamma computation and store it
		if (pout.calcGam) {
			pout.gGam /= 3;
			params.gGam[index] = pout.gGam;
			pout.gGam.w = fmaxf(pout.gGam.w, 1e-5f);
		}
	}

	// finalize velocity computation. we only store it later though, because the rest of this
	// kernel may compute vel.w
	pout.vel.x /= 3;
	pout.vel.y /= 3;
	pout.vel.z /= 3;

	// Loop over FLUID neighbors
	// This is needed:
	// (0) to compute sumPwall _always_
	// (1) k-epsilon in TKE case
	// (2) in IO case, to compute velocity and pressure against wall
	// The contributions are factored out and enabled only when needed
	for_each_neib(PT_FLUID, index, pdata.pos, pdata.gridPos,
		params.cellStart, params.neibsList) {
		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos(params.pos[neib_index]);

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const segment_ndata<Params> ndata(params, pdata, neib_index, relPos);

		if ( !(ndata.r < params.influenceradius && dot3(pdata.normal, relPos) < 0.0f) )
			continue;

		pout.sumpWall += fmax(ndata.press + physical_density(ndata.vel.w,fluid_num(pdata.info))*dot3(d_gravity, relPos), 0.0f)*ndata.w;

		keps_fluid_contrib(params, pdata, ndata, pout);

		io_fluid_contrib<step>(params, pdata, ndata, pout);

		pout.shepard_div += ndata.w;
	}

	impose_io_bc(params, pdata, pout);
	// impose solid-wall boundary conditions if open boundaries are not enabled
	// (if they are enabled, solid-wall BC are imposed in impose_io_bc
	impose_solid_bc<!Params::has_io>(params, pdata, pout);

	// store recomputed velocity + pressure
	params.vel[index] = pout.vel;

	// TODO FIXME splitneibs merge: master here had the code for FLUID particles moving through IO
	// segments
}

//! Computes the boundary condition on segments for SA boundaries during repacking
/*!
 This function computes the boundary condition for density/pressure on segments if the SA boundary type
 is selected.
 \note updates are made in-place because we only read from fluids and vertex particles and only write
 boundary particles data, and no conflict can thus occurr.
*/
template<typename Params,
	int step = Params::step,
	bool initStep = (step <= 0) // handle both step 0 (initialization) and -1 (reinit after repacking)
>
__global__ void
saSegmentBoundaryConditionsRepackDevice(Params params)
{

	using namespace sa_bc;

	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	const particleinfo info = tex1Dfetch(infoTex, index);

	if (!BOUNDARY(info))
		return;

	const sa_bc::segment_pdata	pdata(params, index, info);
	sa_bc::segment_pout<Params>	pout(params, index, info);

	// Loop over VERTEX neighbors.
	// TODO this is only needed
	// (1) to compute gamma (i.e. init step or moving objects if they get too close)
	// (2) to compute the velocity for boundary of moving objects
	// (3) to compute the eulerian velocity for non-IO boundaries in the KEPS case
	// TODO skip this whole block (loop + calcGam update) when not needed, if possible
	// at compile time
	{
		for_each_neib(PT_VERTEX, index, pdata.pos, pdata.gridPos, params.cellStart, params.neibsList) {
			const uint neib_index = neib_iter.neib_index();

			// Compute relative position vector and distance
			const float4 relPos = neib_iter.relPos(params.pos[neib_index]);

			// skip inactive particles
			if (INACTIVE(relPos))
				continue;

			const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);

			if (has_vertex(pdata.verts, id(neib_info))) {

				if (pout.calcGam)
					pout.gGam += params.gGam[neib_index];
			}
		}

		// finalize gamma computation and store it
		if (pout.calcGam) {
			pout.gGam /= 3;
			params.gGam[index] = pout.gGam;
			pout.gGam.w = fmaxf(pout.gGam.w, 1e-5f);
		}
	}

	// Loop over FLUID neighbors
	// This is needed:
	// (0) to compute sumPwall _always_
	// (1) k-epsilon in TKE case
	// (2) in IO case, to compute velocity and pressure against wall
	// The contributions are factored out and enabled only when needed
	for_each_neib(PT_FLUID, index, pdata.pos, pdata.gridPos,
		params.cellStart, params.neibsList) {
		const uint neib_index = neib_iter.neib_index();

		// Compute relative position vector and distance
		// Now relPos is a float4 and neib mass is stored in relPos.w
		const float4 relPos = neib_iter.relPos(params.pos[neib_index]);

		// skip inactive particles
		if (INACTIVE(relPos))
			continue;

		const segment_ndata<Params> ndata(params, pdata, neib_index, relPos);

		if ( !(ndata.r < params.influenceradius && dot3(pdata.normal, relPos) < 0.0f) )
			continue;

		pout.sumpWall += fmax(ndata.press + physical_density(ndata.vel.w,fluid_num(pdata.info))*dot3(d_gravity, relPos), 0.0f)*ndata.w;

		pout.shepard_div += ndata.w;
	}

	// impose solid-wall boundary conditions if open boundaries are not enabled
	// (if they are enabled, solid-wall BC are imposed in impose_io_bc
	impose_solid_bc<!Params::has_io || Params::repacking>(params, pdata, pout);

	// store recomputed velocity + pressure
	params.vel[index] = pout.vel;

}
/// Mark fluid particles that have crossed an open boundary
/** For each fluid particle, detect if it has crossed an open boundary and
 * identify the boundary element that it moved through. The vertices of this element
 * will later be used to redistribute the mass of the gone particle.
 * We abuse the vertex info (which is usually empty for fluid particles) to store the
 * relevant information, and abuse gGam to store the relative weights of the vertices
 * for mass repartition (which is fine because the outgoing particle will be disabled
 * anyway).
 */
template<KernelType kerneltype>
__global__ void
findOutgoingSegmentDevice(
	const	float4		* __restrict__	posArray,
	const	float4		* __restrict__	velArray,
			vertexinfo	* __restrict__	vertices,
			float4		* __restrict__	gGam,
	const	float2		* __restrict__	vertPos0,
	const	float2		* __restrict__	vertPos1,
	const	float2		* __restrict__	vertPos2,
	const	hashKey		* __restrict__	particleHash,
	const	uint		* __restrict__	cellStart,
	const	neibdata	* __restrict__	neibsList,
			uint						numParticles,
			float						influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	const particleinfo info = tex1Dfetch(infoTex, index);

	if (!FLUID(info))
		return;

	const float4 pos = posArray[index];
	if (INACTIVE(pos))
		return;

#if 1
	{
		/* Check that we aren't re-processing a fluid particle that was
		 * already processed. Note that this shouldn't happen!
		 */
		vertexinfo vert = vertices[index];
		if (vert.x | vert.y) {
			printf("%d (id %d, flags %d) already had vertices (%d, %d)\n",
				index, id(info), PART_FLAGS(info), vert.x, vert.y);
			return;
		}
	}
#endif

	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );
	const float4 vel = velArray[index];

	// Data about closest “past” boundary element so far:
	// squared distance to closest boundary so far
	float r2_min = influenceradius*influenceradius;
	// index of the closest boundary so far
	uint index_min = UINT_MAX;
	// normal of the closest boundary so far
	float3 normal_min = make_float3(0.0f);
	// relPos to the closest boundary so far
	float3 relPos_min = make_float3(0.0f);

	for_each_neib(PT_BOUNDARY, index, pos, gridPos, cellStart, neibsList) {
		const uint neib_index = neib_iter.neib_index();
		const particleinfo neib_info = tex1Dfetch(infoTex, neib_index);
		if (!IO_BOUNDARY(neib_info))
			continue; // we only care about IO boundary elements

		const float4 relPos = neib_iter.relPos(posArray[neib_index]);
		const float4 normal = tex1Dfetch(boundTex, neib_index);
		const float3 relVel = as_float3(vel - velArray[neib_index]);
		const float r2 = sqlength3(relPos);

		if (	r2 < r2_min && // we are closer to this element than other elements
			dot3(normal, relPos) <= 0.0f && // we are behind the boundary element
			dot3(normal, relVel) < 0.0f ) // we are moving outwards relative to this element
		{
			r2_min = r2; // new minimum distance
			index_min = neib_index;
			normal_min = make_float3(normal);
			relPos_min = make_float3(relPos);
		}

	}

	// No crossed segment, nothing to do
	if (index_min == UINT_MAX)
		return;

	// Vertex coordinates in the local system
	float3 vx[3];
	calcVertexRelPos(vx, normal_min,
		vertPos0[index_min], vertPos1[index_min], vertPos2[index_min], 1);
	// calcVertexRelPos computes the relative position to the barycenter,
	// we want the one relative to the fluid particle. Fixup:
	vx[0] = relPos_min - vx[0];
	vx[1] = relPos_min - vx[1];
	vx[2] = relPos_min - vx[2];

	float4 vertexWeights;
	getMassRepartitionFactor(vx, normal_min, vertexWeights);
	// preserve the mass, since invalidation of the particle will destroy it
	vertexWeights.w = pos.w;

	// vertex is normally empty for fluid particles, use it to store
	// the vertices of the boundary element (avoiding a neighbor-of-neighbor search
	// later on)
	vertices[index] = vertices[index_min];

	// abuse gamma to store the vertexWeights: since the particle will be disabled,
	// this should not affect computation
	gGam[index] = vertexWeights;
}

/// Normal computation for vertices in the initialization phase
/*! Computes a normal for vertices in the initialization phase. This normal is used in the forces
 *	computation so that gamma can be appropriately calculated for vertices, i.e. particles on a boundary.
 *	\param[out] newGGam : vertex normal vector is computed
 *	\param[in] vertices : pointer to boundary vertices table
 *	\param[in] vertIDToIndex : pointer that associated a vertex id with an array index
 *	\param[in] pinfo : pointer to particle info
 *	\param[in] particleHash : pointer to particle hash
 *	\param[in] cellStart : pointer to indices of first particle in cells
 *	\param[in] neibsList : neighbour list
 *	\param[in] numParticles : number of particles
 */
template<KernelType kerneltype>
__global__ void
computeVertexNormalDevice(
						float4*			boundelement,
				const	vertexinfo*		vertices,
				const	particleinfo*	pinfo,
				const	hashKey*		particleHash,
				const	uint*			cellStart,
				const	neibdata*		neibsList,
				const	uint			numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles
	const particleinfo info = pinfo[index];
	if (!VERTEX(info))
		return;

	float4 pos = make_float4(0.0f);
	uint our_id = id(info);

	// Average norm used in the initial step to compute grad gamma for vertex particles
	// During the simulation this is used for open boundaries to determine whether particles are created
	// For all other boundaries in the keps case this is the average normal of all non-open boundaries used to ensure that the
	// Eulerian velocity is only normal to the fixed wall
	float3 avgNorm = make_float3(0.0f);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Loop over all BOUNDARY neighbors
	for_each_neib(PT_BOUNDARY, index, pos, gridPos, cellStart, neibsList) {
		const uint neib_index = neib_iter.neib_index();
		const particleinfo neib_info = pinfo[neib_index];

		// Skip this neighboring boundary element if it's not in the same boundary
		// classification as us, i.e. if it's an IO boundary element and we are not
		// an IO vertex, or if the boundary element is not IO and we are an IO vertex.
		// The check is done by negating IO_BOUNDARY because IO_BOUNDARY returns
		// the combination of FG_INLET and FG_OUTLET pertaining to the particle,
		// and we don't care about that aspect, we only care about IO vs non-IO
		if (!IO_BOUNDARY(info) != !IO_BOUNDARY(neib_info))
			continue;

		const vertexinfo neib_verts = vertices[neib_index];
		const float4 boundElement = boundelement[neib_index];

		// check if vertex is associated with this segment
		if (has_vertex(neib_verts, our_id)) {
			// in the initial step we need to compute an approximate grad gamma direction
			// for the computation of gamma, in general we need a sort of normal as well
			// for open boundaries to decide whether or not particles are created at a
			// vertex or not, finally for k-epsilon we need the normal to ensure that the
			// velocity in the wall obeys v.n = 0
			avgNorm += as_float3(boundElement)*boundElement.w;
		}
	}

	// normalize average norm. The .w component for vertices is not used
	boundelement[index] = make_float4(normalize(avgNorm), NAN);
}

/// Variables needed by both gradGamma() and Gamma() during gamma initialization
struct InitGammaVars {
	bool skip;
	float3 normal; /* neighbor normal */
	float3 q_vb[3]; /* normalized relative positions of vertices with respect to segment */
	float3 q;

	/// Prepare the variables needed by both gradGamma() and Gamma() during gamma initialization
	template<typename NeibListIterator>
	__device__
	InitGammaVars(
		NeibListIterator const& neib_iter,
		const	float4*	oldPos,
		const	float4*	boundElement,
		const	float2*	vertPos0,
		const	float2*	vertPos1,
		const	float2*	vertPos2,
		const	float	slength,
		const	float	influenceradius,
		const	float	deltap) :
		skip(false)
	{
		const uint neib_index = neib_iter.neib_index();

		const float3 relPos = as_float3(neib_iter.relPos(oldPos[neib_index]));

		if (length(relPos) > influenceradius + deltap*0.5f) {
			skip = true;
			return;
		}

		normal = as_float3(boundElement[neib_index]);
		q = relPos/slength;

		calcVertexRelPos(q_vb, normal,
			vertPos0[neib_index], vertPos1[neib_index], vertPos2[neib_index], slength);
	}
};



/// Initializes gamma using quadrature formula
/*! In the dynamic gamma case gamma is computed using a transport equation. Thus an initial value needs
 *	to be computed. In this kernel this value is determined using a numerical integration. As this integration
 *	has its problem when particles are close to the wall, it's not useful with open boundaries, but at the
 *	initial time-step particles should be far enough away.
 *	\param[out] newGGam : vertex normal vector is computed
 *	\param[in] oldPos : particle positions
 *	\param[in] boundElement : pointer to vertex & segment normals
 *	\param[in] pinfo : pointer to particle info
 *	\param[in] particleHash : pointer to particle hash
 *	\param[in] cellStart : pointer to indices of first particle in cells
 *	\param[in] neibsList : neighbour list
 *	\param[in] slength : smoothing length
 *	\param[in] influenceradius : kernel radius
 *	\param[in] deltap : particle size
 *	\param[in] epsilon : numerical epsilon
 *	\param[in] numParticles : number of particles
 */
template<KernelType kerneltype, ParticleType cptype>
__global__ void
initGammaDevice(
						float4*			newGGam,
				const	float4*			oldGGam,
				const	float4*			oldPos,
				const	float4*			boundElement,
				const	float2*			vertPos0,
				const	float2*			vertPos1,
				const	float2*			vertPos2,
				const	particleinfo*	pinfo,
				const	hashKey*		particleHash,
				const	uint*			cellStart,
				const	neibdata*		neibsList,
				const	float			slength,
				const	float			influenceradius,
				const	float			deltap,
				const	float			epsilon,
				const	uint			numParticles)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles
	const particleinfo info = pinfo[index];
	if (PART_TYPE(info) != cptype)
		return;

	float4 pos = oldPos[index];

	// gamma that is to be computed
	float gam = 1.0f;
	// grad gamma
	float3 gGam = make_float3(0.0f);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// We compute both Gamma and gradGamma, but Gamma needs the direction of gradGamma,
	// so we need to first compute gradGamma fully, and then compute Gamma.
	// This requires two cycles over the neighbors, but since this function is only called
	// once during initialization, the overall performance impact on the whole simulation
	// is low.
	// The only real downside is that the body of the two loops over the neighbors
	// is nearly exactly the same, with only the final two lines being slightly different.
	// The preparation is thus refactored in the constructor of the structure that holds
	// the common variables.


	// Iterate over all BOUNDARY neighbors to compute the gamma gradient
	for_each_neib(PT_BOUNDARY, index, pos, gridPos, cellStart, neibsList) {
		const InitGammaVars gVar(neib_iter, oldPos, boundElement,
			vertPos0, vertPos1, vertPos2,
			slength, influenceradius, deltap);
		if (gVar.skip)
			continue;

		const float ggamma_as = gradGamma<kerneltype>(slength, gVar.q, gVar.q_vb, gVar.normal);
		gGam += ggamma_as*gVar.normal;
	}

	// Iterate over all BOUNDARY neighbors to compute gamma
	for_each_neib(PT_BOUNDARY, index, pos, gridPos, cellStart, neibsList) {
		const InitGammaVars gVar(neib_iter, oldPos, boundElement,
			vertPos0, vertPos1, vertPos2,
			slength, influenceradius, deltap);
		if (gVar.skip)
			continue;

		/* these need to be mutable for gamma */
		float3 q_vb[3] = { gVar.q_vb[0], gVar.q_vb[1], gVar.q_vb[2] };

		const float gamma_as = Gamma<kerneltype, cptype>(slength, gVar.q, q_vb, gVar.normal,
					gGam, epsilon);
		gam -= gamma_as;
	}

//	if (cptype == PT_FLUID && newGGam[index].w == newGGam[index].w)
//		newGGam[index] = oldGGam[index];
//	else
	newGGam[index] = make_float4(gGam.x, gGam.y, gGam.z, gam);
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
__launch_bounds__(BLOCK_SIZE_SA_BOUND, MIN_BLOCKS_SA_BOUND)
initIOmass_vertexCountDevice(
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
	uint vertexCount = 0;

	const float4 pos = make_float4(0.0f); // we don't need pos, so let's just set it to 0
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	uint neibVertIds[MAXNEIBVERTS];
	uint neibVertIdsCount=0;

	// Loop over all BOUNDARY neighbors
	for_each_neib(PT_BOUNDARY, index, pos, gridPos, cellStart, neibsList) {
		const uint neib_index = neib_iter.neib_index();

		const particleinfo neib_info = pinfo[neib_index];

		// only IO boundary neighbours as we need to count the vertices that belong to the same segment as our vertex particle
		if (IO_BOUNDARY(neib_info)) {

			// prepare ids of neib vertices
			const vertexinfo neibVerts = vertices[neib_index];

			// only check adjacent boundaries
			if (has_vertex(neibVerts, id(info))) {
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

	// Loop over all VERTEX neighbors
	for_each_neib(PT_VERTEX, index, pos, gridPos, cellStart, neibsList) {
		const uint neib_index = neib_iter.neib_index();

		const particleinfo neib_info = pinfo[neib_index];

		for (uint j = 0; j<neibVertIdsCount; j++) {
			if (id(neib_info) == neibVertIds[j] && !CORNER(neib_info))
				vertexCount += 1;
		}
	}

	forces[index].w = (float)(vertexCount);
}

template<KernelType kerneltype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SA_BOUND, MIN_BLOCKS_SA_BOUND)
initIOmassDevice(
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

	// Loop over all BOUNDARY neighbors
	for_each_neib(PT_BOUNDARY, index, pos, gridPos, cellStart, neibsList) {
		const uint neib_index = neib_iter.neib_index();

		const particleinfo neib_info = pinfo[neib_index];

		// only IO boundary neighbours as we need to count the vertices that belong to the same segment as our vertex particle
		if (!IO_BOUNDARY(neib_info))
			continue;

		// prepare ids of neib vertices
		const vertexinfo neibVerts = vertices[neib_index];

		// only check adjacent boundaries
		if (has_vertex(neibVerts, id(info))) {
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

	// Loop over all VERTEX neighbors
	for_each_neib(PT_VERTEX, index, pos, gridPos, cellStart, neibsList) {
		const uint neib_index = neib_iter.neib_index();

		const particleinfo neib_info = pinfo[neib_index];

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

/// Compute boundary conditions for vertex particles in the semi-analytical boundary case
/*! This function determines the physical properties of vertex particles in the
 * semi-analytical boundary case. The properties of fluid particles are used to
 * compute the properties of the vertices. Due to this most arrays are read
 * from (the fluid info) and written to (the vertex info) simultaneously inside
 * this function. In the case of open boundaries the vertex mass is updated in
 * this routine and new fluid particles are created on demand. Additionally,
 * the mass of outgoing fluid particles is redistributed to vertex particles
 * herein.
 */
template<typename Params,
	int step = Params::step,
	bool initStep = (step <= 0), // handle both step 0 (initialization) and -1 (reinit after repacking)
	bool lastStep = (step == 2)
>
__global__ void
saVertexBoundaryConditionsDevice(Params params)
{
	using namespace sa_bc;

	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (!VERTEX(info))
		return;

	const sa_bc::vertex_pdata<Params> pdata(params, index, info);
	sa_bc::vertex_pout<Params>	pout(params, index, info);

	// Loop over all FLUID neighbors
	for_each_neib(PT_FLUID, index, pdata.pos, pdata.gridPos, params.cellStart, params.neibsList) {
		const uint neib_index = neib_iter.neib_index();

		const float4 relPos = neib_iter.relPos(params.pos[neib_index]);

		if (INACTIVE(relPos))
			continue;

		const sa_bc::vertex_fluid_ndata ndata(params, neib_index, relPos);

		if (ndata.r < params.influenceradius) {
			pout.sumpWall += fmax(ndata.press + physical_density(ndata.vel.w,fluid_num(pdata.info))*dot3(d_gravity, relPos), 0.0f)*ndata.w;
			pout.shepard_div += ndata.w;

			io_fluid_contrib<step>(params, pdata, ndata, pout);
		}

	}

	// Loop over all BOUNDARY neighbors, if necessary
	vertex_boundary_loop(params, pdata, pout);

	pout.shepard_div = fmax(pout.shepard_div, 0.1f*pdata.gam); // avoid division by 0

	// standard boundary condition. For open boundaries, this may get overwritten
	// further down in impose_vertex_io_bc
	// TODO compute, and only write once
	params.vel[index].w = RHO(pout.sumpWall/pout.shepard_div,fluid_num(info));

	// impose the k-epsilon boundary conditions, if any
	impose_vertex_keps_bc(params, pdata, pout);

	// impose boundary conditions on open boundary vertex, if appropriate,
	// including generation of new fluid particles and destruction of fluid
	// particles that have been marked by the preceding findOutgoingSegmentDevice
	// invokation
	impose_vertex_io_bc(params, pdata, pout);
}

/// Compute boundary conditions for vertex particles in the semi-analytical boundary case
/*! This function determines the physical properties of vertex particles in the
 * semi-analytical boundary case. The properties of fluid particles are used to
 * compute the properties of the vertices. Due to this most arrays are read
 * from (the fluid info) and written to (the vertex info) simultaneously inside
 * this function. In the case of open boundaries the vertex mass is updated in
 * this routine and new fluid particles are created on demand. Additionally,
 * the mass of outgoing fluid particles is redistributed to vertex particles
 * herein.
 */
template<typename Params,
	int step = Params::step,
	bool initStep = (step <= 0), // handle both step 0 (initialization) and -1 (reinit after repacking)
	bool lastStep = (step == 2)
>
__global__ void
saVertexBoundaryConditionsRepackDevice(Params params)
{
	using namespace sa_bc;

	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= params.numParticles)
		return;

	// read particle data from sorted arrays
	// kernel is only run for vertex particles
	const particleinfo info = tex1Dfetch(infoTex, index);
	if (!VERTEX(info))
		return;

	const sa_bc::vertex_pdata<Params> pdata(params, index, info);
	sa_bc::vertex_pout<Params>	pout(params, index, info);

	// Loop over all FLUID neighbors
	for_each_neib(PT_FLUID, index, pdata.pos, pdata.gridPos, params.cellStart, params.neibsList) {
		const uint neib_index = neib_iter.neib_index();

		const float4 relPos = neib_iter.relPos(params.pos[neib_index]);

		if (INACTIVE(relPos))
			continue;

		const sa_bc::vertex_fluid_ndata ndata(params, neib_index, relPos);

		if (ndata.r < params.influenceradius) {
			pout.sumpWall += fmax(ndata.press + physical_density(ndata.vel.w,fluid_num(pdata.info))*dot3(d_gravity, relPos), 0.0f)*ndata.w;
			pout.shepard_div += ndata.w;
		}
	}

	pout.shepard_div = fmax(pout.shepard_div, 0.1f*pdata.gam); // avoid division by 0

	// standard boundary condition
	params.vel[index].w = RHO(pout.sumpWall/pout.shepard_div,fluid_num(info));

}
//! Identify corner vertices on open boundaries
/*!
 Corner vertices are vertices that have segments that are not part of an open boundary. These
 vertices are treated slightly different when imposing the boundary conditions during the
 computation in saVertexBoundaryConditions.
*/
__global__ void
saIdentifyCornerVerticesDevice(
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

	// Loop over all BOUNDARY neighbors
	for_each_neib(PT_BOUNDARY, index, pos, gridPos, cellStart, neibsList) {
		const uint neib_index = neib_iter.neib_index();

		const particleinfo neib_info = pinfo[neib_index];
		const uint neib_obj = object(neib_info);

		// loop only over boundary elements that are not of the same open boundary
		if (!(obj == neib_obj && IO_BOUNDARY(neib_info))) {
			// check if the current vertex is part of the vertices of the segment
			if (has_vertex(vertices[neib_index], id(info))) {
				SET_FLAG(info, FG_CORNER);
				pinfo[index] = info;
				break;
			}
		}
	}
}

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
				if ((vertices.x | vertices.y) != 0) {
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

//! This kernel computes the pressure for DUMMY_BOUNDARY by using Eq.(27) of Adami et al. (2012).
//! It also computes the velocity for the no-slip boundary conditions (Eq. (22) and (23))
template<KernelType kerneltype>
__global__ void
__launch_bounds__(BLOCK_SIZE_SHEPARD, MIN_BLOCKS_SHEPARD)
ComputeDummyParticlesDevice(
	const	float4*	__restrict__ posArray,
		float4*	__restrict__ velArray,
		float4*	__restrict__ dummyVelArray,
			float4*	__restrict__ volArray, // only for SPH_GRENIER
	const	particleinfo* __restrict__ infoArray,
	const	hashKey* __restrict__ particleHash,
	const	neibdata* __restrict__ neibsList,
	const	uint* __restrict__ cellStart,
				const uint		numParticles,
				const float		slength,
				const float		influenceradius)
{
	const uint index = INTMUL(blockIdx.x,blockDim.x) + threadIdx.x;

	if (index >= numParticles)
		return;

	const particleinfo info = infoArray[index];
	const float4 pos = posArray[index];

	// We only operate on active boundary particles
	if (! (ACTIVE(pos) && BOUNDARY(info)) )
		return;

	float4 vel = make_float4(0.0f);

	// Compute grid position of current particle
	const int3 gridPos = calcGridPosFromParticleHash( particleHash[index] );

	// Compute g - a_w where a_w is the particle acceleration
	// TODO FIXME change for moving bodies
	const float3 accel_delta = d_gravity;

	//const float3 accel_delta = d_gravity - compute_body_particle_accel(info, gridPos, make_float3(pos));

	// Persistent variables across getNeibData calls
	float3 pos_corr;

	// Loop over all the neighbors, and you should pay attention to the followings
	// 1, neighbors are not wall particles
	// 2, we need to compute the neighbor's pressure 
	// 3, pressure is normalized by the kernel function

	//kahan summation
	float4 c_vel = make_float4(0.0);
	float c_norm_f = 0.0;


	float norm_factor = 0.0f; // Shepard normalization factor (denominator)
	// Loop only over FLUID neighbors
	for_each_neib(PT_FLUID, index, pos, gridPos, cellStart, neibsList) {

		const uint neib_index = neib_iter.neib_index();

		const float4 relPos = pos_corr - posArray[neib_index];
		const particleinfo neib_info = infoArray[neib_index];

		// Skip inactive neighbors and neighbors which are not FLUID
		if (INACTIVE(relPos) || NOT_FLUID(neib_info) || IO_BOUNDARY(neib_info))
			continue;

		const float r = length3(relPos);

		// skip particles not in influence radius
		if (r >= influenceradius)
			continue;


		const float4 neib_vel = velArray[neib_index];

		const float neib_pressure = P(neib_vel.w, fluid_num(neib_info)); // neib_vel.w = rho_tilde

		const float w = W<kerneltype>(r, slength);

		//kahan summation // TODO FIXME use already implemented functions
		float w_for_kahan = w;
		w_for_kahan -= c_norm_f;
		float t_norm_f = norm_factor + w_for_kahan;
		c_norm_f = t_norm_f - norm_factor - w_for_kahan;
		norm_factor = t_norm_f;

		// the .xyz components are just the sum of the weighted velocities,
		// the .w component is the smoothed pressure, which is computed as:
		// (neib_pressure + neib_rho*dot(gravity, as_float3(relPos)))*w
		// For convenience, we achieve this by multiply neib_vel.w by dot(g, relPos)
		// and adding neib_pressure, so that the shep_vel_P can be obtained with a simple
		// vectorized increment:
		float4 neib_contrib = make_float4(
			neib_vel.x, neib_vel.y, neib_vel.z,
			neib_pressure + physical_density(neib_vel.w, fluid_num(neib_info))*dot(accel_delta, as_float3(relPos)));
		neib_contrib *= w;

		//kahan summation // TODO FIXME use already implemented functions
		neib_contrib -= c_vel;
		float4 t_vel = vel + neib_contrib;
		c_vel = t_vel - vel - neib_contrib;
		vel = t_vel;
	}



	// TODO add hydrostatic pressure



	// Normalize the pressure and the velocity, but only if we actually
	// had neighbors (norm_factor != 0)
	if (norm_factor)
		vel /= norm_factor;

	// now vel.w has the pressure, but we actuall want the density there, so:
	const float rho = RHO(vel.w, fluid_num(info)); // returns rho_tilde

	// finally, the velocity of the particle should actually be 2*v_w - <v_f>
	// where -<v_f> is the opposite of the smoothed velocity of the fluid, which we
	// have now in vel, and v_w is the velocity of the wall at the particle position,
	// which is actually what we have in velArray

	//const float4 wall_vel = velArray[index];
	const float4 wall_vel = make_float4(0,0,0,0); //TODO assing the actual wall velocity

	float4 new_vel;

	new_vel = make_float4(
			2*wall_vel.x - vel.x,
			2*wall_vel.y - vel.y,
			2*wall_vel.z - vel.z,
			vel.w);

	dummyVelArray[index] = new_vel;
	velArray[index].w = rho;
/*
	if (sph_formulation == SPH_GRENIER) {
		float4 vol = volArray[index];
		vol.w = pos.w/vel.w;
		// vol.y = log(vol.w/vol.x);
		volArray[index] = vol;
	}*/
}

/** @} */

} // namespace cubounds

#endif
