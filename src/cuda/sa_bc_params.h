/*  Copyright (c) 2018-2019 INGV, EDF, UniCT, JHU

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

#ifndef _SA_SEGMENT_BC_PARAMS_H
#define _SA_SEGMENT_BC_PARAMS_H

#include "particledefine.h"
#include "simflags.h"

/* Templatized parameters structure for the SA_BOUNDARY boundary conditions kernels
 * (saSegmentBoundaryConditionsDevice, saVertexBoundaryConditionsDevice
 */

#include "neibs_list_params.h"

/**! Parameters common to all \ref saSegmentBoundaryConditionsDevice and
 * \ref saVertexBoundaryConditionsDevice specializations
 */
struct common_sa_bc_params :
	//! Position, mass, info, particle hash, cellStart, neighbors list
	neibs_list_params,
	//! Boundary elements normals
	boundelements_wrapper,
	//! Barycentric vertex positions
	vertPos_params<false>
{
	//! Indices of the vertices neighboring each boundary element
	const	vertexinfo*	__restrict__ vertices;

	//! Particle velocity and density. Both saSegmentBoundaryConditionsDevice and saVertexBoundaryConditionsDevice
	//! will update the densities to match the fluid pressure
			float4	* __restrict__ vel;

	//! Gamma and its gradient. Will be updated during initialization
	//! and in the case of open boundaries and moving objects.
	//! TODO FIXME make const otherwise
			float4	* __restrict__ gGam;

	// TODO these should probably go into constant memory
	const	float	deltap; //! Inter-particle distance

	// Constructor / initializer
	common_sa_bc_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint	_numParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius)
	:
		neibs_list_params(bufread, _numParticles, _slength, _influenceradius),
		boundelements_wrapper(bufread),
		vertPos_params<false>(bufread),
		vel(bufwrite.getData<BUFFER_VEL>()),
		gGam(bufwrite.getData<BUFFER_GRADGAMMA>()),
		vertices(bufread.getData<BUFFER_VERTICES>()),
		deltap(_deltap)
	{}
};

//! Eulerian velocity array, needed in case of k-epsilon viscosity
//! or open boundaries
struct eulervel_sa_bc_params
{
	float4 * __restrict__ eulerVel;

	eulervel_sa_bc_params(float4 * _eulerVel) :
		eulerVel(_eulerVel)
	{}
	eulervel_sa_bc_params(BufferList& bufwrite) :
		eulerVel(bufwrite.getData<BUFFER_EULERVEL>())
	{}
};

//! Parameters needed by the \ref saSegmentBoundaryConditionsDevice kernel
template<KernelType _kerneltype, typename _ViscSpec, flag_t _simflags,
	int _step,
	RunMode _run_mode = SIMULATE,
	bool _repacking = (_run_mode == REPACK),
	bool _has_io = !!(_simflags & ENABLE_INLET_OUTLET),
	bool _has_keps = (_ViscSpec::turbmodel == KEPSILON),
	bool _has_moving = !!(_simflags & ENABLE_MOVING_BODIES),
	bool has_eulerVel = (_has_io || _has_keps),
	typename eulervel_struct =
		typename COND_STRUCT(has_eulerVel, eulervel_sa_bc_params),
	typename keps_struct =
		typename COND_STRUCT(_has_keps, keps_params<true>)
	>
struct sa_segment_bc_params :
	common_sa_bc_params,
	eulervel_struct,
	keps_struct
{
	static constexpr KernelType kerneltype = _kerneltype; //! kernel type
	using ViscSpec = _ViscSpec; //! viscous model specification
	static constexpr flag_t simflags = _simflags; //! simulation flags
	static constexpr int step = _step; //! integration step
	static constexpr bool has_io = _has_io; //! Open boundaries enabled?
	static constexpr bool has_keps = _has_keps; //! Using the k-epsilon viscous model?
	static constexpr bool has_moving = _has_moving; //! Do we have moving objects?
	static constexpr RunMode run_mode = _run_mode; //! run mode: SIMULATE or REPACK
	static constexpr bool repacking = _repacking; //! true if run_mode is REPACK

	// TODO FIXME instead of using sa_segment_bc_params
	// versus sa_vertex_bc_params, consider using sa_bc_params<PT_BOUNDARY>
	// versus sa_bc_params<PT_VERTEX>
	static constexpr ParticleType cptype = PT_BOUNDARY;

	sa_segment_bc_params(
		BufferList const& bufread,
		BufferList & bufwrite,
		const	uint	_numParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius)
	:
		common_sa_bc_params(bufread, bufwrite,
			_numParticles, _deltap, _slength, _influenceradius),
		eulervel_struct(bufwrite),
		keps_struct(bufwrite)
	{}
};

//! Parameters needed with open boundaries
struct sa_io_params
{
	float4 * __restrict__ clonePos; //! Writable pos array
	const float dt; //! ∆t for this (half) time-step

	sa_io_params(BufferList& bufwrite, const float _dt) :
		clonePos(bufwrite.getData<BUFFER_POS>()),
		dt(_dt)
	{}
};

//! Parameters needed when cloning particles
/*! Note that to initialize some of the data for the new particles we might
 * need to access for writing some buffers that should otherwise be read-only,
 * and may even write to buffers shared between states of the particle system.
 * This is OK, but we need to inform the system about the safety of these
 * accesses.
 */
struct sa_cloning_params
{
	// vel, gGam, eulerVel, tke and eps are already writeable in the mother structure,
	// so we don't expose them here too
	float4 * __restrict__ cloneForces; //! Writable forces array
	particleinfo * __restrict__ cloneInfo; //! Writable info array
	hashKey * __restrict__ cloneParticleHash; //! Writable particleHash array
	vertexinfo * __restrict__ cloneVertices; //! Writable vertices array
	float4 * __restrict__ cloneBoundElems; //! Writeable boundary elements array
	uint * __restrict__ nextIDs; //! Next ID for generated particles
	uint * __restrict__ newNumParticles; //! New number of particles
	const uint totParticles; //! Maximum number of particles allowed in the simulation
	const uint deviceId; //! ID of the device the kernel is running on
	const uint numDevices; //! number of devices used for the simulation

	sa_cloning_params(
				BufferList& bufwrite,
				uint	* __restrict__ _newNumParticles,
		const	 uint _totParticles,
		const	 uint _deviceId,
		const	 uint _numDevices)
	:
		cloneForces(bufwrite.getData<BUFFER_FORCES>()),
		cloneInfo(bufwrite.getData<BUFFER_INFO,
			BufferList::AccessSafety::MULTISTATE_SAFE>()),
		cloneParticleHash(bufwrite.getData<BUFFER_HASH,
			BufferList::AccessSafety::MULTISTATE_SAFE>()),
		cloneVertices(bufwrite.getData<BUFFER_VERTICES,
			BufferList::AccessSafety::MULTISTATE_SAFE>()),
		cloneBoundElems(bufwrite.getData<BUFFER_BOUNDELEMENTS,
			BufferList::AccessSafety::MULTISTATE_SAFE>()),
		nextIDs(bufwrite.getData<BUFFER_NEXTID,
			// TODO FIXME rather than being multi-state safe,
			// access to nextIDs should simply set the n+1 state
			BufferList::AccessSafety::MULTISTATE_SAFE>()),
		newNumParticles(_newNumParticles),
		totParticles(_totParticles),
		deviceId(_deviceId),
		numDevices(_numDevices)
	{}
};

//! Parameters needed by the \ref saVertexBoundaryConditionsDevice kernel
template<KernelType _kerneltype, typename _ViscSpec, flag_t _simflags, int _step,
	RunMode _run_mode = SIMULATE,
	bool _repacking = (_run_mode == REPACK),
	bool _has_io = !!(_simflags & ENABLE_INLET_OUTLET),
	bool _has_keps = (_ViscSpec::turbmodel == KEPSILON),
	bool _has_moving = !!(_simflags & ENABLE_MOVING_BODIES),
	bool _last_io_step = (_has_io && (_step == 2)),
	bool has_eulerVel = (_has_io || _has_keps),
	typename eulervel_struct =
		typename COND_STRUCT(has_eulerVel && !_repacking, eulervel_sa_bc_params),
	typename keps_struct =
		typename COND_STRUCT(_has_keps && !_repacking, keps_params<true>),
	typename io_struct =
		typename COND_STRUCT(_has_io && !_repacking, sa_io_params),
	typename clone_struct =
		typename COND_STRUCT(_last_io_step && !_repacking, sa_cloning_params)
	>
struct sa_vertex_bc_params :
	common_sa_bc_params,
	eulervel_struct,
	keps_struct,
	io_struct,
	clone_struct
{
	static constexpr KernelType kerneltype = _kerneltype; //! kernel type
	using ViscSpec = _ViscSpec;
	static constexpr flag_t simflags = _simflags; //! simulation flags
	static constexpr int step = _step; //! integration step
	static constexpr bool has_io = _has_io; //! Open boundaries enabled?
	static constexpr bool has_keps = _has_keps; //! Using the k-epsilon viscous model?
	static constexpr bool has_moving = _has_moving; //! Do we have moving objects?
	static constexpr RunMode run_mode = _run_mode; //! run mode: SIMULATE or REPACK
	static constexpr bool repacking = _repacking; //! true if run_mode is REPACK

	// TODO FIXME instead of using sa_segment_bc_params
	// versus sa_vertex_bc_params, consider using sa_bc_params<PT_BOUNDARY>
	// versus sa_bc_params<PT_VERTEX>
	static constexpr ParticleType cptype = PT_VERTEX;

	sa_vertex_bc_params(
		const	BufferList&	bufread,
				BufferList&	bufwrite,
				uint	* __restrict__ _newNumParticles,
		const	uint	_numParticles,
		const	uint	_totParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius,
		const	uint	_deviceId,
		const	uint	_numDevices,
		const	float	_dt)
	:
		common_sa_bc_params(bufread, bufwrite,
			_numParticles, _deltap, _slength, _influenceradius),
		eulervel_struct(bufwrite),
		keps_struct(bufwrite),
		io_struct(bufwrite, _dt),
		clone_struct(bufwrite, _newNumParticles,
			_totParticles, _deviceId, _numDevices)
	{}
};

template<flag_t simflags>
using repackViscSpec = FullViscSpec<NEWTONIAN, LAMINAR_FLOW, KINEMATIC, MORRIS, ARITHMETIC, simflags, true>;

template<KernelType _kerneltype,
	typename _ViscSpec,
	flag_t _simflags,
	int _step>
using sa_vertex_bc_repack_params = sa_vertex_bc_params<_kerneltype, repackViscSpec<_simflags>,
	  _simflags, _step>;

template<KernelType _kerneltype,
	typename _ViscSpec,
	flag_t _simflags,
	int _step>
using sa_segment_bc_repack_params = sa_segment_bc_params<_kerneltype, repackViscSpec<_simflags>,
	  _simflags, _step>;

//! findOutgoingSegmentDevice params

struct sa_outgoing_bc_params :
	//! Position, mass, info, particle hash, cellStart, neighbors list
	//! FIXME: slength in the neibs_list_params is not used, so ideally we'd want
	//! one without
	neibs_list_params,
	//! Velocity/density
	vel_wrapper,
	//! Boundary elements normals
	boundelements_wrapper,
	//! Barycentric vertex positions
	vertPos_params<false>
{
	//! Indices of the vertices neighboring each boundary element
	//! Will be updated by the fluid particles to store the vertices
	//! through which we're crossing the domain edge
	vertexinfo*	__restrict__ vertices;

	//! Gamma and its gradient. Outgoing fluid particles will abuse this
	//! to store the relative weights of the vertices for mass repartition.
	float4	* __restrict__ gGam;

	sa_outgoing_bc_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		uint				particleRangeEnd,
		float				slength,
		float				influenceradius)
	: neibs_list_params(bufread, particleRangeEnd, slength, influenceradius)
	, vel_wrapper(bufread)
	, boundelements_wrapper(bufread)
	, vertPos_params<false>(bufread)
	, vertices(bufwrite.getData<BUFFER_VERTICES, BufferList::AccessSafety::MULTISTATE_SAFE>())
	, gGam(bufwrite.getData<BUFFER_GRADGAMMA>())
	{}

};

struct sa_init_gamma_params : neibs_list_params, sa_boundary_params
{
	float4 * __restrict__ newGGam;
	const float deltap;
	const float epsilon;

	sa_init_gamma_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint	_numParticles,
		const	float	_slength,
		const	float	_influenceradius,
		const	float	_deltap,
		const	float	_epsilon)
	: neibs_list_params(bufread, _numParticles, _slength, _influenceradius)
	, sa_boundary_params(bufread)
	, newGGam(bufwrite.getData<BUFFER_GRADGAMMA>())
	, deltap(_deltap)
	, epsilon(_epsilon)
	{}
};

#endif // _SA_BC_PARAMS_H

