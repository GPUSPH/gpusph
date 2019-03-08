/*  Copyright 2018 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _SA_SEGMENT_BC_PARAMS_H
#define _SA_SEGMENT_BC_PARAMS_H

#include "particledefine.h"
#include "simflags.h"

/* Templatized parameters structure for the SA_BOUNDARY boundary conditions kernels
 * (saSegmentBoundaryConditionsDevice, saVertexBoundaryConditionsDevice
 */

#include "cond_params.h"

/**! Parameters common to all \ref saSegmentBoundaryConditionsDevice and
 * \ref saVertexBoundaryConditionsDevice specializations
 */
struct common_sa_bc_params
{
	//! Particle position and mass.
	//! saVertexBoundaryConditionsDevice will need to update this in the open boundary case,
	//! but that will be achieved by passing a non-const copy of this array in a different structure
	const	float4	* __restrict__ pos;
	//! Particle velocity and density. Both saSegmentBoundaryConditionsDevice and saVertexBoundaryConditionsDevice
	//! will update the densities to match the fluid pressure
			float4	* __restrict__ vel;
	//! Particle hash, to get the cell grid
	const	hashKey * __restrict__ particleHash;
	//! Start of each cell, for neighbors search
	const	uint	* __restrict__ cellStart;
	//! Neighbors list for each particle
	const	neibdata* __restrict__ neibsList;

	//! Gamma and its gradient. Will be updated during initialization
	//! and in the case of open boundaries and moving objects.
	//! TODO FIXME make const otherwise
			float4	* __restrict__ gGam;
	//! Indices of the vertices neighboring each boundary element
	const	vertexinfo*	__restrict__ vertices;
	//! Barycentric vertex positions
	const	float2	* __restrict__ vertPos0;
	const	float2	* __restrict__ vertPos1;
	const	float2	* __restrict__ vertPos2;

	//! Number of particles to process
	const	uint	numParticles;

	// TODO these should probably go into constant memory
	const	float	deltap; //! Inter-particle distance
	const	float	slength; //! Kernel smoothing length h
	const	float	influenceradius; //! Kernel influence radius

	// Constructor / initializer
	common_sa_bc_params(
		const	float4	* __restrict__ _pos,
				float4	* __restrict__ _vel,
		const	hashKey * __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,

				float4	* __restrict__ _gGam,
		const	vertexinfo*	__restrict__ _vertices,
		const	float2	* const *__restrict__ _vertPos,

		const	uint	_numParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius)
	:
		pos(_pos),
		vel(_vel),
		particleHash(_particleHash),
		cellStart(_cellStart),
		neibsList(_neibsList),
		gGam(_gGam),
		vertices(_vertices),
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2]),
		numParticles(_numParticles),
		deltap(_deltap),
		slength(_slength),
		influenceradius(_influenceradius)
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

//! k-epsilon viscosity arrays
/** Both are read/write to allow imposing boundary conditions */
struct keps_sa_bc_params
{
	float * __restrict__ tke; //! Turbulent Kinetic Energy
	float * __restrict__ eps; //! Turbulent dissipation

	keps_sa_bc_params(
		float *__restrict__ _tke,
		float *__restrict__ _eps)
	:
		tke(_tke), eps(_eps)
	{}
	keps_sa_bc_params(BufferList& bufwrite)
	:
		tke(bufwrite.getData<BUFFER_TKE>()),
		eps(bufwrite.getData<BUFFER_EPSILON>())
	{}
};

//! Parameters needed by the \ref saSegmentBoundaryConditionsDevice kernel
template<KernelType _kerneltype, typename _ViscSpec, flag_t _simflags,
	int _step,
	bool _has_io = !!(_simflags & ENABLE_INLET_OUTLET),
	bool _has_keps = (_ViscSpec::turbmodel == KEPSILON),
	bool _has_moving = !!(_simflags & ENABLE_MOVING_BODIES),
	bool has_eulerVel = (_has_io || _has_keps),
	typename eulervel_struct =
		typename COND_STRUCT(has_eulerVel, eulervel_sa_bc_params),
	typename keps_struct =
		typename COND_STRUCT(_has_keps, keps_sa_bc_params)
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

	// TODO FIXME instead of using sa_segment_bc_params
	// versus sa_vertex_bc_params, consider using sa_bc_params<PT_BOUNDARY>
	// versus sa_bc_params<PT_VERTEX>
	static constexpr ParticleType cptype = PT_BOUNDARY;

	sa_segment_bc_params(
		const	float4	* __restrict__ _pos,
				float4	* __restrict__ _vel,
		const	hashKey * __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,

				float4	* __restrict__ _gGam,
		const	vertexinfo*	__restrict__ _vertices,
		const	float2	* const *__restrict__ _vertPos,

				float4	* __restrict__ _eulerVel,
				float	* __restrict__ _tke,
				float	* __restrict__ _eps,

		const	uint	_numParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius)
	:
		common_sa_bc_params(
			_pos, _vel, _particleHash, _cellStart, _neibsList,
			_gGam, _vertices, _vertPos,
			_numParticles, _deltap, _slength, _influenceradius),
		eulervel_struct(_eulerVel),
		keps_struct(_tke, _eps)
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
	bool _has_io = !!(_simflags & ENABLE_INLET_OUTLET),
	bool _has_keps = (_ViscSpec::turbmodel == KEPSILON),
	bool _has_moving = !!(_simflags & ENABLE_MOVING_BODIES),
	bool _last_io_step = (_has_io && (_step == 2)),
	bool has_eulerVel = (_has_io || _has_keps),
	typename eulervel_struct =
		typename COND_STRUCT(has_eulerVel, eulervel_sa_bc_params),
	typename keps_struct =
		typename COND_STRUCT(_has_keps, keps_sa_bc_params),
	typename io_struct =
		typename COND_STRUCT(_has_io, sa_io_params),
	typename clone_struct =
		typename COND_STRUCT(_last_io_step, sa_cloning_params)
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
		common_sa_bc_params(
			bufread.getData<BUFFER_POS>(),
			bufwrite.getData<BUFFER_VEL>(),
			bufread.getData<BUFFER_HASH>(),
			bufread.getData<BUFFER_CELLSTART>(),
			bufread.getData<BUFFER_NEIBSLIST>(),
			bufwrite.getData<BUFFER_GRADGAMMA>(),
			bufread.getData<BUFFER_VERTICES>(),
			bufread.getRawPtr<BUFFER_VERTPOS>(),
			_numParticles, _deltap, _slength, _influenceradius),
		eulervel_struct(bufwrite),
		keps_struct(bufwrite),
		io_struct(bufwrite, _dt),
		clone_struct(bufwrite, _newNumParticles,
			_totParticles, _deviceId, _numDevices)
	{}
};

#endif // _SA_BC_PARAMS_H

