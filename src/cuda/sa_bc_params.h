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
};

//! Parameters needed by the \ref saSegmentBoundaryConditionsDevice kernel
template<KernelType _kerneltype, ViscosityType _visctype, flag_t _simflags,
	bool _has_io = (_simflags & ENABLE_INLET_OUTLET),
	bool _has_keps = (_visctype == KEPSVISC),
	bool _has_moving = (_simflags & ENABLE_MOVING_BODIES),
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
	static constexpr ViscosityType visctype = _visctype; //! viscous model
	static constexpr flag_t simflags = _simflags; //! simulation flags
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

//! Parameters needed when cloning particles
struct sa_cloning_params
{
	// vel, gGam, eulerVel, tke and eps are already writeable
	float4 * __restrict__ clonePos; //! Writable pos array
	float4 * __restrict__ cloneForces; //! Writable forces array
	particleinfo * __restrict__ cloneInfo; //! Writable info array
	vertexinfo * __restrict__ cloneVertices; //! Writable vertices array
	hashKey * __restrict__ cloneParticleHash; //! Writable particleHash array
	uint * __restrict__ newNumParticles; //! New number of particles
	uint * __restrict__ nextIDs; //! Next ID for generated particles
	const uint totParticles; //! Maximum number of particles allowed in the simulation
	const uint deviceId; //! ID of the device the kernel is running on
	const uint numDevices; //! number of devices used for the simulation

	const float dt; //! ∆t for this (half) time-step

	sa_cloning_params(
		float4 * __restrict__ _clonePos,
		float4 * __restrict__ _cloneForces,
		particleinfo * __restrict__ _cloneInfo,
		vertexinfo * __restrict__ _cloneVertices,
		hashKey * __restrict__ _cloneParticleHash,
		uint * __restrict__ _newNumParticles,
		uint * __restrict__ _nextIDs,
		const uint _totParticles,
		const uint _deviceId,
		const uint _numDevices,
		const float _dt)
	:
		clonePos(_clonePos),
		cloneForces(_cloneForces),
		cloneInfo(_cloneInfo),
		cloneVertices(_cloneVertices),
		cloneParticleHash(_cloneParticleHash),
		newNumParticles(_newNumParticles),
		nextIDs(_nextIDs),
		totParticles(_totParticles),
		deviceId(_deviceId),
		numDevices(_numDevices),
		dt(_dt)
	{}
};

//! Parameters needed by the \ref saVertexBoundaryConditionsDevice kernel
template<KernelType _kerneltype, ViscosityType _visctype, flag_t _simflags,
	bool _has_io = (_simflags & ENABLE_INLET_OUTLET),
	bool _has_keps = (_visctype == KEPSVISC),
	bool _has_moving = (_simflags & ENABLE_MOVING_BODIES),
	bool has_eulerVel = (_has_io || _has_keps),
	typename eulervel_struct =
		typename COND_STRUCT(has_eulerVel, eulervel_sa_bc_params),
	typename keps_struct =
		typename COND_STRUCT(_has_keps, keps_sa_bc_params),
	typename io_struct =
		typename COND_STRUCT(_has_io, sa_cloning_params)
	>
struct sa_vertex_bc_params :
	common_sa_bc_params,
	eulervel_struct,
	keps_struct,
	io_struct
{
	static constexpr KernelType kerneltype = _kerneltype; //! kernel type
	static constexpr ViscosityType visctype = _visctype; //! viscous model
	static constexpr flag_t simflags = _simflags; //! simulation flags
	static constexpr bool has_io = _has_io; //! Open boundaries enabled?
	static constexpr bool has_keps = _has_keps; //! Using the k-epsilon viscous model?
	static constexpr bool has_moving = _has_moving; //! Do we have moving objects?

	// TODO FIXME instead of using sa_segment_bc_params
	// versus sa_vertex_bc_params, consider using sa_bc_params<PT_BOUNDARY>
	// versus sa_bc_params<PT_VERTEX>
	static constexpr ParticleType cptype = PT_VERTEX;

	sa_vertex_bc_params(
				float4	* __restrict__ _pos,
				float4	* __restrict__ _vel,
				particleinfo	* __restrict__ _info,
				hashKey * __restrict__ _particleHash,
		const	uint	* __restrict__ _cellStart,
		const	neibdata	* __restrict__ _neibsList,

				float4	* __restrict__ _gGam,
				vertexinfo*	__restrict__ _vertices,
		const	float2	* const *__restrict__ _vertPos,

				float4	* __restrict__ _eulerVel,
				float	* __restrict__ _tke,
				float	* __restrict__ _eps,

				float4	* __restrict__ _forces,

		const	uint	_numParticles,
				uint	* __restrict__ _newNumParticles,
				uint	* __restrict__ _nextIDs,
		const	uint	_totParticles,
		const	float	_deltap,
		const	float	_slength,
		const	float	_influenceradius,
		const	uint	_deviceId,
		const	uint	_numDevices,
		const	float	_dt)
	:
		common_sa_bc_params(
			_pos, _vel, _particleHash, _cellStart, _neibsList,
			_gGam, _vertices, _vertPos,
			_numParticles, _deltap, _slength, _influenceradius),
		eulervel_struct(_eulerVel),
		keps_struct(_tke, _eps),
		io_struct(_pos, _forces, _info, _vertices, _particleHash,
			_newNumParticles, _nextIDs, _totParticles, _deviceId, _numDevices, _dt)
	{}
};

#endif // _SA_BC_PARAMS_H

