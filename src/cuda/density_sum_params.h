/*  Copyright 2014 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _DENSITY_SUM_PARAMS_H
#define _DENSITY_SUM_PARAMS_H

#include "particledefine.h"
#include "simflags.h"

/* To achieve good performance, each combination of kernel, boundary, formulation
   etc is a different specialization, something which in itself makes for a huge
   number of kernels.
   To make things more complicated, some incarnations require a different set
   of parameters compared to other, and instantiate different sets of variables.
   All of this should be managed as automagically as possible while trying to
   rely only on the most readable features of C++ and, if possible, using the
   preprocessor as little as possible.

   To this end, we set up a mechanism that allows us to build structure
   templates in which the number and quality of members depends on the
   specialization.
*/

#include "common_params.h"

// We now have the tools to assemble the structure that will be used to pass parameters to the density_sum kernel

/* Now we define structures that hold the parameters to be passed
   to the density_sum kernel. These are defined in chunks, and then ‘merged’
   into the template structure that is actually used as argument to the kernel.
   Each struct must define an appropriate constructor / initializer for its const
   members
*/

/// Parameters common to all density_sum kernel specializations
struct common_density_sum_params :
	Pos_params<false>, ///< old (in) and new (in) position
	Vel_params<true>, ///< old (in) and new (in/out) velocity
	gGam_params<true> ///< old (in) and new (in/out) gamma and its gradient
{
	const	hashKey	*particleHash;		///< particle's hash (in)
	const	particleinfo	*info;		///< particle's information
	const	neibdata	*neibsList;
	const	uint	*cellStart;
			float4	*forces;			///< derivative of particle's velocity and density (in/out)

	const	uint	numParticles;		///< total number of particles
	const	float	dt;					///< time step (dt or dt/2 depending on integration phase)
	const	float	t;					///< simulation time
	const	uint	step;				///< integrator step
	const	float	slength;
	const	float	influenceradius;

	// Constructor / initializer
	common_density_sum_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint		_numParticles,
		const	float		_dt,
		const	float		_t,
		const	uint		_step,
		const	float		_slength,
		const	float		_influenceradius)
	:
		Pos_params<false>(bufread, bufwrite),
		Vel_params<true>(bufread, bufwrite),
		gGam_params<true>(bufread, bufwrite),
		particleHash(bufread.getData<BUFFER_HASH>()),
		info(bufread.getData<BUFFER_INFO>()),
		neibsList(bufread.getData<BUFFER_NEIBSLIST>()),
		cellStart(bufread.getData<BUFFER_CELLSTART>()),
		forces(bufwrite.getData<BUFFER_FORCES>()),
		numParticles(_numParticles),
		dt(_dt),
		t(_t),
		step(_step),
		slength(_slength),
		influenceradius(_influenceradius)
	{}
};

/// Additional parameters passed only to the kernel with BOUNDARY neighbors
template<flag_t simflags,
	bool enable_moving = !!(simflags & ENABLE_MOVING_BODIES)>
struct boundary_density_sum_params :
	BoundElement_params<false>,
	vertPos_params<false>
{
	boundary_density_sum_params(
		BufferList const&	bufread,
		BufferList const&	bufwrite) // the ‘write’ copy is read-onl too, actually
	:
		// if we have moving boundaries, we want both the old and new boundary elements,
		// otherwise we know they'll be the same
		BoundElement_params<false>(bufread, enable_moving ? bufwrite : bufread),
		vertPos_params<false>(bufread)
	{}

	boundary_density_sum_params(boundary_density_sum_params const&) = default;
};

/// The actual density_sum_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	ParticleType _ntype,
	flag_t _simflags,
	// if we have open boundaries, we also want the old and new Eulerian velocity
	// (read-only)
	typename io_params = typename
		COND_STRUCT(_simflags & ENABLE_INLET_OUTLET, EulerVel_params<false>),
	typename boundary_params = typename
		COND_STRUCT(_ntype == PT_BOUNDARY, boundary_density_sum_params<_simflags>)
	>
struct density_sum_params :
	common_density_sum_params,
	io_params,
	boundary_params
{
	static const KernelType kerneltype = _kerneltype;
	static const ParticleType ntype = _ntype;
	static const flag_t simflags = _simflags;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the density_sum kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	density_sum_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint		_numParticles,
		const	float		_dt,
		const	float		_t,
		const	uint		_step,
		const	float		_slength,
		const	float		_influenceradius)
	:
		common_density_sum_params(bufread, bufwrite,
			_numParticles, _dt, _t, _step, _slength, _influenceradius),
		io_params(bufread, bufwrite),
		boundary_params(bufread, bufwrite)
	{}
};

/// Common params for integrateGammaDevice in the dynamic gamma case
template<flag_t simflags>
struct common_dynamic_integrate_gamma_params :
	Pos_params<false>, ///< old (in) and new (in) position
	Vel_params<false>, ///< old (in) and new (in) position
	gGam_params<true>, ///< old (in) and new (in/out) gamma and its gradient
	boundary_density_sum_params<simflags> ///< boundary elements (old and new) and vert pos
{
	const	particleinfo * __restrict__ info; ///< particle info
	const	hashKey	* __restrict__ particleHash; ///< particle hash
	const	neibdata *__restrict__ neibsList;
	const	uint	* __restrict__ cellStart;
	const	uint	particleRangeEnd; ///< max number of particles
	const	float	dt; ///< time step (dt or dt/2 depending on integrator step)
	const	float	t; ///< simulation time
	const	uint	step; ///< integrator step
	const	float	slength;
	const	float	influenceradius;

	common_dynamic_integrate_gamma_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint	_particleRangeEnd, ///< max number of particles
		const	float	_dt, ///< time step (dt or dt/2)
		const	float	_t, ///< simulation time
		const	uint	_step, ///< integrator step
		const	float	_slength,
		const	float	_influenceradius)
	:
		Pos_params<false>(bufread, bufwrite),
		Vel_params<false>(bufread, bufwrite),
		gGam_params<true>(bufread, bufwrite),
		boundary_density_sum_params<simflags>(bufread, bufwrite),
		info(bufread.getData<BUFFER_INFO>()),
		particleHash(bufread.getData<BUFFER_HASH>()),
		neibsList(bufread.getData<BUFFER_NEIBSLIST>()),
		cellStart(bufread.getData<BUFFER_CELLSTART>()),
		particleRangeEnd(_particleRangeEnd),
		dt(_dt), t(_t), step(_step),
		slength(_slength), influenceradius(_influenceradius)
	{}

	common_dynamic_integrate_gamma_params(common_dynamic_integrate_gamma_params const&) = default;
};

/// integrateGammaDevice parameters specific for !USING_DYNAMIC_GAMMA
/* TODO merge common elements betwene common_dynamic_integrate_gamma_params
 * and quadrature_gamma_params */
template<flag_t simflags, bool has_moving_bodies = !!(simflags & ENABLE_MOVING_BODIES)>
struct quadrature_gamma_params :
	gGam_params<true>, ///< old (in) and new (in/out) gamma and its gradient
	vertPos_params<false>
{
	const	float4	* __restrict__ newPos; ///< positions at step n+1
	const	particleinfo * __restrict__ info; ///< particle info
	const	float4	* __restrict__ newBoundElement; ///< boundary elements at step n+1
	const	hashKey	* __restrict__ particleHash; ///< particle hash
	const	neibdata *__restrict__ neibsList;
	const	uint	* __restrict__ cellStart;
	const	uint	particleRangeEnd; ///< max number of particles
	const	float	epsilon;
	const	float	slength;
	const	float	influenceradius;

	quadrature_gamma_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint	_particleRangeEnd, ///< max number of particles
		const	float	_epsilon,
		const	float	_slength,
		const	float	_influenceradius)
	:
		gGam_params<true>(bufread, bufwrite),
		vertPos_params<false>(bufread),
		newPos(bufwrite.getData<BUFFER_POS>()),
		info(bufread.getData<BUFFER_INFO>()),
		particleHash(bufread.getData<BUFFER_HASH>()),
		newBoundElement(has_moving_bodies ?
			bufwrite.getConstData<BUFFER_BOUNDELEMENTS>() :
			bufread.getData<BUFFER_BOUNDELEMENTS>()),
		neibsList(bufread.getData<BUFFER_NEIBSLIST>()),
		cellStart(bufread.getData<BUFFER_CELLSTART>()),
		particleRangeEnd(_particleRangeEnd),
		epsilon(_epsilon),
		slength(_slength),
		influenceradius(_influenceradius)
	{}

	/* Constructor from an quadrature_gamma_params with different cptype */
	quadrature_gamma_params(quadrature_gamma_params const&) = default;
};


template<ParticleType _cptype, KernelType _kerneltype, flag_t _simflags,
	RunMode _run_mode = SIMULATE,
	bool _repacking = (_run_mode == REPACK),
	bool dynamic = USING_DYNAMIC_GAMMA(_simflags),
	bool has_io = !!(_simflags & ENABLE_INLET_OUTLET),
	typename dynamic_gamma_params = typename
		COND_STRUCT(dynamic, common_dynamic_integrate_gamma_params<_simflags>),
	// for open boundaries we also want the old and new Euler vel
	typename dynamic_io_gamma_params = typename
		COND_STRUCT(dynamic && has_io, EulerVel_params<false>),
	typename quadrature_params = typename
		COND_STRUCT(!dynamic, quadrature_gamma_params<_simflags>)
	>
struct integrate_gamma_params :
	dynamic_gamma_params,
	dynamic_io_gamma_params,
	quadrature_params
{
	static constexpr KernelType kerneltype = _kerneltype;
	static constexpr ParticleType cptype = _cptype;
	static constexpr flag_t simflags = _simflags;
	static constexpr RunMode run_mode = _run_mode;
	static constexpr bool repacking = _repacking;

	integrate_gamma_params(
		BufferList const&	bufread,
		BufferList &		bufwrite,
		const	uint	_particleRangeEnd, ///< max number of particles
		const	float	_dt, ///< time step (dt or dt/2)
		const	float	_t, ///< simulation time
		const	uint	_step, ///< integrator step
		const	float	_epsilon, ///< epsilon for gamma tolerance
		const	float	_slength,
		const	float	_influenceradius)
	:
		dynamic_gamma_params(bufread, bufwrite,
				_particleRangeEnd,
				_dt, _t, _step,
				_slength, _influenceradius),
		dynamic_io_gamma_params(bufread, bufwrite),
		quadrature_params(bufread, bufwrite,
			_particleRangeEnd, _epsilon, _slength, _influenceradius)
	{}

	/* Constructor from an integrate_gamma_params with different cptype */
	template<typename OtherParams>
	integrate_gamma_params(OtherParams const& p) :
		dynamic_gamma_params(p),
		dynamic_io_gamma_params(p),
		quadrature_params(p)
	{}
};

template<ParticleType _cptype, KernelType _kerneltype, flag_t _simflags>
using integrate_gamma_repack_params = integrate_gamma_params<_cptype, _kerneltype,
	  _simflags, REPACK>;

#endif // _DENSITY_SUM_PARAMS_H

