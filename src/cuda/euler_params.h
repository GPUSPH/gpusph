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

#ifndef _EULER_PARAMS_H
#define _EULER_PARAMS_H

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

#include "cond_params.h"

// We now have the tools to assemble the structure that will be used to pass parameters to the euler kernel

/* Now we define structures that hold the parameters to be passed
   to the euler kernel. These are defined in chunks, and then ‘merged’
   into the template structure that is actually used as argument to the kernel.
   Each struct must define an appropriate constructor / initializer for its const
   members
*/

/// Parameters common to all forces kernel specializations
struct common_euler_params
{
			float4	*newPos;		///< updated particle's position (out)
			float4	*newVel;			///< updated particle's velocity (out)
	const	float4	*oldPos;			///< previous particle's position (in)
	const	hashKey	*particleHash;	///< particle's hash (in)
	const	float4	*oldVel;			///< previous particle's velocity (in/out)
	const	particleinfo	*info;		///< particle's information
	const	float4	*forces;			///< derivative of particle's velocity and density (in)
	const	uint	numParticles;			///< total number of particles
	const	float	full_dt;			///< time step (dt)
	const	float	half_dt;			///< half of time step (dt/2)
	const	float	t;				///< simulation time
	const	uint		step;			///< integrator step //parametro template di euler params struttura collettiva

	// Constructor / initializer
	common_euler_params(
				float4		*_newPos,
				float4		*_newVel,
		const	float4		*_oldPos,
		const	hashKey		*_particleHash,
		const	float4		*_oldVel,
		const	particleinfo	*_info,
		const	float4		*_forces,
		const	uint			_numParticles,
		const	float		_full_dt,
		const	float		_half_dt,
		const	float		_t,
		const	uint			_step) :
		newPos(_newPos),
		newVel(_newVel),
		oldPos(_oldPos),
		particleHash(_particleHash),
		oldVel(_oldVel),
		info(_info),
		forces(_forces),
		numParticles(_numParticles),
		full_dt(_full_dt),
		half_dt(_half_dt),
		t(_t),
		step(_step)
	{}
};

/// Additional parameters passed only to kernels with XSPH enabled
struct xsph_euler_params
{
	const	float4	*xsph;
	xsph_euler_params(const	float4 *_xsph) :
		xsph(_xsph)
	{}
};

/// Additional parameters passed only to kernels with SA_BOUNDARY
struct sa_boundary_euler_params
{
			float4	*oldgGam;
			float4	*newgGam;
	const	float2	*contupd;
			float4	*oldVelRW;
			float4	*newEulerVel;
			float4	*newBoundElement;
	const	float2	*vertPos0;
	const	float2	*vertPos1;
	const	float2	*vertPos2;
	const	float4	*oldEulerVel;
	const	float	slength;
	const	float	influenceradius;
	const	neibdata	*neibsList;
	const	uint	*cellStart;

	// Constructor / initializer
	sa_boundary_euler_params(
				float4	*_oldgGam,
				float4	*_newgGam,
		const	float2	*_contupd,
		const	float4	*_oldVel,
				float4	*_newEulerVel,
				float4	*_newBoundElement,
		const	float2	* const _vertPos[],
		const	float4	*_oldEulerVel,
		const	float	_slength,
		const	float	_influenceradius,
		const	neibdata	*_neibsList,
		const	uint	*_cellStart) :
		oldgGam(_oldgGam),
		newgGam(_newgGam),
		contupd(_contupd),
		oldVelRW(const_cast<float4*>(_oldVel)),
		newEulerVel(_newEulerVel),
		newBoundElement(_newBoundElement),
		vertPos0(_vertPos[0]),
		vertPos1(_vertPos[1]),
		vertPos2(_vertPos[2]),
		oldEulerVel(_oldEulerVel),
		slength(_slength),
		influenceradius(_influenceradius),
		neibsList(_neibsList),
		cellStart(_cellStart)
	{}
};


/// Additional parameters passed only to kernels with KEPSVISC
struct kepsvisc_euler_params
{
	float			*newTKE;	///< updated values of k, for k-e model (out)
	float			*newEps;	///< updated values of e, for k-e model (out)
	const	float	*oldTKE;		///< previous values of k, for k-e model (in)
	const	float	*oldEps;		///< previous values of e, for k-e model
	const	float3	*keps_dkde;	///< derivative of ??? (in)

	// Constructor / initializer
	kepsvisc_euler_params(
			float		*_newTKE,
			float		*_newEps,
			const float	*_oldTKE,
			const float	*_oldEps,
			const float3	*_keps_dkde):
			newTKE(_newTKE),
			newEps(_newEps),
			oldTKE(_oldTKE),
			oldEps(_oldEps),
			keps_dkde(_keps_dkde)
	{}
};


/// Additional parameters passed only to kernels with SPH_GRENIER formulation
struct grenier_euler_params
{
			float4	*newVol;			///< updated particle's voume (out)
	const	float4	*oldVol;			///< previous particle's volume (in)

	// Constructor / initializer
	grenier_euler_params(
				float4 *_newVol,
			const float4 *_oldVol) :
			newVol(_newVol),
			oldVol(_oldVol)
	{}
};

/// The actual euler_params struct, which concatenates all of the above, as appropriate.
template<KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	BoundaryType _boundarytype,
	ViscosityType _visctype,
	flag_t _simflags>
struct euler_params :
	common_euler_params,
	COND_STRUCT(_simflags & ENABLE_XSPH, xsph_euler_params),
	COND_STRUCT(_boundarytype == SA_BOUNDARY, sa_boundary_euler_params),
	COND_STRUCT(_visctype == KEPSVISC, kepsvisc_euler_params),
	COND_STRUCT(_sph_formulation == SPH_GRENIER, grenier_euler_params)
{
	static const KernelType kerneltype = _kerneltype;
	static const SPHFormulation sph_formulation = _sph_formulation;
	static const BoundaryType boundarytype = _boundarytype;
	static const ViscosityType visctype = _visctype;
	static const flag_t simflags = _simflags;

	// This structure provides a constructor that takes as arguments the union of the
	// parameters that would ever be passed to the euler kernel.
	// It then delegates the appropriate subset of arguments to the appropriate
	// structs it derives from, in the correct order
	euler_params(
		// common
				float4		*_newPos,
				float4		*_newVel,
		const	float4		*_oldPos,
		const	hashKey		*_particleHash,
		const	float4		*_oldVel,
		const	particleinfo	*_info,
		const	float4		*_forces,
		const	uint			_numParticles,
		const	float		_full_dt,
		const	float		_half_dt,
		const	float		_t,
		const	uint			_step,

		// XSPH
		const	float4	*_xsph,

		// SA_BOUNDARY
				float4	*_oldgGam,
				float4	*_newgGam,
		const	float2	*_contupd,
				float4	*_newEulerVel,
				float4	*_newBoundElement,
		const	float2	* const _vertPos[],
		const	float4	*_oldEulerVel,
		const	float	_slength,
		const	float	_influenceradius,
		const	neibdata	*_neibsList,
		const	uint	*_cellStart,

		// KEPSVISC
				float	*_newTKE,
				float	*_newEps,
		const	float	*_oldTKE,
		const	float	*_oldEps,
		const	float3	*_keps_dkde,

		// SPH_GRENIER
				float4	*_newVol,
		const	float4	*_oldVol) :

		common_euler_params(_newPos, _newVel, _oldPos, _particleHash,
			_oldVel, _info, _forces, _numParticles, _full_dt, _half_dt, _t, _step),
		COND_STRUCT(simflags & ENABLE_XSPH, xsph_euler_params)(_xsph),
		COND_STRUCT(boundarytype == SA_BOUNDARY, sa_boundary_euler_params)
			(_oldgGam, _newgGam, _contupd, _oldVel, _newEulerVel, _newBoundElement,
			_vertPos, _oldEulerVel, _slength, _influenceradius, _neibsList, _cellStart),
		COND_STRUCT(visctype == KEPSVISC, kepsvisc_euler_params)(_newTKE, _newEps,  _oldTKE, _oldEps, _keps_dkde),
		COND_STRUCT(sph_formulation == SPH_GRENIER, grenier_euler_params)(_newVol, _oldVol)
	{}
};

#endif // _EULER_PARAMS_H

