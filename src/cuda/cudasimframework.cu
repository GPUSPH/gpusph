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

#ifndef _CUDASIMFRAMEWORK_H
#define _CUDASIMFRAMEWORK_H

#include "simframework.h"

#include "predcorr_alloc_policy.h"

#include "simflags.h"

#include "buildneibs.cu"
#include "euler.cu"
#include "forces.cu"

// This class holds the implementation and interface of CUDASimFramework,
// the CUDA simulation framework for GPUPSH. (In fact, the only simulation
// framework in GPUSPH, presently).

// The CUDASimFramework is a template class depending on KernelType, ViscosityType,
// BoundaryType, Periodicity and simulation flags, in order to allow concrete
// instantiation of only the needed specializations of the appropriate engines.

// To allow the user to specify any (or none, or all) of the template parameters,
// in any order, we separate the implementation from the user-visible class that
// instantiates it. This is not strictly necessary, but it makes the code more
// readable by separating the optional, named template arguments management
// from the implementation of the actual framework.


/* CUDABoundaryConditionsSelector */

// We begin with an auxiliary functor to select a BoundaryConditionsEngine,
// which currently simply selects the CUDABoundaryConditionsEngine in the case of
// SA_BOUNDARY BoundaryType, and returns NULL otherwise.

// General case
template<KernelType kerneltype, ViscosityType visctype,
	BoundaryType boundarytype, flag_t simflags>
struct CUDABoundaryConditionsSelector
{
	typedef CUDABoundaryConditionsEngine<kerneltype, visctype, boundarytype, simflags> BCEtype;
	static BCEtype* select()
	{ return NULL; } // default, no BCE
};

// SA_BOUNDARY specialization
template<KernelType kerneltype, ViscosityType visctype, flag_t simflags>
struct CUDABoundaryConditionsSelector<kerneltype, visctype, SA_BOUNDARY, simflags>
{
	typedef CUDABoundaryConditionsEngine<kerneltype, visctype, SA_BOUNDARY, simflags> BCEtype;
	static BCEtype* select()
	{ return new BCEtype(); } // TODO FIXME when we have proper BCEs
};

/// Some combinations of frameworks for kernels are invalid/
/// unsupported/untested and we want to prevent the user from
/// using them, by (1) catching the error as soon as possible
/// during compilation and (2) give an error message that is
/// as descriptive as possible (non-trivial with C++).
/// Point (2) is particularly hard to realize with nvcc because
/// it doesn't print out the actual line with the error, so we
/// need some indirection; we achieve this by making the
/// CUDASimFramework subclass a template class InvalidOptionCombination
/// whose instantiation will fail in case of invalid option combinations;
/// this failure is due to trying to subclass an IncompleteType class
/// which is not defined except in the not invalid case.
/// nvcc will then show the error for InvalidOptionCombination, which
/// is hopefully descriptive enough for users.
template<bool invalid>
class IncompleteType;

template<>
class IncompleteType<false>
{};

template<bool invalid>
class InvalidOptionCombination : IncompleteType<invalid>
{};

/* CUDASimFrameworkImpl */

// Here we define the implementation for the CUDASimFramework. The use of *Impl is
// to allow the user-facing CUDASimFramework to be the one that allows named and optional
// template parameters

template<
	KernelType kerneltype,
	SPHFormulation sph_formulation,
	ViscosityType visctype,
	BoundaryType boundarytype,
	Periodicity periodicbound,
	flag_t simflags,
	bool invalid_combination = (
		// Currently, we consider invalid only the case
		// of SA_BOUNDARY

		// TODO extend to include all unsupported/untested combinations for other boundary conditions

		boundarytype == SA_BOUNDARY && (
			// viscosity
			visctype == KINEMATICVISC		||	// untested
			visctype == SPSVISC 			||	// untested
			visctype == ARTVISC 			||	// untested (use is discouraged, use Ferrari correction)
			// kernel
			! (kerneltype == WENDLAND) 		||	// only the Wendland kernel is allowed in SA_BOUNDARY
												// all other kernels would require their respective
												// gamma and grad gamma formulation
			// formulation
			sph_formulation == SPH_GRENIER	||	// multi-fluid is currently not implemented
			// flags
			simflags & ENABLE_XSPH			||	// untested
			simflags & ENABLE_DEM			||	// not implemented (flat wall formulation is in an old branch)
			(simflags & ENABLE_INLET_OUTLET && ((~simflags) & ENABLE_DENSITY_SUM))
												// inlet outlet works only with the summation density
		)
	)
>
class CUDASimFrameworkImpl : public SimFramework,
	private InvalidOptionCombination<invalid_combination>
{
public:
	CUDASimFrameworkImpl() : SimFramework()
	{
		m_neibsEngine = new CUDANeibsEngine<sph_formulation, boundarytype, periodicbound, true>();
		m_integrationEngine = new CUDAPredCorrEngine<sph_formulation, boundarytype, kerneltype, simflags>();
		m_viscEngine = new CUDAViscEngine<visctype, kerneltype, boundarytype>();
		m_forcesEngine = new CUDAForcesEngine<kerneltype, sph_formulation, visctype, boundarytype, simflags>();
		m_bcEngine = CUDABoundaryConditionsSelector<kerneltype, visctype, boundarytype, simflags>::select();

		// TODO should be allocated by the integration scheme
		m_allocPolicy = new PredCorrAllocPolicy();

		m_simparams = new SimParams(kerneltype, sph_formulation, visctype,
			boundarytype, periodicbound, simflags);
	}

protected:
	AbstractFilterEngine* newFilterEngine(FilterType filtertype, int frequency)
	{
		switch (filtertype) {
		case SHEPARD_FILTER:
			return new CUDAFilterEngine<SHEPARD_FILTER, kerneltype, boundarytype>(frequency);
		case MLS_FILTER:
			return new CUDAFilterEngine<MLS_FILTER, kerneltype, boundarytype>(frequency);
		case INVALID_FILTER:
			throw runtime_error("Invalid filter type");
		}
		throw runtime_error("Unknown filter type");
	}

	AbstractPostProcessEngine* newPostProcessEngine(PostProcessType pptype, flag_t options=NO_FLAGS)
	{
		switch (pptype) {
		case VORTICITY:
			return new CUDAPostProcessEngine<VORTICITY, kerneltype>(options);
		case TESTPOINTS:
			return new CUDAPostProcessEngine<TESTPOINTS, kerneltype>(options);
		case SURFACE_DETECTION:
			return new CUDAPostProcessEngine<SURFACE_DETECTION, kerneltype>(options);
		case CALC_PRIVATE:
			return new CUDAPostProcessEngine<CALC_PRIVATE, kerneltype>(options);
		case INVALID_POSTPROC:
			throw runtime_error("Invalid filter type");
		}
		throw runtime_error("Unknown filter type");
	}

};

/* CUDASimFramework user-facing interface */

// We want to allow the user to create a CUDASimFramework by omitting any of the template
// parameters, and to override them in any order. For example, if the user wants to
// override only the kernel and the periodicity, and to enable XSPH, they should be able to
// write something like:
//
//	m_simframework = new CUDASimFramework<
//		withKernel<WENDLAND>,
//		withFlags<ENABLE_XSPH | ENABLE_DEM>,
//		withPeriodicity<PERIODIC_X>
//	>();
//
// NOTE: the withFlags<> will override the default flags, not add to them,
// so in case of flag override, the default ones should be included manually.
// (TODO is there a way to avoid this? In this case we would need to provide two
// 'withFlags' for the user, one to override the default flags and one to add to them.)

// TODO we may want to put the implementation of the named template options into its own
// header file.

// To get to the named, optional parameter template API we will need a couple of auxiliary
// classes. The main mechanism is essentially inspired by the named template arguments
// mechanism shown in http://www.informit.com/articles/article.aspx?p=31473 with some
// additions to take into account that our template arguments are not typenames, but
// values of different types.

// The first auxiliary class is TypeValue: a class template to carry a value and its type
// (note that the value should be convertible to enum): this will be used to specify the
// default values for the parameters, as well as to allow their overriding by the user.
// It is needed because we want to allow parameters to be specified in any order,
// and this means that we need a common 'carrier' for our specific types.

template<typename T, T _val>
struct TypeValue
{
		typedef T type;
		enum { value = _val };
		operator T() { return _val; }; // allow automatic conversion to the type
};

// We will rely on multiple inheritance to group the arguments, and we need to be
// able to specify the same class multiple times (which is forbidden by the standard),
// so we will wrap the type in a "multiplexer":

template<typename T, int idx>
struct MultiplexSubclass : public T
{};

// Template arguments are collected into this class: it will subclass
// all of the template arguments, that must therefore have a common base class
// (see below), and uses the multiplexer class above in case two ore more arguments
// are actually the same class. The number of supported template arguments
// should match that of the CUDASimFramework

template<typename Arg1, typename Arg2, typename Arg3,
	typename Arg4, typename Arg5, typename Arg6>
struct ArgSelector :
	public MultiplexSubclass<Arg1,1>,
	public MultiplexSubclass<Arg2,2>,
	public MultiplexSubclass<Arg3,3>,
	public MultiplexSubclass<Arg4,4>,
	public MultiplexSubclass<Arg5,5>,
	public MultiplexSubclass<Arg6,6>
{};

// Now we set the defaults for each argument
struct TypeDefaults
{
	typedef TypeValue<KernelType, WENDLAND> Kernel;
	typedef TypeValue<SPHFormulation, SPH_F1> Formulation;
	typedef TypeValue<ViscosityType, ARTVISC> Viscosity;
	typedef TypeValue<BoundaryType, LJ_BOUNDARY> Boundary;
	typedef TypeValue<Periodicity, PERIODIC_NONE> Periodic;
	typedef TypeValue<flag_t, ENABLE_DTADAPT> Flags;
};

// The user-visible name template parameters will all subclass TypeDefaults,
// and override specific typedefs
// NOTE: inheritance must be virtual so that there will be no resolution
// ambiguity.

// No override: these are the default themselves
struct DefaultArg : virtual public TypeDefaults
{};

// Kernel override
template<KernelType kerneltype>
struct kernel : virtual public TypeDefaults
{ typedef TypeValue<KernelType, kerneltype> Kernel; };

// Formulation override
template<SPHFormulation sph_formulation>
struct formulation : virtual public TypeDefaults
{ typedef TypeValue<SPHFormulation, sph_formulation> Formulation; };

// Viscosity override
template<ViscosityType visctype>
struct viscosity : virtual public TypeDefaults
{ typedef TypeValue<ViscosityType, visctype> Viscosity; };

// Boundary override
template<BoundaryType boundarytype>
struct boundary : virtual public TypeDefaults
{ typedef TypeValue<BoundaryType, boundarytype> Boundary; };

// Periodic override
template<Periodicity periodicbound>
struct periodicity : virtual public TypeDefaults
{ typedef TypeValue<Periodicity, periodicbound> Periodic; };

// Flags override
template<flag_t simflags>
struct flags : virtual public TypeDefaults
{ typedef TypeValue<flag_t, simflags> Flags; };

// And that's all!
template<
	typename Arg1 = DefaultArg,
	typename Arg2 = DefaultArg,
	typename Arg3 = DefaultArg,
	typename Arg4 = DefaultArg,
	typename Arg5 = DefaultArg,
	typename Arg6 = DefaultArg>
class CUDASimFramework : public
	CUDASimFrameworkImpl<
#define ARGS ArgSelector<Arg1, Arg2, Arg3, Arg4, Arg5, Arg6>
		KernelType(ARGS::Kernel::value),
		SPHFormulation(ARGS::Formulation::value),
		ViscosityType(ARGS::Viscosity::value),
		BoundaryType(ARGS::Boundary::value),
		Periodicity(ARGS::Periodic::value),
		flag_t(ARGS::Flags::value)>
#undef ARGS
{};

#endif

/* vim: set ft=cuda sw=4 ts=4 : */
