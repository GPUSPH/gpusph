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

#include "bounds.cu"
#include "buildneibs.cu"
#include "euler.cu"
#include "forces.cu"
#include "post_process.cu"

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
	KernelType _kerneltype,
	SPHFormulation _sph_formulation,
	ViscosityType _visctype,
	BoundaryType _boundarytype,
	Periodicity _periodicbound,
	flag_t _simflags,
	bool invalid_combination = (
		// Currently, we consider invalid only the case
		// of SA_BOUNDARY

		// TODO extend to include all unsupported/untested combinations for other boundary conditions

		_boundarytype == SA_BOUNDARY && (
			// viscosity
			_visctype == KINEMATICVISC		||	// untested
			_visctype == SPSVISC			||	// untested
			_visctype == ARTVISC			||	// untested (use is discouraged, use Ferrari correction)
			// kernel
			! (_kerneltype == WENDLAND)		||	// only the Wendland kernel is allowed in SA_BOUNDARY
												// all other kernels would require their respective
												// gamma and grad gamma formulation
			// formulation
			_sph_formulation == SPH_GRENIER	||	// multi-fluid is currently not implemented
			// flags
			_simflags & ENABLE_XSPH			||	// untested
			_simflags & ENABLE_DEM			||	// not implemented (flat wall formulation is in an old branch)
			(_simflags & ENABLE_INLET_OUTLET && !(_simflags & ENABLE_DENSITY_SUM)) ||
												// inlet outlet works only with the summation density
			(_simflags & ENABLE_GAMMA_QUADRATURE && _simflags & ENABLE_DENSITY_SUM) ||
												// enable density sum only works with the dynamic equation for gamma
			(!(_simflags & ENABLE_GAMMA_QUADRATURE) && !(_simflags & ENABLE_DENSITY_SUM))
												// this has to be changed but for the moment it is only possible to use gamma quadrature
												// when computing drho/dt=div u
		)
	)
>
class CUDASimFrameworkImpl : public SimFramework,
	private InvalidOptionCombination<invalid_combination>
{
	static const KernelType kerneltype = _kerneltype;
	static const SPHFormulation sph_formulation = _sph_formulation;
	static const ViscosityType visctype = _visctype;
	static const BoundaryType boundarytype = _boundarytype;
	static const Periodicity periodicbound = _periodicbound;
	static const flag_t simflags = _simflags;

public:
	CUDASimFrameworkImpl() : SimFramework()
	{
		m_neibsEngine = new CUDANeibsEngine<sph_formulation, boundarytype, periodicbound, true>();
		m_integrationEngine = new CUDAPredCorrEngine<sph_formulation, boundarytype, kerneltype, visctype, simflags>();
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
			return new CUDAPostProcessEngine<VORTICITY, kerneltype, simflags>(options);
		case TESTPOINTS:
			return new CUDAPostProcessEngine<TESTPOINTS, kerneltype, simflags>(options);
		case SURFACE_DETECTION:
			return new CUDAPostProcessEngine<SURFACE_DETECTION, kerneltype, simflags>(options);
		case FLUX_COMPUTATION:
			return new CUDAPostProcessEngine<FLUX_COMPUTATION, kerneltype, simflags>(options);
		case CALC_PRIVATE:
			return new CUDAPostProcessEngine<CALC_PRIVATE, kerneltype, simflags>(options);
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
// As an alternative, a class that adds to the defaults is provided too.

// TODO we may want to put the implementation of the named template options into its own
// header file.

// To get to the named, optional parameter template API we will need a couple of auxiliary
// classes. The main mechanism is essentially inspired by the named template arguments
// mechanism shown in http://www.informit.com/articles/article.aspx?p=31473 with some
// additions to take into account that our template arguments are not typenames, but
// values of different types, and to allow inheritance from previous arguments selectors.

// The first auxiliary class is TypeValue: a class template to carry a value and its type:
// this will be used to specify the default values for the parameters, as well
// as to allow their overriding by the user. It is needed because we want to
// allow parameters to be specified in any order, and this means that we need a
// common 'carrier' for our specific types.

template<typename T, T _val>
struct TypeValue
{
		typedef T type;
		static const T value = _val;
		operator T() { return _val; }; // allow automatic conversion to the type
};

// We will rely on multiple inheritance to group the arguments, and we need to be
// able to specify the same class multiple times (which is forbidden by the standard),
// so we will wrap the type in a "multiplexer":

template<typename T, int idx>
struct MultiplexSubclass : virtual public T
{};

// Template arguments are collected into this class: it will subclass
// all of the template arguments, that must therefore have a common base class
// (see below), and uses the multiplexer class above in case two ore more arguments
// are actually the same class. The number of supported template arguments
// should match that of the CUDASimFramework

template<typename Arg1, typename Arg2, typename Arg3,
	typename Arg4, typename Arg5, typename Arg6>
struct ArgSelector :
	virtual public MultiplexSubclass<Arg1,1>,
	virtual public MultiplexSubclass<Arg2,2>,
	virtual public MultiplexSubclass<Arg3,3>,
	virtual public MultiplexSubclass<Arg4,4>,
	virtual public MultiplexSubclass<Arg5,5>,
	virtual public MultiplexSubclass<Arg6,6>
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
// NOTE: in order to allow the combination of a named parameter struct with
// an existing (specific) ArgSelector, we allow them to be assigned a different
// parent, in order to avoid resolution ambiguity in constructs such as:
// ArgSelector<OldArgSelector, formulation<OTHER_FORMULATION> >

// No override: these are the default themselves
struct DefaultArg : virtual public TypeDefaults
{};

// Kernel override
template<KernelType kerneltype, typename ParentArgs=TypeDefaults>
struct kernel : virtual public ParentArgs
{
	typedef TypeValue<KernelType, kerneltype> Kernel;

	template<typename NewParent> struct reparent :
		virtual public kernel<kerneltype, NewParent> {};
};

// Formulation override
template<SPHFormulation sph_formulation, typename ParentArgs=TypeDefaults>
struct formulation : virtual public ParentArgs
{
	typedef TypeValue<SPHFormulation, sph_formulation> Formulation;

	template<typename NewParent> struct reparent :
		virtual public formulation<sph_formulation, NewParent> {};
};

// Viscosity override
template<ViscosityType visctype, typename ParentArgs=TypeDefaults>
struct viscosity : virtual public ParentArgs
{
	typedef TypeValue<ViscosityType, visctype> Viscosity;

	template<typename NewParent> struct reparent :
		virtual public viscosity<visctype, NewParent> {};
};

// Boundary override
template<BoundaryType boundarytype, typename ParentArgs=TypeDefaults>
struct boundary : virtual public ParentArgs
{
	typedef TypeValue<BoundaryType, boundarytype> Boundary;

	template<typename NewParent> struct reparent :
		virtual public boundary<boundarytype, NewParent> {};
};

// Periodic override
template<Periodicity periodicbound, typename ParentArgs=TypeDefaults>
struct periodicity : virtual public ParentArgs
{
	typedef TypeValue<Periodicity, periodicbound> Periodic;

	template<typename NewParent> struct reparent :
		virtual public periodicity<periodicbound, NewParent> {};
};

// Flags override
template<flag_t simflags, typename ParentArgs=TypeDefaults>
struct flags : virtual public ParentArgs
{
	typedef TypeValue<flag_t, simflags> Flags;

	template<typename NewParent> struct reparent :
		virtual public flags<simflags, NewParent> {};
};

// Add flags: this is an override that adds the new simflags
// to the ones of the parent.
template<flag_t simflags, typename ParentArgs=TypeDefaults>
struct add_flags : virtual public ParentArgs
{
	typedef TypeValue<flag_t, ParentArgs::Flags::value | simflags> Flags;

	template<typename NewParent> struct reparent :
		virtual public add_flags<simflags, NewParent> {};
};

/// We want to give users the possibility to change options (e.g. enable flags)
/// conditionally at runtime. For this, we need a way to pack collection of
/// overrides to be selected by a switch statement (currently limited to three
/// options)
template<typename _A, typename _B, typename _C>
struct TypeSwitch {
	typedef _A A;
	typedef _B B;
	typedef _C C;
};

/// Comfort method to allow the user to select one of three flags at runtime
template<flag_t F0, flag_t F1, flag_t F2>
struct FlagSwitch :
	TypeSwitch<
		add_flags<F0>,
		add_flags<F1>,
		add_flags<F2>
	>
{};

/// Our CUDASimFramework is actualy a factory for CUDASimFrameworkImpl*,
/// generating one when assigned to a SimFramework*. This is to allow us
/// to change the set of options at runtime without setting up/tearing down
/// the whole simframework every time an option is changed (setting up/tearing
/// down the factory itself is much cheaper as there is no associated storage, so
/// it's mostly just compile-time juggling).
template<
	typename Arg1 = DefaultArg,
	typename Arg2 = DefaultArg,
	typename Arg3 = DefaultArg,
	typename Arg4 = DefaultArg,
	typename Arg5 = DefaultArg,
	typename Arg6 = DefaultArg>
class CUDASimFramework {
	/// The collection of arguments for our current setup
	typedef ArgSelector<Arg1, Arg2, Arg3, Arg4, Arg5, Arg6> Args;

	/// Comfort static defines
	static const KernelType kerneltype = Args::Kernel::value;
	static const SPHFormulation sph_formulation = Args::Formulation::value;
	static const ViscosityType visctype = Args::Viscosity::value;
	static const BoundaryType boundarytype = Args::Boundary::value;
	static const Periodicity periodicbound = Args::Periodic::value;
	static const flag_t simflags = Args::Flags::value;

	/// The CUDASimFramework implementation of the current setup
	typedef CUDASimFrameworkImpl<
			kerneltype,
			sph_formulation,
			visctype,
			boundarytype,
			periodicbound,
			simflags> CUDASimFrameworkType;

	/// A comfort auxiliary class that overrides Args (the current setup)
	/// with the Extra named option
	template<typename Extra> struct Override :
		virtual public Args,
		virtual public Extra::template reparent<Args>
	{};

	/// A method to produce a new factory with an overridden parameter
	template<typename Extra>
	CUDASimFramework< Override<Extra> > extend() {
		return CUDASimFramework< Override<Extra> >();
	}

public:
	/// Conversion operator: this produces the actual implementation of the
	/// simframework
	operator SimFramework *()
	{
		// return the intended framework
		return new CUDASimFrameworkType();
	}

	/// Runtime selectors.

	/// Note that they must return a SimFramework* because otherwise the type
	/// returned would depend on the runtime selection, which is not possible.
	/// As a result we cannot chain runtime selectors, and must instead provide
	/// further runtime selectors with multiple (pairs of) overrides

	/// Select an override only if a boolean option is ture
	template<typename Extra>
	SimFramework * select_options(bool selector, Extra)
	{
		if (selector)
			return extend<Extra>();
		return *this;
	}

	/// Select one of three overrides in a Switch, based on the value of
	/// selector. TODO refine
	template<typename Switch>
	SimFramework * select_options(int selector, Switch)
	{
		switch (selector) {
		case 0:
			return extend< typename Switch::A >();
		case 1:
			return extend< typename Switch::B >();
		case 2:
			return extend< typename Switch::C >();
		}
		throw std::runtime_error("invalid selector value");
	}

	/// Chained selectors (for multiple overrides)
	template<typename Extra, typename Sel2, typename Other>
	SimFramework * select_options(bool selector, Extra, Sel2 selector2, Other)
	{
		if (selector)
			return extend<Extra>().select_options(selector2, Other());
		return this->select_options(selector2, Other());
	}

	/// Chained selectors (for multiple overrides)
	template<typename Switch, typename Sel2, typename Other>
	SimFramework * select_options(int selector, Switch, Sel2 selector2, Other)
	{
		switch (selector) {
		case 0:
			return extend< typename Switch::A >().select_options(selector2, Other());
		case 1:
			return extend< typename Switch::B >().select_options(selector2, Other());
		case 2:
			return extend< typename Switch::C >().select_options(selector2, Other());
		}
		throw std::runtime_error("invalid selector value");
	}

};

#endif

/* vim: set ft=cuda sw=4 ts=4 : */
