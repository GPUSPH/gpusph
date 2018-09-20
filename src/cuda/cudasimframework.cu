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
#include "textures.cuh"

#include "sph_core.cu"
#include "phys_core.cu"
#include "buildneibs.cu"
#include "geom_core.cu"
#include "boundary_conditions.cu"
#include "euler.cu"
#include "forces.cu"
#include "visc.cu"
#include "post_process.cu"

using namespace std;

using namespace std;

// This class holds the implementation and interface of CUDASimFramework,
// the CUDA simulation framework for GPUPSH. (In fact, the only simulation
// framework in GPUSPH, presently).

// The CUDASimFramework is a template class depending on KernelType, ViscSpec,
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
template<KernelType kerneltype, typename ViscSpec,
	BoundaryType boundarytype, flag_t simflags>
struct CUDABoundaryConditionsSelector
{
	typedef CUDABoundaryConditionsEngine<kerneltype, ViscSpec, boundarytype, simflags> BCEtype;
	static BCEtype* select()
	{ return NULL; } // default, no BCE
};

// SA_BOUNDARY specialization
template<KernelType kerneltype, typename ViscSpec, flag_t simflags>
struct CUDABoundaryConditionsSelector<kerneltype, ViscSpec, SA_BOUNDARY, simflags>
{
	typedef CUDABoundaryConditionsEngine<kerneltype, ViscSpec, SA_BOUNDARY, simflags> BCEtype;
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
	DensityDiffusionType _densitydiffusiontype,
	RheologyType _rheologytype,
	TurbulenceModel _turbmodel,
	ComputationalViscosityType _compvisc,
	ViscousModel _viscmodel,
	AverageOperator _viscavgop,
	LegacyViscosityType _legacyvisctype,
	BoundaryType _boundarytype,
	Periodicity _periodicbound,
	flag_t _simflags,
	bool _is_const_visc = (_legacyvisctype == KINEMATICVISC) || (
		IS_SINGLEFLUID(_simflags) &&
		(_rheologytype == NEWTONIAN) &&
		(_turbmodel != KEPSILON)
	),
	bool invalid_combination =
		// Currently, we consider invalid only the case
		// of SA_BOUNDARY

		// TODO extend to include all unsupported/untested combinations for other boundary conditions

		(_legacyvisctype == KINEMATICVISC && IS_MULTIFLUID(_simflags)) || // kinematicvisc model only made sense for single-fluid
		(_turbmodel == KEPSILON && _boundarytype != SA_BOUNDARY) || // k-epsilon only supported in SA currently
		(_boundarytype == SA_BOUNDARY && (
			// viscosity
			_viscavgop != ARITHMETIC		||	// untested
			_turbmodel == SPS			||	// untested
			_turbmodel == ARTIFICIAL		||	// untested (use is discouraged, use density diffusion instead)
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
			(_simflags & ENABLE_DENSITY_SUM && _simflags & ENABLE_GAMMA_QUADRATURE)
												// enable density sum only works with the dynamic equation for gamma,
												// so gamma quadrature must be disabled
		)
	)
	||
	(
	!(_boundarytype == SA_BOUNDARY) && _simflags & ENABLE_DENSITY_SUM
												// density sum is untested with boundary conditions other than SA
	)
>
class CUDASimFrameworkImpl : public SimFramework,
	private InvalidOptionCombination<invalid_combination>
{
public:
	static const KernelType kerneltype = _kerneltype;
	static const SPHFormulation sph_formulation = _sph_formulation;
	static const DensityDiffusionType densitydiffusiontype = _densitydiffusiontype;

	static const RheologyType rheologytype = _rheologytype;
	static const TurbulenceModel turbmodel = _turbmodel;
	static const ComputationalViscosityType compvisc = _compvisc;
	static const ViscousModel viscmodel = _viscmodel;
	static const AverageOperator viscavgop = _viscavgop;
	static const bool is_const_visc = _is_const_visc;

	using ViscSpec = FullViscSpec<_rheologytype, _turbmodel, _compvisc,
	      _viscmodel, _viscavgop, _simflags, _is_const_visc>;

	static const BoundaryType boundarytype = _boundarytype;
	static const Periodicity periodicbound = _periodicbound;
	static const flag_t simflags = _simflags;

public:
	CUDASimFrameworkImpl() : SimFramework()
	{
		m_neibsEngine = new CUDANeibsEngine<sph_formulation, boundarytype, periodicbound, true>();
		m_integrationEngine = new CUDAPredCorrEngine<sph_formulation, boundarytype, kerneltype, ViscSpec, simflags>();
		m_viscEngine = new CUDAViscEngine<ViscSpec, kerneltype, boundarytype>();
		m_forcesEngine = new CUDAForcesEngine<kerneltype, sph_formulation, densitydiffusiontype, ViscSpec, boundarytype, simflags>();
		m_bcEngine = CUDABoundaryConditionsSelector<kerneltype, ViscSpec, boundarytype, simflags>::select();

		// TODO should be allocated by the integration scheme
		m_allocPolicy = new PredCorrAllocPolicy();

		m_simparams = new SimParams(this);
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
			return new CUDAPostProcessEngine<VORTICITY, kerneltype, boundarytype, simflags>(options);
		case TESTPOINTS:
			return new CUDAPostProcessEngine<TESTPOINTS, kerneltype, boundarytype, simflags>(options);
		case SURFACE_DETECTION:
			return new CUDAPostProcessEngine<SURFACE_DETECTION, kerneltype, boundarytype, simflags>(options);
		case FLUX_COMPUTATION:
			return new CUDAPostProcessEngine<FLUX_COMPUTATION, kerneltype, boundarytype, simflags>(options);
		case CALC_PRIVATE:
			return new CUDAPostProcessEngine<CALC_PRIVATE, kerneltype, boundarytype, simflags>(options);
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
	constexpr operator T() const { return _val; }; // allow automatic conversion to the type
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
	typename Arg4, typename Arg5, typename Arg6,
	typename Arg7, typename Arg8, typename Arg9,
	typename Arg10, typename Arg11, typename Arg12>
struct ArgSelector :
	virtual public MultiplexSubclass<Arg1,1>,
	virtual public MultiplexSubclass<Arg2,2>,
	virtual public MultiplexSubclass<Arg3,3>,
	virtual public MultiplexSubclass<Arg4,4>,
	virtual public MultiplexSubclass<Arg5,5>,
	virtual public MultiplexSubclass<Arg6,6>,
	virtual public MultiplexSubclass<Arg7,7>,
	virtual public MultiplexSubclass<Arg9,9>,
	virtual public MultiplexSubclass<Arg10,10>,
	virtual public MultiplexSubclass<Arg11,11>,
	virtual public MultiplexSubclass<Arg12,12>
{};

// Now we set the defaults for each argument
struct TypeDefaults
{
	typedef TypeValue<KernelType, WENDLAND> Kernel;
	typedef TypeValue<SPHFormulation, SPH_F1> Formulation;
	typedef TypeValue<DensityDiffusionType, DENSITY_DIFFUSION_NONE> DensityDiffusion;
	typedef TypeValue<RheologyType, INVISCID> Rheology;
	typedef TypeValue<TurbulenceModel, ARTIFICIAL> Turbulence;
	typedef TypeValue<ComputationalViscosityType, KINEMATIC> ComputationalViscosity;
	typedef TypeValue<ViscousModel, MORRIS> ViscModel;
	typedef TypeValue<AverageOperator, ARITHMETIC> ViscAveraging;
	typedef TypeValue<LegacyViscosityType, INVALID_VISCOSITY> LegacyViscType;
	typedef TypeValue<BoundaryType, LJ_BOUNDARY> Boundary;
	typedef TypeValue<Periodicity, PERIODIC_NONE> Periodic;
	typedef TypeValue<flag_t, ENABLE_NONE> Flags;
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

#define DEFINE_ARGSELECTOR(selector, SelectorType, ArgName) \
template<SelectorType value__, typename ParentArgs=TypeDefaults> \
struct selector : virtual public ParentArgs \
{ \
	typedef TypeValue<SelectorType, value__> ArgName; \
	template<typename NewParent> struct reparent : \
		virtual public selector<value__, NewParent> {}; \
}

// Kernel override
DEFINE_ARGSELECTOR(kernel, KernelType, Kernel);

// Formulation override
DEFINE_ARGSELECTOR(formulation, SPHFormulation, Formulation);

// Density diffusion override
DEFINE_ARGSELECTOR(densitydiffusion, DensityDiffusionType, DensityDiffusion);

// Rheology override
DEFINE_ARGSELECTOR(rheology, RheologyType, Rheology);

// Turbulence model override
DEFINE_ARGSELECTOR(turbulence_model, TurbulenceModel, Turbulence);

// ComputationalViscosity override
DEFINE_ARGSELECTOR(computational_visc, ComputationalViscosityType, ComputationalViscosity);

// ViscousModel override
DEFINE_ARGSELECTOR(visc_model, ViscousModel, ViscModel);

// AverageOperator override
DEFINE_ARGSELECTOR(visc_average, AverageOperator, ViscAveraging);

template<LegacyViscosityType visctype, typename ParentArgs=TypeDefaults>
struct viscosity : virtual public ParentArgs
{
	// propagate the information about the fact that the user
	// specified the given legacy type
	typedef TypeValue<LegacyViscosityType, visctype> LegacyViscType;

	// set the corresponding viscous model parameters
	using Spec = typename ConvertLegacyVisc<visctype>::type;
	typedef TypeValue<RheologyType, Spec::rheologytype> Rheology;
	typedef TypeValue<TurbulenceModel, Spec::turbmodel> Turbulence;
	typedef TypeValue<ComputationalViscosityType, Spec::compvisc> ComputationalViscosity;
	typedef TypeValue<ViscousModel, Spec::viscmodel> ViscModel;
	typedef TypeValue<AverageOperator, Spec::avgop> ViscAveraging;

	template<typename NewParent> struct reparent :
		virtual public viscosity<visctype, NewParent> {};
};

// Boundary override
DEFINE_ARGSELECTOR(boundary, BoundaryType, Boundary);

// Periodic override
DEFINE_ARGSELECTOR(periodicity, Periodicity, Periodic);

// Flags override
DEFINE_ARGSELECTOR(flags, flag_t, Flags);

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
	typename Arg6 = DefaultArg,
	typename Arg7 = DefaultArg,
	typename Arg8 = DefaultArg,
	typename Arg9 = DefaultArg,
	typename Arg10 = DefaultArg,
	typename Arg11 = DefaultArg,
	typename Arg12 = DefaultArg>
class CUDASimFramework {
	/// The collection of arguments for our current setup
	typedef ArgSelector<Arg1, Arg2, Arg3, Arg4, Arg5, Arg6,
		Arg7, Arg8, Arg9, Arg10, Arg11, Arg12> Args;

	/// Comfort static defines
	static const KernelType kerneltype = Args::Kernel::value;
	static const SPHFormulation sph_formulation = Args::Formulation::value;
	static const DensityDiffusionType densitydiffusiontype = Args::DensityDiffusion::value;

	static const RheologyType rheologytype = Args::Rheology::value;
	static const TurbulenceModel turbmodel = Args::Turbulence::value;
	static const ComputationalViscosityType compvisc = Args::ComputationalViscosity::value;
	static const ViscousModel viscmodel = Args::ViscModel::value;
	static const AverageOperator viscavgop = Args::ViscAveraging::value;

	static const BoundaryType boundarytype = Args::Boundary::value;
	static const Periodicity periodicbound = Args::Periodic::value;
	static const flag_t simflags = Args::Flags::value;

	/// The CUDASimFramework implementation of the current setup
	typedef CUDASimFrameworkImpl<
			kerneltype,
			sph_formulation,
			densitydiffusiontype,
			rheologytype,
			turbmodel,
			compvisc,
			viscmodel,
			viscavgop,
			Args::LegacyViscType::value,
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
		throw runtime_error("invalid selector value");
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
		throw runtime_error("invalid selector value");
	}

	/// Chained selectors (for multiple overrides)
	template<typename E1, typename E2, typename SelO, typename Other>
	SimFramework * select_options(bool s1, E1, bool s2, E2, SelO sel_o, Other)
	{
		if (s1)
			return extend<E1>().select_options(s2, E2(), sel_o, Other());
		return this->select_options(s2, E2(), sel_o, Other());
	}


};

#endif

/* vim: set ft=cuda sw=4 ts=4 : */
