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

#include "simflags.h"
#include "buildneibs.cuh"
#include "euler.cuh"
#include "forces.cuh"

// TODO FIXME find a way to make template parameters optional / not order dependent,
// most likely with an auxiliary (set of) constructors

// Auxiliary functor to select a BoundaryConditionsEngine, currently
// simply selects the CUDABoundaryConditionsEngine (which is only for SA) in the
// SA_BOUNDARY case, NULL otherwise

template<KernelType kerneltype, ViscosityType visctype,
	BoundaryType boundarytype, flag_t simflags>
struct CUDABoundaryConditionsSelector
{
	typedef CUDABoundaryConditionsEngine<kerneltype, visctype, boundarytype, simflags> BCEtype;
	static BCEtype* select()
	{ return NULL; } // default, no BCE
};

template<KernelType kerneltype, ViscosityType visctype, flag_t simflags>
struct CUDABoundaryConditionsSelector<kerneltype, visctype, SA_BOUNDARY, simflags>
{
	typedef CUDABoundaryConditionsEngine<kerneltype, visctype, SA_BOUNDARY, simflags> BCEtype;
	static BCEtype* select()
	{ return new BCEtype(); } // TODO fix when we have proper BCEs
};


template<
	KernelType kerneltype,
	SPHFormulation sph_formulation,
	ViscosityType visctype,
	BoundaryType boundarytype,
	Periodicity periodicbound,
	flag_t simflags>
class CUDASimFramework : public SimFramework
{
public:
	CUDASimFramework() {
		m_neibsEngine = new CUDANeibsEngine<boundarytype, periodicbound, true>();
		m_integrationEngine = new CUDAPredCorrEngine<boundarytype, simflags & ENABLE_XSPH>();
		m_viscEngine = new CUDAViscEngine<visctype, kerneltype, boundarytype>();
		m_forcesEngine = new CUDAForcesEngine
			<kerneltype, sph_formulation, visctype, boundarytype, simflags>();

		m_bcEngine = CUDABoundaryConditionsSelector
			<kerneltype, visctype, boundarytype, simflags>::select();
		m_postprocEngine = NULL; // TODO

		m_simparams.kerneltype = kerneltype;
		m_simparams.sph_formulation = sph_formulation;
		m_simparams.visctype = visctype;
		m_simparams.boundarytype = boundarytype;
		m_simparams.periodicbound = periodicbound;
		m_simparams.xsph = simflags & ENABLE_XSPH;
		m_simparams.dtadapt = simflags & ENABLE_DTADAPT;
		m_simparams.usedem = simflags & ENABLE_DEM;
		m_simparams.movingBoundaries = simflags & ENABLE_MOVING_BODIES;
		m_simparams.floatingObjects = simflags & ENABLE_FLOATING_BODIES;
		m_simparams.inoutBoundaries = simflags & ENABLE_INLET_OUTLET;
		m_simparams.ioWaterdepthComputation = simflags & ENABLE_WATER_DEPTH;
	}

	template<FilterType filtertype> AbstractFilterEngine*
	addFilterEngine(int frequency)
	{
		FilterEngineSet::iterator found(m_filterEngines.find(filtertype));
		if (found == m_filterEngines.end()) {
			AbstractFilterEngine *flt = new CUDAFilterEngine<filtertype, kerneltype, boundarytype>();
			m_filterEngines[filtertype] = flt;
			return flt;
		} else {
			// TODO message about override?
			found->second->set_frequency(frequency);
			return found->second;
		}
	}

};

#endif

/* vim: set ft=cuda sw=4 ts=4 : */
