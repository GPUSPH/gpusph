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

#ifndef _SIMFRAMEWORK_H
#define _SIMFRAMEWORK_H

/* The SimFramework encompasses the engines and flags of a simulation */

#include <map>
#include <vector>
#include <stdexcept>

#include "neibsengine.h"
#include "filterengine.h"
#include "integrationengine.h"
#include "viscengine.h"
#include "forcesengine.h"

typedef std::map<FilterType, AbstractFilterEngine *> FilterEngineSet;

// (ordered) list of filters and frequencies pairs
typedef std::vector< std::pair<FilterType, uint> > FilterFreqList;

class SimFramework
{
protected:
	AbstractNeibsEngine *m_neibsEngine;
	FilterEngineSet m_filterEngines;
	AbstractIntegrationEngine *m_integrationEngine;
	AbstractViscEngine *m_viscEngine;
	AbstractForcesEngine *m_forcesEngine;

	SimParams m_simparams;
public:
	AbstractNeibsEngine *getNeibsEngine()
	{ return m_neibsEngine; }
	FilterEngineSet const& getFilterEngines() const
	{ return m_filterEngines; }
	AbstractIntegrationEngine *getIntegrationEngine()
	{ return m_integrationEngine; }
	AbstractViscEngine *getViscEngine()
	{ return m_viscEngine; }
	AbstractForcesEngine *getForcesEngine()
	{ return m_forcesEngine; }

	// add a filter engine, and keep the frequency in the
	// simparams in sync
	template<FilterType filtertype> AbstractFilterEngine*
	addFilterEngine(int frequency)
	{
		throw std::runtime_error("Abstract addFilterEngine called!?");
	}

	SimParams const& get_simparams() const
	{ return m_simparams; }
	SimParams& get_simparams()
	{ return m_simparams; }
};
#endif
