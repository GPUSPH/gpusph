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

#include "neibsengine.h"
#include "integrationengine.h"
#include "forcesengine.h"

// TODO IntegrationScheme, ViscEngine, vector<PostProcessEngine>

class SimFramework
{
protected:
	AbstractNeibsEngine *m_neibsEngine;
	AbstractIntegrationEngine *m_integrationEngine;
	AbstractForcesEngine *m_forcesEngine;

	SimParams m_simparams;
public:
	AbstractNeibsEngine *getNeibsEngine()
	{ return m_neibsEngine; }
	AbstractIntegrationEngine *getIntegrationEngine()
	{ return m_integrationEngine; }
	AbstractForcesEngine *getForcesEngine()
	{ return m_forcesEngine; }

	SimParams const& get_simparams() const
	{ return m_simparams; }
	SimParams& get_simparams()
	{ return m_simparams; }
};
#endif
