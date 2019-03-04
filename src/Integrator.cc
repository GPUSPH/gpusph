/*  Copyright 2019 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#include <string>
#include <stdexcept>

#include "GlobalData.h"

#include "Integrator.h"

// Include all known integrator type headers, for the make_integrator switch
#include "PredictorCorrectorIntegrator.h"

/*! \file
 * Integrator implementation. For the time being this implements the predictor/corrector integration scheme only,
 * it will be refactored later to include other as well.
 */

using namespace std;

shared_ptr<Integrator>
Integrator::instance(IntegratorType type, GlobalData const* gdata)
{
	switch (type)
	{
	case PREDITOR_CORRECTOR:
		return make_shared<PredictorCorrector>(gdata);
	default:
		throw out_of_range("no known interator #" + to_string(type));
	}
}

Integrator::Phase *
Integrator::enter_phase(size_t phase_idx)
{
	if (phase_idx >= m_phase.size())
		throw runtime_error("trying to enter non-existing phase #" + to_string(phase_idx));

	m_phase_idx = phase_idx;
	if (gdata->debug.print_step)
		cout << "Entering phase " << current_phase()->name() << endl;
	Phase *phase = current_phase();
	phase->reset();
	return phase;
}
