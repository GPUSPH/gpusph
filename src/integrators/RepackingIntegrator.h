/*  Copyright (c) 2019 INGV, EDF, UniCT, JHU

    Istituto Nazionale di Geofisica e Vulcanologia, Sezione di Catania, Italy
    Électricité de France, Paris, France
    Università di Catania, Catania, Italy
    Johns Hopkins University, Baltimore (MD), USA

    This file is part of GPUSPH. Project founders:
        Alexis Hérault, Giuseppe Bilotta, Robert A. Dalrymple,
        Eugenio Rustico, Ciro Del Negro
    For a full list of authors and project partners, consult the logs
    and the project website <https://www.gpusph.org>

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

/*! \file
 * Repacking integrator
 */

#include "Integrator.h"

#include "particledefine.h"

// TODO this is currently obtained copying over the PredictorCorrector
// as a template, common code should be shared instead!
class RepackingIntegrator : public Integrator
{
	// Phases implemented in this integrator, will be used as index
	// in the m_phase list
	enum PhaseCode {
		POST_UPLOAD, // post-upload phase

		NEIBS_LIST, // neibs list construction phase

		INITIALIZATION, // initialization phase, after upload and neibs list, but before the main integrator loop

		BEGIN_TIME_STEP, // integrator step initialization phase
		REPACKING, // repacking phase
		FINISH_REPACKING, // end-of-repacking phase
		PREPARE_SIMULATION, // rerun the initialization phase, in preparation for the actual simulation

		NUM_PHASES, // number of phases
	};

	bool m_entered_main_cycle;
	bool m_finished_main_cycle;

	static std::string getCurrentStateForStep(int step_num);
	static std::string getNextStateForStep(int step_num);

	// initialize the command sequence for boundary models (one per boundary)
	template<BoundaryType boundarytype>
	void initializeBoundaryConditionsSequence(Phase *this_phase, StepInfo const& step);

	Phase* initializeInitializationSequence(StepInfo const& step_num);
	Phase* initializeRepackingSequence(StepInfo const& step_num);

	template<PhaseCode phase>
	void initializePhase();

	// Determine the phase following phase cur
	PhaseCode phase_after(PhaseCode cur);

public:
	// From the generic Integrator we only reimplement
	// the constructor, that will initialize the integrator phases
	RepackingIntegrator(GlobalData const* _gdata);

	void start() override
	{ enter_phase(POST_UPLOAD); }

	// Override the end-of-simulation call: after the repacking main loop,
	// we need to reset free surface boundary particles and recompute gamma.
	void we_are_done() override
	{ m_finished_main_cycle = true; }

	Phase *next_phase() override;
};
