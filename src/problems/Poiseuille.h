/*  Copyright (c) 2018-2019 INGV, EDF, UniCT, JHU

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
#ifndef POISEUILLE_H_
#define POISEUILLE_H_

#define PROBLEM_API 1
#include "Problem.h"

#ifndef POISEUILLE_PROBLEM
#define POISEUILLE_PROBLEM Poiseuille
#endif

class POISEUILLE_PROBLEM: public Problem {

	private:
		// box dimensions. The origin is assumed to be in the middle,
		// so that the domains extents are [-lx/2, lx/2], [-ly/2, ly/2], [-lz/2, lz/2]
		const double lz, ly, lx; // declared in reverse order because we initialize lz first

		// fluid density: defaults at 1, can be set to something different
		// to ensure the correctness of the use of kinematic vs dynamic viscosity
		const float rho;
		// kinematic viscosity of the fluid
		const float kinvisc;
		// driving force magnitude: can be changed to check the behavior for
		// different Reynolds numbers
		const float driving_force;

		// yield strength of the fluid
		// this will be autocomputed (if appropriate for the rheology)
		// to achieve a plug about half the channel height
		const float ys;
		// maximum theoretical flow velocity (computed from other flow parameters)
		const float max_vel;
		// Reynolds number (computed from other flow parameters)
		const float Re;

	public:
		POISEUILLE_PROBLEM(GlobalData *);
		void initializeParticles(BufferList &, const uint);
		float compute_poiseuille_vel(float) const;
};
#endif

