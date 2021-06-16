/*  Copyright (c) 2021 INGV, EDF, UniCT, JHU

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

#ifndef LIDDRIVENCAVITY2D_H
#define	LIDDRIVENCAVITY2D_H

#define PROBLEM_API 1
#include "Problem.h"

class LidDrivenCavity2D: public Problem {
	const double lid_vel;
	const double lead_in_time;
public:
	LidDrivenCavity2D(GlobalData *);
	// we override the moving bodies callback to set the lid velocity
	void moving_bodies_callback(
		const uint index,
		Object *object,
		const double t0,
		const double t1,
		const float3& force,
		const float3& torque,
		const KinematicData& initial_kdata,
		KinematicData& kdata,
		double3& dx,
		EulerParameters& dr) override;
};
#endif	/* LIDDRIVENCAVITY2D_H */

