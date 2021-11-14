/*  Copyright (c) 2011-2019 INGV, EDF, UniCT, JHU

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

#ifndef _SLIDINGWEDGE_H
#define	_SLIDINGWEDGE_H

#define PROBLEM_API 1
#include "Problem.h"

class SlidingWedge: public Problem {
private:
	const bool use_ccsph;
	const uint dim;
	const uint ppm; // particles per meter
	const DensityDiffusionType rho_diff;

	const double H; // water depth
	const double wedge_mass; // mass of the wedge
	Vector slope_dir; // slope direction
	const double tstart; // time at which the wedge is released

	void setup_framework();
public:
	SlidingWedge(GlobalData *);

	// Prescribed motion callback
	void moving_body_dynamics_callback
		( const uint index ///< sequential index of the moving body
		, ObjectPtr object ////< pointer to the moving body object
		, const double t0 ///< time at the beginning of the timestep
		, const double t1 ///< time at the end of the timestep
		, const double dt ///< timestep
		, const int step  ///< integration step (0 = predictor, 1 = corrector)
		, float3 const& force ///< force exherted on the body by the fluid
		, float3 const& torque ///< torque exherted on the body by the fluid
		, KinematicData const& initial_kdata // kinematic data at time t = 0
		, KinematicData const& kdata0 // kinematic data at time t = t0
		, KinematicData& kdata ///< kinematic body data at time t = t1 (computed by the callback)
		, AccelerateData& adata ///< acceleration at time t = t1 (computed by the callback)
		, double3& dx ///< translation to be applied at time t = t1
		, EulerParameters& dr ////< rotation to be applied at time t = t1
		) override;
};
#endif
