/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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


#ifndef OILJET_H_
#define OILJET_H_

#define PROBLEM_API 1
#include "Problem.h"

class OilJet: public Problem {
	private:
		double		lx, ly, lz;		// dimension of water tank
		double		water_level;	// water level
		double		inner_diam;		// pipe inner diameter
		double		pipe_length;	// pipe length

		// Moving boundary data
		double		piston_amplitude, piston_omega;
		double3     piston_origin;
		double		piston_tstart, piston_tend;
		double		piston_vel;

		int			layers;		// Number of particles layers for dynamic boundaries
	public:
		OilJet(GlobalData *);

		void moving_bodies_callback(const uint, Object*, const double, const double, const float3&,
									const float3&, const KinematicData &, KinematicData &,
									double3&, EulerParameters&);
};



#endif /* OILJET_H_ */
