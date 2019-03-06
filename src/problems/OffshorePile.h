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

#ifndef _OFFSHOREPILE_H
#define	_OFFSHOREPILE_H

#include "XProblem.h"

class OffshorePile: public XProblem {
	private:
		double		piston_height, piston_width;
		double		h_length, height, slope_length, beta;
		double		H;		// still water level
		double		lx, ly, lz;		// dimension of water tank
		double		x0;
		double		periodic_offset_y;
		double		cyl_xpos, cyl_height, cyl_diam, cyl_rho;

		// Moving boundary data
		double		piston_amplitude, piston_omega;
		double3		piston_origin;
		double		piston_tstart, piston_tend;

		int			layers;		// Number of particles layers for dynamic boundaries
	public:
		OffshorePile(GlobalData *);
		virtual void moving_bodies_callback(const uint index, Object* object,
				const double t0, const double t1,
				const float3& force, const float3& torque,
				const KinematicData& initial_kdata,
				KinematicData& kdata, double3& dx, EulerParameters& dr);

};
#endif	/* _OFFSHOREPILE_H */
