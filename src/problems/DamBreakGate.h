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

/*
 * File:   DamBreakGate.h
 * Author: Alexis; Tony
 *
 * Created on 28 janvier 2009; Feb 2011-2013
 */

#ifndef _DAMBREAKGATE_H
#define	_DAMBREAKGATE_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"
#include "Vector.h"

class DamBreakGate: public Problem {
	private:
		Cube		experiment_box;
		Cube		obstacle;
		Rect		gate;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	obstacle_parts;
		PointVect	gate_parts;
		float		H;  // still watr level

	public:
		DamBreakGate(GlobalData *);
		virtual ~DamBreakGate(void);

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void moving_bodies_callback(const uint, Object*, const double, const double, const float3&,
			 	 	 	 	 	 	const float3&, const KinematicData &, KinematicData &,
			 	 	 	 	 	 	double3&, EulerParameters&);
		void release_memory(void);
};
#endif	/* _DAMBREAKGATE_H */


