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

#include "Problem.h"
#include "Point.h"
#include "Cylinder.h"
#include "Vector.h"
#include "Cube.h"

class OffshorePile: public Problem {
	private:
		PointVect	parts;
		PointVect	boundary_parts;

		Cylinder	cyl;
		dJointID	joint;
		Cube        piston;
		double		piston_height, piston_width;
		double		h_length, height, slope_length, beta;
		double		H;		// still water level
		double		lx, ly, lz;		// dimension of water tank
		double 		x0;
		double		cyl_xpos, cyl_height, cyl_diam, cyl_rho;

		// Moving boundary data
		double		piston_amplitude, piston_omega;
		double3     piston_origin;
		double		piston_tstart, piston_tend;

		int 		layers;		// Number of particles layers for dynamic boundaries
	public:
		OffshorePile(GlobalData *);
		~OffshorePile(void);
		int fill_parts(void);

		void copy_to_array(BufferList &);

		void moving_bodies_callback(const uint, Object*, const double, const double, const float3&,
									const float3&, const KinematicData &, KinematicData &,
									double3&, EulerParameters&);

		void release_memory(void);
};
#endif	/* _OffshorePile_H */
