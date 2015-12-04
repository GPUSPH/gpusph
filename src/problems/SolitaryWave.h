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
 * File:   SolitaryWave.h
 * Author: rad
 *
 * Created on February 7, 2011-2013, 2:39 PM
 */

#ifndef SolitaryWave_H
#define	SolitaryWave_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"
#include "Cylinder.h"
#include "Vector.h"
#include "Cone.h"

class SolitaryWave: public Problem {
	private:
		int			icyl, icone, wmakertype;
		Cube		experiment_box;
		Rect		experiment_box1;
		Rect		piston;
		int			i_use_bottom_plane;
		PointVect	parts;
		PointVect	boundary_parts;

		Cylinder	cyl[10];
		Cone 		cone;

		double 		lx, ly, lz;	// Dimension of the computational domain
		double		h_length;	// Horizontal part of the experimental domain
		double		slope_length;	// Length of the inclined plane
		double		height;		// Still water (with z origin on the horizontal part)
		double		beta;		// Angle of the inclined plane
        double		H;		// still water level
		double		Hbox;	// height of experiment box

		// Moving boundary data
		double		a, b, c;
		double 		piston_tstart, piston_tend;
		double		piston_initial_crotx;

	public:
		SolitaryWave(GlobalData *);
		~SolitaryWave(void);
		int fill_parts(void);
		void copy_planes(PlaneList &);

		void copy_to_array(BufferList &);
		void moving_bodies_callback(const uint, Object*, const double, const double, const float3&,
			 	 	 	 	 	 	const float3&, const KinematicData &, KinematicData &,
			 	 	 	 	 	 	double3&, EulerParameters&);

		void release_memory(void);
};
#endif	/* _SolitaryWave_H */

