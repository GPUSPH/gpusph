/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

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

#ifndef ENERGYGENERATOR_H
#define	ENERGYGENERATOR_H

#include "Problem.h"
#include "Point.h"
#include "Vector.h"
#include "Cube.h"
#include "Rect.h"
#include "Cylinder.h"
#include "Torus.h"
#include "Sphere.h"


class EnergyGenerator: public Problem {
	private:
		bool		use_bottom_plane;
		Cube		experiment_box;
		Rect        bottom_rect;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	paddle_parts;

		Cylinder	cyl;
		Sphere 		torus;
		double		paddle_length;
		double		paddle_width;
		double		h_length, height, slope_length, beta;
	    double		H;		// still water level
		double		lx, ly, lz;		// dimension of experiment box

	public:
		EnergyGenerator(const Options &);
		~EnergyGenerator(void);
		
		int fill_parts(void);
		uint fill_planes(void);
		void copy_planes(float4*, float*);

		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);
		MbCallBack& mb_callback(const float, const float, const int);

		void release_memory(void);
};
#endif	/* ENERGYGENERATOR_H */

