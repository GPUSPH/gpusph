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
#include "Cube.h"
#include "Rect.h"
#include "Cylinder.h"
#include "Vector.h"
#include "Cone.h"
#include "Torus.h"
#include "Circle.h"
#include "Sphere.h"
#include "Cone.h"

class EnergyGenerator: public Problem {
	private:
		//Testpoints
		int			wmakertype;
		Cube		experiment_box;
		Rect        experiment_box1;
		int			i_use_bottom_plane;
		
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	paddle_parts;

		Cylinder	cyl1, cyl2;
		Torus		torus;
		Circle		circle;
		Sphere		sphere;
		Cone		cone;
		Cube		cube1, cube2;

		float		paddle_length;
		float		paddle_width;
		float		h_length, height, slope_length, beta;
	    float		H;		// still water level
		float		Hbox;	// height of experiment box

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

