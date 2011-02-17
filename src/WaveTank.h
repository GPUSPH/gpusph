/*
 * File:   WaveTank.h
 * Author: Tony (following Alexis' PaddleTest3D)
 *
 * Created on 29 janvier 2009, 22:42
 */

#ifndef _WAVETANK_H
#define	_WAVETANK_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"
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

#include "Cylinder.h"
#include "Vector.h"
#include "Cone.h"

class WaveTank: public Problem {
	private:
		int			icyl, icone, wmakertype;
		Cube		experiment_box;
		Rect        experiment_box1;
		int			i_use_bottom_plane;
		Point		p1,p2;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	paddle_parts, gate_parts;

		Cylinder	cyl1, cyl2, cyl3, cyl4;
		Cylinder	cyl5, cyl6, cyl7;
		Cylinder	cyl8, cyl9, cyl10;
		Cylinder	cyl11;
		Cone 		cone;
		float		paddle_length;
		float		paddle_width;
		float		h_length, height, slope_length, beta;
	    float		H;		// still water level
		float		Hbox;	// height of experiment box


	public:
		WaveTank(const Options &);
		~WaveTank(void);
		int fill_parts(void);
		uint fill_planes(void);
		void copy_planes(float4*, float*);

		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);
		MbCallBack& mb_callback(const float, const float, const int);

		void release_memory(void);
};
#endif	/* _WAVETANK_H */

