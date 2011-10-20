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
/*
 * File:   DamBreak3D.h
 * Author: alexis
 *
 * Created on 28 janvier 2009, 00:44
 */

#ifndef _DAMBREAKOBJECTS_H
#define	_DAMBREAKOBJECTS_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Sphere.h"
#include "Cone.h"
#include "Torus.h"


class DamBreakObjects: public Problem {
	private:
		Cube		experiment_box;
		Cube		obstacle;
		Cone		object1;
		Cube		object2;
		Sphere		object3;
		Torus		object4;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	obstacle_parts;
		double		H;				// still water level
		double		lx, ly, lz;		// dimension of experiment box
		bool		wet;			// set wet to true have a wet bed experiment
		

	public:
		DamBreakObjects(const Options &);
		~DamBreakObjects(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);

		void release_memory(void);
};
#endif	/* _DAMBREAK3D_H */

