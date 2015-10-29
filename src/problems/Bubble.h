/*  Copyright 2015 Giuseppe Bilotta, Alexis Herault, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

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

#ifndef _BUBBLE_H
#define	_BUBBLE_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"

class Bubble: public Problem {
	private:
		Cube		experiment_box;
		Cube		fluid;
		PointVect	fluid_parts;
		PointVect	boundary_parts;
		uint		dyn_layers; // number of layers for DYN_BOUNDARY
		double3		extra_offset; // offset caused by DYN_BOUNDARY
		float		H;  // still water level
		float		R;  // sphere radius
		double		lx, ly, lz;		// dimension of experiment box

	public:
		Bubble(GlobalData *);
		~Bubble(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_planes(PlaneList&);
		void copy_to_array(BufferList &);

		void release_memory(void);
};
#endif	/* _BUBBLE_H */


