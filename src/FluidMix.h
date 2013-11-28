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

#ifndef _FLUIDMIX_H
#define	_FLUIDMIX_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"
#include "Cylinder.h"
#include "Vector.h"
#include "Cone.h"


class FluidMix: public Problem {
	private:
		Cube		experiment_box;
		PointVect	parts0, parts1;
		PointVect	boundary_parts;
		float		lx, ly, lz;
	    float		H;		// still water level


	public:
		FluidMix(const Options &);
		virtual ~FluidMix(void);
		int fill_parts(void);

		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *, uint*);

		void release_memory(void);
};
#endif	


