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
 * File:   testtopo.h
 * Author: alexis
 *
 * Created on 20 mars 2009, 20:33
 */

#ifndef _TESTTOPO_H
#define	_TESTTOPO_H

#include "Problem.h"
#include "Point.h"
#include "TopoCube.h"

class TestTopo: public Problem {
	private:
		TopoCube	experiment_box;
		PointVect	boundary_parts;
		PointVect	piston_parts;
		PointVect	parts;
		float		H;  // still watr level
		float		north, south, east, west;
		float		nsres, ewres;

	public:
		TestTopo(const Options &);

		~TestTopo(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4*, float4*, particleinfo*);

		void release_memory(void);
};
#endif	/* _TESTTOPO_H */

