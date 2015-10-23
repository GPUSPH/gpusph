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
		TopoCube	*experiment_box;
		PointVect	boundary_parts;
		PointVect	piston_parts;
		PointVect	parts;
		double		H;				// still water level

	public:
		TestTopo(GlobalData *);

		virtual ~TestTopo(void);

		int fill_parts(void);

		void copy_to_array(BufferList &);
		void copy_planes(PlaneList& planes);

		// override standard split
		void fillDeviceMap();

		void release_memory(void);
};
#endif	/* _TESTTOPO_H */

