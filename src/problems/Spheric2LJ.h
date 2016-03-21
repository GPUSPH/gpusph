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

#ifndef _SPHERIC2LJ_H
#define	_SPHERIC2LJ_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"

class Spheric2LJ: public Problem {
	private:
		Cube		experiment_box;
		Cube		obstacle;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	test_points;
		PointVect	obstacle_parts;
		float		H;				// still water level
		double		lx, ly, lz;		// dimension of experiment box
		bool		wet;			// set wet to true have a wet bed experiment
		bool		m_usePlanes;	// use planes or boundaries

	public:
		Spheric2LJ(GlobalData *);
		virtual ~Spheric2LJ(void);

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void copy_planes(PlaneList &);

		// override standard split
		void fillDeviceMap();

		void release_memory(void);
};
#endif	/* _SPHERIC2LJ_H */

