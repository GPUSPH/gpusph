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

#ifndef _OPENCHANNEL_H
#define	_OPENCHANNEL_H

#include "Problem.h"
#include "Point.h"
#include "Rect.h"
#include "Cube.h"

class OpenChannel: public Problem {
	private:
		bool		use_side_walls; // use sidewalls or not
		Rect		rect1, rect2, rect3;
		Cube		experiment_box;
		PointVect	parts;
		PointVect	boundary_parts;
		uint		dyn_layers;
		double3		dyn_offset;
		double		a, h, l;  // experiment box dimension
		double		H; // still water level


	public:
		OpenChannel(GlobalData *);
		virtual ~OpenChannel(void);

		int fill_parts(void);
		void copy_to_array(BufferList &);

		void release_memory(void);
};


#endif	/* _POWERLAW_H */
