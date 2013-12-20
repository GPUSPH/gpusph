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
 * File:   FishPass.h
 * Author: rustico
 *
 */

#ifndef _WATERFALL_H
#define	_WATERFALL_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"

class Waterfall: public Problem {
	private:
		// parameters - see constructor for details
		float WORLD_LENGTH, WORLD_WIDTH;
		float SIDES_HEIGHT, STEP_HEIGHT;
		float WATER_LEVEL;

		Rect upperFloor, loweFloor, step;
		Rect right_side, left_side;

		Cube upperFluid, lowerFluid;

		PointVect fluid_parts;
		PointVect floor_parts;
		PointVect walls_parts;
		float H;  // still watr level
		//double		lx, ly, lz;		// dimension of experiment box
		//bool		wet;			// set wet to true have a wet bed experiment

	public:
		Waterfall(const Options &);
		virtual ~Waterfall(void);

		int fill_parts(void);
		void copy_to_array(float4 *, float4 *, particleinfo *);

		// override standard split
		void fillDeviceMap(GlobalData* gdata);

		void release_memory(void);
};
#endif	/* _WATERFALL_H */

