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

#ifndef _FISHPASS_H
#define	_FISHPASS_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"

class FishPass: public Problem {
	private:
		// parameters - see constructor for details
		float WALLS_THICKNESS;
		float A_LENGTH, A_YOFFSET;
		float B_LENGTH;
		float C_LENGTH, C_XOFFSET;
		float POOL_LENGTH, POOL_WIDTH;
		uint POOLS;
		float ROUNDNESS_RAY;
		float POOL_WATER_LEVEL;
		float INFLOW_BOX_HEIGHT, INFLOW_BOX_LENGTH;
		float INFLOW_CHANNEL_HEIGHT, INFLOW_WATER_LEVEL;
		float OUTFLOW_BOX_HEIGHT, OUTFLOW_BOX_LENGTH;
		float OUTFLOW_WATER_LEVEL;
		float INLET_LENGTH, OUTLET_LENGTH, OUTLET_HEIGHT, OUTLET_ZOFFSET;
		float INLET_SIDE_WALLS_DISTANCE;
		float POST_INFLOW_YOFFSET, PRE_OUTFLOW_YOFFSET;
		float HORIZONTAL_SLOPE_LENGTH;
		float TOTAL_HEIGHT, TOTAL_LENGTH;
		float WORLD_OFFSET_X, WORLD_OFFSET_Y, WORLD_OFFSET_Z;
		float OUTFLOW_RAMP_HEIGHT, OUTFLOW_RAMP_LENGTH;
		bool MAX_WALL_HEIGHT;
		float3 INITIAL_VELOCITY;
		bool PERIODIC;

		Rect slope_floor, inflow_floor, outflow_floor, outflow_ramp;
		Rect right_side, left_side, back_side, front_side;
		Rect inflow_lid;
		// arrays; for each pool, 3 walls {a,b,c}; for each wall, Front Back Side rects
		Rect *pAf, *pAb, *pAs;
		Rect *pBf, *pBb, *pBs;
		Rect *pCf, *pCb, *pCs;

		Cube inboxFluid, outboxFluid;

		PointVect fluid_parts;
		PointVect floor_parts;
		PointVect sides_parts;
		PointVect walls_parts;
		float H;  // still watr level
		//double		lx, ly, lz;		// dimension of experiment box
		//bool		wet;			// set wet to true have a wet bed experiment

		void addWalls(uint wIndex, float Ypos);
		void addWater(float Ypos, float specialSize = -1.0F);
		void drawWalls();
		float getAbsoluteFloorHeight(float Ypos);
		void fillHFrustum(float3 base1Origin, float2 base1Size, float3 base2Origin, float2 base2Size);
	public:
		FishPass(const Options &);
		virtual ~FishPass(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);

		void release_memory(void);
};
#endif	/* _FISHPASS_H */

