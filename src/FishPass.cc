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

#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif
#include <cmath>
#include <iostream>

#include "FishPass.h"
#include "Cube.h"
#include "Point.h"
#include "Vector.h"


FishPass::FishPass(const Options &options) : Problem(options)
{
	// Fish-pass specific parameters - all sizes are in meters. After the value, the name in the plan is written
	// Wall A: the shorter one; B: the longer; C: small, orthogonal to B
	// Axes: X width towards right (looking at direction of flow), Y length (towards flow), Z opposite to gravity
	// TODO: table for 3 sizes?
	// TODO: the aperture and its projection are optional since they are implictly available depending on
	//	 the other parameters. However, it is more useful to the model if the aperture is the input instead;
	//	 its projection and the length of B should be computed accordingly.
	// TODO: the number of particles can be optimized to increase the amount of fluid fitting in memory. Ex:
	//	- Optional front and back sides
	//	- Do not fill sides below the floor. There are several ways to optimize this
	//	- Reduce inflow, outflow and offsets as much as possible
	//	- ...

	set_deltap(0.02f);
	m_physparams.r0 = m_deltap;
	float r0 = m_physparams.r0;

	// SHRINK_FACTOR = 1.0f; // for scaled models; but it's not that easy
	// total length, all pools included - can be computed as (#pools + size_inlet + size_outlet + pool_length + various thicknesses)
	//LENGTH = 20.0F;
	// total width; since side walls have no thickness, it is equale to the pool width
	//WIDTH = 0.785F;
	// thickness of walls A, B, C
	WALLS_THICKNESS = 0.029; // wall thickness
	// length of wall A measured from side wall
	A_LENGTH = 0.150F; // bu
	// Y offset added to A's position with respect to B; the measure in the plan does not include A's nor B's thickness
	A_YOFFSET = 0.067F + WALLS_THICKNESS; // a (+ thickness of B wall)
	// length of wall B measured from side wall
	B_LENGTH = 0.543F; // t
	// length of wall C measured from B wall; the measure in the plan includes B in its thickness
	C_LENGTH = 0.239F - WALLS_THICKNESS; // c (- thickness of B wall)
	// X offset subtracted to C's position to make its right wall oblique
	C_XOFFSET = 0.05F; // g

	// if true, walls will all as tall as the ceiling; if false, they will be twice the water level
	MAX_WALL_HEIGHT = true;

	// from each B wall (back face) to its next (back face). NOTE: in the map
	// lb does not include B's thickness, while this does include it
	POOL_LENGTH = 0.990F + WALLS_THICKNESS; // lb (+ thickness)

	// number of pools
	POOLS = 9;

	// aperture, orthogonal to the flow - it can be computed from all the other parameters as a check
	// APERTURE = 0.122F; // s
	// X projection of the aperture - it can be computed from all the other parameters as a check
	// X_PROJ_APERTURE = 0.092F; // bs
	// applies to walls A, B and C; no roundness if zero
	ROUNDNESS_RAY = 0.010F; // R
	// make A wall this shorter to preserve aperture size with no roundness
	// ROUNDNESS_CORRECTION = 0.010;
	// internally between side walls; side walls have no thickness
	POOL_WIDTH = 0.785F; // b
	// water level in each pool measured from pool center
	POOL_WATER_LEVEL = 0.25F;

	// size of the in-flow gronud box; the width is POOL_WIDTH
	INFLOW_BOX_HEIGHT = 0.392F - (0.0279F * 2.0F);
	INFLOW_BOX_LENGTH = 1.96F;
	// size of the channel and its water level
	INFLOW_WATER_LEVEL = POOL_WATER_LEVEL;
	INFLOW_CHANNEL_HEIGHT = INFLOW_WATER_LEVEL * 2;

	// same as the in-flow gronud "box"
	OUTFLOW_BOX_HEIGHT = 0.0F;
	//OUTFLOW_BOX_LENGTH = 4.03F;
	OUTFLOW_BOX_LENGTH = 1.0F;
	// should the following water level be a parameter or an outcome?
	OUTFLOW_WATER_LEVEL = POOL_WATER_LEVEL;

	// distances of the pools (B's front) from the edges of the slope
	// WARNING: should be enough to avoid C and fluid compenetration
	POST_INFLOW_YOFFSET = (C_LENGTH + WALLS_THICKNESS + r0) * 2;
	// forcing specific slope length for exact slope
	PRE_OUTFLOW_YOFFSET = 14.3F - (1.019F * 2.0F) - POST_INFLOW_YOFFSET - (POOLS * POOL_LENGTH);
	//PRE_OUTFLOW_YOFFSET = 5.0F - POST_INFLOW_YOFFSET - (POOLS * POOL_LENGTH);
	//PRE_OUTFLOW_YOFFSET = A_YOFFSET + WALLS_THICKNESS + r0;

	// Outflow ramp before the freefall
	OUTFLOW_RAMP_HEIGHT = 0.21F;
	OUTFLOW_RAMP_LENGTH = 0.312F;

	// OUTLET_HEIGHT and INLET_LEGTH will be updated after the physical parameters have been initialized
	OUTLET_LENGTH = 0.2F;
	OUTLET_ZOFFSET = -0.02F; // how much lower should it be with respect to the end of the ramp

	// periodic (gravity driven) or with inlet and outlet
	PERIODIC = false;

	// vars useful for code readability
	HORIZONTAL_SLOPE_LENGTH =
		POST_INFLOW_YOFFSET + POOLS * POOL_LENGTH + PRE_OUTFLOW_YOFFSET;
	// total height from the very bottom to the highest wall, including inlet and outlet channels
	TOTAL_HEIGHT = INFLOW_BOX_HEIGHT + INFLOW_CHANNEL_HEIGHT;
	// total length including inflow, pools, outflow, offsets
	TOTAL_LENGTH = INFLOW_BOX_LENGTH + HORIZONTAL_SLOPE_LENGTH + OUTFLOW_BOX_LENGTH + (PERIODIC ? 0.0f : OUTFLOW_RAMP_LENGTH + OUTLET_LENGTH);

	// TODO: TOTAL_HEIGHT includes the height of the outflow box! This should be
	//	 either drawed or not included. Check if there is any problem with this

	// where to shift the origin
	WORLD_OFFSET_X = - POOL_WIDTH / 2.0F;
	WORLD_OFFSET_Y = - TOTAL_LENGTH / 2.0F;
	// centers correctly along Z also when OUTFLOW_BOX_HEIGHT != 0
	WORLD_OFFSET_Z = - (TOTAL_HEIGHT + OUTFLOW_BOX_HEIGHT)/2.0F;

	// Initial velocity for all fluid particles. Note: velocity only, no pressure component
	INITIAL_VELOCITY = make_float3(0.0F, 0.01F, 0.0F);

	// initialize to 0; it will be set later
	INLET_SIDE_WALLS_DISTANCE = 0;

	// uncomment one or more of the following to disable centering
	//WORLD_OFFSET_X = 0.0F;
	WORLD_OFFSET_Y = 0.0F;
	//WORLD_OFFSET_Z = 0.0F;

	/*
	              |post_off |      |      |       | pre_off |
	  ____________|______   |        POOLS        |         |
	 |            |      ```|`''''''-------......,,,,,,_____|_____________
	 |            |         |                     |         |             |
	 | inflow box |         |                     |         | outflow box |
	 |____________|_________|_____________________|_________|_____________|
	              |<----------------slope------------------>|
	 */

	// here "wall" == {A, B, C}
	uint numWalls = POOLS + 1;
	pAf = new Rect[numWalls];
	pAb = new Rect[numWalls];
	pAs = new Rect[numWalls];
	pBf = new Rect[numWalls];
	pBb = new Rect[numWalls];
	pBs = new Rect[numWalls];
	pCf = new Rect[numWalls];
	pCb = new Rect[numWalls];
	pCs = new Rect[numWalls];

	// TODO: check that deltap is not too big with respect to
	// - WALLS_THICKNESS: must be greater or particles will interact through the wall (exit if so)
	// - Roundness: polygons or lines must be fillable with deltap (r0) spacing
	if (WALLS_THICKNESS < 2*m_deltap)
		printf("WARNING: wall thickness smaller than twice deltap!\n");

	// Check if there is a decent distance from the first wall to the beginning of the slope. It is possible to skip this check if
	// the filling will handle correctly the possibility for the C wall to intersect the inflow channel.
	if (POST_INFLOW_YOFFSET < C_LENGTH + WALLS_THICKNESS)
			printf("WARNING: (post_inflow_offset < c_length + thickness) might result in wrong filling!\n");

	// Same as above, but checking from the last A wall to the end of the slope ( = being of outflow channel)
	if (PRE_OUTFLOW_YOFFSET < A_YOFFSET + WALLS_THICKNESS)
			printf("WARNING: (pre_outflow_offset < a_yoffset + thickness) might result in wrong filling!\n");

	// The inflow channel should be higher than the outflow
	if (INFLOW_BOX_HEIGHT <= OUTFLOW_BOX_HEIGHT)
			printf("WARNING: inflow_box_height is not higher than outflow_box_height!\n");

	// The inflow channel should be higher than the outflow
	if (OUTFLOW_WATER_LEVEL <= OUTFLOW_RAMP_HEIGHT)
			printf("WARNING: outflow_water_level is not higher than outflow_ramp_height!\n");

	//m_size = make_float3(lx, ly, lz);
	float world_width = POOL_WIDTH;
	float world_length = TOTAL_LENGTH; // + margin?
	float world_height = TOTAL_HEIGHT;

	// SPH parameters
	//set_deltap(0.02f); // remember: deltap needs to be set at the beginning of the constructor if it is used for setting geometry
	m_simparams.slength = 1.3f*m_deltap;
	m_simparams.kernelradius = 2.0f;
	m_simparams.kerneltype = WENDLAND;
	m_simparams.dt = 0.0001f;
	m_simparams.xsph = false;
	m_simparams.dtadapt = true;
	m_simparams.dtadaptfactor = 0.3;
	m_simparams.buildneibsfreq = 10;
	m_simparams.shepardfreq = 0;
	m_simparams.mlsfreq = 10;
	//m_simparams.visctype = ARTVISC;
	m_simparams.visctype = KINEMATICVISC;
	//m_simparams.visctype = DYNAMICVISC;
	m_simparams.boundarytype= LJ_BOUNDARY;
	m_simparams.tend = 20.0f;

	// Free surface detection
	m_simparams.surfaceparticle = false;
	m_simparams.savenormals = false;

	// We have no moving boundary
	m_simparams.mbcallback = false;

	// Physical parameters
	H = POOL_WATER_LEVEL;
	m_physparams.gravity = make_float3(0.0, 0.0, -9.81f);
	float g = length(m_physparams.gravity);
	// set_density moved after setting maxvel

	//set p1coeff,p2coeff, epsxsph here if different from 12.,6., 0.5
	m_physparams.dcoeff = 5.0f*g*H;
	m_physparams.r0 = m_deltap;

	// BC when using MK boundary condition: Coupled with m_simsparams.boundarytype=MK_BOUNDARY
	#define MK_par 2
	m_physparams.MK_K = g*H;
	m_physparams.MK_d = 1.1*m_deltap/MK_par;
	m_physparams.MK_beta = MK_par;
	#undef MK_par

	m_physparams.kinematicvisc = 1.0e-6f;
	m_physparams.artvisccoeff = 0.3f;
	m_physparams.epsartvisc = 0.01*m_simparams.slength*m_simparams.slength;


	// Size and origin of the simulation domain
	m_size = make_float3(world_width, world_length, world_height);
	m_origin = make_float3(WORLD_OFFSET_X, WORLD_OFFSET_Y, WORLD_OFFSET_Z);

	m_writerType = VTKWRITER;

	// Y periodicity
	m_simparams.periodicbound = PERIODIC;

	if (PERIODIC) {
		m_physparams.dispvect = make_float3(0.0F, TOTAL_LENGTH, 0.0F);
		m_physparams.minlimit = make_float3(m_origin.x, m_origin.y, m_origin.z);
		m_physparams.maxlimit = make_float3(m_origin.x + m_size.x, m_origin.y + m_size.y, m_origin.z + m_size.z);

		// extra Z offset on Y periodicity (lifting)
		m_physparams.dispOffset = make_float3(0.0F, 0.0F, OUTFLOW_BOX_HEIGHT-INFLOW_BOX_HEIGHT);
	} else { // ! PERIODIC
		// inlet
		// NOTE: in case a higher accuracy is required, use the rounded (inflow/deltap) instead of the actual deltap
		INLET_LENGTH = 8*m_deltap*m_simparams.kernelradius;
		// Particles stuck in local minima (LJ walls) are forced to move anyway, and when released by the inlet they are shot
		// away. To avoid this, the distance from the walls parallel to the direction of the flow is increased by m_deltap/2
		INLET_SIDE_WALLS_DISTANCE = r0/2;
		float3 inlet_min = make_float3(WORLD_OFFSET_X + INLET_SIDE_WALLS_DISTANCE,
				WORLD_OFFSET_Y + INLET_SIDE_WALLS_DISTANCE,
				WORLD_OFFSET_Z + INFLOW_BOX_HEIGHT + INLET_SIDE_WALLS_DISTANCE);
		float3 inlet_max = make_float3(WORLD_OFFSET_X + POOL_WIDTH - INLET_SIDE_WALLS_DISTANCE,
				WORLD_OFFSET_Y + INLET_LENGTH + INLET_SIDE_WALLS_DISTANCE,
				WORLD_OFFSET_Z + TOTAL_HEIGHT - INLET_SIDE_WALLS_DISTANCE); // here adding INLET_SIDE_WALLS_DISTANCE is only useful for lids
		float4 inflow_vel = make_float4(0.0F, 0.1F, 0.0F, NAN);
		//float4 inflow_vel = make_float4(NAN, 0.1F, NAN, NAN);
		add_inlet(inlet_min, inlet_max, inflow_vel);
		printf("Added inlet: min (%g, %g, %g), max (%g, %g, %g)\n",
			inlet_min.x, inlet_min.y, inlet_min.z,
			inlet_max.x, inlet_max.y, inlet_max.z);// */

		// outlet
		OUTLET_HEIGHT = m_simparams.slength*m_simparams.kernelradius;
		// extend the outlet backwards to touch the back of the ramp
		const float outlet_deltaY = OUTFLOW_RAMP_LENGTH * OUTLET_ZOFFSET / OUTFLOW_RAMP_HEIGHT + r0/2;
		float3 outlet_min = make_float3(WORLD_OFFSET_X + r0/2,
				WORLD_OFFSET_Y + TOTAL_LENGTH - OUTLET_LENGTH + outlet_deltaY,
				WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT + OUTFLOW_RAMP_HEIGHT + OUTLET_ZOFFSET - OUTLET_HEIGHT);
		float3 outlet_max = make_float3(WORLD_OFFSET_X + POOL_WIDTH - r0/2,
				WORLD_OFFSET_Y + TOTAL_LENGTH,
				WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT + OUTFLOW_RAMP_HEIGHT + OUTLET_ZOFFSET);
		float3 outlet_dir = make_float3(0, 0, -1.0F);
		add_outlet(outlet_min, outlet_max, outlet_dir);
	}

	//m_maxvel = sqrt(m_physparams.gravity*H);
	//m_maxvel = 3.0f;
	m_maxvel = sqrt(g * abs(INFLOW_BOX_HEIGHT - OUTFLOW_BOX_HEIGHT));
	m_physparams.set_density(0,1000.0, 7.0f, 20.f*m_maxvel);

	// Scales for drawing
	m_maxrho = density(H,0);
	m_minrho = m_physparams.rho0[0];
	m_minvel = 0.0f;

	// Drawing and saving times
	m_displayinterval = 0.001f;
	m_writefreq = 20;
	m_screenshotfreq = 20;

	const float S = (INFLOW_BOX_HEIGHT - OUTFLOW_BOX_HEIGHT) / HORIZONTAL_SLOPE_LENGTH;
	printf("Fishpass initialized - %d pools, slope %g%\n", POOLS, S);

	// Name of problem used for directory creation
	m_name = "FishPass";
	create_problem_dir();
}


FishPass::~FishPass(void)
{
	release_memory();
}


void FishPass::release_memory(void)
{
	delete[] pAf;
	delete[] pAb;
	delete[] pAs;
	delete[] pBf;
	delete[] pBb;
	delete[] pBs;
	delete[] pCf;
	delete[] pCb;
	delete[] pCs;

	fluid_parts.clear();
	floor_parts.clear();
	sides_parts.clear();
	walls_parts.clear();
}

/// Returns the heigh of the floor at given Y (*including* OUTFLOW_BOX_HEIGHT)
float FishPass::getAbsoluteFloorHeight(float Ypos)
{
	if (Ypos < 0.0F) return 0.0F;
	if (Ypos > TOTAL_LENGTH) return 0.0F;

	if (Ypos <= INFLOW_BOX_LENGTH)
		return INFLOW_BOX_HEIGHT;
	if (Ypos >= INFLOW_BOX_LENGTH + HORIZONTAL_SLOPE_LENGTH)
		return OUTFLOW_BOX_HEIGHT;

	float delta_slope = OUTFLOW_BOX_HEIGHT - INFLOW_BOX_HEIGHT;
	float delta_ypos = Ypos - INFLOW_BOX_LENGTH;

	return INFLOW_BOX_HEIGHT + delta_slope*(delta_ypos / HORIZONTAL_SLOPE_LENGTH);
}

// Ypos is the Y coordinate of B's back wall
void FishPass::addWalls(uint wIndex, float Ypos)
{
	float floor_height;

	float r0 = m_physparams.r0;

	// starting Y of back walls. Warning: modified throughout the method
	float AY = Ypos + A_YOFFSET; // includes B's thickness
	float BY = Ypos;
	float CY = Ypos;

	//const float WALL_HEIGHT = INFLOW_CHANNEL_HEIGHT;
	// WALL_HEIGHT refers to B's back wall; it is absolute, not relative to the ground/slope
	float WALL_HEIGHT = min( getAbsoluteFloorHeight(AY) + POOL_WATER_LEVEL * 2.0f, TOTAL_HEIGHT);
	if (MAX_WALL_HEIGHT) WALL_HEIGHT = TOTAL_HEIGHT;

	// p..b and p..f have filled edges, r0 distance from walls; p..s no borders

	// *** A WALL ***

	// A, back wall
	floor_height = getAbsoluteFloorHeight(AY);
	pAb[wIndex] = Rect(
		Point(WORLD_OFFSET_X + POOL_WIDTH - r0, WORLD_OFFSET_Y + AY, WORLD_OFFSET_Z + floor_height + r0/2),
		Vector(- (A_LENGTH - r0), 0, 0),
		Vector(0, 0, WALL_HEIGHT - floor_height - r0/2)
	);
	pAb[wIndex].SetPartMass(r0, m_physparams.rho0[0]);
	pAb[wIndex].Fill(walls_parts, r0, true);

	// A, front wall
	AY += WALLS_THICKNESS;
	floor_height = getAbsoluteFloorHeight(AY);
	pAf[wIndex] = Rect(
		Point(WORLD_OFFSET_X + POOL_WIDTH - r0, WORLD_OFFSET_Y + AY, WORLD_OFFSET_Z + floor_height + r0/2),
		Vector(- (A_LENGTH - r0), 0, 0),
		Vector(0, 0, WALL_HEIGHT - floor_height - r0/2)
	);
	pAf[wIndex].SetPartMass(r0, m_physparams.rho0[0]);
	pAf[wIndex].Fill(walls_parts, r0, true);

	// A, side wall, same AY and height as front; so going backwards and no border
	pAs[wIndex] = Rect(
		Point(WORLD_OFFSET_X + POOL_WIDTH - A_LENGTH, WORLD_OFFSET_Y + AY, WORLD_OFFSET_Z + floor_height + r0/2),
		Vector(0, -WALLS_THICKNESS, 0),
		Vector(0, 0, WALL_HEIGHT - floor_height)
	);
	pAs[wIndex].SetPartMass(r0, m_physparams.rho0[0]);
	pAs[wIndex].Fill(walls_parts, r0, false, true); // border false, fill true

	// *** B WALL ***

	// B, back wall
	floor_height = getAbsoluteFloorHeight(BY);
	pBb[wIndex] = Rect(
		Point(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + BY, WORLD_OFFSET_Z + floor_height + r0/2),
		Vector(B_LENGTH - r0 - C_XOFFSET - WALLS_THICKNESS, 0, 0),
		Vector(0, 0, WALL_HEIGHT - floor_height - r0/2)
	);
	pBb[wIndex].SetPartMass(r0, m_physparams.rho0[0]);
	pBb[wIndex].Fill(walls_parts, r0, true);

	// B, front wall
	BY += WALLS_THICKNESS;
	floor_height = getAbsoluteFloorHeight(BY);
	pBf[wIndex] = Rect(
		Point(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + BY, WORLD_OFFSET_Z + floor_height + r0/2),
		Vector(B_LENGTH - r0, 0, 0),
		Vector(0, 0, WALL_HEIGHT - floor_height - r0/2)
	);
	pBf[wIndex].SetPartMass(r0, m_physparams.rho0[0]);
	pBf[wIndex].Fill(walls_parts, r0, true);

	// B, side wall, same BY and height as front; so going backwards and no border
	pBs[wIndex] = Rect(
		Point(WORLD_OFFSET_X + B_LENGTH, WORLD_OFFSET_Y + BY, WORLD_OFFSET_Z + floor_height + r0/2),
		Vector(0, -WALLS_THICKNESS, 0),
		Vector(0, 0, WALL_HEIGHT - floor_height - r0)
	);
	pBs[wIndex].SetPartMass(r0, m_physparams.rho0[0]);
	pBs[wIndex].Fill(walls_parts, r0, false, true); // border false, fill true

	// *** C WALL ***
	// from Ypos going backwards (-Y)

	// C, back wall
	floor_height = getAbsoluteFloorHeight(CY);
	pCb[wIndex] = Rect(
		Point(WORLD_OFFSET_X + B_LENGTH - WALLS_THICKNESS - C_XOFFSET, WORLD_OFFSET_Y + CY - r0, WORLD_OFFSET_Z + floor_height + r0/2),
		Vector(0, - (C_LENGTH - r0), 0),
		Vector(0, 0, WALL_HEIGHT - floor_height - r0/2)
	);
	pCb[wIndex].SetPartMass(r0, m_physparams.rho0[0]);
	pCb[wIndex].Fill(walls_parts, r0, true);

	// C, front wall; same ypos
	pCf[wIndex] = Rect(
		Point(WORLD_OFFSET_X + B_LENGTH, WORLD_OFFSET_Y + CY, WORLD_OFFSET_Z + floor_height + r0/2),
		Vector(- C_XOFFSET, - C_LENGTH, 0),
		Vector(0, 0, WALL_HEIGHT - floor_height - r0/2)
	);
	pCf[wIndex].SetPartMass(r0, m_physparams.rho0[0]);
	pCf[wIndex].Fill(walls_parts, r0, true);

	// C, side wall, no border
	CY -= C_LENGTH;
	floor_height = getAbsoluteFloorHeight(CY);
	pCs[wIndex] = Rect(
		Point(WORLD_OFFSET_X + B_LENGTH - C_XOFFSET, WORLD_OFFSET_Y + CY, WORLD_OFFSET_Z + floor_height + r0/2),
		    Vector(-WALLS_THICKNESS, 0, 0),
		    Vector(0, 0, WALL_HEIGHT - floor_height - r0/2)
	);
	pCs[wIndex].SetPartMass(r0, m_physparams.rho0[0]);
	pCs[wIndex].Fill(walls_parts, r0, false, true); // border false, fill true
}

// Ypos is the Y coordinate of B's back wall
// specialSize is the length of the last "aa" portion (useful for the first pool). If not specified, the regular size is used.
void FishPass::addWater(float Ypos, float aaSize) {
	float r0 = m_physparams.r0;
	if (aaSize == -1.0F)
		aaSize = POOL_LENGTH - A_YOFFSET - WALLS_THICKNESS - C_LENGTH - 2* r0;
	// "aa" portion
	float aaYPos = Ypos - C_LENGTH - r0 - aaSize;
	fillHFrustum(	make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + aaYPos, WORLD_OFFSET_Z + getAbsoluteFloorHeight(aaYPos) + r0),
					make_float2(POOL_WIDTH - 2*r0, POOL_WATER_LEVEL - r0),
					make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + aaYPos + aaSize, WORLD_OFFSET_Z + getAbsoluteFloorHeight(aaYPos + aaSize) + r0),
					make_float2(POOL_WIDTH - 2*r0, OUTFLOW_WATER_LEVEL - r0) );
	// "a" portion
	float aYPos = Ypos - C_LENGTH;
	fillHFrustum(	make_float3(WORLD_OFFSET_X + B_LENGTH - C_XOFFSET + r0, WORLD_OFFSET_Y + aYPos, WORLD_OFFSET_Z + getAbsoluteFloorHeight(aYPos) + r0),
					make_float2(POOL_WIDTH - 2*r0 - B_LENGTH + C_XOFFSET, POOL_WATER_LEVEL - r0),
					make_float3(WORLD_OFFSET_X + B_LENGTH + r0, WORLD_OFFSET_Y + Ypos, WORLD_OFFSET_Z + getAbsoluteFloorHeight(Ypos) + r0),
					make_float2(POOL_WIDTH - 2*r0 - B_LENGTH, OUTFLOW_WATER_LEVEL - r0) );
	// "b" portion
	float bYPos = Ypos - C_LENGTH;
	fillHFrustum(	make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + bYPos, WORLD_OFFSET_Z + getAbsoluteFloorHeight(bYPos) + r0),
					make_float2(B_LENGTH - C_XOFFSET - WALLS_THICKNESS - 2*r0, POOL_WATER_LEVEL - r0),
					make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + Ypos - r0, WORLD_OFFSET_Z + getAbsoluteFloorHeight(Ypos - r0) + r0),
					make_float2(B_LENGTH - C_XOFFSET - WALLS_THICKNESS - 2*r0, POOL_WATER_LEVEL - r0) );
	// "c" portion
	float cYPos = Ypos + m_deltap;
	fillHFrustum(	make_float3(WORLD_OFFSET_X + B_LENGTH + r0, WORLD_OFFSET_Y + cYPos, WORLD_OFFSET_Z + getAbsoluteFloorHeight(cYPos) + r0),
					make_float2(POOL_WIDTH - B_LENGTH - 2*r0, POOL_WATER_LEVEL - r0),
					make_float3(WORLD_OFFSET_X + B_LENGTH + r0, WORLD_OFFSET_Y + cYPos + A_YOFFSET - r0 - m_deltap, WORLD_OFFSET_Z + getAbsoluteFloorHeight(cYPos + A_YOFFSET - r0) + r0),
					make_float2(POOL_WIDTH - B_LENGTH - 2*r0, POOL_WATER_LEVEL - r0) );
	// "d" portion
	float dYPos = Ypos + WALLS_THICKNESS + r0;
	float dFinalPos = Ypos + A_YOFFSET - m_deltap;
	// "d" portion can be very thin. The method fillHFrustum() behaves correctly if dest is < than origin, but if this happens here it means particles are too close to the wall
	if (dFinalPos > dYPos)
		fillHFrustum(	make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + dYPos, WORLD_OFFSET_Z + getAbsoluteFloorHeight(dYPos) + r0),
						make_float2(B_LENGTH - r0, POOL_WATER_LEVEL - r0),
						make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + dFinalPos, WORLD_OFFSET_Z + getAbsoluteFloorHeight(dFinalPos) + r0),
						make_float2(B_LENGTH - r0, POOL_WATER_LEVEL - r0) );
	// "e" portion
	float eYPos = Ypos + A_YOFFSET;
	float eFinalPos = eYPos + WALLS_THICKNESS;
	fillHFrustum(	make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + eYPos, WORLD_OFFSET_Z + getAbsoluteFloorHeight(eYPos) + r0),
					make_float2(POOL_WIDTH - A_LENGTH - 2*r0, POOL_WATER_LEVEL - r0),
					make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + eFinalPos, WORLD_OFFSET_Z + getAbsoluteFloorHeight(eFinalPos) + r0),
					make_float2(POOL_WIDTH - A_LENGTH - 2*r0, POOL_WATER_LEVEL - r0) );

}

void FishPass::drawWalls()
{
	//glColor3f(1.0, 0, 0);
	for (uint wi = 0; wi <= POOLS; wi++) // remember: walls == POOLS + 1
	{
		pAf[wi].GLDraw();
		pBf[wi].GLDraw();
		pCf[wi].GLDraw();
		pAb[wi].GLDraw();
		pBb[wi].GLDraw();
		pCb[wi].GLDraw();
		pAs[wi].GLDraw();
		pBs[wi].GLDraw();
		pCs[wi].GLDraw();
	}
}

// Fill a horizontal frustum (height parallel with Y axis) with fluid particles.
// It is given the 3D origins of the rectangles and their X and Z sizes.
void FishPass::fillHFrustum(float3 base1Origin, float2 base1Size, float3 base2Origin, float2 base2Size) {
	// NOTE: every float2.y is interpreted as Z size

	float deltay = base2Origin.y - base1Origin.y;
	uint nsteps;
	// 1st case: first rectangle has lower Y
	if (deltay > 0.0F)
		nsteps = (uint) floorf( deltay/m_deltap + 0.5 );
	else
		nsteps = (uint) - ceilf( deltay/m_deltap - 0.5 );

	float xpos_step = (base2Origin.x - base1Origin.x) / nsteps;
	float ypos_step = deltay/nsteps;
	float zpos_step = (base2Origin.z - base1Origin.z) / nsteps;
	float xsize_step = (base2Size.x - base1Size.x) / nsteps;
	float zsize_step = (base2Size.y - base1Size.y) / nsteps;

	Rect curr;

	// special case: if frustum is so short that there is only one rect, then the rect is an average of base1 and base2
	if (nsteps == 0) {
		curr =  Rect(
			Point( (base1Origin.x + base2Origin.x)/2.0F, (base1Origin.y + base2Origin.y)/2.0F, (base1Origin.z + base2Origin.z)/2.0F ),
			Vector( (base1Size.x + base2Size.x)/2.0F, 0, 0),
			Vector(0, 0, (base1Size.y + base2Size.y)/2.0F )
		);
		curr.SetPartMass(m_deltap, m_physparams.rho0[0]);
		curr.Fill(fluid_parts, m_deltap, true);
	} else

	for (uint curr_step = 0; curr_step <= nsteps; curr_step++) {
		curr =  Rect(
			Point(base1Origin.x + curr_step*xpos_step, base1Origin.y + curr_step*ypos_step, base1Origin.z + curr_step*zpos_step ),
			Vector(base1Size.x + curr_step*xsize_step, 0, 0),
			Vector(0, 0, base1Size.y + curr_step*zsize_step)
		);
		curr.SetPartMass(m_deltap, m_physparams.rho0[0]);
		curr.Fill(fluid_parts, m_deltap, true);
	}
}

int FishPass::fill_parts()
{
	float r0 = m_physparams.r0;
	int totParts = 0;

	const float HALFDP = m_deltap/2.0F;
	const float HALFR0 = r0/2.0F;

	inflow_floor = Rect(
		Point(WORLD_OFFSET_X + HALFR0, WORLD_OFFSET_Y + (PERIODIC ? HALFR0 : 0), WORLD_OFFSET_Z + INFLOW_BOX_HEIGHT), // starts at Y 0; if periodic, at HALFR0
		Vector(POOL_WIDTH - r0, 0, 0), // has HALFR0 distance from each side
		Vector(0, INFLOW_BOX_LENGTH - r0 - (PERIODIC ? HALFR0 : 0), 0) // ends r0 before INFLOW_BOX_LENGTH
	);
	inflow_floor.SetPartMass(r0, m_physparams.rho0[0]);
	inflow_floor.Fill(floor_parts, r0, true);

	slope_floor = Rect(
		Point(WORLD_OFFSET_X + HALFR0, WORLD_OFFSET_Y + INFLOW_BOX_LENGTH, WORLD_OFFSET_Z + INFLOW_BOX_HEIGHT), // starts at Y INFLOW_BOX_LENGTH
		Vector(POOL_WIDTH - r0, 0, 0), // has HALFR0 distance from each side
		Vector(0, HORIZONTAL_SLOPE_LENGTH - r0,
		       OUTFLOW_BOX_HEIGHT - INFLOW_BOX_HEIGHT) // ends r0 before the outflow channel
	);
	slope_floor.SetPartMass(r0, m_physparams.rho0[0]);
	slope_floor.Fill(floor_parts, r0, true);

	outflow_floor = Rect(
		Point(WORLD_OFFSET_X + HALFR0, WORLD_OFFSET_Y + INFLOW_BOX_LENGTH + HORIZONTAL_SLOPE_LENGTH,
				WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT), // starts exactly with the outflow channel
		Vector(POOL_WIDTH - r0, 0, 0),  // has HALFR0 distance from each side
		Vector(0, OUTFLOW_BOX_LENGTH - (PERIODIC ? HALFR0 : r0), 0) // if periodic, ends HALFR0 before
	);
	outflow_floor.SetPartMass(r0, m_physparams.rho0[0]);
	outflow_floor.Fill(floor_parts, r0, true);

	if (!PERIODIC) {
		outflow_ramp = Rect(
			Point(WORLD_OFFSET_X + HALFR0, WORLD_OFFSET_Y + INFLOW_BOX_LENGTH + HORIZONTAL_SLOPE_LENGTH + OUTFLOW_BOX_LENGTH,
					WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT), // starts at the end of the outflow channel
			Vector(POOL_WIDTH - r0, 0, 0), // has HALFR0 distance from each side
			Vector(0, OUTFLOW_RAMP_LENGTH, OUTFLOW_RAMP_HEIGHT)
		);
		outflow_ramp.SetPartMass(r0, m_physparams.rho0[0]);
		outflow_ramp.Fill(floor_parts, r0, true);
	}

	totParts += floor_parts.size();

	left_side = Rect(
		Point(WORLD_OFFSET_X + 0, WORLD_OFFSET_Y + (PERIODIC ? HALFR0 : 0), WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT),
		Vector(0, 0, TOTAL_HEIGHT - OUTFLOW_BOX_HEIGHT),
		Vector(0, TOTAL_LENGTH - (PERIODIC ? r0 : 0.0F), 0)
	);
	left_side.SetPartMass(r0, m_physparams.rho0[0]);
	left_side.Fill(sides_parts, r0, true);

	right_side = Rect(
		Point(WORLD_OFFSET_X + POOL_WIDTH, WORLD_OFFSET_Y + (PERIODIC ? HALFR0 : 0), WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT),
		Vector(0, 0, TOTAL_HEIGHT - OUTFLOW_BOX_HEIGHT),
		Vector(0, TOTAL_LENGTH - (PERIODIC ? r0 : 0.0F), 0)
	);
	right_side.SetPartMass(r0, m_physparams.rho0[0]);
	right_side.Fill(sides_parts, r0, true);

	// non need for back side in periodic mode
	if (!PERIODIC) {
		back_side = Rect(
			Point(WORLD_OFFSET_X + 0, WORLD_OFFSET_Y + 0, WORLD_OFFSET_Z + INFLOW_BOX_HEIGHT),
			Vector(POOL_WIDTH, 0, 0),
			Vector(0, 0, INFLOW_CHANNEL_HEIGHT)
		);
		back_side.SetPartMass(r0, m_physparams.rho0[0]);
		back_side.Fill(sides_parts, r0, false, true); // false -> borders, true -> fill

		/*
		 front_side = Rect(
			Point(WORLD_OFFSET_X + 0, WORLD_OFFSET_Y + TOTAL_LENGTH, WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT),
			Vector(POOL_WIDTH, 0, 0),
			//Vector(0, 0, INFLOW_CHANNEL_HEIGHT)
			Vector(0, 0, TOTAL_HEIGHT - OUTFLOW_BOX_HEIGHT)
		);
		front_side.SetPartMass(r0, m_physparams.rho0[0]);
		front_side.Fill(sides_parts, r0, false, true); // false -> borders, true -> fill
		*/
	}

	// the lid was only needed for splashes when only discharging
	// NOTE: if this will be ever activated again, one should check the overlap with the sides
	/* inflow_lid = Rect(
		Point(WORLD_OFFSET_X + 0, WORLD_OFFSET_Y + 0, WORLD_OFFSET_Z + TOTAL_HEIGHT), // - OUTFLOW_BOX_HEIGHT),
		Vector(POOL_WIDTH, 0, 0),
		Vector(0, INFLOW_BOX_LENGTH + POST_INFLOW_YOFFSET + A_YOFFSET, 0)
	);
	inflow_lid.SetPartMass(r0, m_physparams.rho0[0]);
	inflow_lid.Fill(sides_parts, r0, false, true); // false -> borders, true -> fill */

	totParts += sides_parts.size();

	// add pools - each addWalls() adds walls A, B, C
	float currY = INFLOW_BOX_LENGTH + POST_INFLOW_YOFFSET;
	for (int pool=0; pool <= POOLS; pool++) {
		addWalls(pool, currY);
		currY += POOL_LENGTH;
	}
	totParts += walls_parts.size();

	// useful?
	//boundary_parts.reserve(2000);
	//parts.reserve(14000);

	// now we fill the fluid: inflow, pools, pre-outflow, outflow
	// assumption: on the Y axis, every portion of filling starts at [last_edge] and ends to [next_edge - r0]
	// (or + m_deltap if previous if fluid)

	float Ystart = (PERIODIC ? HALFDP : INLET_SIDE_WALLS_DISTANCE + HALFDP);

	inboxFluid = Cube(
		Point(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + Ystart, WORLD_OFFSET_Z + r0 + INFLOW_BOX_HEIGHT),
		Vector(0, INFLOW_BOX_LENGTH - Ystart - m_deltap, 0), // fill along Y until the slope, leaving m_deltap to next fluid
		//Vector(0, INLET_LENGTH - m_deltap, 0),
		Vector(POOL_WIDTH - 2*r0, 0, 0),
		Vector(0, 0, INFLOW_WATER_LEVEL - r0) );
	inboxFluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	inboxFluid.Fill(fluid_parts, m_deltap, true);

	// yes, we could cycle over the pools only once instead
	currY = INFLOW_BOX_LENGTH + POST_INFLOW_YOFFSET;
	const float firstAAlength = POST_INFLOW_YOFFSET - C_LENGTH - r0;
	for (int pool=0; pool <= POOLS; pool++) {
		if (pool == 0)
			addWater(currY, firstAAlength);
		else
			addWater(currY);
		currY += POOL_LENGTH;
	}

	// pre-outflow: is there any?
	// PRE_OUTFLOW_YOFFSET refers to the last back rect of B wall. Subtract A_YOFFSET and WALLS_THICKNESS; r0 or m_deltap distance
	// from A wall and E portion of the fluid; m_deltap from the outflow channel
	const float preout_size = PRE_OUTFLOW_YOFFSET - A_YOFFSET - WALLS_THICKNESS - max(m_deltap, r0) - m_deltap;
	if (preout_size >= m_deltap) {
		// starts at (TOTAL_LENGTH - OUTFLOW_BOX_LENGTH - preout_size); we also leave m_deltap before the outflow channel
		const float preout_start = TOTAL_LENGTH - OUTFLOW_BOX_LENGTH - preout_size - m_deltap - (PERIODIC ? 0.0F : OUTFLOW_RAMP_LENGTH + OUTLET_LENGTH);
		fillHFrustum(	make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + preout_start, WORLD_OFFSET_Z + getAbsoluteFloorHeight(preout_start) + r0),
						make_float2(POOL_WIDTH - 2*r0, POOL_WATER_LEVEL - r0),
						make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + preout_start + preout_size, WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT + r0),
						make_float2(POOL_WIDTH - 2*r0, OUTFLOW_WATER_LEVEL - r0) );
	}

	float YLeftspace = (PERIODIC ? HALFDP : m_deltap);

	outboxFluid = Cube(
		Point(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + INFLOW_BOX_LENGTH + HORIZONTAL_SLOPE_LENGTH, WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT + r0),
		Vector(0, OUTFLOW_BOX_LENGTH - YLeftspace, 0),
		Vector(POOL_WIDTH - 2*r0, 0, 0),
		Vector(0, 0, OUTFLOW_WATER_LEVEL - r0) );
	outboxFluid.SetPartMass(m_deltap, m_physparams.rho0[0]);
	outboxFluid.Fill(fluid_parts, m_deltap, true);

	// fill the ramp, if any
	if (!PERIODIC) {
		const float ramp_start = TOTAL_LENGTH - OUTFLOW_RAMP_LENGTH - OUTLET_LENGTH;
		const float final_height = OUTFLOW_BOX_HEIGHT + OUTFLOW_WATER_LEVEL - OUTFLOW_RAMP_HEIGHT - r0;
		if (final_height > 0.0F)
			fillHFrustum(	make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + ramp_start, WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT + r0),
							make_float2(POOL_WIDTH - 2*r0, POOL_WATER_LEVEL - r0),
							make_float3(WORLD_OFFSET_X + r0, WORLD_OFFSET_Y + ramp_start + OUTFLOW_RAMP_LENGTH, WORLD_OFFSET_Z + OUTFLOW_BOX_HEIGHT + OUTFLOW_RAMP_HEIGHT + r0),
							make_float2(POOL_WIDTH - 2*r0, final_height) );
	}

	/*fillHFrustum(	make_float3(r0, r0, r0+INFLOW_BOX_HEIGHT),
					make_float2(POOL_WIDTH - 2*r0, INFLOW_WATER_LEVEL - 2*r0),
					make_float3(r0, INFLOW_BOX_LENGTH, r0+INFLOW_BOX_HEIGHT),
					make_float2(POOL_WIDTH - 2*r0, INFLOW_WATER_LEVEL - 2*r0) );*/

	// this is equivalent to the previous fillHFrustum() call
	/*fluid1 = Cube(
		Point(r0, r0, r0+INFLOW_BOX_HEIGHT),
		Vector(POOL_WIDTH - 2*r0, 0, 0),
		Vector(0, INFLOW_BOX_LENGTH - r0, 0),
		Vector(0, 0, INFLOW_WATER_LEVEL - 2*r0) );

	fluid1.SetPartMass(m_deltap, m_physparams.rho0[0]);
	fluid1.Fill(fluid_parts, m_deltap, true);*/

/*	fluid2 = Cube(
		Point(r0, r0 + INFLOW_BOX_LENGTH, r0+INFLOW_BOX_HEIGHT),
		      Vector(B_LENGTH - WALLS_THICKNESS - C_XOFFSET - 2*r0, 0, 0),
		      Vector(0, POST_INFLOW_YOFFSET - 2*r0, 0),
		      Vector(0, 0, INFLOW_WATER_LEVEL - 2*r0) );

	fluid2.SetPartMass(m_deltap, m_physparams.rho0[0]);
	fluid2.Fill(fluid_parts, m_deltap, true); */

	totParts += fluid_parts.size();

	return totParts;
}


void FishPass::draw_boundary(float t)
{
	glColor3f(0.0, 1.0, 0.0);
	inflow_floor.GLDraw();
	slope_floor.GLDraw();
	outflow_floor.GLDraw();
	outflow_ramp.GLDraw();
	//glColor3f(1.0, 0.0, 0.0);
	right_side.GLDraw();
	left_side.GLDraw();
	back_side.GLDraw();
	front_side.GLDraw();
	inflow_lid.GLDraw();

	glColor3f(0.5, 0.5, 0.5);
	drawWalls();
}


void FishPass::copy_to_array(float4 *pos, float4 *vel, particleinfo *info)
{
	std::cout << "Floor parts: " << floor_parts.size() << "\n";
	for (uint i = 0; i < floor_parts.size(); i++) {
		pos[i] = make_float4(floor_parts[i]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,1,i);
	}
	int j = floor_parts.size();
	std::cout << "Floor part mass:" << pos[j-1].w << "\n";

	std::cout << "Side parts: " << sides_parts.size() << "\n";
	for (uint i = j; i < j + sides_parts.size(); i++) {
		pos[i] = make_float4(sides_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,2,i);
	}
	j += sides_parts.size();
	std::cout << "Side part mass:" << pos[j-1].w << "\n";

	/*std::cout << "Obstacle parts: " << obstacle_parts.size() << "\n";
	for (uint i = j; i < j + obstacle_parts.size(); i++) {
		pos[i] = make_float4(obstacle_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,1,i);
	}
	j += obstacle_parts.size();
	std::cout << "Obstacle part mass:" << pos[j-1].w << "\n";*/

	std::cout << "Wall parts: " << walls_parts.size() << "\n";
	for (uint i = j; i < j + walls_parts.size(); i++) {
		pos[i] = make_float4(walls_parts[i-j]);
		vel[i] = make_float4(0, 0, 0, m_physparams.rho0[0]);
		info[i]= make_particleinfo(BOUNDPART,3,i);
	}
	j += walls_parts.size();
	std::cout << "Wall part mass:" << pos[j-1].w << "\n";

	std::cout << "Fluid parts: " << fluid_parts.size() << "\n";
	for (uint i = j; i < j + fluid_parts.size(); i++) {
		pos[i] = make_float4(fluid_parts[i-j]);
		vel[i] = make_float4(INITIAL_VELOCITY.x, INITIAL_VELOCITY.y, INITIAL_VELOCITY.z, m_physparams.rho0[0]);
		info[i]= make_particleinfo(FLUIDPART,0,i);
	}
	j += fluid_parts.size();
	std::cout << "Fluid part mass:" << pos[j-1].w << "\n";
}
