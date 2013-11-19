/* 
 * File:   FallingCube.h
 * Author: alexis
 *
 * Created on 31 août 2011-2013, 17:32
 */

#ifndef FALLINGCUBES_H
#define	FALLINGCUBES_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"

#include "ode/ode.h"

class FallingCubes: public Problem {
	private:
		Cube		experiment_box;
		PointVect	parts;
		PointVect	boundary_parts;
		float		H;  // still water level
		double		lx, ly, lz;		// dimension of experiment box
		// ODE and rigid body stuff
		Cube		cube[3];
		dGeomID		planes[5];

	public:
		FallingCubes(const Options &);
		virtual ~FallingCubes(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *, uint *);

		void ODE_near_callback(void *, dGeomID, dGeomID);

		void release_memory(void);
};


#endif	/* FALLINGCUBES_H */

