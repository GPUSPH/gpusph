/* 
 * File:   FallingCube.h
 * Author: alexis
 *
 * Created on 15 août 2011, 17:32
 */

#ifndef FALLINGCUBE_H
#define	FALLINGCUBE_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "RigidBody.h"

class FallingCube: public Problem {
	private:
		Cube		experiment_box;
		Cube		obstacle;
		PointVect	parts;
		PointVect	boundary_parts;
		float		H;  // still water level

	public:
		FallingCube(const Options &);
		~FallingCube(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);

		void release_memory(void);
};


#endif	/* FALLINGCUBE_H */

