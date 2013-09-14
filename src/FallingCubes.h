/* 
 * File:   FallingCube.h
 * Author: alexis
 *
 * Created on 31 août 2011, 17:32
 */

#ifndef FALLINGCUBES_H
#define	FALLINGCUBES_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"


class FallingCubes: public Problem {
	private:
		Cube		experiment_box;
		Cube		cube[3];
		PointVect	parts;
		PointVect	boundary_parts;
		float		H;  // still water level
		double		lx, ly, lz;		// dimension of experiment box

	public:
		FallingCubes(const Options &);
		virtual ~FallingCubes(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *, uint *);

		void release_memory(void);
};


#endif	/* FALLINGCUBES_H */

