/*
 * File:   DamBreak3D.h
 * Author: alexis
 *
 * Created on 28 janvier 2009, 00:44
 */

#ifndef _DAMBREAK3D_H
#define	_DAMBREAK3D_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"

class DamBreak3D: public Problem {
	private:
		Cube		experiment_box;
		Cube		obstacle;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	obstacle_parts;
		float		H;  // still watr level

	public:
		DamBreak3D(const Options &);
		~DamBreak3D(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);

		void release_memory(void);
};
#endif	/* _DAMBREAK3D_H */

