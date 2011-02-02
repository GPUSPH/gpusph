/*
 * File:   lavaflow.h
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
		TopoCube	experiment_box;
		PointVect	boundary_parts;
		PointVect	piston_parts;
		PointVect	parts;
		float		H;  // still watr level
		float		north, south, east, west;
		float		nsres, ewres;

	public:
		TestTopo(const Options &);

		~TestTopo(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4*, float4*, particleinfo*);

		void release_memory(void);
};
#endif	/* _TESTTOPO_H */

