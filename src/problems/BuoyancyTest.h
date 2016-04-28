/*
 * BuoyancyTest.h
 *
 *  Created on: 20 juin 2014
 *      Author: alexisherault
 */

#ifndef BUOYANCYTEST_H_
#define BUOYANCYTEST_H_

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Sphere.h"
#include "Torus.h"

class BuoyancyTest: public Problem {
	private:
		PointVect	parts;
		PointVect	boundary_parts;
		Cube		cube;
		Sphere		sphere;
		Torus 		torus;
		float		H;  // still water level
		double		lx, ly, lz;		// dimension of experiment box

	public:
		BuoyancyTest(GlobalData *);
		virtual ~BuoyancyTest(void);

		int fill_parts(void);
		void copy_to_array(BufferList &);

		void release_memory(void);
};
#endif /* BUOYANCYTEST_H_ */
