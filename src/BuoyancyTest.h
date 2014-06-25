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
#include "ode/ode.h"

class BuoyancyTest: public Problem {
	private:
		Cube		experiment_box, cube;
		PointVect	parts;
		PointVect	boundary_parts;
		float		H;  // still water level
		double		lx, ly, lz;		// dimension of experiment box
		double		olx, oly, olz;	// dimension of the floating cube
		dGeomID		planes[5];
		dJointID	joint;

	public:
		BuoyancyTest(const GlobalData *);
		virtual ~BuoyancyTest(void);

		int fill_parts(void);
		void copy_to_array(BufferList &);
		void ODE_near_callback(void *, dGeomID, dGeomID);

		void release_memory(void);
};
#endif /* BUOYANCYTEST_H_ */
