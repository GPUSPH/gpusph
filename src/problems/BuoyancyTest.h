/*
 * BuoyancyTest.h
 *
 *  Created on: 20 juin 2014
 *      Author: alexisherault
 */

#ifndef BUOYANCYTEST_H_
#define BUOYANCYTEST_H_

#include "XProblem.h"
#include "Point.h"
#include "Cube.h"
#include "Sphere.h"
#include "Torus.h"

class BuoyancyTest: public XProblem {

	public:
		BuoyancyTest(GlobalData *);
};
#endif /* BUOYANCYTEST_H_ */
