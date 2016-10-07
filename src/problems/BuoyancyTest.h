/*
 * BuoyancyTest.h
 *
 *  Created on: 20 juin 2014
 *      Author: alexisherault
 */

#ifndef XBUOYANCYTEST_H_
#define XBUOYANCYTEST_H_

#include "XProblem.h"
#include "Point.h"
#include "Cube.h"
#include "Sphere.h"
#include "Torus.h"

class XBuoyancyTest: public XProblem {

	public:
		XBuoyancyTest(GlobalData *);
};
#endif /* XBUOYANCYTEST_H_ */
