/* 
 * File:   Object.h
 * Author: alexis
 *
 * Created on 8 septembre 2011, 16:21
 */

#ifndef OBJECT_H
#define	OBJECT_H

#include "Point.h"

class Object {
	public:

		Object(void) {};
		~Object(void) {};

		virtual double SetPartMass(double, double) = 0;
		virtual void SetPartMass(double) = 0;

		virtual void FillBorder(PointVect& , double) = 0;
		virtual void Fill(PointVect& , double) = 0;

		virtual void GLDraw(void) = 0;
};
#endif	/* OBJCET_H */

