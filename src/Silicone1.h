/* 
 * File:   Silicone1.h
 * Author: alexis
 *
 * Created on 9 juin 2011-2013, 23:46
 */

#ifndef SILICONE1_H
#define	SILICONE1_H
#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"
#include "Cylinder.h"
#include "Vector.h"
#include "Cone.h"

class Silicone1: public Problem {
	private:
		Cube		experiment_box;
		int			i_use_bottom_plane;
		PointVect	parts;
		PointVect	boundary_parts;

		Cylinder	cyl;
	    float		H;		// still water level


	public:
		Silicone1(const Options &);
		virtual ~Silicone1(void);
		int fill_parts(void);
		uint fill_planes(void);
		void copy_planes(float4*, float*);

		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);

		void release_memory(void);
};
#endif	/* SILICONE1_H */

