/* 
 * File:   SolitaryWave.h
 * Author: rad
 *
 * Created on February 7, 2011, 2:39 PM
 */

#ifndef SolitaryWave_H
#define	SolitaryWave_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"
#include "Cylinder.h"
#include "Vector.h"
#include "Cone.h"

class SolitaryWave: public Problem {
	private:
		int			icyl, icone, wmakertype;
		Cube		experiment_box;
		Rect		experiment_box1;
		int			i_use_bottom_plane;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	piston_parts;
		PointVect	gate_parts;

		Cylinder	cyl1, cyl2, cyl3, cyl4;
		Cylinder	cyl5, cyl6, cyl7;
		Cylinder	cyl8, cyl9, cyl10;
		Cone 		cone;
		float		h_length, height, slope_length, beta;
        float		H;		// still water level
		float		Hbox;	// height of experiment box

		// Moving boundary data
		float		m_S, m_Hoh, m_tau;

	public:
		SolitaryWave(const Options &);
		~SolitaryWave(void);
		int fill_parts(void);
		uint fill_planes(void);
		void copy_planes(float4*, float*);

		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);
		MbCallBack& mb_callback(const float, const float, const int);

		void release_memory(void);
};
#endif	/* _SolitaryWave_H */

