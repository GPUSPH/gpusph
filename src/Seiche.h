/* 
 * File:   Seiche.h
 * Author: rad
 *
 * Created on February 11, 2011, 2:48 PM
 */

#ifndef SEICHE_H
#define	SEICHE_H

#include "Problem.h"
#include "Point.h"
#include "Rect.h"
#include "Cube.h"

class Seiche: public Problem {
	private:
		Cube		experiment_box;
		PointVect	parts;
		PointVect	boundary_parts;
		float		h, w, l;
		float		H; // still water level
		float		m_gtstart, m_gtend;

	public:
		Seiche(const Options &);
		~Seiche(void);

		int  fill_parts(void);
		uint fill_planes(void);
		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);
		void copy_planes(float4*, float*);
        float3 g_callback(const float);

		void release_memory(void);
};


#endif	/* SEICHE_H */

