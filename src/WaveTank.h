/*
 * File:   WaveTank.h
 * Author: Tony (following Alexis' PaddleTest3D)
 *
 * Created on 29 janvier 2009, 22:42
 */

#ifndef _WAVETANK_H
#define	_WAVETANK_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"
#include "Cylinder.h"
#include "Vector.h"
#include "Cone.h"

class WaveTank: public Problem {
	private:
		int			icyl, icone, wmakertype;
		Cube		experiment_box;
		Rect        experiment_box1, experiment_box6;
		int			i_use_bottom_plane;
		Rect		paddle;
		Point		paddle_origin;
		Point		p1,p2;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	piston_parts;
		PointVect	paddle_parts;
		PointVect	gate_parts;

		Cylinder	cyl1, cyl2, cyl3, cyl4;
		Cylinder	cyl5, cyl6, cyl7;
		Cylinder	cyl8, cyl9, cyl10;
		Cylinder	cyl11;
		Cone 		cone;
		float		paddle_length;
		float		paddle_width;
		float		h_length, height, slope_length, beta;
	    float		H;		// still water level
		float		Hbox;	// height of experiment box

		// Moving boundary data
		float		m_S, m_Hoh, m_tau;
		float		m_mbposx;
		float		m_mbomega;
		float		m_mbamplitude;
		bool		m_mbnextimeupdate;
		float		m_mbtstart, m_mbtend;

	public:
		WaveTank(const Options &);
		~WaveTank(void);
		int fill_parts(void);
		uint fill_planes(void);
		void copy_planes(float4*, float*);

		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);
	    MbCallBack& mb_callback(const float, const float);

		void release_memory(void);
};
#endif	/* _WAVETANK_H */

