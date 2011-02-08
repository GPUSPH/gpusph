/*
 * File:   DamBreakGate.h
 * Author: Alexis; Tony
 *
 * Created on 28 janvier 2009; Feb 2011
 */

#ifndef _DAMBREAKGATE_H
#define	_DAMBREAKGATE_H

#include "Problem.h"
#include "Point.h"
#include "Cube.h"
#include "Rect.h"
#include "Vector.h"

class DamBreakGate: public Problem {
	private:
		Cube		experiment_box;
		Cube		obstacle;
		Rect		gate, actual_gate;
		PointVect	parts;
		PointVect	boundary_parts;
		PointVect	obstacle_parts;
		PointVect	gate_parts;
		float		H;  // still watr level

		// Moving boundary data
		float3		m_mbv;
		float3		m_gateorigin;

		bool		m_mbnextimeupdate;
		float		m_mbtstart, m_mbtend;

	public:
		DamBreakGate(const Options &);
		~DamBreakGate(void);

		int fill_parts(void);
		void draw_boundary(float);
		void copy_to_array(float4 *, float4 *, particleinfo *);
	        MbCallBack& mb_callback(const float, const float);
		void release_memory(void);
};
#endif	/* _DAMBREAKGATE_H */


