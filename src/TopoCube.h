/*
 * File:   Cube.h
 * Author: alexis
 *
 * Created on 14 juin 2008, 18:04
 */

#ifndef _TOPOCUBE_H
#define	_TOPOCUBE_H

#include "Point.h"
#include "Vector.h"

class TopoCube {
	private:
		Point	origin;
		Vector	vx, vy, vz;
		float*	m_dem;
		int		m_ncols, m_nrows;
		float	m_nsres, m_ewres;
		bool	m_interpol;
		float	m_H;

	public:
		TopoCube(void);
		~TopoCube(void);

		void SetCubeDem(float H, float *dem, int ncols, int nrows, float nsres, float ewres, bool interpol);

		double SetPartMass(double dx, double rho);
		void SetPartMass(double mass);
		void FillBorder(PointVect& points, double dx, int face_num, bool fill_edges);
		void FillBorder(PointVect& points, double dx, bool fill_top_face);
		void FillDem(PointVect& points, double dx);
		float DemInterpol(float x, float y);
		float DemDist(float x, float y, float z, float dx);
		void Fill(PointVect& points, double H, double dx, bool faces_filled);
		void GLDrawQuad(const Point& p1, const Point& p2, const Point& p3, const Point& p4);
		void GLDraw(void);
};

#endif	/* _CUBE_H */

