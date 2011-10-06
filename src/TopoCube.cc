/*  Copyright 2011 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

	Istituto de Nazionale di Geofisica e Vulcanologia
          Sezione di Catania, Catania, Italy

    Universita di Catania, Catania, Italy

    Johns Hopkins University, Baltimore, MD

    This file is part of GPUSPH.

    GPUSPH is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    GPUSPH is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with GPUSPH.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif
#include <math.h>
#include <iostream>

#include "TopoCube.h"
#include "Point.h"
#include "Vector.h"
#include "Rect.h"


TopoCube::TopoCube(void)
{
	m_origin = Point(0, 0, 0);
	m_vx = Vector(0, 0, 0);
	m_vy = Vector(0, 0, 0);
	m_vz = Vector(0, 0, 0);
}


TopoCube::~TopoCube(void)
{
	delete[] m_dem;
}


void TopoCube::SetCubeDem(const double H, const float *dem, const int ncols, const int nrows,
							const double nsres, const double ewres, const bool interpol)
{
	m_ncols = ncols;
	m_nrows = nrows;
	m_nsres = nsres;
	m_ewres = ewres;
	m_interpol = interpol;
	m_H = H;
	m_origin = Point(0.0, 0.0, 0.0, 0.0);
	m_vx = Vector(((float) m_ncols - 1)*m_ewres, 0.0, 0.0);
	m_vy = Vector(0.0, ((float) m_nrows - 1)*m_nsres,0.0);
	m_vz = Vector(0.0, 0.0, H);
	m_dem = new float[m_ncols*m_nrows];

	for (int i = 0; i < ncols*nrows; i++) {
		m_dem[i] = dem[i];
		}
}


double
TopoCube::SetPartMass(const double dx, const double rho)
{
	const double mass = dx*dx*dx*rho;

	m_center(3) = mass;
	return mass;
}


void
TopoCube::FillBorder(PointVect& points, const double dx, const int face_num, const bool fill_edges)
{
	Point   rorigin;
	Vector  v;

	m_origin(3) = m_center(3);
	switch(face_num){
		case 0:
			rorigin = m_origin;
			v = m_vx;
			break;
		case 1:
			rorigin = m_origin + m_vx;
			v = m_vy;
			break;
		case 2:
			rorigin = m_origin + m_vx + m_vy;
			v = -m_vx;
			break;
		case 3:
			rorigin = m_origin + m_vy;
			v = -m_vy;
			break;
	}

	const int n = (int) (v.norm()/dx);
	const double delta = v.norm()/((double) n);
	int nstart = 0;
	int nend = n;
	if (!fill_edges) {
		nstart++;
		nend--;
	}
	for (int i = nstart; i <= nend; i++) {
		const double x = rorigin(0) + (double) i/((double) n)*v(0);
		const double y = rorigin(1) + (double) i/((double) n)*v(1);
		double z = DemInterpol(x, y);
		while (z < m_H - dx) {
			z += dx;
			Point p(x, y, z, m_origin(3));
			points.push_back(p);
			}
		}
}


void
TopoCube::FillDem(PointVect& points, double dx)
{
	int nx = (int) (m_vx.norm()/dx);
	double deltax = m_vx.norm()/((double) nx);
	int ny = (int) (m_vy.norm()/dx);
	double deltay = m_vy.norm()/((double) ny);

	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			const double x = m_origin(0) + (double) i/((double) nx)*m_vx(0) + (double) j/((double) ny)*m_vy(0);
			const double y = m_origin(1) + (double) i/((double) nx)*m_vx(1) + (double) j/((double) ny)*m_vy(1);
			const double z = DemInterpol(x, y);
			Point p(x, y, z, m_origin(3));
			points.push_back(p);
			}
		}
}


double
TopoCube::DemInterpol(const double x, const double y)  // x and y ranging in [0, ncols/ewres]x[0, nrows/nsres]
{
	const double xb = x/m_ewres;
	const double yb = y/m_nsres;
	int i = floor(xb);
	int j = floor(yb);
	const double a = xb - (float) i;
	const double b = yb - (float) j;
	double z = (1 - a)*(1 - b)*m_dem[i + j*m_ncols];
	if (i < m_ncols - 1)
		z +=  a*(1 - b)*m_dem[i + 1 + j*m_ncols];
	if (j < m_nrows - 1)
		z +=  (1 - a)*b*m_dem[i + (j + 1)*m_ncols];
	if (i < m_ncols - 1 && j < m_nrows - 1)
		z += a*b*m_dem[i + 1 + (j + 1)*m_ncols];
	return z;
}



double
TopoCube::DemDist(const double x, const double y, const double z, double dx)
{
	if (dx > 0.5*std::min(m_nsres, m_ewres))
		dx = 0.5*std::min(m_nsres, m_ewres);
	const double z0 = DemInterpol(x/m_ewres, y/m_nsres);
	const double z1 = DemInterpol((x + dx)/m_ewres, y/m_nsres);
	const double z2 = DemInterpol(x/m_ewres, (y + dx)/m_nsres);
	// A(x, y, z0) B(x + h, y, z1) C(x, y + h, z2)
	// AB(h, 0, z1 - z0) AC(0, h, z2 - z0)
	// AB^AC = ( -h*(z1 - z0), -h*(z2 - z0), h*h)
	const double a = dx*(z0 - z1);
	const double b = dx*(z0 - z2);
	const double c = dx*dx;
	const double d = - a*x - b*y - c*z0;
	const double l = sqrt(a*a + b*b + c*c);

	// Getting distance along the normal
	double r = fabs(a*x + b*y + c*z + d)/l;
	if (z <= z0)
		r = 0;
	return r;
}


int
TopoCube::Fill(PointVect& points, const double H, const double dx, const bool faces_filled, const bool fill)
{
	int nparts;
	
	const int nx = (int) (m_vx.norm()/dx);
	const double deltax = m_vx.norm()/((double) nx);
	const int ny = (int) (m_vy.norm()/dx);
	const double deltay = m_vy.norm()/((double) ny);

	int startx = 0;
	int starty = 0;
	int endx = nx;
	int endy = ny;
	if (faces_filled) {
		startx ++;
		starty++;
		endx--;
		endy--;
	}

	for (int i = startx; i <= endx; i++) {
		for (int j = starty; j <= endy; j++) {
			float x = m_origin(0) + (float) i/((float) nx)*m_vx(0) + (float) j/((float) ny)*m_vy(0);
			float y = m_origin(1) + (float) i/((float) nx)*m_vx(1) + (float) j/((float) ny)*m_vy(1);
			float z = DemInterpol(x, y);
			while (z < H - dx) {
				z += dx;
				Point p(x, y, z, m_origin(3));
				nparts ++;
				if (fill)
					points.push_back(p);
				}
			}
		}
	
	return nparts;
}


// TODO:: FIXME
bool 
TopoCube::IsInside(const Point&, const double) const
{
	return false;
}


void
TopoCube::GLDraw(void) const
{
	Point p1, p2, p3, p4;
	p1 = m_origin;
	p2 = p1 + m_vx;
	p3 = p2 + m_vz;
	p4 = p3 - m_vx;
	GLDrawQuad(p1, p2, p3, p4);
	p1 = m_origin + m_vy;
	p2 = p1 + m_vx;
	p3 = p2 + m_vz;
	p4 = p3 - m_vx;
	GLDrawQuad(p1, p2, p3, p4);
	p1 = m_origin;
	p2 = p1 + m_vy;
	p3 = p2 + m_vz;
	p4 = p3 - m_vy;
	GLDrawQuad(p1, p2, p3, p4);
	p1 = m_origin + m_vx;
	p2 = p1 + m_vy;
	p3 = p2 + m_vz;
	p4 = p3 - m_vy;
	GLDrawQuad(p1, p2, p3, p4);
	p1 = m_origin;
	p2 = p1 + m_vx;
	p3 = p2 + m_vy;
	p4 = p3 - m_vx;
	GLDrawQuad(p1, p2, p3, p4);
	p1 = m_origin + m_vz;
	p2 = p1 + m_vx;
	p3 = p2 + m_vy;
	p4 = p3 - m_vx;
	GLDrawQuad(p1, p2, p3, p4);

	for (int y = 0; y < m_nrows - 1; y++) {
		for (int x = 0; x < m_ncols -1; x++) {
			glBegin(GL_QUADS);
			{
			#define dem_idx(x, y) ((x) + (y)*m_ncols)
			#define vertex(x, y) (x)*m_ewres, (y)*m_nsres , z

			float z = m_dem[dem_idx(x, y)];
			glVertex3f(vertex(x, y));

			z = m_dem[dem_idx(x+1, y)];
			glVertex3f(vertex(x+1, y));

			z = m_dem[dem_idx(x+1, y+1)];
			glVertex3f(vertex(x+1, y+1));

			z = m_dem[dem_idx(x, y+1)];
			glVertex3f(vertex(x, y+1));

			#undef dem_idx
			#undef vertex
			}
			glEnd();
		}
	}
}


void
TopoCube::GLDraw(const EulerParameters& ep, const Point& cg) const
{
	GLDraw();
}
	