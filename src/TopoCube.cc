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

#include <math.h>
#include <iostream>
#include <math.h>
#ifdef __APPLE__
#include <OpenGl/gl.h>
#else
#include <GL/gl.h>
#endif

#include "TopoCube.h"
#include "Point.h"
#include "Vector.h"
#include "Rect.h"


TopoCube::TopoCube(void)
{
	origin = Point(0, 0, 0);
	vx = Vector(0, 0, 0);
	vy = Vector(0, 0, 0);
	vz = Vector(0, 0, 0);
}


TopoCube::~TopoCube(void)
{
	delete[] m_dem;
}


void TopoCube::SetCubeDem(float H, float *dem, int ncols, int nrows,
							float nsres, float ewres, bool interpol)
{
	m_ncols = ncols;
	m_nrows = nrows;
	m_nsres = nsres;
	m_ewres = ewres;
	m_interpol = interpol;
	m_H = H;
	origin = Point(0.0, 0.0, 0.0, 0.0);
	vx = Vector(((float) m_ncols - 1)*m_ewres, 0.0, 0.0);
	vy = Vector(0.0, ((float) m_nrows - 1)*m_nsres,0.0);
	vz = Vector(0.0, 0.0, H);
	m_dem = new float[m_ncols*m_nrows];

	for (int i = 0; i < ncols*nrows; i++) {
		m_dem[i] = dem[i];
		}
}


double
TopoCube::SetPartMass(double dx, double rho)
{
	int nx = (int) (vx.norm()/dx);
	double deltax = vx.norm()/((double) nx);
	int ny = (int) (vy.norm()/dx);
	double deltay = vy.norm()/((double) ny);
	double mass = deltax*deltay*dx*rho;

	origin(3) = mass;
	return mass;
}


void
TopoCube::SetPartMass(double mass)
{
	origin(3) = mass;
}


void
TopoCube::FillBorder(PointVect& points, double dx, int face_num, bool fill_edges)
{
	Point   rorigin;
	Vector  v;

	switch(face_num){
		case 0:
			rorigin = origin;
			v = vx;
			break;
		case 1:
			rorigin = origin + vx;
			v = vy;
			break;
		case 2:
			rorigin = origin + vx + vy;
			v = -vx;
			break;
		case 3:
			rorigin = origin + vy;
			v = -vy;
			break;
	}

	int n = (int) (v.norm()/dx);
	double delta = v.norm()/((double) n);
	int nstart = 0;
	int nend = n;
	if (!fill_edges) {
		nstart++;
		nend--;
	}
	for (int i = nstart; i <= nend; i++) {
		float x = rorigin(0) + (float) i/((float) n)*v(0);
		float y = rorigin(1) + (float) i/((float) n)*v(1);
		float z = DemInterpol(x, y);
		while (z < m_H - dx) {
			z += dx;
			Point p(x, y, z, origin(3));
			points.push_back(p);
			}
		}
}


void
TopoCube::FillDem(PointVect& points, double dx)
{
	int nx = (int) (vx.norm()/dx);
	double deltax = vx.norm()/((double) nx);
	int ny = (int) (vy.norm()/dx);
	double deltay = vy.norm()/((double) ny);

	for (int i = 0; i <= nx; i++) {
		for (int j = 0; j <= ny; j++) {
			float x = origin(0) + (float) i/((float) nx)*vx(0) + (float) j/((float) ny)*vy(0);
			float y = origin(1) + (float) i/((float) nx)*vx(1) + (float) j/((float) ny)*vy(1);
			float z = DemInterpol(x, y);
			Point p(x, y, z, origin(3));
			points.push_back(p);
			}
		}
}


float
TopoCube::DemInterpol(float x, float y)  // x and y ranging in [0, ncols/ewres]x[0, nrows/nsres]
{
	float xb = x/m_ewres;
	float yb = y/m_nsres;
	int i = floor(xb);
	int j = floor(yb);
	float a = xb - (float) i;
	float b = yb - (float) j;
	float z = (1 - a)*(1 - b)*m_dem[i + j*m_ncols];
	if (i < m_ncols - 1)
		z +=  a*(1 - b)*m_dem[i + 1 + j*m_ncols];
	if (j < m_nrows - 1)
		z +=  (1 - a)*b*m_dem[i + (j + 1)*m_ncols];
	if (i < m_ncols - 1 && j < m_nrows - 1)
		z += a*b*m_dem[i + 1 + (j + 1)*m_ncols];
	return z;
}



float
TopoCube::DemDist(float x, float y, float z, float dx)
{
	if (dx > 0.5*std::min(m_nsres, m_ewres))
		dx = 0.5*std::min(m_nsres, m_ewres);
	float z0 = DemInterpol(x/m_ewres, y/m_nsres);
	float z1 = DemInterpol((x + dx)/m_ewres, y/m_nsres);
	float z2 = DemInterpol(x/m_ewres, (y + dx)/m_nsres);
	// A(x, y, z0) B(x + h, y, z1) C(x, y + h, z2)
	// AB(h, 0, z1 - z0) AC(0, h, z2 - z0)
	// AB^AC = ( -h*(z1 - z0), -h*(z2 - z0), h*h)
	float a = dx*(z0 - z1);
	float b = dx*(z0 - z2);
	float c = dx*dx;
	float d = - a*x - b*y - c*z0;
	float l = sqrt(a*a + b*b + c*c);

	// Getting distance along the normal
	float r = fabs(a*x + b*y + c*z + d)/l;
	if (z <= z0)
		r = 0;
	return r;
}


void
TopoCube::Fill(PointVect& points, double H, double dx, bool faces_filled)
{
	int nx = (int) (vx.norm()/dx);
	double deltax = vx.norm()/((double) nx);
	int ny = (int) (vy.norm()/dx);
	double deltay = vy.norm()/((double) ny);

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
			float x = origin(0) + (float) i/((float) nx)*vx(0) + (float) j/((float) ny)*vy(0);
			float y = origin(1) + (float) i/((float) nx)*vx(1) + (float) j/((float) ny)*vy(1);
			float z = DemInterpol(x, y);
			while (z < H - dx) {
				z += dx;
				Point p(x, y, z, origin(3));
				points.push_back(p);
				}
			}
		}
	return;
}


void
TopoCube::GLDrawQuad(const Point& p1, const Point& p2,
		const Point& p3, const Point& p4)
{
	glVertex3f((float) p1(0), (float) p1(1), (float) p1(2));
	glVertex3f((float) p2(0), (float) p2(1), (float) p2(2));
	glVertex3f((float) p3(0), (float) p3(1), (float) p3(2));
	glVertex3f((float) p4(0), (float) p4(1), (float) p4(2));
}


void
TopoCube::GLDraw(void)
{
	Point p1, p2, p3, p4;
	glBegin(GL_QUADS);
	{
		p1 = origin;
		p2 = p1 + vx;
		p3 = p2 + vz;
		p4 = p3 - vx;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin + vy;
		p2 = p1 + vx;
		p3 = p2 + vz;
		p4 = p3 - vx;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin;
		p2 = p1 + vy;
		p3 = p2 + vz;
		p4 = p3 - vy;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin + vx;
		p2 = p1 + vy;
		p3 = p2 + vz;
		p4 = p3 - vy;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin;
		p2 = p1 + vx;
		p3 = p2 + vy;
		p4 = p3 - vx;
		GLDrawQuad(p1, p2, p3, p4);
		p1 = origin + vz;
		p2 = p1 + vx;
		p3 = p2 + vy;
		p4 = p3 - vx;
		GLDrawQuad(p1, p2, p3, p4);
	}
	glEnd();

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
