/*  Copyright 2011-2013 Alexis Herault, Giuseppe Bilotta, Robert A. Dalrymple, Eugenio Rustico, Ciro Del Negro

    Istituto Nazionale di Geofisica e Vulcanologia
        Sezione di Catania, Catania, Italy

    Università di Catania, Catania, Italy

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

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "Rect.h"


/// Default onstructor
Rect::Rect(void)
{
	m_origin = Point(0, 0, 0);
	m_vx = Vector(0, 0, 0);
	m_vy = Vector(0, 0, 0);
}


/// Constructor
/*!	The parallelipiped is built from a starting point and
 	two vectors
	\param origin : starting point of the parallelipided
	\param vx : first vector
	\param vy : first vector
*/
Rect::Rect(const Point& origin, const Vector& vx, const Vector& vy)
{
	if (fabs(vx*vy) > 1.e-8*vx.norm()*vy.norm()) {
		//std::cout << "Trying to construct a rectangle with non perpendicular vectors\n";
		//exit(1);
	}

	m_origin = origin;
	m_vx = vx;
	m_vy = vy;
	m_lx = vx.norm();
	m_ly = vy.norm();
	m_center = m_origin + 0.5*m_vx + 0.5*m_vy;

	m_vz = m_vx.cross(m_vy);
	m_vz.normalize();

	Vector axis;
	double mat[9];
	mat[0] = m_vx(0)/m_lx;
	mat[3] = m_vx(1)/m_lx;
	mat[6] = m_vx(2)/m_lx;
	mat[1] = m_vy(0)/m_ly;
	mat[4] = m_vy(1)/m_ly;
	mat[7] = m_vy(2)/m_ly;
	mat[2] = m_vz(0);
	mat[5] = m_vz(1);
	mat[8] = m_vz(2);

	double trace = mat[0] + mat[4] + mat[8];
	double cs = 0.5*(trace - 1.0);
	double angle = acos(cs);  // in [0,PI]

	if (angle > 0.0)
	{
		if (angle < M_PI)
		{
			axis(0) = mat[7] - mat[5];
			axis(1) = mat[2] - mat[6];
			axis(2) = mat[3] - mat[1];
			axis /= axis.norm();
		}
		else
		{
			// angle is PI
			double halfInverse;
			if (mat[0] >= mat[4])
			{
				// r00 >= r11
				if (mat[0] >= mat[8])
				{
					// r00 is maximum diagonal term
					axis(0) = 0.5*sqrt(1.0 + mat[0] - mat[4] - mat[8]);
					halfInverse = 0.5/axis(0);
					axis(1) = halfInverse*mat[1];
					axis(2) = halfInverse*mat[2];
				}
				else
				{
					// r22 is maximum diagonal term
					axis(2) = 0.5*sqrt(1.0 + mat[8] - mat[0] - mat[4]);
					halfInverse = 0.5/axis(2);
					axis(0) = halfInverse*mat[2];
					axis(1) = halfInverse*mat[5];
				}
			}
			else
			{
				// r11 > r00
				if (mat[4] >= mat[8])
				{
					// r11 is maximum diagonal term
					axis(1) = 0.5*sqrt(1.0 + + mat[4] - mat[0] - mat[8]);
					halfInverse  = 0.5/axis(1);
					axis(0) = halfInverse*mat[1];
					axis(2) = halfInverse*mat[5];
				}
				else
				{
					// r22 is maximum diagonal term
					axis(2) = 0.5*sqrt(1.0 + mat[8] - mat[0] - mat[4]);
					halfInverse = 0.5/axis(2);
					axis(0) = halfInverse*mat[2];
					axis(1) = halfInverse*mat[5];
				}
			}
		}
	}
	else
	{
		// The angle is 0 and the matrix is the identity.  Any axis will
		// work, so just use the x-axis.
		axis(0) = 1.0;
		axis(1) = 0.0;
		axis(2) = 0.0;
	}

	m_ep = EulerParameters(axis, angle);
	m_ep.ComputeRot();
}


Rect::Rect(const Point &origin, const double lx, const double ly, const EulerParameters &ep)
{
	m_origin = origin;

	m_lx = lx;
	m_ly = ly;

	m_vx = Vector(lx,0,0);
	m_vy = Vector(0,ly,0);
	m_vz = m_vx.cross(m_vy);
	m_vz.normalize();
	setEulerParameters(ep);
}


double
Rect::Volume(const double dx) const
{
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double volume = lx*ly*dx;
	return volume;
}


void
Rect::SetInertia(const double dx)
{
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	const double lz = dx;
	m_inertia[0] = m_mass/12.0*(ly*ly + lz*lz);
	m_inertia[1] = m_mass/12.0*(lx*lx + lz*lz);
	m_inertia[2] = m_mass/12.0*(lx*lx + ly*ly);
}


void
Rect::FillBorder(PointVect& points, const double dx,
		const bool populate_first, const bool populate_last, const int edge_num)
{
	Point		origin;
	Vector		dir;

	m_origin(3) = m_center(3);
	switch(edge_num){
		case 0:
			origin = m_origin;
			dir = m_vx;
			break;
		case 1:
			origin = m_origin + m_vx;
			dir = m_vy;
			break;
		case 2:
			origin = m_origin + m_vx + m_vy;
			dir = - m_vx;
			break;
		case 3:
			origin = m_origin + m_vy;
			dir = - m_vy;
			break;
	}

	int nx = (int) (dir.norm()/dx);
	int startx = 0;
	int endx = nx;

	if (!populate_first){
		startx++;
	}

	if (!populate_last){
		endx--;
	}

	for (int i = startx; i <= endx; i++) {
		Point p = origin + i*dir/nx;
		points.push_back(p);
	}
}


void
Rect::FillBorder(PointVect& points, const double dx)
{
	m_origin(3) = m_center(3);
	FillBorder(points, dx, false, false, 0);
	FillBorder(points, dx, true, true, 1);
	FillBorder(points, dx, false, false, 2);
	FillBorder(points, dx, true, true, 3);
}


int
Rect::Fill(PointVect& points, const double dx, const bool fill_edges, const bool fill)
{
	m_origin(3) = m_center(3);
	int nparts = 0;

	int nx = max((int) (m_lx/dx), 1);
	int ny = max((int) (m_ly/dx), 1);
	int startx = 0;
	int starty = 0;
	int endx = nx;
	int endy = ny;

	if (!fill_edges){
		startx++;
		starty++;
		endx --;
		endy --;
	}

	for (int i = startx; i <= endx; i++)
		for (int j = starty; j <= endy; j++) {
			Point p = m_origin + i*m_vx/nx + j*m_vy/ny;
			if (fill)
				points.push_back(p);
			nparts++;
		}

	return nparts;
}


int
Rect::Fill(PointVect& points, const double dx, const bool fill)
{
	return Fill(points, dx, true, fill);
}


void
Rect::Fill(PointVect& points, const double dx, const bool* edges_to_fill)
{
	m_origin(3) = m_center(3);

	Fill(points, dx, false, true);

	for (int border_num = 0; border_num < 4; border_num++) {
		if (edges_to_fill[border_num])
			FillBorder(points, dx, true, false, border_num);
		}

	return;
}


void
Rect::Fill(PointVect& bpoints, PointVect& belems, PointVect& vpoints, std::vector<uint4>& vindexes, const double dx, const int face_num, std::vector<uint> edgeparts[6][4])
{
	m_origin(3) = m_center(3)/2;

	int nx = (int) (m_lx/dx);
	int ny = (int) (m_ly/dx);
	int startx = 0;
	int starty = 0;
	int endx = nx;
	int endy = ny;
	double belm_surf = m_lx/nx * m_ly/ny / 2.0;

	size_t predef_vparts = 0;
	Point belm(0, 0, 0, belm_surf);

	//Fill near-edge regions with boundary particles (elements) and set connectivity for them
	switch(face_num){
		case 0: //back face
			belm.SetCoord(0, 1, 0);
			break;
		case 1: //left face
			startx++;
			belm.SetCoord(-1, 0, 0);

			predef_vparts = edgeparts[0][2].size();
			if(predef_vparts)
			for(uint i=0; i<predef_vparts-1; i++) {
				uint vpoint_index = edgeparts[0][2][i];
				Point vp = vpoints[vpoint_index];

				Point bp1 = vp + m_vx/(nx*3) + 2*m_vy/(ny*3);
				bpoints.push_back(bp1);

				uint4 vertices1 = {vpoint_index, vpoint_index + 1, vpoints.size() + i + 1, 0};
				vindexes.push_back(vertices1);

				Point bp2 = vp + 2*m_vx/(nx*3) + m_vy/(ny*3);
				bpoints.push_back(bp2);

				uint4 vertices2 = {vpoint_index, vpoints.size() + i, vpoints.size() + i + 1, 0};
				vindexes.push_back(vertices2);

				belems.push_back(belm);
				belems.push_back(belm);
			}

			break;
		case 2: //front face
			startx++;
			belm.SetCoord(0, -1, 0);

			predef_vparts = edgeparts[1][2].size();
			if(predef_vparts)
			for(uint i=0; i<predef_vparts-1; i++) {
				uint vpoint_index = edgeparts[1][2][i];
				Point vp = vpoints[vpoint_index];

				Point bp1 = vp + m_vx/(nx*3) + 2*m_vy/(ny*3);
				bpoints.push_back(bp1);

				uint4 vertices1 = {vpoint_index, vpoint_index + 1, vpoints.size() + i + 1, 0};
				vindexes.push_back(vertices1);

				Point bp2 = vp + 2*m_vx/(nx*3) + m_vy/(ny*3);
				bpoints.push_back(bp2);

				uint4 vertices2 = {vpoint_index, vpoints.size() + i, vpoints.size() + i + 1, 0};
				vindexes.push_back(vertices2);

				belems.push_back(belm);
				belems.push_back(belm);
			}

			break;
		case 3: //right face
			startx++;
			endx--;
			belm.SetCoord(1, 0, 0);

			predef_vparts = edgeparts[2][2].size();
			if(predef_vparts)
			for(uint i=0; i<predef_vparts-1; i++) {
				uint vpoint_index = edgeparts[2][2][i];
				Point vp = vpoints[vpoint_index];

				Point bp1 = vp + m_vx/(nx*3) + 2*m_vy/(ny*3);
				bpoints.push_back(bp1);

				uint4 vertices1 = {vpoint_index, vpoint_index + 1, vpoints.size() + i + 1, 0};
				vindexes.push_back(vertices1);

				Point bp2 = vp + 2*m_vx/(nx*3) + m_vy/(ny*3);
				bpoints.push_back(bp2);

				uint4 vertices2 = {vpoint_index, vpoints.size() + i, vpoints.size() + i + 1, 0};
				vindexes.push_back(vertices2);
			}

			predef_vparts = edgeparts[0][0].size();
			if(predef_vparts)
			for(uint i=0; i<predef_vparts-1; i++) {
				uint vpoint_index = edgeparts[0][0][i];
				Point vp = vpoints[vpoint_index];

				Point bp1 = vp - 2*m_vx/(nx*3) + 2*m_vy/(ny*3);
				bpoints.push_back(bp1);

				uint4 vertices1 = {vpoints.size() + (endy-starty+1)*(endx-startx) + i, vpoints.size() + (endy-starty+1)*(endx-startx)+ i + 1, vpoint_index + 1, 0};
				vindexes.push_back(vertices1);

				Point bp2 = vp - m_vx/(nx*3) + m_vy/(ny*3);
				bpoints.push_back(bp2);

				uint4 vertices2 = {vpoints.size() + (endy-starty+1)*(endx-startx) + i, vpoint_index, vpoint_index + 1, 0};
				vindexes.push_back(vertices2);
			}

			for(uint i=0; i<4*(predef_vparts-1); i++)
				belems.push_back(belm);

			break;
		case 4: //bottom face
			startx++;
			starty++;
			endx--;
			endy--;
			belm.SetCoord(0, 0, 1);

			predef_vparts = edgeparts[3][1].size();
			if(predef_vparts)
			for(uint i=1; i<predef_vparts; i++) {
				uint vpoint0_index = edgeparts[3][1][i-1];
				uint vpoint1_index = edgeparts[3][1][i];
				Point vp = vpoints[vpoint1_index];

				Point bp1 = vp + 2*m_vx/(nx*3) + m_vy/(ny*3);
				bpoints.push_back(bp1);

				uint4 vertices1 = {vpoint1_index, vpoints.size() + (endy-starty) - i, vpoints.size() + (endy-starty) - i + 1, 0};
				vindexes.push_back(vertices1);

				Point bp2 = vp + m_vx/(nx*3) + 2*m_vy/(ny*3);
				bpoints.push_back(bp2);

				uint4 vertices2 = {vpoint0_index, vpoint1_index, vpoints.size() + (endy-starty) - i + 1, 0};
				vindexes.push_back(vertices2);
			}

			predef_vparts = edgeparts[2][1].size();
			if(predef_vparts)
			for(uint i=1; i<predef_vparts; i++) {
				uint vpoint0_index = edgeparts[2][1][i-1];
				uint vpoint1_index = edgeparts[2][1][i];
				Point vp = vpoints[vpoint1_index];

				Point bp1 = vp + 2*m_vx/(nx*3) - 2*m_vy/(ny*3);
				bpoints.push_back(bp1);

				uint4 vertices1 = {vpoint0_index, vpoints.size()-1 + (endy-starty+1)*(endx-startx-i+1), vpoints.size()-1 + (endy-starty+1)*(endx-startx-i+2), 0};
				if(i == predef_vparts-1)
					vertices1.y = edgeparts[3][1][0];
				vindexes.push_back(vertices1);

				Point bp2 = vp + m_vx/(nx*3) - 1*m_vy/(ny*3);
				bpoints.push_back(bp2);

				uint4 vertices2 = {vpoint0_index, vpoints.size()-1 + (endy-starty+1)*(endx-startx-i+1), vpoint1_index, 0};
				if(i == predef_vparts-1)
					vertices2.y = edgeparts[3][1][0];
				vindexes.push_back(vertices2);
			}

			predef_vparts = edgeparts[1][1].size();
			if(predef_vparts)
			for(uint i=1; i<predef_vparts; i++) {
				uint vpoint0_index = edgeparts[1][1][i-1];
				uint vpoint1_index = edgeparts[1][1][i];
				Point vp = vpoints[vpoint1_index];

				Point bp1 = vp - m_vx/(nx*3) - 2*m_vy/(ny*3);
				bpoints.push_back(bp1);

				uint4 vertices1 = {vpoint1_index, vpoint0_index, vpoints.size()-1 + (endy-starty+1)*(endx-startx) + i, 0};
				vindexes.push_back(vertices1);

				Point bp2 = vp - 2*m_vx/(nx*3) - m_vy/(ny*3);
				bpoints.push_back(bp2);

				uint4 vertices2 = {vpoint1_index,vpoints.size()-1 + (endy-starty+1)*(endx-startx) + i,vpoints.size()-1 + (endy-starty+1)*(endx-startx) + i + 1, 0};
				if(i == predef_vparts-1)
					vertices2.z = edgeparts[2][1][0];
				vindexes.push_back(vertices2);
			}

			predef_vparts = edgeparts[0][1].size();
			if(predef_vparts)
			for(uint i=1; i<predef_vparts; i++) {
				uint vpoint1_index = edgeparts[0][1][i];
				uint vpoint0_index = edgeparts[0][1][i-1];
				Point vp = vpoints[vpoint1_index];

				Point bp1 = vp - m_vx/(nx*3) + m_vy/(ny*3);
				bpoints.push_back(bp1);

				uint4 vertices1 = {vpoint0_index, vpoint1_index, vpoints.size() + (endy-starty+1)*(i-1), 0};
				if(i == predef_vparts-1)
					vertices1.z = edgeparts[1][1][0];
				vindexes.push_back(vertices1);

				Point bp2 = vp - 2*m_vx/(nx*3) + 2*m_vy/(ny*3);
				bpoints.push_back(bp2);

				uint4 vertices2 = {vpoint0_index, vpoints.size() + (endy-starty+1)*(i-2), vpoints.size() + (endy-starty+1)*(i-1), 0};
				if(i == 1)
					vertices2.y = edgeparts[3][1][endx-startx];
				if(i == predef_vparts-1)
					vertices2.z = edgeparts[1][1][0];
				vindexes.push_back(vertices2);
			}

			for(uint i=0; i<8*(predef_vparts-2); i++)
				belems.push_back(belm);

			break;
		case 5: //top face
			startx++;
			starty++;
			endx--;
			endy--;
			belm.SetCoord(0, 0, -1);

			break;
		}

	//Fill plane surface with boundary particles (except some near-edge regions)
	for (int i = startx; i <= endx; i++)
	for (int j = starty; j <= endy; j++) {
			Point vp = m_origin + i*m_vx/nx + j*m_vy/ny;
			int nvertex = vpoints.size();

			//Save vertex particles located at the edges of planes
			if (i == 0) {
				edgeparts[face_num][0].push_back(nvertex);
				vp(3) /= 2.0;
			}
			if (i == nx) {
				edgeparts[face_num][2].push_back(nvertex);
				vp(3) /= 2.0;
			}
			if (j == 0) {
				edgeparts[face_num][1].push_back(nvertex);
				vp(3) /= 2.0;
			}
			if (j == ny) {
				edgeparts[face_num][3].push_back(nvertex);
			}

			vpoints.push_back(vp);

			//Fill rectangular plane with boundary particles and set connectivity for them
			if (i != endx && j != endy) {
				Point bp1 = vp + m_vx/(nx*3) + 2*m_vy/(ny*3);
				bpoints.push_back(bp1);

				uint4 vertices1 = {nvertex, nvertex + 1, nvertex + (endy-starty) + 2, 0};
				vindexes.push_back(vertices1);

				Point bp2 = vp + 2*m_vx/(nx*3) + m_vy/(ny*3);
				bpoints.push_back(bp2);

				uint4 vertices2 = {nvertex, nvertex + (endy-starty) + 1, nvertex + (endy-starty) + 2, 0};
				vindexes.push_back(vertices2);

				belems.push_back(belm);
				belems.push_back(belm);
			}
	}

	return;
}

/// Fill a rectangle with layers of particles, from the surface
/// to the direction of the normal vector. Use a negative
/// value of layers to FillIn the opposite direction
void
Rect::FillIn(PointVect &points, const double dx, const int layers)
{
	int _layers = abs(layers);

	Vector unitshift(layers > 0 ? m_vz : -m_vz);
	unitshift.normalize();

	Fill(points, dx, true);

	// NOTE: pre-decrementing causes (_layers-1) layers to be filled. This
	// is correct since the first layer was already filled
	while (--_layers > 0) {
		Rect layer(m_origin + dx*_layers*unitshift, m_vx, m_vy);
		layer.SetPartMass(m_center(3));
		layer.Fill(points, dx, true);
	}
}

bool
Rect::IsInside(const Point& p, const double dx) const
{
	Point lp = m_ep.TransposeRot(p - m_origin);
	const double lx = m_lx + dx;
	const double ly = m_ly + dx;
	bool inside = false;
	if (lp(0) > -dx && lp(0) < lx && lp(1) > -dx && lp(1) < ly &&
		lp(2) > -dx && lp(2) < dx)
		inside = true;

	return inside;
}

void Rect::setEulerParameters(const EulerParameters &ep)
{
	m_ep = ep;
	m_ep.ComputeRot();

	m_vx = m_lx*m_ep.Rot(Vector(1, 0, 0));
	m_vy = m_ly*m_ep.Rot(Vector(0, 1, 0));
	m_vz = m_vx.cross(m_vy);
	m_vz.normalize();

	m_center = m_origin + m_ep.Rot(Vector(0.5*m_lx, 0.5*m_ly, 0.0));
}

void Rect::getBoundingBox(Point &output_min, Point &output_max)
{
	getBoundingBoxOfCube(output_min, output_max, m_origin,
		m_vx, m_vy, Vector(0, 0, 0) );
}

void Rect::shift(const double3 &offset)
{
	const Point poff = Point(offset);
	m_center += poff;
	m_origin += poff;
}
